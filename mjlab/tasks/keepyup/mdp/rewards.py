"""Reward functions for the keepy up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


class bounce_event_reward:
    """Reward for successful bounce events.
    
    Tracks paddle-ball contact state and detects bounce cycles:
    no_contact -> contact -> no_contact
    
    When a bounce completes, rewards based on the ball's upward velocity
    at the moment it separates from the paddle.
    """
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]
        self.max_reward_velocity = cfg.params.get("max_reward_velocity", 3.0)
        
        # Track contact state: 0 = no contact, 1 = in contact
        self.prev_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        ball_cfg: SceneEntityCfg,
        max_reward_velocity: float = 3.0,
        min_upward_velocity: float = 1.0,
    ) -> torch.Tensor:
        """Compute bounce event reward."""
        # Get contact sensor data
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None
        
        # Current contact state (any contact detected)
        current_contact = contact_sensor.data.found.squeeze(-1) > 0  # [B]
        
        # Detect bounce completion: was in contact, now not in contact
        bounce_completed = self.prev_contact & ~current_contact  # [B]
        
        # Get ball's vertical velocity at separation
        ball: Entity = env.scene[ball_cfg.name]
        ball_vel_z = ball.data.root_link_lin_vel_w[:, 2]  # [B]
        
        # Reward is proportional to upward velocity, clamped to max.
        # Require a minimum upward velocity so tiny "micro-bounces" are ignored.
        upward_vel = torch.clamp(ball_vel_z, 0.0, max_reward_velocity)
        valid_bounce = upward_vel >= min_upward_velocity
        
        # Apply reward only when bounce completes
        reward = torch.where(
            bounce_completed & valid_bounce, upward_vel, torch.zeros_like(upward_vel)
        )
        
        # Update state for next step
        self.prev_contact = current_contact
        
        # Log diagnostics every step
        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_contact_rate"] = current_contact.float().mean()
        log["Metrics/bounces_per_step"] = bounce_completed.float().mean()

        # Log bounce-quality metrics when bounce events occur
        num_bounces = torch.sum(bounce_completed.float())
        if num_bounces > 0:
            mean_bounce_vel = torch.sum(upward_vel * bounce_completed.float()) / num_bounces
            log["Metrics/bounce_velocity_mean"] = mean_bounce_vel
        
        return reward


def ball_alive(
    env: ManagerBasedRlEnv,
    max_distance: float = 1.2,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Reward for keeping the ball within a base-frame distance threshold.

    Returns 1.0 if the ball is within ``max_distance`` of the robot base frame,
    otherwise 0.0.
    """
    from mjlab.tasks.keepyup.mdp.observations import ball_pos_in_base_frame

    ball_pos_b = ball_pos_in_base_frame(env, robot_cfg, ball_cfg)  # [B, 3]
    ball_dist = torch.norm(ball_pos_b, dim=-1)  # [B]
    return (ball_dist <= max_distance).float()


class sustained_contact_penalty:
    """Penalty for sustaining contact with the ball (anti-balancing).
    
    Tracks consecutive steps where paddle-ball contact is active.
    Applies an escalating penalty if contact persists beyond a threshold.
    """
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]
        self.threshold = cfg.params.get("threshold", 10)
        
        # Track consecutive contact steps per environment
        self.consecutive_contact = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
        
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        threshold: int = 10,
    ) -> torch.Tensor:
        """Compute sustained contact penalty."""
        # Get contact sensor data
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None
        
        # Current contact state
        in_contact = contact_sensor.data.found.squeeze(-1) > 0  # [B]
        
        # Update consecutive contact counter
        self.consecutive_contact = torch.where(
            in_contact,
            self.consecutive_contact + 1,
            torch.zeros_like(self.consecutive_contact),
        )
        
        # Penalty is proportional to how many steps beyond threshold
        penalty = torch.clamp(
            self.consecutive_contact - threshold,
            min=0,
        ).float()
        
        # Log metrics
        max_consecutive = torch.max(self.consecutive_contact.float())
        mean_consecutive = torch.mean(self.consecutive_contact.float())
        env.extras["log"]["Metrics/max_consecutive_contact"] = max_consecutive
        env.extras["log"]["Metrics/mean_consecutive_contact"] = mean_consecutive
        
        return penalty


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize robot self-collisions detected by a contact sensor."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found.squeeze(-1)


def ball_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Reward for maintaining ball at target height.
    
    Exponential reward centered at target height.
    """
    ball: Entity = env.scene[ball_cfg.name]
    ball_z = ball.data.root_link_pos_w[:, 2]  # [B]
    
    height_error = torch.square(ball_z - target_height)
    return torch.exp(-height_error / (std ** 2))


class paddle_height_ceiling_penalty:
    """Soft penalty when paddle rises above a height ceiling.

    This discourages "carrying" behavior where the policy follows the ball upward
    with the paddle instead of producing cleaner impulse-like contacts.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_paddle_height: float = 1.05,
        deadband: float = 0.02,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]  # [B]
        excess = torch.clamp(paddle_z - (max_paddle_height + deadband), min=0.0)
        return excess


class paddle_face_up_reward:
    """Reward keeping paddle face roughly upward (parallel to floor)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_geoms("paddle_geom")
        if len(ids) == 0:
            raise RuntimeError("paddle_geom not found on robot entity")
        self._paddle_geom_idx: int = ids[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        min_alignment: float = 0.85,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]

        # For mjGEOM_CYLINDER, local +Z is the face normal.
        paddle_quat_w = robot.data.geom_quat_w[:, self._paddle_geom_idx, :]  # [B, 4]
        local_normal = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(
            env.num_envs, 1
        )
        normal_w = quat_apply(paddle_quat_w, local_normal)  # [B, 3]

        # Reward face-up alignment (normal close to +Z), with dead-zone below threshold.
        alignment = normal_w[:, 2]  # dot(normal_w, world_up)
        reward = torch.clamp((alignment - min_alignment) / (1.0 - min_alignment), min=0.0)

        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_up_alignment_mean"] = alignment.mean()
        return reward


class paddle_ball_distance:
    """Reward for keeping the paddle close to the ball.

    Provides smooth gradient signal via ``exp(-dist^2 / std^2)`` so the
    policy is guided toward the ball even before any contact occurs.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float = 0.2,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]

        paddle_pos_w = robot.data.site_pos_w[:, self._paddle_site_idx, :]  # [B, 3]
        ball_pos_w = ball.data.root_link_pos_w  # [B, 3]

        distance = torch.norm(ball_pos_w - paddle_pos_w, dim=-1)  # [B]
        # Log diagnostic distance statistics so we can verify spawn/proximity.
        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_ball_dist_mean"] = distance.mean()
        log["Metrics/paddle_ball_dist_min"] = distance.min()
        return torch.exp(-torch.square(distance) / (std ** 2))
