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
        self.steps_since_rewarded_bounce = torch.full(
            (env.num_envs,), 10_000, dtype=torch.int32, device=env.device
        )
        
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.prev_contact[:] = False
            self.steps_since_rewarded_bounce[:] = 10_000
        else:
            self.prev_contact[env_ids] = False
            self.steps_since_rewarded_bounce[env_ids] = 10_000

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        ball_cfg: SceneEntityCfg,
        max_reward_velocity: float = 3.0,
        min_upward_velocity: float = 1.0,
        min_apex_height: float = 1.15,
        min_apex_gain: float = 0.20,
        min_reward_interval_steps: int = 8,
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
        ball_pos_z = ball.data.root_link_pos_w[:, 2]  # [B]
        
        # Reward is proportional to upward velocity, clamped to max.
        # Require a minimum upward velocity so tiny "micro-bounces" are ignored.
        upward_vel = torch.clamp(ball_vel_z, 0.0, max_reward_velocity)
        valid_bounce = upward_vel >= min_upward_velocity
        predicted_apex = ball_pos_z + torch.square(upward_vel) / (2.0 * 9.81)
        apex_ok = (predicted_apex >= min_apex_height) & (
            (predicted_apex - ball_pos_z) >= min_apex_gain
        )
        cooldown_ok = self.steps_since_rewarded_bounce >= min_reward_interval_steps
        
        # Apply reward only for meaningful, separated bounce events.
        rewarded_bounce = bounce_completed & valid_bounce & apex_ok & cooldown_ok
        reward = torch.where(rewarded_bounce, upward_vel, torch.zeros_like(upward_vel))
        
        # Update state for next step
        self.prev_contact = current_contact
        self.steps_since_rewarded_bounce = torch.where(
            rewarded_bounce,
            torch.zeros_like(self.steps_since_rewarded_bounce),
            self.steps_since_rewarded_bounce + 1,
        )
        
        # Log diagnostics every step
        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_contact_rate"] = current_contact.float().mean()
        log["Metrics/bounces_per_step"] = bounce_completed.float().mean()
        log["Metrics/rewarded_bounces_per_step"] = rewarded_bounce.float().mean()
        log["Metrics/predicted_apex_mean"] = predicted_apex.mean()

        # Log bounce-quality metrics when bounce events occur
        num_bounces = torch.sum(rewarded_bounce.float())
        if num_bounces > 0:
            mean_bounce_vel = (
                torch.sum(upward_vel * rewarded_bounce.float()) / num_bounces
            )
            log["Metrics/bounce_velocity_mean"] = mean_bounce_vel
        
        return reward


class bounce_quality_reward:
    """Reward quality of each bounce based on apex and upward velocity targets."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]
        self.prev_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.steps_since_rewarded = torch.full(
            (env.num_envs,), 10_000, dtype=torch.int32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.prev_contact[:] = False
            self.steps_since_rewarded[:] = 10_000
        else:
            self.prev_contact[env_ids] = False
            self.steps_since_rewarded[env_ids] = 10_000

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
        target_apex_height: float = 1.42,
        apex_std: float = 0.12,
        target_upward_velocity: float = 1.75,
        velocity_std: float = 0.40,
        min_upward_velocity: float = 0.4,
        min_reward_interval_steps: int = 8,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None

        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        bounce_completed = self.prev_contact & ~current_contact

        ball: Entity = env.scene[ball_cfg.name]
        ball_z = ball.data.root_link_pos_w[:, 2]
        ball_vz = ball.data.root_link_lin_vel_w[:, 2]
        upward_vz = torch.clamp(ball_vz, min=0.0)
        predicted_apex = ball_z + torch.square(upward_vz) / (2.0 * 9.81)

        apex_score = torch.exp(-torch.square(predicted_apex - target_apex_height) / (apex_std ** 2))
        vel_score = torch.exp(
            -torch.square(upward_vz - target_upward_velocity) / (velocity_std ** 2)
        )
        quality = apex_score * vel_score

        cooldown_ok = self.steps_since_rewarded >= min_reward_interval_steps
        valid = upward_vz >= min_upward_velocity
        rewarded = bounce_completed & cooldown_ok & valid
        reward = torch.where(rewarded, quality, torch.zeros_like(quality))

        self.prev_contact = current_contact
        self.steps_since_rewarded = torch.where(
            rewarded, torch.zeros_like(self.steps_since_rewarded), self.steps_since_rewarded + 1
        )

        log = env.extras.setdefault("log", {})
        log["Metrics/predicted_apex_mean"] = predicted_apex.mean()
        log["Metrics/bounce_quality_mean"] = quality.mean()
        log["Metrics/rewarded_bounce_quality_mean"] = torch.where(
            rewarded, quality, torch.zeros_like(quality)
        ).sum() / torch.clamp(rewarded.float().sum(), min=1.0)
        return reward


class bounce_discovery_reward:
    """Reward any upward rebound event, with curriculum-tightened thresholds."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.prev_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.steps_since_rewarded = torch.full(
            (env.num_envs,), 10_000, dtype=torch.int32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.prev_contact[:] = False
            self.steps_since_rewarded[:] = 10_000
        else:
            self.prev_contact[env_ids] = False
            self.steps_since_rewarded[env_ids] = 10_000

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
        min_upward_velocity: float = 0.08,
        min_apex_height: float = 0.95,
        min_apex_gain: float = 0.04,
        target_upward_velocity: float = 1.0,
        min_reward_interval_steps: int = 3,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None

        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        bounce_completed = self.prev_contact & ~current_contact

        ball: Entity = env.scene[ball_cfg.name]
        ball_z = ball.data.root_link_pos_w[:, 2]
        ball_vel = ball.data.root_link_lin_vel_w
        ball_vz = ball_vel[:, 2]
        upward_vz = torch.clamp(ball_vz, min=0.0)
        predicted_apex = ball_z + torch.square(upward_vz) / (2.0 * 9.81)
        apex_gain = predicted_apex - ball_z

        cooldown_ok = self.steps_since_rewarded >= min_reward_interval_steps
        valid = (
            (upward_vz >= min_upward_velocity)
            & (predicted_apex >= min_apex_height)
            & (apex_gain >= min_apex_gain)
        )
        rewarded = bounce_completed & valid & cooldown_ok

        # Provide a gentle shaping signal from weak to stronger rebounds.
        vel_scale = torch.clamp(
            (upward_vz - min_upward_velocity) / max(target_upward_velocity - min_upward_velocity, 1e-6),
            min=0.0,
            max=1.0,
        )
        reward = torch.where(rewarded, vel_scale, torch.zeros_like(vel_scale))

        self.prev_contact = current_contact
        self.steps_since_rewarded = torch.where(
            rewarded, torch.zeros_like(self.steps_since_rewarded), self.steps_since_rewarded + 1
        )

        log = env.extras.setdefault("log", {})
        log["Metrics/discovery_bounce_events"] = bounce_completed.float().mean()
        log["Metrics/discovery_rewarded_events"] = rewarded.float().mean()
        log["Metrics/discovery_predicted_apex_mean"] = predicted_apex.mean()
        return reward


class lateral_drift_penalty:
    """Penalize lateral ball speed shortly after contact release."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.prev_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.post_bounce_window = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.prev_contact[:] = False
            self.post_bounce_window[:] = 0
        else:
            self.prev_contact[env_ids] = False
            self.post_bounce_window[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
        post_bounce_window_steps: int = 5,
        deadband: float = 0.05,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None
        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        bounce_completed = self.prev_contact & ~current_contact

        # Open a short phase window immediately after release, then decay.
        new_window = torch.full_like(self.post_bounce_window, post_bounce_window_steps)
        decayed_window = torch.clamp(self.post_bounce_window - 1, min=0)
        self.post_bounce_window = torch.where(bounce_completed, new_window, decayed_window)
        active = self.post_bounce_window > 0

        ball: Entity = env.scene[ball_cfg.name]
        vxy = torch.norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1)
        penalty = torch.clamp(vxy - deadband, min=0.0)

        self.prev_contact = current_contact

        log = env.extras.setdefault("log", {})
        log["Metrics/lateral_speed_mean"] = vxy.mean()
        log["Metrics/lateral_penalty_active_rate"] = active.float().mean()
        return torch.where(active, penalty, torch.zeros_like(penalty))


class under_ball_alignment_reward:
    """Reward paddle XY alignment under a descending ball near strike zone."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_xy: float = 0.12,
        min_descending_speed: float = 0.05,
        strike_zone_z_min: float = 0.82,
        strike_zone_z_max: float = 1.22,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]

        paddle_pos = robot.data.site_pos_w[:, self._paddle_site_idx, :]
        ball_pos = ball.data.root_link_pos_w
        ball_vz = ball.data.root_link_lin_vel_w[:, 2]

        descending = ball_vz < -min_descending_speed
        in_zone = (ball_pos[:, 2] >= strike_zone_z_min) & (ball_pos[:, 2] <= strike_zone_z_max)
        active = descending & in_zone

        xy_dist = torch.norm(ball_pos[:, :2] - paddle_pos[:, :2], dim=-1)
        score = torch.exp(-torch.square(xy_dist) / (std_xy ** 2))
        return torch.where(active, score, torch.zeros_like(score))


class strike_plane_hold_reward:
    """Reward staying near a strike plane while the ball is not in strike phase."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_paddle_height: float = 0.95,
        std: float = 0.08,
        ascending_vz_threshold: float = 0.05,
        far_descending_height: float = 1.25,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]

        paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]
        ball_z = ball.data.root_link_pos_w[:, 2]
        ball_vz = ball.data.root_link_lin_vel_w[:, 2]

        # Active when ball is rising or still high before descending into strike zone.
        active = (ball_vz > ascending_vz_threshold) | (ball_z > far_descending_height)
        score = torch.exp(-torch.square(paddle_z - target_paddle_height) / (std ** 2))
        return torch.where(active, score, torch.zeros_like(score))


class upward_chase_penalty:
    """Penalize upward paddle motion while the ball is ascending."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]
        self._prev_paddle_z = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._initialized[:] = False
            self._prev_paddle_z[:] = 0.0
        else:
            self._initialized[env_ids] = False
            self._prev_paddle_z[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        ball_ascending_threshold: float = 0.05,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]

        paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]
        ball_vz = ball.data.root_link_lin_vel_w[:, 2]

        not_init = ~self._initialized
        if not_init.any():
            self._prev_paddle_z[not_init] = paddle_z[not_init]
            self._initialized[not_init] = True

        paddle_vz = (paddle_z - self._prev_paddle_z) / env.step_dt
        self._prev_paddle_z = paddle_z.clone()

        active = ball_vz > ball_ascending_threshold
        penalty = torch.clamp(paddle_vz, min=0.0)
        return torch.where(active, penalty, torch.zeros_like(penalty))


class apex_clearance_target_reward:
    """Reward a target ball-paddle clearance around ball apex."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_clearance: float = 0.45,
        std: float = 0.10,
        apex_vz_window: float = 0.18,
        min_ball_height: float = 1.00,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]

        paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]
        ball_z = ball.data.root_link_pos_w[:, 2]
        ball_vz = ball.data.root_link_lin_vel_w[:, 2]

        clearance = ball_z - paddle_z
        near_apex = (torch.abs(ball_vz) <= apex_vz_window) & (ball_z >= min_ball_height)
        score = torch.exp(-torch.square(clearance - target_clearance) / (std ** 2))
        return torch.where(near_apex, score, torch.zeros_like(score))


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
        
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.consecutive_contact[:] = 0
        else:
            self.consecutive_contact[env_ids] = 0

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
        log = env.extras.setdefault("log", {})
        log["Metrics/max_consecutive_contact"] = max_consecutive
        log["Metrics/mean_consecutive_contact"] = mean_consecutive
        
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


def ball_height_above_ceiling_penalty(
    env: ManagerBasedRlEnv,
    ceiling_height: float = 1.55,
    deadband: float = 0.02,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Soft penalty when the ball flies too high above a ceiling."""
    ball: Entity = env.scene[ball_cfg.name]
    ball_z = ball.data.root_link_pos_w[:, 2]  # [B]
    return torch.clamp(ball_z - (ceiling_height + deadband), min=0.0)


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
