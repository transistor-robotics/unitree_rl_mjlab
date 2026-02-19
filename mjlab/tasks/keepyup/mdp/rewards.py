"""Reward functions for the keepy up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def ball_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 1.0,
    std: float = 0.22,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Reward for keeping the ball near a target vertical height."""
    ball: Entity = env.scene[ball_cfg.name]
    ball_z = ball.data.root_link_pos_w[:, 2]
    height_error_sq = torch.square(ball_z - target_height)
    reward = torch.exp(-height_error_sq / (std**2))

    log = env.extras.setdefault("log", {})
    log["Metrics/ball_height_mean"] = ball_z.mean()
    return reward


class bounce_rhythm_reward:
    """Reward consistent timing between bounces
    Keep track of number of bounces
    Calculate consistency based on last N bounces
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params.get("sensor_name", "paddle_ball_contact")
        self.prev_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.steps_since_bounce = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
        self.ema_interval = torch.full(
            (env.num_envs,), 18.0, dtype=torch.float32, device=env.device
        )
        self.have_interval = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.prev_contact[:] = False
            self.steps_since_bounce[:] = 0
            self.ema_interval[:] = 18.0
            self.have_interval[:] = False
        else:
            self.prev_contact[env_ids] = False
            self.steps_since_bounce[env_ids] = 0
            self.ema_interval[env_ids] = 18.0
            self.have_interval[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "paddle_ball_contact",
        target_interval_steps: float | None = None,
        interval_std_steps: float = 4.0,
        ema_alpha: float = 0.2,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None

        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        bounce_completed = self.prev_contact & ~current_contact
        interval = self.steps_since_bounce.float().clamp(min=1.0)

        target_interval = (
            torch.full_like(interval, float(target_interval_steps))
            if target_interval_steps is not None
            else self.ema_interval
        )
        interval_error_sq = torch.square(interval - target_interval)
        interval_score = torch.exp(-interval_error_sq / (interval_std_steps**2))

        rewarded = bounce_completed & self.have_interval
        reward = torch.where(rewarded, interval_score, torch.zeros_like(interval_score))

        new_ema = (1.0 - ema_alpha) * self.ema_interval + ema_alpha * interval
        self.ema_interval = torch.where(bounce_completed, new_ema, self.ema_interval)
        self.have_interval = self.have_interval | bounce_completed
        self.steps_since_bounce = torch.where(
            bounce_completed,
            torch.zeros_like(self.steps_since_bounce),
            self.steps_since_bounce + 1,
        )
        self.prev_contact = current_contact

        log = env.extras.setdefault("log", {})
        log["Metrics/bounce_interval_ema_mean"] = self.ema_interval.mean()
        log["Metrics/bounce_rhythm_rewarded_events"] = rewarded.float().mean()
        return reward


class bounce_reward:
    """Reward for the ball hitting off the paddle and reversing its direction"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params.get("sensor_name", "paddle_ball_contact")
        self.prev_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.prev_contact[:] = False
        else:
            self.prev_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "paddle_ball_contact",
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
        min_upward_velocity: float = 0.1,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None

        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        bounce_completed = self.prev_contact & ~current_contact

        ball: Entity = env.scene[ball_cfg.name]
        upward_vz = torch.clamp(ball.data.root_link_lin_vel_w[:, 2], min=0.0)
        rewarded = bounce_completed & (upward_vz >= min_upward_velocity)
        reward = rewarded.float()

        self.prev_contact = current_contact

        log = env.extras.setdefault("log", {})
        log["Metrics/bounce_events_per_step"] = bounce_completed.float().mean()
        log["Metrics/rewarded_bounces_per_step"] = rewarded.float().mean()
        return reward


def ball_paddle_tracking_reward(
    env: ManagerBasedRlEnv,
    std_xy: float = 0.14,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Reward when paddle horizontal position tracks the ball."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    ids, _names = robot.find_sites("paddle_face")
    if len(ids) == 0:
        raise RuntimeError("paddle_face site not found on robot entity")
    paddle_site_idx = ids[0]

    paddle_pos = robot.data.site_pos_w[:, paddle_site_idx, :]
    ball_pos = ball.data.root_link_pos_w

    horizontal_dist = torch.norm(paddle_pos[:, :2] - ball_pos[:, :2], dim=-1)
    reward = torch.exp(-torch.square(horizontal_dist) / (std_xy**2))

    log = env.extras.setdefault("log", {})
    log["Metrics/ball_paddle_horizontal_dist_mean"] = horizontal_dist.mean()
    return reward


class paddle_height_consistency_reward:
    """Reward maintaining a consistent paddle Z (height) value over time
    Need to allow for small, sharp impulses when ball is being hit
    Apply smoothing?
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]
        self.sensor_name = cfg.params.get("sensor_name", "paddle_ball_contact")
        self.ema_height = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.impulse_cooldown = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
        self.prev_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.ema_height[:] = 0.0
            self.initialized[:] = False
            self.impulse_cooldown[:] = 0
            self.prev_contact[:] = False
        else:
            self.ema_height[env_ids] = 0.0
            self.initialized[env_ids] = False
            self.impulse_cooldown[env_ids] = 0
            self.prev_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "paddle_ball_contact",
        ema_alpha: float = 0.04,
        base_std: float = 0.035,
        impulse_std: float = 0.12,
        impulse_window_steps: int = 3,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]

        contact_sensor: ContactSensor = env.scene[sensor_name]
        assert contact_sensor.data.found is not None
        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        new_impulse = self.prev_contact & ~current_contact

        not_init = ~self.initialized
        if not_init.any():
            self.ema_height[not_init] = paddle_z[not_init]
            self.initialized[not_init] = True

        self.ema_height = (1.0 - ema_alpha) * self.ema_height + ema_alpha * paddle_z
        self.impulse_cooldown = torch.where(
            new_impulse,
            torch.full_like(self.impulse_cooldown, impulse_window_steps),
            torch.clamp(self.impulse_cooldown - 1, min=0),
        )

        effective_std = torch.where(
            self.impulse_cooldown > 0,
            torch.full_like(paddle_z, impulse_std),
            torch.full_like(paddle_z, base_std),
        )
        error_sq = torch.square(paddle_z - self.ema_height)
        reward = torch.exp(-error_sq / torch.square(effective_std))

        self.prev_contact = current_contact

        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_height_ema_mean"] = self.ema_height.mean()
        return reward


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
        reward = torch.clamp(
            (alignment - min_alignment) / (1.0 - min_alignment), min=0.0
        )

        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_up_alignment_mean"] = alignment.mean()
        return reward


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize robot self-collisions detected by a contact sensor."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found.squeeze(-1)


def paddle_robot_collision_cost(
    env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
    """Penalize collisions between the paddle geometry and robot body."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found.squeeze(-1)