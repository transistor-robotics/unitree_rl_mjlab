"""Reward functions for the keepy up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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

class ball_trajectory_consistency_reward:
    """
    Reward keeping a consistent straight up/down trajectory.

    The reward combines:
      1) Verticality: low horizontal velocity relative to total speed.
      2) Line consistency: ball stays close to a smoothed XY trajectory centerline.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.ball_name = cfg.params.get("ball_cfg", _DEFAULT_BALL_CFG).name
        self.ema_xy = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.ema_xy[:] = 0.0
            self.initialized[:] = False
        else:
            self.ema_xy[env_ids] = 0.0
            self.initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
        vel_xy_std: float = 0.55,
        pos_xy_std: float = 0.09,
        ema_alpha: float = 0.06,
        min_speed: float = 0.15,
        verticality_weight: float = 0.6,
    ) -> torch.Tensor:
        ball: Entity = env.scene[self.ball_name if self.ball_name else ball_cfg.name]
        ball_pos_xy = ball.data.root_link_pos_w[:, :2]
        ball_vel = ball.data.root_link_lin_vel_w
        speed = torch.norm(ball_vel, dim=-1)
        vel_xy = torch.norm(ball_vel[:, :2], dim=-1)

        # Initialize EMA at first call after reset so initial error is zero.
        not_init = ~self.initialized
        if not_init.any():
            self.ema_xy[not_init] = ball_pos_xy[not_init]
            self.initialized[not_init] = True

        pos_xy_error = torch.norm(ball_pos_xy - self.ema_xy, dim=-1)
        line_consistency = torch.exp(-torch.square(pos_xy_error) / (pos_xy_std**2))
        verticality = torch.exp(-torch.square(vel_xy) / (vel_xy_std**2))

        # Avoid rewarding near-static apex phases where direction is undefined.
        active = (speed >= min_speed).float()
        reward = (
            verticality_weight * verticality + (1.0 - verticality_weight) * line_consistency
        ) * active

        self.ema_xy = (1.0 - ema_alpha) * self.ema_xy + ema_alpha * ball_pos_xy

        log = env.extras.setdefault("log", {})
        log["Metrics/ball_traj_xy_error_mean"] = pos_xy_error.mean()
        log["Metrics/ball_traj_xy_speed_mean"] = vel_xy.mean()
        log["Metrics/ball_traj_active_ratio"] = active.mean()
        return reward

class paddle_height_consistency_reward:
    """Reward for keeping paddle height near a fixed target."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        robot: Entity = env.scene["robot"]
        ids, _names = robot.find_sites("paddle_face")
        if len(ids) == 0:
            raise RuntimeError("paddle_face site not found on robot entity")
        self._paddle_site_idx: int = ids[0]
        self.sensor_name = cfg.params.get("sensor_name", "paddle_ball_contact")
        self.impulse_cooldown = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
        self.prev_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self.impulse_cooldown[:] = 0
            self.prev_contact[:] = False
        else:
            self.impulse_cooldown[env_ids] = 0
            self.prev_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        target_height: float = 0.8,
        base_std: float = 0.08,
        impulse_std: float = 0.16,
        impulse_window_steps: int = 3,
        sensor_name: str | None = None,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ) -> torch.Tensor:
        robot: Entity = env.scene[robot_cfg.name]
        paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]

        contact_sensor: ContactSensor = env.scene[sensor_name or self.sensor_name]
        assert contact_sensor.data.found is not None
        current_contact = contact_sensor.data.found.squeeze(-1) > 0
        new_impulse = self.prev_contact & ~current_contact

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

        height_error_sq = torch.square(paddle_z - target_height)
        reward = torch.exp(-height_error_sq / torch.square(effective_std))
        self.prev_contact = current_contact

        log = env.extras.setdefault("log", {})
        log["Metrics/paddle_height_mean"] = paddle_z.mean()
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