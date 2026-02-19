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
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
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


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize robot self-collisions detected by a contact sensor."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found.squeeze(-1)


def paddle_robot_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize collisions between the paddle geometry and robot body."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found.squeeze(-1)


# class bounce_event_reward:
#     """Reward for successful bounce events.

#     Tracks paddle-ball contact state and detects bounce cycles:
#     no_contact -> contact -> no_contact

#     When a bounce completes, rewards based on the ball's upward velocity
#     at the moment it separates from the paddle.
#     """

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         self.sensor_name = cfg.params["sensor_name"]
#         self.max_reward_velocity = cfg.params.get("max_reward_velocity", 3.0)

#         # Track contact state: 0 = no contact, 1 = in contact
#         self.prev_contact = torch.zeros(
#             env.num_envs, dtype=torch.bool, device=env.device
#         )
#         self.steps_since_rewarded_bounce = torch.full(
#             (env.num_envs,), 10_000, dtype=torch.int32, device=env.device
#         )

#     def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#         if env_ids is None:
#             self.prev_contact[:] = False
#             self.steps_since_rewarded_bounce[:] = 10_000
#         else:
#             self.prev_contact[env_ids] = False
#             self.steps_since_rewarded_bounce[env_ids] = 10_000

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         sensor_name: str,
#         ball_cfg: SceneEntityCfg,
#         max_reward_velocity: float = 3.0,
#         min_upward_velocity: float = 1.0,
#         min_apex_height: float = 1.15,
#         min_apex_gain: float = 0.20,
#         min_reward_interval_steps: int = 8,
#     ) -> torch.Tensor:
#         """Compute bounce event reward."""
#         # Get contact sensor data
#         contact_sensor: ContactSensor = env.scene[sensor_name]
#         assert contact_sensor.data.found is not None

#         # Current contact state (any contact detected)
#         current_contact = contact_sensor.data.found.squeeze(-1) > 0  # [B]

#         # Detect bounce completion: was in contact, now not in contact
#         bounce_completed = self.prev_contact & ~current_contact  # [B]

#         # Get ball's vertical velocity at separation
#         ball: Entity = env.scene[ball_cfg.name]
#         ball_vel_z = ball.data.root_link_lin_vel_w[:, 2]  # [B]
#         ball_pos_z = ball.data.root_link_pos_w[:, 2]  # [B]

#         # Reward is proportional to upward velocity, clamped to max.
#         # Require a minimum upward velocity so tiny "micro-bounces" are ignored.
#         upward_vel = torch.clamp(ball_vel_z, 0.0, max_reward_velocity)
#         valid_bounce = upward_vel >= min_upward_velocity
#         predicted_apex = ball_pos_z + torch.square(upward_vel) / (2.0 * 9.81)
#         apex_ok = (predicted_apex >= min_apex_height) & (
#             (predicted_apex - ball_pos_z) >= min_apex_gain
#         )
#         cooldown_ok = self.steps_since_rewarded_bounce >= min_reward_interval_steps

#         # Apply reward only for meaningful, separated bounce events.
#         rewarded_bounce = bounce_completed & valid_bounce & apex_ok & cooldown_ok
#         reward = torch.where(rewarded_bounce, upward_vel, torch.zeros_like(upward_vel))

#         # Update state for next step
#         self.prev_contact = current_contact
#         self.steps_since_rewarded_bounce = torch.where(
#             rewarded_bounce,
#             torch.zeros_like(self.steps_since_rewarded_bounce),
#             self.steps_since_rewarded_bounce + 1,
#         )

#         # Log diagnostics every step
#         log = env.extras.setdefault("log", {})
#         log["Metrics/paddle_contact_rate"] = current_contact.float().mean()
#         log["Metrics/bounces_per_step"] = bounce_completed.float().mean()
#         log["Metrics/rewarded_bounces_per_step"] = rewarded_bounce.float().mean()
#         log["Metrics/predicted_apex_mean"] = predicted_apex.mean()

#         # Log bounce-quality metrics when bounce events occur
#         num_bounces = torch.sum(rewarded_bounce.float())
#         if num_bounces > 0:
#             mean_bounce_vel = (
#                 torch.sum(upward_vel * rewarded_bounce.float()) / num_bounces
#             )
#             log["Metrics/bounce_velocity_mean"] = mean_bounce_vel

#         return reward


# class bounce_quality_reward:
#     """Reward quality of each bounce based on apex and upward velocity targets."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         self.sensor_name = cfg.params["sensor_name"]
#         self.prev_contact = torch.zeros(
#             env.num_envs, dtype=torch.bool, device=env.device
#         )
#         self.steps_since_rewarded = torch.full(
#             (env.num_envs,), 10_000, dtype=torch.int32, device=env.device
#         )

#     def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#         if env_ids is None:
#             self.prev_contact[:] = False
#             self.steps_since_rewarded[:] = 10_000
#         else:
#             self.prev_contact[env_ids] = False
#             self.steps_since_rewarded[env_ids] = 10_000

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         sensor_name: str,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#         target_apex_height: float = 1.42,
#         apex_std: float = 0.50,
#         target_upward_velocity: float = 1.75,
#         velocity_std: float = 0.60,
#         vel_weight: float = 0.0,
#         vert_std: float = 2.0,
#         vert_weight: float = 0.0,
#         min_upward_velocity: float = 0.05,
#         min_reward_interval_steps: int = 5,
#     ) -> torch.Tensor:
#         """Compute bounce quality reward.

#         Three curriculum-controlled knobs blend in over training:

#         - ``apex_std``: How close the predicted apex must be to ``target_apex_height``.
#           Wide early (forgiving), narrows later (strict). The apex score is always
#           the primary driver.

#         - ``vel_weight`` + ``velocity_std``: How much upward speed matters. At
#           ``vel_weight=0`` the velocity Gaussian is ignored entirely; at 1 it
#           contributes fully. Blended as: ``lerp(1.0, vel_score, vel_weight)``.

#         - ``vert_weight`` + ``vert_std``: How vertical the ball's trajectory must be.
#           Scored via lateral speed ``vxy = sqrt(vx^2 + vy^2)`` at separation — a
#           perfectly vertical bounce has vxy=0 and scores 1.0. At ``vert_weight=0``
#           this is completely ignored; at 1 it fully penalises sideways deflections.
#           Blended as: ``lerp(1.0, vert_score, vert_weight)``.

#         Combined: ``quality = apex_score * lerp(1, vel_score, vel_weight)
#                                            * lerp(1, vert_score, vert_weight)``
#         """
#         contact_sensor: ContactSensor = env.scene[sensor_name]
#         assert contact_sensor.data.found is not None

#         current_contact = contact_sensor.data.found.squeeze(-1) > 0
#         bounce_completed = self.prev_contact & ~current_contact

#         ball: Entity = env.scene[ball_cfg.name]
#         ball_vel_w = ball.data.root_link_lin_vel_w
#         ball_z = ball.data.root_link_pos_w[:, 2]
#         ball_vz = ball_vel_w[:, 2]
#         ball_vxy = torch.norm(ball_vel_w[:, :2], dim=-1)

#         upward_vz = torch.clamp(ball_vz, min=0.0)
#         predicted_apex = ball_z + torch.square(upward_vz) / (2.0 * 9.81)

#         # Primary: how close will the ball apex to the target height?
#         apex_score = torch.exp(
#             -torch.square(predicted_apex - target_apex_height) / (apex_std**2)
#         )

#         # Secondary (curriculum-blended): how fast is the ball going upward?
#         vel_score = torch.exp(
#             -torch.square(upward_vz - target_upward_velocity) / (velocity_std**2)
#         )
#         vel_contrib = vel_weight * vel_score + (1.0 - vel_weight)

#         # Tertiary (curriculum-blended): how vertical is the trajectory?
#         vert_score = torch.exp(-torch.square(ball_vxy) / (vert_std**2))
#         vert_contrib = vert_weight * vert_score + (1.0 - vert_weight)

#         quality = apex_score * vel_contrib * vert_contrib

#         cooldown_ok = self.steps_since_rewarded >= min_reward_interval_steps
#         valid = upward_vz >= min_upward_velocity
#         rewarded = bounce_completed & cooldown_ok & valid
#         reward = torch.where(rewarded, quality, torch.zeros_like(quality))

#         self.prev_contact = current_contact
#         self.steps_since_rewarded = torch.where(
#             rewarded,
#             torch.zeros_like(self.steps_since_rewarded),
#             self.steps_since_rewarded + 1,
#         )

#         log = env.extras.setdefault("log", {})
#         log["Metrics/predicted_apex_mean"] = predicted_apex.mean()
#         log["Metrics/ball_vxy_mean"] = ball_vxy.mean()
#         log["Metrics/bounce_events_per_step"] = bounce_completed.float().mean()
#         log["Metrics/rewarded_bounces_per_step"] = rewarded.float().mean()
#         log["Metrics/rewarded_bounce_quality_mean"] = torch.where(
#             rewarded, quality, torch.zeros_like(quality)
#         ).sum() / torch.clamp(rewarded.float().sum(), min=1.0)
#         return reward


# class bounce_discovery_reward:
#     """Reward any upward rebound event, with curriculum-tightened thresholds."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         self.prev_contact = torch.zeros(
#             env.num_envs, dtype=torch.bool, device=env.device
#         )
#         self.steps_since_rewarded = torch.full(
#             (env.num_envs,), 10_000, dtype=torch.int32, device=env.device
#         )

#     def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#         if env_ids is None:
#             self.prev_contact[:] = False
#             self.steps_since_rewarded[:] = 10_000
#         else:
#             self.prev_contact[env_ids] = False
#             self.steps_since_rewarded[env_ids] = 10_000

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         sensor_name: str,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#         min_upward_velocity: float = 0.08,
#         min_apex_height: float = 0.95,
#         min_apex_gain: float = 0.04,
#         target_upward_velocity: float = 1.0,
#         min_reward_interval_steps: int = 3,
#     ) -> torch.Tensor:
#         contact_sensor: ContactSensor = env.scene[sensor_name]
#         assert contact_sensor.data.found is not None

#         current_contact = contact_sensor.data.found.squeeze(-1) > 0
#         bounce_completed = self.prev_contact & ~current_contact

#         ball: Entity = env.scene[ball_cfg.name]
#         ball_z = ball.data.root_link_pos_w[:, 2]
#         ball_vel = ball.data.root_link_lin_vel_w
#         ball_vz = ball_vel[:, 2]
#         upward_vz = torch.clamp(ball_vz, min=0.0)
#         predicted_apex = ball_z + torch.square(upward_vz) / (2.0 * 9.81)
#         apex_gain = predicted_apex - ball_z

#         cooldown_ok = self.steps_since_rewarded >= min_reward_interval_steps
#         valid = (
#             (upward_vz >= min_upward_velocity)
#             & (predicted_apex >= min_apex_height)
#             & (apex_gain >= min_apex_gain)
#         )
#         rewarded = bounce_completed & valid & cooldown_ok

#         # Provide a gentle shaping signal from weak to stronger rebounds.
#         vel_scale = torch.clamp(
#             (upward_vz - min_upward_velocity)
#             / max(target_upward_velocity - min_upward_velocity, 1e-6),
#             min=0.0,
#             max=1.0,
#         )
#         reward = torch.where(rewarded, vel_scale, torch.zeros_like(vel_scale))

#         self.prev_contact = current_contact
#         self.steps_since_rewarded = torch.where(
#             rewarded,
#             torch.zeros_like(self.steps_since_rewarded),
#             self.steps_since_rewarded + 1,
#         )

#         log = env.extras.setdefault("log", {})
#         log["Metrics/discovery_bounce_events"] = bounce_completed.float().mean()
#         log["Metrics/discovery_rewarded_events"] = rewarded.float().mean()
#         log["Metrics/discovery_predicted_apex_mean"] = predicted_apex.mean()
#         return reward


# class under_ball_alignment_reward:
#     """Reward paddle XY alignment under a descending ball near strike zone."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_sites("paddle_face")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_face site not found on robot entity")
#         self._paddle_site_idx: int = ids[0]

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         std_xy: float = 0.12,
#         min_descending_speed: float = 0.05,
#         strike_zone_z_min: float = 0.82,
#         strike_zone_z_max: float = 1.22,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]
#         ball: Entity = env.scene[ball_cfg.name]

#         paddle_pos = robot.data.site_pos_w[:, self._paddle_site_idx, :]
#         ball_pos = ball.data.root_link_pos_w
#         ball_vz = ball.data.root_link_lin_vel_w[:, 2]

#         descending = ball_vz < -min_descending_speed
#         in_zone = (ball_pos[:, 2] >= strike_zone_z_min) & (
#             ball_pos[:, 2] <= strike_zone_z_max
#         )
#         active = descending & in_zone

#         xy_dist = torch.norm(ball_pos[:, :2] - paddle_pos[:, :2], dim=-1)
#         score = torch.exp(-torch.square(xy_dist) / (std_xy**2))
#         return torch.where(active, score, torch.zeros_like(score))


# class strike_plane_hold_reward:
#     """Reward staying near a strike plane while the ball is not in strike phase."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_sites("paddle_face")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_face site not found on robot entity")
#         self._paddle_site_idx: int = ids[0]

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         target_paddle_height: float = 0.95,
#         std: float = 0.08,
#         ascending_vz_threshold: float = 0.05,
#         far_descending_height: float = 1.25,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]
#         ball: Entity = env.scene[ball_cfg.name]

#         paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]
#         ball_z = ball.data.root_link_pos_w[:, 2]
#         ball_vz = ball.data.root_link_lin_vel_w[:, 2]

#         # Active when ball is rising or still high before descending into strike zone.
#         active = (ball_vz > ascending_vz_threshold) | (ball_z > far_descending_height)
#         score = torch.exp(-torch.square(paddle_z - target_paddle_height) / (std**2))
#         return torch.where(active, score, torch.zeros_like(score))


# class upward_chase_penalty:
#     """Penalize upward paddle motion while the ball is ascending."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_sites("paddle_face")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_face site not found on robot entity")
#         self._paddle_site_idx: int = ids[0]
#         self._prev_paddle_z = torch.zeros(env.num_envs, device=env.device)
#         self._initialized = torch.zeros(
#             env.num_envs, dtype=torch.bool, device=env.device
#         )

#     def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#         if env_ids is None:
#             self._initialized[:] = False
#             self._prev_paddle_z[:] = 0.0
#         else:
#             self._initialized[env_ids] = False
#             self._prev_paddle_z[env_ids] = 0.0

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         ball_ascending_threshold: float = 0.05,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]
#         ball: Entity = env.scene[ball_cfg.name]

#         paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]
#         ball_vz = ball.data.root_link_lin_vel_w[:, 2]

#         not_init = ~self._initialized
#         if not_init.any():
#             self._prev_paddle_z[not_init] = paddle_z[not_init]
#             self._initialized[not_init] = True

#         paddle_vz = (paddle_z - self._prev_paddle_z) / env.step_dt
#         self._prev_paddle_z = paddle_z.clone()

#         active = ball_vz > ball_ascending_threshold
#         penalty = torch.clamp(paddle_vz, min=0.0)
#         return torch.where(active, penalty, torch.zeros_like(penalty))


# class apex_clearance_target_reward:
#     """Reward a target ball-paddle clearance around ball apex."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_sites("paddle_face")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_face site not found on robot entity")
#         self._paddle_site_idx: int = ids[0]

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         target_clearance: float = 0.45,
#         std: float = 0.10,
#         apex_vz_window: float = 0.18,
#         min_ball_height: float = 1.00,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]
#         ball: Entity = env.scene[ball_cfg.name]

#         paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]
#         ball_z = ball.data.root_link_pos_w[:, 2]
#         ball_vz = ball.data.root_link_lin_vel_w[:, 2]

#         clearance = ball_z - paddle_z
#         near_apex = (torch.abs(ball_vz) <= apex_vz_window) & (ball_z >= min_ball_height)
#         score = torch.exp(-torch.square(clearance - target_clearance) / (std**2))
#         return torch.where(near_apex, score, torch.zeros_like(score))


# def ball_alive(
#     env: ManagerBasedRlEnv,
#     max_distance: float = 1.2,
#     robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#     ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
# ) -> torch.Tensor:
#     """Reward for keeping the ball within a base-frame distance threshold.

#     Returns 1.0 if the ball is within ``max_distance`` of the robot base frame,
#     otherwise 0.0.
#     """
#     from mjlab.tasks.keepyup.mdp.observations import ball_pos_in_base_frame

#     ball_pos_b = ball_pos_in_base_frame(env, robot_cfg, ball_cfg)  # [B, 3]
#     ball_dist = torch.norm(ball_pos_b, dim=-1)  # [B]
#     return (ball_dist <= max_distance).float()


# class sustained_contact_penalty:
#     """Penalty for sustaining contact with the ball (anti-balancing).

#     Tracks consecutive steps where paddle-ball contact is active.
#     Applies an escalating penalty if contact persists beyond a threshold.
#     """

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         self.sensor_name = cfg.params["sensor_name"]
#         self.threshold = cfg.params.get("threshold", 10)

#         # Track consecutive contact steps per environment
#         self.consecutive_contact = torch.zeros(
#             env.num_envs, dtype=torch.int32, device=env.device
#         )

#     def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#         if env_ids is None:
#             self.consecutive_contact[:] = 0
#         else:
#             self.consecutive_contact[env_ids] = 0

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         sensor_name: str,
#         threshold: int = 10,
#     ) -> torch.Tensor:
#         """Compute sustained contact penalty."""
#         # Get contact sensor data
#         contact_sensor: ContactSensor = env.scene[sensor_name]
#         assert contact_sensor.data.found is not None

#         # Current contact state
#         in_contact = contact_sensor.data.found.squeeze(-1) > 0  # [B]

#         # Update consecutive contact counter
#         self.consecutive_contact = torch.where(
#             in_contact,
#             self.consecutive_contact + 1,
#             torch.zeros_like(self.consecutive_contact),
#         )

#         # Penalty is proportional to how many steps beyond threshold
#         penalty = torch.clamp(
#             self.consecutive_contact - threshold,
#             min=0,
#         ).float()

#         # Log metrics
#         max_consecutive = torch.max(self.consecutive_contact.float())
#         mean_consecutive = torch.mean(self.consecutive_contact.float())
#         log = env.extras.setdefault("log", {})
#         log["Metrics/max_consecutive_contact"] = max_consecutive
#         log["Metrics/mean_consecutive_contact"] = mean_consecutive

#         return penalty


# def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
#     """Penalize robot self-collisions detected by a contact sensor."""
#     sensor: ContactSensor = env.scene[sensor_name]
#     assert sensor.data.found is not None
#     return sensor.data.found.squeeze(-1)


# def ball_height_reward(
#     env: ManagerBasedRlEnv,
#     target_height: float,
#     std: float,
#     ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
# ) -> torch.Tensor:
#     """Reward for maintaining ball at target height.

#     Exponential reward centered at target height.
#     """
#     ball: Entity = env.scene[ball_cfg.name]
#     ball_z = ball.data.root_link_pos_w[:, 2]  # [B]

#     height_error = torch.square(ball_z - target_height)
#     return torch.exp(-height_error / (std**2))


# def ball_height_above_ceiling_penalty(
#     env: ManagerBasedRlEnv,
#     ceiling_height: float = 1.55,
#     deadband: float = 0.02,
#     ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
# ) -> torch.Tensor:
#     """Soft penalty when the ball flies too high above a ceiling."""
#     ball: Entity = env.scene[ball_cfg.name]
#     ball_z = ball.data.root_link_pos_w[:, 2]  # [B]
#     return torch.clamp(ball_z - (ceiling_height + deadband), min=0.0)


# class paddle_height_ceiling_penalty:
#     """Soft penalty when paddle rises above a height ceiling.

#     This discourages "carrying" behavior where the policy follows the ball upward
#     with the paddle instead of producing cleaner impulse-like contacts.
#     """

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_sites("paddle_face")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_face site not found on robot entity")
#         self._paddle_site_idx: int = ids[0]

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         max_paddle_height: float = 1.05,
#         deadband: float = 0.02,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]
#         paddle_z = robot.data.site_pos_w[:, self._paddle_site_idx, 2]  # [B]
#         excess = torch.clamp(paddle_z - (max_paddle_height + deadband), min=0.0)
#         return excess


# class paddle_face_up_reward:
#     """Reward keeping paddle face roughly upward (parallel to floor)."""

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_geoms("paddle_geom")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_geom not found on robot entity")
#         self._paddle_geom_idx: int = ids[0]

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#         min_alignment: float = 0.85,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]

#         # For mjGEOM_CYLINDER, local +Z is the face normal.
#         paddle_quat_w = robot.data.geom_quat_w[:, self._paddle_geom_idx, :]  # [B, 4]
#         local_normal = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(
#             env.num_envs, 1
#         )
#         normal_w = quat_apply(paddle_quat_w, local_normal)  # [B, 3]

#         # Reward face-up alignment (normal close to +Z), with dead-zone below threshold.
#         alignment = normal_w[:, 2]  # dot(normal_w, world_up)
#         reward = torch.clamp(
#             (alignment - min_alignment) / (1.0 - min_alignment), min=0.0
#         )

#         log = env.extras.setdefault("log", {})
#         log["Metrics/paddle_up_alignment_mean"] = alignment.mean()
#         return reward


# class paddle_ball_distance:
#     """Reward for keeping the paddle close to the ball.

#     Provides smooth gradient signal via ``exp(-dist^2 / std^2)`` so the
#     policy is guided toward the ball even before any contact occurs.
#     """

#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
#         robot: Entity = env.scene["robot"]
#         ids, _names = robot.find_sites("paddle_face")
#         if len(ids) == 0:
#             raise RuntimeError("paddle_face site not found on robot entity")
#         self._paddle_site_idx: int = ids[0]

#     def __call__(
#         self,
#         env: ManagerBasedRlEnv,
#         std: float = 0.2,
#         robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
#         ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
#     ) -> torch.Tensor:
#         robot: Entity = env.scene[robot_cfg.name]
#         ball: Entity = env.scene[ball_cfg.name]

#         paddle_pos_w = robot.data.site_pos_w[:, self._paddle_site_idx, :]  # [B, 3]
#         ball_pos_w = ball.data.root_link_pos_w  # [B, 3]

#         distance = torch.norm(ball_pos_w - paddle_pos_w, dim=-1)  # [B]
#         # Log diagnostic distance statistics so we can verify spawn/proximity.
#         log = env.extras.setdefault("log", {})
#         log["Metrics/paddle_ball_dist_mean"] = distance.mean()
#         log["Metrics/paddle_ball_dist_min"] = distance.min()
#         return torch.exp(-torch.square(distance) / (std**2))
