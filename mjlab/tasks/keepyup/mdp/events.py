"""Keepyup-specific event helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("ball")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


class randomize_paddle_mount_position:
    """Randomize paddle body mount position in the hand frame at reset."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._env = env
        params = cfg.params
        robot_cfg = params.get("robot_cfg", _DEFAULT_ROBOT_CFG)
        paddle_body_name = params.get("paddle_body_name", "paddle")

        robot: Entity = env.scene[robot_cfg.name]
        body_ids, _ = robot.find_bodies(paddle_body_name)
        if len(body_ids) == 0:
            raise RuntimeError(f"{paddle_body_name} body not found on robot entity")
        self._paddle_body_idx = body_ids[0]

        default_body_pos = env.sim.get_default_field("body_pos")
        self._default_mount_pos = default_body_pos[self._paddle_body_idx].clone().to(env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        x_range: tuple[float, float] = (-0.02, 0.02),
        y_range: tuple[float, float] = (-0.02, 0.02),
        z_range: tuple[float, float] = (-0.01, 0.01),
        **_,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.int)

        num = len(env_ids)
        if num == 0:
            return

        offsets = torch.stack(
            [
                sample_uniform(
                    torch.full((num,), x_range[0], device=env.device),
                    torch.full((num,), x_range[1], device=env.device),
                    (num,),
                    device=env.device,
                ),
                sample_uniform(
                    torch.full((num,), y_range[0], device=env.device),
                    torch.full((num,), y_range[1], device=env.device),
                    (num,),
                    device=env.device,
                ),
                sample_uniform(
                    torch.full((num,), z_range[0], device=env.device),
                    torch.full((num,), z_range[1], device=env.device),
                    (num,),
                    device=env.device,
                ),
            ],
            dim=-1,
        )  # [B, 3]

        model_body_pos = env.sim.model.body_pos
        mount_pos = self._default_mount_pos.unsqueeze(0) + offsets

        if len(model_body_pos.shape) == 3:
            model_body_pos[env_ids, self._paddle_body_idx, :] = mount_pos
        elif len(model_body_pos.shape) == 2:
            model_body_pos[self._paddle_body_idx, :] = mount_pos[0]
        else:
            raise RuntimeError(f"Unexpected body_pos shape: {tuple(model_body_pos.shape)}")


def reset_root_state_uniform_no_env_origins(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset root state without adding scene env_origins offsets.

    Keepyup uses a fixed-base robot. In multi-env training, adding env_origins to
    the floating ball can shift it away from the robot frame and eliminate contact.
    This reset keeps pose sampling in each world's local frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )

    if asset.is_fixed_base:
        if not asset.is_mocap:
            raise ValueError(
                f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
            )
        default_root_state = asset.data.default_root_state
        assert default_root_state is not None
        root_states = default_root_state[env_ids].clone()
        positions = root_states[:, 0:3] + pose_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
        )
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)
        asset.write_mocap_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        return

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + pose_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    if velocity_range is None:
        velocity_range = {}
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_ball_targeted_to_paddle(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    paddle_radius: float = 0.075,
    hit_probability: float = 0.8,
    hit_radius_fraction: float = 0.7,
    miss_radius_range: tuple[float, float] = (0.08, 0.13),
    entry_angle_deg_range: tuple[float, float] = (0.0, 28.0),
    time_to_impact_range: tuple[float, float] = (0.28, 0.55),
    min_downward_speed_at_impact: float = 0.1,
    angle_search_steps: int = 21,
) -> None:
    """Reset ball by solving ballistic velocity to target paddle impact.

    This samples a spawn point and an impact point, then computes initial velocity so
    the ball reaches the impact point under gravity. It allows controlling impact
    angle while choosing how often the ball hits the paddle directly.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    ball: Entity = env.scene[asset_cfg.name]
    robot: Entity = env.scene[robot_cfg.name]

    # Resolve paddle geom for center pose in world frame.
    paddle_geom_ids, _ = robot.find_geoms("paddle_geom")
    if len(paddle_geom_ids) == 0:
        raise RuntimeError("paddle_geom not found on robot entity")
    paddle_geom_idx = paddle_geom_ids[0]

    default_root_state = ball.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    num = len(env_ids)
    device = env.device

    # Spawn sampling ranges.
    x_rng = pose_range.get("x", (0.0, 0.0))
    y_rng = pose_range.get("y", (0.0, 0.0))
    z_rng = pose_range.get("z", (1.2, 1.2))
    roll_rng = pose_range.get("roll", (0.0, 0.0))
    pitch_rng = pose_range.get("pitch", (0.0, 0.0))
    yaw_rng = pose_range.get("yaw", (0.0, 0.0))

    # Paddle frame basis in world.
    paddle_center_w = robot.data.geom_pos_w[env_ids, paddle_geom_idx, :]  # [B, 3]
    paddle_quat_w = robot.data.geom_quat_w[env_ids, paddle_geom_idx, :]  # [B, 4]
    basis_x_w = quat_apply(
        paddle_quat_w, torch.tensor([1.0, 0.0, 0.0], device=device).repeat(num, 1)
    )
    basis_y_w = quat_apply(
        paddle_quat_w, torch.tensor([0.0, 1.0, 0.0], device=device).repeat(num, 1)
    )

    # Impact point on (or near) paddle plane.
    hit_mask = torch.rand(num, device=device) < hit_probability
    u = torch.rand(num, device=device)
    theta = sample_uniform(
        torch.zeros(num, device=device),
        torch.full((num,), 2.0 * torch.pi, device=device),
        (num,),
        device=device,
    )
    hit_radius = hit_radius_fraction * paddle_radius * torch.sqrt(u)
    miss_r = sample_uniform(
        torch.full((num,), miss_radius_range[0], device=device),
        torch.full((num,), miss_radius_range[1], device=device),
        (num,),
        device=device,
    )
    radius = torch.where(hit_mask, hit_radius, miss_r)
    impact_offset_w = (
        radius.unsqueeze(-1)
        * (torch.cos(theta).unsqueeze(-1) * basis_x_w + torch.sin(theta).unsqueeze(-1) * basis_y_w)
    )
    impact_point_w = paddle_center_w + impact_offset_w

    # Spawn pose sampled in world (no env_origins offsets).
    spawn_x = sample_uniform(
        torch.full((num,), x_rng[0], device=device),
        torch.full((num,), x_rng[1], device=device),
        (num,),
        device=device,
    )
    spawn_y = sample_uniform(
        torch.full((num,), y_rng[0], device=device),
        torch.full((num,), y_rng[1], device=device),
        (num,),
        device=device,
    )
    spawn_z = sample_uniform(
        torch.full((num,), z_rng[0], device=device),
        torch.full((num,), z_rng[1], device=device),
        (num,),
        device=device,
    )
    spawn_pos_w = torch.stack([spawn_x, spawn_y, spawn_z], dim=-1)

    # Desired entry angle from vertical at impact. Search for a time that best matches it.
    target_angle_deg = sample_uniform(
        torch.full((num,), entry_angle_deg_range[0], device=device),
        torch.full((num,), entry_angle_deg_range[1], device=device),
        (num,),
        device=device,
    )
    target_angle = torch.deg2rad(target_angle_deg)

    t_candidates = torch.linspace(
        time_to_impact_range[0], time_to_impact_range[1], angle_search_steps, device=device
    )  # [K]
    delta = (impact_point_w - spawn_pos_w)[:, None, :]  # [B, K, 3]
    t = t_candidates[None, :]  # [1, K]
    g = torch.tensor([0.0, 0.0, -9.81], device=device)

    # Ballistic solve for v0 given candidate times.
    v0_all = (delta - 0.5 * g[None, None, :] * (t**2).unsqueeze(-1)) / t.unsqueeze(-1)
    v_imp_all = v0_all + g[None, None, :] * t.unsqueeze(-1)

    vxy = torch.norm(v_imp_all[..., :2], dim=-1)  # [B, K]
    vz_down = torch.clamp(-v_imp_all[..., 2], min=1e-6)  # [B, K], positive means downward
    angle_all = torch.atan2(vxy, vz_down)  # 0 = straight down

    # Prefer solutions with downward impact speed above minimum.
    valid_downward = vz_down >= min_downward_speed_at_impact
    angle_err = torch.abs(angle_all - target_angle[:, None])
    angle_err = torch.where(valid_downward, angle_err, angle_err + 1e6)

    best_idx = torch.argmin(angle_err, dim=1)  # [B]
    v0 = v0_all[torch.arange(num, device=device), best_idx, :]  # [B, 3]

    # Orientations (allow optional pose randomization).
    roll = sample_uniform(
        torch.full((num,), roll_rng[0], device=device),
        torch.full((num,), roll_rng[1], device=device),
        (num,),
        device=device,
    )
    pitch = sample_uniform(
        torch.full((num,), pitch_rng[0], device=device),
        torch.full((num,), pitch_rng[1], device=device),
        (num,),
        device=device,
    )
    yaw = sample_uniform(
        torch.full((num,), yaw_rng[0], device=device),
        torch.full((num,), yaw_rng[1], device=device),
        (num,),
        device=device,
    )
    q_delta = quat_from_euler_xyz(roll, pitch, yaw)
    quat_w = quat_mul(root_states[:, 3:7], q_delta)

    # Write pose and velocity.
    ball.write_root_link_pose_to_sim(
        torch.cat([spawn_pos_w, quat_w], dim=-1), env_ids=env_ids
    )
    root_vel = root_states[:, 7:13].clone()
    root_vel[:, 0:3] = v0
    ball.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)


class spawn_ball_targeted_after_delay:
    """Spawn the ball once per episode after a delay.

    On reset this term parks the ball below the play area. The interval callback
    then performs exactly one targeted ballistic spawn for each environment.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._env = env
        params = cfg.params
        self._asset_cfg = params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        self._robot_cfg = params.get("robot_cfg", _DEFAULT_ROBOT_CFG)
        self._pose_range = dict(params.get("pose_range", {}))
        self._paddle_radius = float(params.get("paddle_radius", 0.075))
        self._hit_probability = float(params.get("hit_probability", 0.8))
        self._hit_radius_fraction = float(params.get("hit_radius_fraction", 0.7))
        self._miss_radius_range = tuple(params.get("miss_radius_range", (0.08, 0.13)))
        self._entry_angle_deg_range = tuple(params.get("entry_angle_deg_range", (0.0, 28.0)))
        self._time_to_impact_range = tuple(params.get("time_to_impact_range", (0.28, 0.55)))
        self._min_downward_speed_at_impact = float(
            params.get("min_downward_speed_at_impact", 0.1)
        )
        self._angle_search_steps = int(params.get("angle_search_steps", 21))

        self._pending_spawn = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(len(self._pending_spawn), device=self._pending_spawn.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(len(self._pending_spawn), device=self._pending_spawn.device)[env_ids]
        else:
            env_ids = env_ids.to(device=self._pending_spawn.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        # Mark these environments for a one-shot delayed spawn.
        self._pending_spawn[env_ids] = True

        # Park the ball out of play immediately after reset so it cannot interact
        # while joints settle.
        ball: Entity = self._env.scene[self._asset_cfg.name]
        default_root_state = ball.data.default_root_state
        assert default_root_state is not None
        parked_state = default_root_state[env_ids].clone()
        parked_state[:, 0] = 0.0
        parked_state[:, 1] = 0.0
        parked_state[:, 2] = -2.0
        parked_state[:, 7:13] = 0.0
        ball.write_root_link_pose_to_sim(parked_state[:, 0:7], env_ids=env_ids)
        ball.write_root_link_velocity_to_sim(parked_state[:, 7:13], env_ids=env_ids)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        spawn_ids = env_ids[self._pending_spawn[env_ids]]
        if len(spawn_ids) == 0:
            return

        reset_ball_targeted_to_paddle(
            env=env,
            env_ids=spawn_ids,
            pose_range=self._pose_range,
            asset_cfg=self._asset_cfg,
            robot_cfg=self._robot_cfg,
            paddle_radius=self._paddle_radius,
            hit_probability=self._hit_probability,
            hit_radius_fraction=self._hit_radius_fraction,
            miss_radius_range=self._miss_radius_range,
            entry_angle_deg_range=self._entry_angle_deg_range,
            time_to_impact_range=self._time_to_impact_range,
            min_downward_speed_at_impact=self._min_downward_speed_at_impact,
            angle_search_steps=self._angle_search_steps,
        )

        self._pending_spawn[spawn_ids] = False
