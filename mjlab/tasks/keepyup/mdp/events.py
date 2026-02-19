"""Keepyup-specific event helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("ball")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")

# Spawn ranges in the paddle local frame:
# - local +X: frontal axis (front/back relative to paddle)
# - local +Z: lateral axis (left/right relative to paddle)
BALL_SPAWN_MAX_LATERAL_RANGE = (-0.15, 0.3)  # 15 cm left, 30 cm right
BALL_SPAWN_MAX_FRONTAL_RANGE = (-0.05, 0.15)  # 5 cm behind, 15 cm in front


def reset_ball(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    spawn_height: float = 0.70,
    lateral_spawn_variance: float = 0.0,
    frontal_spawn_variance: float = 0.0,
    throw_origin_distance: float = 0.0,
):
    # EC: TODO -> Add variance to angle of entry to mimic a human gently tossing the ball to the agent
    """Reset and spawn ball relative to the paddle.

    Notes:
      - MuJoCo world +Z is vertical (up/down).
      - ``spawn_height`` always offsets along world +Z.
      - ``frontal/lateral`` variances are sampled in the paddle local frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    ball: Entity = env.scene[asset_cfg.name]
    robot: Entity = env.scene[robot_cfg.name]

    paddle_geom_ids, _ = robot.find_geoms("paddle_geom")
    if len(paddle_geom_ids) == 0:
        raise RuntimeError("paddle_geom not found on robot entity")
    paddle_geom_idx = paddle_geom_ids[0]

    default_root_state = ball.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    num = len(env_ids)
    device = env.device

    # Find world-space paddle center and orientation.
    paddle_center_w = robot.data.geom_pos_w[env_ids, paddle_geom_idx, :]  # [B, 3]
    paddle_quat_w = robot.data.geom_quat_w[env_ids, paddle_geom_idx, :]  # [B, 4]

    # Frontal and lateral spawn offsets are sampled as fractions of max ranges.
    frontal_scale = max(0.0, min(1.0, float(frontal_spawn_variance)))
    lateral_scale = max(0.0, min(1.0, float(lateral_spawn_variance)))
    frontal_low = BALL_SPAWN_MAX_FRONTAL_RANGE[0] * frontal_scale
    frontal_high = BALL_SPAWN_MAX_FRONTAL_RANGE[1] * frontal_scale
    lateral_low = BALL_SPAWN_MAX_LATERAL_RANGE[0] * lateral_scale
    lateral_high = BALL_SPAWN_MAX_LATERAL_RANGE[1] * lateral_scale
    frontal_offset = sample_uniform(
        torch.full((num,), frontal_low, device=device),
        torch.full((num,), frontal_high, device=device),
        (num,),
        device=device,
    )
    lateral_offset = sample_uniform(
        torch.full((num,), lateral_low, device=device),
        torch.full((num,), lateral_high, device=device),
        (num,),
        device=device,
    )

    # Build world-space basis vectors from paddle geometry frame.
    basis_x_w = quat_apply(
        paddle_quat_w, torch.tensor([1.0, 0.0, 0.0], device=device).repeat(num, 1)
    )
    basis_z_w = quat_apply(
        paddle_quat_w, torch.tensor([0.0, 0.0, 1.0], device=device).repeat(num, 1)
    )

    # Target point is above paddle center with lateral/frontal variance.
    target_pos_w = paddle_center_w.clone()
    target_pos_w[:, 2] += spawn_height
    target_pos_w += (
        frontal_offset.unsqueeze(-1) * basis_x_w
        + lateral_offset.unsqueeze(-1) * basis_z_w
    )

    # Throw origin is shifted along paddle frontal axis.
    throw_dist = float(throw_origin_distance)
    spawn_pos_w = target_pos_w + throw_dist * basis_x_w

    # Choose horizontal velocity so the ball reaches target_xy after free-fall time.
    g = 9.81
    t_fall = max((2.0 * spawn_height / g) ** 0.5, 1e-6)
    delta_xy = target_pos_w[:, :2] - spawn_pos_w[:, :2]
    v0_xy = delta_xy / t_fall
    v0 = torch.zeros((num, 3), device=device)
    v0[:, :2] = v0_xy

    ball.write_root_link_pose_to_sim(
        torch.cat([spawn_pos_w, root_states[:, 3:7]], dim=-1), env_ids=env_ids
    )
    root_vel = root_states[:, 7:13].clone()
    root_vel[:] = 0.0
    root_vel[:, :3] = v0
    ball.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)


def reset_arm_then_ball(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    left_arm_cfg: SceneEntityCfg,
    position_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    spawn_height: float = 0.70,
    lateral_spawn_variance: float = 0.0,
    frontal_spawn_variance: float = 0.0,
    throw_origin_distance: float = 0.0,
) -> None:
    """Reset left arm joints, sync kinematics, then spawn ball above paddle."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    # First restore robot joints to their reset pose.
    reset_joints_by_offset(
        env=env,
        env_ids=env_ids,
        position_range=position_range,
        velocity_range=velocity_range,
        asset_cfg=left_arm_cfg,
    )
    # Flush/reset kinematics so paddle pose reads reflect the just-reset arm pose.
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)

    # Then spawn the ball relative to the refreshed paddle transform.
    reset_ball(
        env=env,
        env_ids=env_ids,
        asset_cfg=asset_cfg,
        robot_cfg=robot_cfg,
        spawn_height=spawn_height,
        lateral_spawn_variance=lateral_spawn_variance,
        frontal_spawn_variance=frontal_spawn_variance,
        throw_origin_distance=throw_origin_distance,
    )