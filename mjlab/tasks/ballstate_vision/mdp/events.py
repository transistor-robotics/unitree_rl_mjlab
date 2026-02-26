"""Events for the ball-state vision estimator task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def reset_ball_vertical_bounce(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  spawn_pos_w: tuple[float, float, float] = (0.0, 0.0, 0.25),
  initial_lin_vel_w: tuple[float, float, float] = (0.0, 0.0, 2.5),
) -> None:
  """Reset the ping pong ball to a deterministic vertical bounce state."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  ball: Entity = env.scene[ball_cfg.name]
  default_root_state = ball.data.default_root_state
  assert default_root_state is not None
  root_state = default_root_state[env_ids].clone()

  root_state[:, 0:3] = torch.tensor(spawn_pos_w, device=env.device, dtype=torch.float32)
  root_state[:, 3:7] = torch.tensor(
    [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
  )
  root_state[:, 7:10] = torch.tensor(
    initial_lin_vel_w, device=env.device, dtype=torch.float32
  )
  root_state[:, 10:13] = 0.0

  ball.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)

