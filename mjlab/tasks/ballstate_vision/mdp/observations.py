"""Observation helpers for the ball-state vision estimator task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def depth_patch(
  env: "ManagerBasedRlEnv",
  sensor_name: str = "depth_camera",
  crop_top: int = 12,
  crop_bottom: int = 108,
  crop_left: int = 20,
  crop_right: int = 140,
  max_depth_m: float = 0.30,
  out_h: int = 32,
  out_w: int = 32,
) -> torch.Tensor:
  """Get cropped, clipped, downsampled depth patch as flattened vector."""
  sensor: CameraSensor = env.scene[sensor_name]
  data = sensor.capture()
  if data.depth is None:
    raise RuntimeError(f"Camera sensor '{sensor_name}' has no depth output.")

  depth = data.depth[..., 0]  # [B, H, W]
  depth = depth[:, crop_top:crop_bottom, crop_left:crop_right]
  depth = torch.clamp(depth, min=0.0, max=float(max_depth_m)) / max(float(max_depth_m), 1e-6)
  depth = depth.unsqueeze(1)  # [B,1,H,W]
  depth_ds = F.interpolate(depth, size=(out_h, out_w), mode="bilinear", align_corners=False)
  return depth_ds.flatten(start_dim=1)


def ball_state_gt_camera_frame(
  env: "ManagerBasedRlEnv",
  sensor_name: str = "depth_camera",
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
  """Ground-truth ball state in camera frame as [x,y,z,vx,vy,vz]."""
  sensor: CameraSensor = env.scene[sensor_name]
  data = sensor.capture()

  ball: Entity = env.scene[ball_cfg.name]
  ball_pos_w = ball.data.root_link_pos_w
  ball_vel_w = ball.data.root_link_lin_vel_w

  rel_pos_w = ball_pos_w - data.pos_w
  pos_cam = quat_apply_inverse(data.quat_w, rel_pos_w)
  vel_cam = quat_apply_inverse(data.quat_w, ball_vel_w)
  return torch.cat([pos_cam, vel_cam], dim=-1)

