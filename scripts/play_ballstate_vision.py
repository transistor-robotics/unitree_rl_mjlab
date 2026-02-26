"""Play script for visualizing ball-state estimator predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.models import BallStateEstimator
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayBallstateVisionCfg:
  task_id: str = "Mjlab-BallState-Vision-Unitree-G1"
  checkpoint: str = "logs/supervised_ballstate_vision/model_latest.pt"
  num_envs: int = 64
  history: int = 4
  depth_h: int = 32
  depth_w: int = 32
  device: str | None = None
  viewer: str = "native"


class BallstatePolicy:
  """No-action policy wrapper that runs estimator inference for visualization."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    estimator: BallStateEstimator,
    history: int,
    depth_h: int,
    depth_w: int,
  ):
    self.env = env
    self.estimator = estimator
    self.history = history
    self.depth_h = depth_h
    self.depth_w = depth_w
    self._depth_hist: torch.Tensor | None = None

  def __call__(self, obs) -> torch.Tensor:
    policy_obs = obs["policy"]  # TensorDict field.
    depth = policy_obs.view(policy_obs.shape[0], self.depth_h, self.depth_w)
    if self._depth_hist is None:
      self._depth_hist = depth.unsqueeze(1).repeat(1, self.history, 1, 1)
    else:
      self._depth_hist = torch.roll(self._depth_hist, shifts=-1, dims=1)
      self._depth_hist[:, -1] = depth

    with torch.no_grad():
      pred_cam = self.estimator(self._depth_hist)

    cam_sensor = self.env.scene["depth_camera"]
    cam_data = cam_sensor.data
    pred_pos_w = quat_apply(cam_data.quat_w, pred_cam[:, :3]) + cam_data.pos_w
    gt_pos_w = self.env.scene["ball"].data.root_link_pos_w

    self.env._pred_ball_pos_w = pred_pos_w.detach()
    self.env._gt_ball_pos_w = gt_pos_w.detach()

    return torch.zeros(
      (self.env.num_envs, self.env.action_manager.total_action_dim),
      dtype=torch.float32,
      device=self.env.device,
    )


def _attach_debug_visualization(env: ManagerBasedRlEnv) -> None:
  def _update_visualizers(visualizer) -> None:
    pred = getattr(env, "_pred_ball_pos_w", None)
    gt = getattr(env, "_gt_ball_pos_w", None)
    if pred is None or gt is None or pred.shape[0] == 0:
      return
    pred_np = pred[0].detach().cpu().numpy()
    gt_np = gt[0].detach().cpu().numpy()
    visualizer.add_sphere(
      center=pred_np,
      radius=0.03,
      color=(0.0, 1.0, 1.0, 0.7),
      label="pred_ball",
    )
    visualizer.add_sphere(
      center=gt_np,
      radius=0.02,
      color=(1.0, 1.0, 0.0, 0.7),
      label="gt_ball",
    )

  env.update_visualizers = _update_visualizers  # type: ignore[attr-defined]


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(PlayBallstateVisionCfg)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  _attach_debug_visualization(env)

  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

  estimator = BallStateEstimator(
    history=cfg.history, input_h=cfg.depth_h, input_w=cfg.depth_w
  ).to(device)
  ckpt = torch.load(Path(cfg.checkpoint).expanduser().resolve(), map_location=device)
  estimator.load_state_dict(ckpt["model_state_dict"])
  estimator.eval()

  policy = BallstatePolicy(
    env=wrapped.unwrapped,
    estimator=estimator,
    history=cfg.history,
    depth_h=cfg.depth_h,
    depth_w=cfg.depth_w,
  )

  if cfg.viewer == "viser":
    ViserPlayViewer(wrapped, policy).run()
  else:
    NativeMujocoViewer(wrapped, policy).run()

  wrapped.close()


if __name__ == "__main__":
  main()

