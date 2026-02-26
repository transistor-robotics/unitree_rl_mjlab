"""Supervised trainer for depth-to-ballstate estimation."""

from __future__ import annotations

import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.models import BallStateEstimator
from mjlab.tasks.registry import load_env_cfg


@dataclass(frozen=True)
class TrainBallstateVisionCfg:
  task_id: str = "Mjlab-BallState-Vision-Unitree-G1"
  num_envs: int = 256
  max_steps: int = 20_000
  log_every: int = 100
  save_every: int = 2_000
  history: int = 4
  depth_h: int = 32
  depth_w: int = 32
  learning_rate: float = 1.0e-3
  vel_loss_weight: float = 0.5
  device: str | None = None
  out_dir: str = "logs/supervised_ballstate_vision"


def _reshape_depth(policy_obs: torch.Tensor, h: int, w: int) -> torch.Tensor:
  return policy_obs.view(policy_obs.shape[0], h, w)


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(TrainBallstateVisionCfg)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  model = BallStateEstimator(
    history=cfg.history, input_h=cfg.depth_h, input_w=cfg.depth_w
  ).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

  out_dir = Path(cfg.out_dir).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  obs, _ = env.reset()
  depth = _reshape_depth(obs["policy"], cfg.depth_h, cfg.depth_w)
  depth_hist = depth.unsqueeze(1).repeat(1, cfg.history, 1, 1)

  running_pos = 0.0
  running_vel = 0.0

  for step in range(1, cfg.max_steps + 1):
    labels = obs["labels"]
    pred = model(depth_hist)

    pos_loss = F.smooth_l1_loss(pred[:, :3], labels[:, :3])
    vel_loss = F.smooth_l1_loss(pred[:, 3:], labels[:, 3:])
    loss = pos_loss + cfg.vel_loss_weight * vel_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    running_pos += float(pos_loss.item())
    running_vel += float(vel_loss.item())

    zero_action = torch.zeros(
      (cfg.num_envs, env.action_manager.total_action_dim),
      dtype=torch.float32,
      device=device,
    )
    obs, _, _, _, _ = env.step(zero_action)

    depth = _reshape_depth(obs["policy"], cfg.depth_h, cfg.depth_w)
    depth_hist = torch.roll(depth_hist, shifts=-1, dims=1)
    depth_hist[:, -1] = depth

    if step % cfg.log_every == 0:
      avg_pos = running_pos / cfg.log_every
      avg_vel = running_vel / cfg.log_every
      print(
        f"[step {step:6d}] pos_loss={avg_pos:.5f} vel_loss={avg_vel:.5f} "
        f"rmse_pos={avg_pos**0.5:.4f} rmse_vel={avg_vel**0.5:.4f}"
      )
      running_pos = 0.0
      running_vel = 0.0

    if step % cfg.save_every == 0:
      ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "cfg": asdict(cfg),
      }
      torch.save(ckpt, out_dir / f"model_{step}.pt")

  torch.save(
    {
      "model_state_dict": model.state_dict(),
      "step": cfg.max_steps,
      "cfg": asdict(cfg),
    },
    out_dir / "model_latest.pt",
  )
  env.close()
  print(f"[done] Saved checkpoints under {out_dir}")


if __name__ == "__main__":
  os.environ.setdefault("MUJOCO_GL", "egl")
  main()

