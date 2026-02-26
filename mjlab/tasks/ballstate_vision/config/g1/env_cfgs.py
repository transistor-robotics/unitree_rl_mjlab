"""Task registration env config for ball-state vision (G1 namespace)."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.ballstate_vision.ballstate_vision_env_cfg import (
  make_ballstate_vision_env_cfg,
)


def unitree_g1_ballstate_vision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_ballstate_vision_env_cfg()
  if play:
    cfg.episode_length_s = int(1e9)
  return cfg

