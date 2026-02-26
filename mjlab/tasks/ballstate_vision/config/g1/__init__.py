"""G1 ball-state vision task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_ballstate_vision_env_cfg
from .rl_cfg import unitree_g1_ballstate_vision_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-BallState-Vision-Unitree-G1",
  env_cfg=unitree_g1_ballstate_vision_env_cfg(),
  play_env_cfg=unitree_g1_ballstate_vision_env_cfg(play=True),
  rl_cfg=unitree_g1_ballstate_vision_ppo_cfg(),
)

