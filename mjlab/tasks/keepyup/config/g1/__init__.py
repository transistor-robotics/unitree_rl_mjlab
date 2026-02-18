"""G1 keepy up task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_keepyup_env_cfg
from .rl_cfg import unitree_g1_keepyup_ppo_cfg

register_mjlab_task(
    task_id="Mjlab-KeepyUp-Unitree-G1",
    env_cfg=unitree_g1_keepyup_env_cfg(),
    play_env_cfg=unitree_g1_keepyup_env_cfg(play=True),
    rl_cfg=unitree_g1_keepyup_ppo_cfg(),
)
