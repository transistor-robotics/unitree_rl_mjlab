"""Environment configuration for supervised ball-state vision estimation."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.ballstate_vision import mdp
from mjlab.tasks.keepyup.ball import get_pingpong_ball_cfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.viewer import ViewerConfig


def make_ballstate_vision_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create environment config for vision-only ball state estimation."""
  observations = {
    "policy": ObservationGroupCfg(
      terms={
        "depth_patch": ObservationTermCfg(
          func=mdp.depth_patch,
          params={
            "sensor_name": "depth_camera",
            "crop_top": 12,
            "crop_bottom": 108,
            "crop_left": 20,
            "crop_right": 140,
            "max_depth_m": 0.30,
            "out_h": 32,
            "out_w": 32,
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "labels": ObservationGroupCfg(
      terms={
        "ball_state_gt": ObservationTermCfg(
          func=mdp.ball_state_gt_camera_frame,
          params={"sensor_name": "depth_camera"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  events = {
    "reset_ball_vertical": EventTermCfg(
      func=mdp.reset_ball_vertical_bounce,
      mode="reset",
      params={
        "ball_cfg": SceneEntityCfg("ball"),
        "spawn_pos_w": (0.0, 0.0, 0.25),
        "initial_lin_vel_w": (0.0, 0.0, 2.5),
      },
    ),
  }

  rewards: dict[str, RewardTermCfg] = {}
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  curriculum: dict[str, CurriculumTermCfg] = {}

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
      entities={"ball": get_pingpong_ball_cfg()},
      sensors=(
        CameraSensorCfg(
          name="depth_camera",
          camera_name=None,
          parent_body=None,
          pos=(0.0, 0.0, 1.5),
          quat=(1.0, 0.0, 0.0, 0.0),
          fovy=58.0,
          width=160,
          height=120,
          data_types=("depth",),
        ),
      ),
    ),
    observations=observations,
    actions={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 0.7),
      distance=2.8,
      elevation=-20.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=20,
      njmax=120,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=10,
    episode_length_s=6.0,
  )

