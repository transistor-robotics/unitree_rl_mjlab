"""Base environment configuration for the keepy up task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.keepyup import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_keepyup_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base keepy up task configuration.

    This factory function creates the base configuration that is then
    customized per robot in robot-specific config files.
    """

    ##
    # Observations
    ##
    locked_joint_names = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    left_arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=(
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ),
    )

    policy_terms = {
        "left_arm_joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": left_arm_cfg},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "left_arm_joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": left_arm_cfg},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "ball_state": ObservationTermCfg(
            func=mdp.ball_state_from_rgbd,
            # Camera-like perception model for sim-to-real transfer.
            params={
                "camera_fps": 30.0,
                "dropout_prob": 0.08,
                "pos_noise_std": 0.012,
                "vel_noise_std": 0.10,
                "outlier_prob": 0.01,
                "outlier_std": 0.05,
                "vel_ema_alpha": 0.35,
                "stale_vel_decay": 0.98,
                "max_speed": 6.0,
            },
        ),
        "ball_visible": ObservationTermCfg(
            func=mdp.ball_visible,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    # Critic gets privileged ball kinematics that are unavailable in deployment.
    critic_terms = {
        **policy_terms,
        "ball_pos_gt": ObservationTermCfg(
            func=mdp.ball_pos_in_base_frame,
        ),
        "ball_vel_gt": ObservationTermCfg(
            func=mdp.ball_vel_in_base_frame,
        ),
        "ball_ang_vel": ObservationTermCfg(
            func=mdp.ball_ang_vel_in_base_frame,
        ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    ##
    # Actions
    ##

    # Only control the 7 left arm joints
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": mdp.SlewRateJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ),
            scale=0.25,  # Will be overridden per-robot with proper actuator scales
            use_default_offset=True,
            # Keep default uncapped here; configure explicit caps per-robot.
            max_velocity=float("inf"),
            clip_to_joint_limits=True,
        )
    }

    ##
    # Events
    ##

    events = {
        "reset_arm_then_ball": EventTermCfg(
            func=mdp.reset_arm_then_ball,
            mode="reset",
            params={
                "left_arm_cfg": left_arm_cfg,
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "min_spawn_height": 1.6,
                "lateral_spawn_variance": 0.0,
                "frontal_spawn_variance": 0.0,
                "max_throw_origin_distance": 0.0,
            },
        ),
        "lock_non_left_arm_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=locked_joint_names,
                ),
            },
        ),
        "randomize_ball_bounciness": EventTermCfg(
            func=mdp.randomize_field,
            mode="reset",
            params={
                "field": "geom_solref",
                # Tiny randomization in damping ratio only (axis 1):
                # lower => bouncier, higher => less bouncy.
                "ranges": {1: (0.45, 0.58)},
                "distribution": "uniform",
                "operation": "abs",
                "asset_cfg": SceneEntityCfg(
                    "ball",
                    geom_names=("ball_geom",),
                ),
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        ######################
        # Task-space rewards #
        ######################
        "total_bounces": RewardTermCfg(
            func=mdp.bounce_reward,
            weight=0.9,
            params={"sensor_name": "paddle_ball_contact"},
        ),
        "ball_height": RewardTermCfg(
            func=mdp.ball_height_reward,
            weight=3.1,
            ####
            params={"target_height": 1.4},
        ),
        "bounce_rhythm": RewardTermCfg(
            func=mdp.bounce_rhythm_reward,
            weight=0.18,
            params={"sensor_name": "paddle_ball_contact"},
        ),
        "ball_paddle_tracking": RewardTermCfg(
            func=mdp.ball_paddle_tracking_reward, weight=0.09
        ),
        "paddle_height_consistency": RewardTermCfg(
            func=mdp.paddle_height_consistency_reward,
            weight=2.8,
            params={"target_height": 0.85},
        ),
        "ball_trajectory_consistency": RewardTermCfg(
            func=mdp.ball_trajectory_consistency_reward, weight=1.1
        ),
        #####################
        # Non task-specific #
        #####################
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.8,
            params={"sensor_name": "self_collision"},
        ),
        "paddle_robot_collisions": RewardTermCfg(
            func=mdp.paddle_robot_collision_cost,
            weight=-0.9,
            params={"sensor_name": "paddle_robot_collision"},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-5.0,
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        "ball_state_noise": CurriculumTermCfg(
            func=mdp.ball_state_noise_schedule,
            params={
                # Spawn-only ablation: keep vision fixed at easiest settings.
                "stages": [
                    {
                        "step": 0,
                        "camera_fps": 200.0,
                        "update_prob": None,
                        "dropout_prob": 0.0,
                        "pos_noise_std": 0.0015,
                        "vel_noise_std": 0.015,
                        "outlier_prob": 0.0,
                        "outlier_std": 0.0,
                        "stale_vel_decay": 1.0,
                    },
                ],
                "term_name": "ball_state",
                "groups": ("policy", "critic"),
            },
        ),
        "ball_spawn_difficulty": CurriculumTermCfg(
            func=mdp.PerformanceGatedSpawnSchedule,
            params={
                "event_term_name": "reset_arm_then_ball",
                "reward_term_name": "total_bounces",
                "promote_bounces": 3.0,
                "rollback_bounces": 1.0,
                "promote_threshold": 0.8,
                "rollback_threshold": 0.3,
                "promote_patience": 6,
                "rollback_patience": 3,
                "ema_alpha": 0.1,
                "min_episodes_per_window": 64,
                # Variances are normalized [0, 1] fractions of max spawn ranges.
                # Spawn height is sampled in [1.6, min_spawn_height].
                "stages": [
                    {
                        "min_spawn_height": 1.5,
                        "lateral_spawn_variance": 0.2,
                        "frontal_spawn_variance": 0.2,
                        "max_throw_origin_distance": 0.05,
                    },
                    {
                        "min_spawn_height": 1.3,
                        "lateral_spawn_variance": 0.5,
                        "frontal_spawn_variance": 0.5,
                        "max_throw_origin_distance": 0.1,
                    },
                    {
                        "min_spawn_height": 1.05,
                        "lateral_spawn_variance": 0.82,
                        "frontal_spawn_variance": 0.65,
                        "max_throw_origin_distance": 0.12,
                    },
                    {
                        "min_spawn_height": 0.83,
                        "lateral_spawn_variance": 1.0,
                        "frontal_spawn_variance": 1.0,
                        "max_throw_origin_distance": 0.15,
                    },
                    {
                        "min_spawn_height": 0.65,
                        "lateral_spawn_variance": 1.0,
                        "frontal_spawn_variance": 1.0,
                        "max_throw_origin_distance": 0.2,
                    },
                ],
            },
        ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "ball_hit_ground": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                # Ball radius is 0.02 m; terminate when center gets close to floor.
                "minimum_height": 0.05,
                "asset_cfg": SceneEntityCfg("ball"),
            },
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(50.0)},
        ),
    }

    ##
    # Scene and simulation
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(
                terrain_type="plane",
            ),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.5,
            elevation=-20.0,
            azimuth=180.0,
        ),
        sim=SimulationCfg(
            nconmax=50,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.002,  # 500 Hz physics for better ball contact
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=10,  # 50 Hz policy control
        episode_length_s=10.0,
    )
