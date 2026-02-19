"""Base environment configuration for the keepy up task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
        "joint_pos": JointPositionActionCfg(
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
                "spawn_height": 0.0,
                "lateral_spawn_variance": 0.0,
                "frontal_spawn_variance": 0.0,
                "throw_origin_distance": 0.0,
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
            weight=3.0,
            params={"sensor_name": "paddle_ball_contact"},
        ),
        "ball_height": RewardTermCfg(
            func=mdp.ball_height_reward, weight=1.8, params={"target_height": 1.4}
        ),
        "bounce_rhythm": RewardTermCfg(
            func=mdp.bounce_rhythm_reward,
            weight=1.2,
            params={"sensor_name": "paddle_ball_contact"},
        ),
        "ball_paddle_tracking": RewardTermCfg(
            func=mdp.ball_paddle_tracking_reward, weight=0.7
        ),
        "paddle_height_consistency": RewardTermCfg(
            func=mdp.paddle_height_consistency_reward,
            weight=0.7,
            params={"sensor_name": "paddle_ball_contact"},
        ),
        #####################
        # Non task-specific #
        #####################
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.45,
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
                # Stages are in env steps (common_step_counter).
                # Approximate iteration mapping assumes num_steps_per_env ~= 24.
                "stages": [
                    # Stage 0: near-oracle bootstrap.
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
                    # Stage 1: mild realism.
                    {
                        "step": 300 * 24,
                        "camera_fps": 35.0,
                        "update_prob": None,
                        "dropout_prob": 0.02,
                        "pos_noise_std": 0.006,
                        "vel_noise_std": 0.06,
                        "outlier_prob": 0.003,
                        "outlier_std": 0.03,
                        "stale_vel_decay": 0.995,
                    },
                    # Stage 2: medium realism.
                    {
                        "step": 900 * 24,
                        "camera_fps": 27.5,
                        "update_prob": None,
                        "dropout_prob": 0.05,
                        "pos_noise_std": 0.010,
                        "vel_noise_std": 0.09,
                        "outlier_prob": 0.008,
                        "outlier_std": 0.04,
                        "stale_vel_decay": 0.99,
                    },
                    # Stage 3: target deployment realism (~20 fps effective).
                    {
                        "step": 1500 * 24,
                        "camera_fps": 20.0,
                        "update_prob": None,
                        "dropout_prob": 0.08,
                        "pos_noise_std": 0.012,
                        "vel_noise_std": 0.10,
                        "outlier_prob": 0.01,
                        "outlier_std": 0.05,
                        "stale_vel_decay": 0.98,
                    },
                ],
                "term_name": "ball_state",
                "groups": ("policy", "critic"),
            },
        ),
        "ball_spawn_difficulty": CurriculumTermCfg(
            func=mdp.ball_spawn_difficulty_schedule,
            params={
                "event_term_name": "reset_arm_then_ball",
                # Variances are normalized [0, 1] fractions of max spawn ranges.
                "stages": [
                    {
                        # Stage 0: no lateral/frontal offset.
                        # Spawn height quite high to give plenty of time to react
                        "step": 0,
                        "lateral_spawn_variance": 0.2,
                        "frontal_spawn_variance": 0.2,
                        "throw_origin_distance": 0.0,
                        "spawn_height": 1.4,
                    },
                    {
                        # Stage 1: light randomness around paddle center.
                        "step": 300 * 24,
                        "lateral_spawn_variance": 0.6,
                        "frontal_spawn_variance": 0.4,
                        "throw_origin_distance": 0.15,
                        "spawn_height": 1.2,
                    },
                    {
                        # Stage 2: moderate offset variance.
                        "step": 900 * 24,
                        "lateral_spawn_variance": 1.0,
                        "frontal_spawn_variance": 0.7,
                        "throw_origin_distance": 0.5,
                        "spawn_height": 1.0,
                    },
                    {
                        # Stage 3: full configured spawn variance.
                        "step": 1600 * 24,
                        "lateral_spawn_variance": 1.0,
                        "frontal_spawn_variance": 1.0,
                        "throw_origin_distance": 0.8,
                        "spawn_height": 0.7,
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
        # "ball_out_of_frame": TerminationTermCfg(
        #     func=mdp.ball_out_of_frame,
        #     params={
        #         "grace_steps": 90,
        #         "max_distance": 1.2,
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ball_cfg": SceneEntityCfg("ball"),
        #     },
        # ),
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
