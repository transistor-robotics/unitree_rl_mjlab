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
    
    left_arm_joint_names = (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )
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
        joint_names=left_arm_joint_names,
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
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=left_arm_joint_names,
                ),
            },
        ),
        "reset_left_elbow_variation": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                # Randomize elbow start around its nominal target.
                "position_range": (-0.09, 0.09),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("left_elbow_joint",),
                ),
            },
        ),
        "reset_left_wrist_variation": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                # Randomize wrist orientation by +/- 6 degrees on roll/pitch/yaw.
                "position_range": (-math.radians(6.0), math.radians(6.0)),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        "left_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                        "left_wrist_yaw_joint",
                    ),
                ),
            },
        ),
        "reset_non_left_arm_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=locked_joint_names,
                ),
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
        "reset_ball": EventTermCfg(
            func=mdp.reset_ball_targeted_to_paddle,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    # Spawn volume sampled in world frame (no env_origins offsets).
                    "x": (0.17, 0.24),
                    "y": (0.02, 0.15),
                    "z": (1.4, 1.55),
                },
                # Targeted ballistic reset settings.
                "hit_probability": 0.8,
                "entry_angle_deg_range": (0.0, 25.0),
                "hit_radius_fraction": 0.7,
                "miss_radius_range": (0.085, 0.13),
                # Curriculum stage 0 overrides this; base default matches final stage.
                "time_to_impact_range": (0.30, 0.52),
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
        "bounce_quality": RewardTermCfg(
            func=mdp.bounce_quality_reward,
            weight=12.0,
            params={
                "sensor_name": "paddle_ball_contact",
                "ball_cfg": SceneEntityCfg("ball"),
                # Camera-aware apex shaping: keep arc in a lower, repeatable band.
                "target_apex_height": 1.42,
                "apex_std": 0.18,
                "target_upward_velocity": 1.75,
                "velocity_std": 0.60,
                "min_upward_velocity": 0.2,
                "min_reward_interval_steps": 5,
            },
        ),
        "bounce_discovery": RewardTermCfg(
            func=mdp.bounce_discovery_reward,
            weight=3.0,
            params={
                "sensor_name": "paddle_ball_contact",
                "ball_cfg": SceneEntityCfg("ball"),
                # Forgiving early signal: reward any clean upward rebound.
                "min_upward_velocity": 0.08,
                "min_apex_height": 0.95,
                "min_apex_gain": 0.04,
                "target_upward_velocity": 1.0,
                "min_reward_interval_steps": 3,
            },
        ),
        "lateral_drift": RewardTermCfg(
            func=mdp.lateral_drift_penalty,
            weight=-0.25,
            params={
                "sensor_name": "paddle_ball_contact",
                "ball_cfg": SceneEntityCfg("ball"),
                "post_bounce_window_steps": 5,
                "deadband": 0.05,
            },
        ),
        "under_ball_alignment": RewardTermCfg(
            func=mdp.under_ball_alignment_reward,
            weight=0.12,
            params={
                "std_xy": 0.12,
                "min_descending_speed": 0.05,
                "strike_zone_z_min": 0.82,
                "strike_zone_z_max": 1.22,
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "strike_plane_hold": RewardTermCfg(
            func=mdp.strike_plane_hold_reward,
            weight=0.04,
            params={
                "target_paddle_height": 0.80,
                "std": 0.08,
                "ascending_vz_threshold": 0.05,
                "far_descending_height": 1.25,
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "upward_chase": RewardTermCfg(
            func=mdp.upward_chase_penalty,
            weight=-1.2,
            params={
                "ball_ascending_threshold": 0.10,
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "sustained_contact": RewardTermCfg(
            func=mdp.sustained_contact_penalty,
            weight=-3.5,
            params={
                "sensor_name": "paddle_ball_contact",
                "threshold": 1,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.45,
            params={"sensor_name": "self_collision"},
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
                "event_term_name": "reset_ball",
                # Start very easy (nearly straight-down center hits), then broaden.
                "stages": [
                    {
                        "step": 0,
                        "pose_range": {
                            "x": (0.195, 0.215),
                            "y": (0.065, 0.085),
                            "z": (1.50, 1.54),
                        },
                        "hit_probability": 1.0,
                        "hit_radius_fraction": 0.25,
                        "miss_radius_range": (0.09, 0.10),
                        "entry_angle_deg_range": (0.0, 2.0),
                        "time_to_impact_range": (1.05, 1.20),
                    },
                    {
                        "step": 300 * 24,
                        "pose_range": {
                            "x": (0.19, 0.225),
                            "y": (0.055, 0.10),
                            "z": (1.48, 1.55),
                        },
                        "hit_probability": 0.95,
                        "hit_radius_fraction": 0.40,
                        "miss_radius_range": (0.09, 0.11),
                        "entry_angle_deg_range": (0.0, 8.0),
                        "time_to_impact_range": (0.90, 1.10),
                    },
                    {
                        "step": 900 * 24,
                        "pose_range": {
                            "x": (0.18, 0.235),
                            "y": (0.04, 0.12),
                            "z": (1.45, 1.55),
                        },
                        "hit_probability": 0.88,
                        "hit_radius_fraction": 0.55,
                        "miss_radius_range": (0.085, 0.12),
                        "entry_angle_deg_range": (0.0, 16.0),
                        "time_to_impact_range": (0.65, 0.90),
                    },
                    {
                        "step": 1500 * 24,
                        "pose_range": {
                            "x": (0.17, 0.24),
                            "y": (0.02, 0.15),
                            "z": (1.40, 1.55),
                        },
                        "hit_probability": 0.80,
                        "hit_radius_fraction": 0.70,
                        "miss_radius_range": (0.085, 0.13),
                        "entry_angle_deg_range": (0.0, 25.0),
                        "time_to_impact_range": (0.30, 0.52),
                    },
                ],
            },
        ),
        "bounce_reward_shaping": CurriculumTermCfg(
            func=mdp.bounce_reward_shaping_schedule,
            params={
                "stages": [
                    # Stage 0: strong discovery signal, very forgiving thresholds.
                    {
                        "step": 0,
                        "discovery_weight": 3.0,
                        "lateral_weight": -0.25,
                        "under_ball_weight": 0.12,
                        "strike_plane_weight": 0.04,
                        "min_upward_velocity": 0.08,
                        "min_apex_height": 0.95,
                        "min_apex_gain": 0.04,
                        "target_upward_velocity": 1.00,
                    },
                    # Stage 1: start tightening after discovery.
                    {
                        "step": 300 * 24,
                        "discovery_weight": 2.4,
                        "lateral_weight": -0.30,
                        "under_ball_weight": 0.11,
                        "strike_plane_weight": 0.038,
                        "min_upward_velocity": 0.12,
                        "min_apex_height": 1.05,
                        "min_apex_gain": 0.08,
                        "target_upward_velocity": 1.20,
                    },
                    # Stage 2: intermediate strictness.
                    {
                        "step": 900 * 24,
                        "discovery_weight": 1.8,
                        "lateral_weight": -0.35,
                        "under_ball_weight": 0.10,
                        "strike_plane_weight": 0.035,
                        "min_upward_velocity": 0.18,
                        "min_apex_height": 1.15,
                        "min_apex_gain": 0.12,
                        "target_upward_velocity": 1.40,
                    },
                    # Stage 3: strictest discovery gate, lower helper reliance.
                    {
                        "step": 1500 * 24,
                        "discovery_weight": 1.2,
                        "lateral_weight": -0.40,
                        "under_ball_weight": 0.09,
                        "strike_plane_weight": 0.03,
                        "min_upward_velocity": 0.24,
                        "min_apex_height": 1.22,
                        "min_apex_gain": 0.16,
                        "target_upward_velocity": 1.60,
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
        "ball_out_of_frame": TerminationTermCfg(
            func=mdp.ball_out_of_frame,
            params={
                "grace_steps": 90,
                "max_distance": 1.2,
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
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
            distance=2.0,
            elevation=-10.0,
            azimuth=45.0,
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
