"""Base environment configuration for the keepy up task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
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
        "ball_pos": ObservationTermCfg(
            func=mdp.ball_pos_in_base_frame,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "ball_vel": ObservationTermCfg(
            func=mdp.ball_vel_in_base_frame,
            noise=Unoise(n_min=-0.1, n_max=0.1),
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
    
    # Critic gets additional ball angular velocity (not available in real)
    critic_terms = {
        **policy_terms,
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
                "position_range": (-0.15, 0.15),
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
                # Randomize wrist orientation by +/- 10 degrees on roll/pitch/yaw.
                "position_range": (-math.radians(10.0), math.radians(10.0)),
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
                "time_to_impact_range": (0.30, 0.52),
                "hit_radius_fraction": 0.7,
                "miss_radius_range": (0.085, 0.13),
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
        "bounce_event": RewardTermCfg(
            func=mdp.bounce_event_reward,
            weight=10.0,
            params={
                "sensor_name": "paddle_ball_contact",
                "ball_cfg": SceneEntityCfg("ball"),
                "max_reward_velocity": 3.0,
                "min_upward_velocity": 1.1,
                "min_apex_height": 1.15,
                "min_apex_gain": 0.20,
                "min_reward_interval_steps": 8,
            },
        ),
        "ball_alive": RewardTermCfg(
            func=mdp.ball_alive,
            weight=0.1,
            params={
                "max_distance": 1.2,
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "paddle_ball_distance": RewardTermCfg(
            func=mdp.paddle_ball_distance,
            weight=0.2,
            params={
                "std": 0.2,
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "ball_height": RewardTermCfg(
            func=mdp.ball_height_reward,
            weight=1.0,
            params={
                "target_height": 1.35,
                "std": 0.25,
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "sustained_contact": RewardTermCfg(
            func=mdp.sustained_contact_penalty,
            weight=-2.0,
            params={
                "sensor_name": "paddle_ball_contact",
                "threshold": 2,
            },
        ),
        "paddle_height_ceiling": RewardTermCfg(
            func=mdp.paddle_height_ceiling_penalty,
            weight=-0.5,
            params={
                "max_paddle_height": 1.05,
                "deadband": 0.02,
                "robot_cfg": SceneEntityCfg("robot"),
            },
        ),
        "paddle_face_up": RewardTermCfg(
            func=mdp.paddle_face_up_reward,
            weight=0.5,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "min_alignment": 0.85,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.35,
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
        "posture": RewardTermCfg(
            func=mdp.posture,
            weight=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=left_arm_joint_names,
                ),
                "std": {".*": 0.3},  # Moderate tolerance for arm motion
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
                "grace_steps": 50,
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
