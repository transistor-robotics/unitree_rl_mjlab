"""Observation functions for the keepy up task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def ball_pos_in_base_frame(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Compute ball position relative to robot pelvis in body frame.
    
    Returns:
        torch.Tensor: Ball position in robot base frame [num_envs, 3]
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    
    # Get ball position in world frame
    ball_pos_w = ball.data.root_link_pos_w  # [num_envs, 3]
    
    # Get robot root position and orientation
    robot_pos_w = robot.data.root_link_pos_w  # [num_envs, 3]
    robot_quat_w = robot.data.root_link_quat_w  # [num_envs, 4]
    
    # Compute relative position in world frame
    rel_pos_w = ball_pos_w - robot_pos_w  # [num_envs, 3]
    
    # Transform to robot body frame
    rel_pos_b = quat_apply_inverse(robot_quat_w, rel_pos_w)
    
    return rel_pos_b


def ball_vel_in_base_frame(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Compute ball linear velocity in robot body frame.
    
    Returns:
        torch.Tensor: Ball velocity in robot base frame [num_envs, 3]
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    
    # Get ball velocity in world frame
    ball_vel_w = ball.data.root_link_lin_vel_w  # [num_envs, 3]
    
    # Get robot orientation
    robot_quat_w = robot.data.root_link_quat_w  # [num_envs, 4]
    
    # Transform velocity to robot body frame
    ball_vel_b = quat_apply_inverse(robot_quat_w, ball_vel_w)
    
    return ball_vel_b


def ball_visible(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Check if ball is within the head camera's frustum.
    
    Uses Intel RealSense D455i specifications:
    - Camera tilt: 42.4 degrees from vertical (downward)
    - Horizontal FOV: 87 degrees (~43.5 deg half-angle)
    - Vertical FOV: 55.2 degrees (~27.6 deg half-angle)
    
    Returns:
        torch.Tensor: Visibility flag, 1.0 if visible, 0.0 otherwise [num_envs, 1]
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    
    # Get head_camera site pose
    # Create a temporary config to resolve the site ID
    robot_ids, robot_site_names = robot.find_sites("head_camera")
    if len(robot_ids) == 0:
        # Fallback: if head_camera site isn't found, assume ball is always visible
        return torch.ones(env.num_envs, 1, device=env.device)
    
    head_camera_idx = robot_ids[0]  # Get the first (and only) match
    
    head_camera_pos_w = robot.data.site_pos_w[:, head_camera_idx, :]  # [B, 3]
    head_camera_quat_w = robot.data.site_quat_w[:, head_camera_idx, :]  # [B, 4]
    
    # Get ball position
    ball_pos_w = ball.data.root_link_pos_w  # [B, 3]
    
    # Compute ball position relative to camera in camera frame
    rel_pos_w = ball_pos_w - head_camera_pos_w
    rel_pos_cam = quat_apply_inverse(head_camera_quat_w, rel_pos_w)  # [B, 3]
    
    # In camera frame: +Z is forward (optical axis), +X is right, +Y is down
    # Ball must be in front of camera (positive Z)
    forward_check = rel_pos_cam[:, 2] > 0.05  # At least 5cm in front
    
    # Compute angles from optical axis
    # Horizontal angle: atan2(x, z) - angle in XZ plane
    horizontal_angle = torch.atan2(torch.abs(rel_pos_cam[:, 0]), rel_pos_cam[:, 2])
    h_fov_half = math.radians(87.0 / 2.0)  # 43.5 degrees
    horizontal_check = horizontal_angle < h_fov_half
    
    # Vertical angle: atan2(y, z) - angle in YZ plane
    vertical_angle = torch.atan2(torch.abs(rel_pos_cam[:, 1]), rel_pos_cam[:, 2])
    v_fov_half = math.radians(55.2 / 2.0)  # 27.6 degrees
    vertical_check = vertical_angle < v_fov_half
    
    # Distance check: ball must be within reasonable tracking range (e.g., 2 meters)
    distance = torch.norm(rel_pos_cam, dim=-1)
    distance_check = distance < 2.0
    
    # Combine all checks
    visible = forward_check & horizontal_check & vertical_check & distance_check
    
    return visible.float().unsqueeze(-1)  # [B, 1]


def left_arm_joint_pos_rel(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Get left arm joint positions relative to default pose.
    
    Returns:
        torch.Tensor: Joint positions [num_envs, 7] for the 7 left arm DOF
    """
    robot: Entity = env.scene[robot_cfg.name]
    
    # Configure for left arm joints only
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
    
    # Get current joint positions for left arm
    current_pos = robot.data.joint_pos[:, left_arm_cfg.joint_ids]  # [B, 7]
    
    # Get default joint positions for left arm
    default_pos = robot.data.default_joint_pos[:, left_arm_cfg.joint_ids]  # [B, 7]
    
    # Return relative position (current - default)
    return current_pos - default_pos


def left_arm_joint_vel(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Get left arm joint velocities.
    
    Returns:
        torch.Tensor: Joint velocities [num_envs, 7] for the 7 left arm DOF
    """
    robot: Entity = env.scene[robot_cfg.name]
    
    # Configure for left arm joints only
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
    
    # Get joint velocities for left arm
    return robot.data.joint_vel[:, left_arm_cfg.joint_ids]  # [B, 7]


def ball_ang_vel_in_base_frame(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Compute ball angular velocity in robot body frame.
    
    This is for the critic only (not available in real deployment).
    
    Returns:
        torch.Tensor: Ball angular velocity in robot base frame [num_envs, 3]
    """
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    
    # Get ball angular velocity in world frame
    ball_angvel_w = ball.data.root_link_ang_vel_w  # [num_envs, 3]
    
    # Get robot orientation
    robot_quat_w = robot.data.root_link_quat_w  # [num_envs, 4]
    
    # Transform angular velocity to robot body frame
    ball_angvel_b = quat_apply_inverse(robot_quat_w, ball_angvel_w)
    
    return ball_angvel_b
