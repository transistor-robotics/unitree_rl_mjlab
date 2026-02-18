"""G1-specific environment configuration for keepy up task."""

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.keepyup.ball import get_pingpong_ball_cfg
from mjlab.tasks.keepyup.keepyup_env_cfg import make_keepyup_env_cfg
from mjlab.tasks.keepyup.paddle import get_g1_with_paddle_cfg


def unitree_g1_keepyup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 keepy up environment configuration.
    
    Args:
        play: If True, configure for visualization/deployment (infinite episodes, no corruption)
        
    Returns:
        Complete environment configuration for G1 keepy up task
    """
    # Start with base config
    cfg = make_keepyup_env_cfg()
    
    # Set entities: G1 with paddle + ping pong ball
    cfg.scene.entities = {
        "robot": get_g1_with_paddle_cfg(),
        "ball": get_pingpong_ball_cfg(),
    }
    
    # Configure paddle-ball contact sensor
    paddle_ball_contact = ContactSensorCfg(
        name="paddle_ball_contact",
        primary=ContactMatch(
            mode="geom",
            pattern="paddle_geom",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="geom",
            pattern="ball_geom",
            entity="ball",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    cfg.scene.sensors = (paddle_ball_contact, self_collision)
    
    # Set action scale for left arm joints using G1's actuator-specific scales
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    
    # Extract scales for left arm joints from G1_ACTION_SCALE
    left_arm_scale = {
        "left_shoulder_pitch_joint": G1_ACTION_SCALE[".*_shoulder_pitch_joint"],
        "left_shoulder_roll_joint": G1_ACTION_SCALE[".*_shoulder_roll_joint"],
        "left_shoulder_yaw_joint": G1_ACTION_SCALE[".*_shoulder_yaw_joint"],
        "left_elbow_joint": G1_ACTION_SCALE[".*_elbow_joint"],
        "left_wrist_roll_joint": G1_ACTION_SCALE[".*_wrist_roll_joint"],
        "left_wrist_pitch_joint": G1_ACTION_SCALE[".*_wrist_pitch_joint"],
        "left_wrist_yaw_joint": G1_ACTION_SCALE[".*_wrist_yaw_joint"],
    }
    
    joint_pos_action.scale = left_arm_scale
    
    # Play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)
        
        # Disable observation corruption
        cfg.observations["policy"].enable_corruption = False
    
    return cfg
