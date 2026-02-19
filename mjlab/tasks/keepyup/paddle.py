"""G1 robot with paddle entity configuration."""

import math

import mujoco

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import EntityCfg

# Initial state:
# - start from G1 home keyframe defaults (stable standing morphology in this model)
# - override right elbow and left wrist roll for keepyup setup
# - keep all unspecified joints at 0.0
_KEEPYUP_JOINT_POS = {
    # Keep right arm tucked further down/out of the way than HOME_KEYFRAME.
    "right_elbow_joint": 1.0,
    "left_elbow_joint": 0.3,
    # Pronate and adjust left wrist to hold paddle parallel with floor
    "left_wrist_roll_joint": 1.48,
    "left_wrist_yaw_joint": 0.6,
    "left_wrist_pitch_joint": -0.2,
    # Start from HOME standing defaults for remaining joints.
    **{
        k: v
        for k, v in g1_constants.HOME_KEYFRAME.joint_pos.items()
        if k != ".*_elbow_joint"
    },
}

KEEPYUP_INIT_STATE = EntityCfg.InitialStateCfg(
    # Fixed-base keepyup runs should be visually grounded in world frame.
    # With the freejoint removed, setting pelvis z=0 places feet near floor.
    pos=(0.0, 0.0, 0.0),
    joint_pos=_KEEPYUP_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_g1_with_paddle_cfg(fixed_base: bool = True) -> EntityCfg:
    """Create G1 robot configuration with paddle attached to left hand.

    Modifications to standard G1:
    1. Uses zero-joint initial state (safe standing) so locked joints are consistent
    2. Adds a paddle body welded to left_wrist_yaw_link
    3. Adds paddle_face site at paddle surface center
    4. Adds head_camera site on torso for frustum computation

    Returns:
        EntityCfg: Configuration for G1 robot with paddle attached
    """

    def create_g1_with_paddle_spec() -> mujoco.MjSpec:
        """Build MuJoCo spec for G1 with paddle."""
        # Start with standard G1 spec
        spec = g1_constants.get_spec()

        # For keepyup we isolate the arm behavior: remove floating-base dynamics
        # so the robot cannot tip/fall while calibrating paddle and ball behavior.
        if fixed_base:
            free_joint = spec.joint("floating_base_joint")
            spec.delete(free_joint)

        # ------------------------------------------------------------------
        # Attach paddle to left hand
        # ------------------------------------------------------------------
        left_wrist_yaw = spec.body("left_wrist_yaw_link")

        # Paddle body welded to left hand (no joint = fixed attachment).
        # Calibration target:
        # - center of paddle face is 130 mm from palm center
        # - from top-down on a pronated left hand, paddle is on the hand's right side
        #
        # We approximate palm-center in wrist frame at (0.12, 0.0, 0.0), then offset
        # 13 cm along Z as "right side" of left hand.
        palm_center_in_wrist = (0.12, 0.0, 0.0)
        paddle_center_from_palm = (0.0, 0.0, 0.13)
        paddle_pos = (
            palm_center_in_wrist[0] + paddle_center_from_palm[0],
            palm_center_in_wrist[1] + paddle_center_from_palm[1],
            palm_center_in_wrist[2] + paddle_center_from_palm[2],
        )
        paddle_body = left_wrist_yaw.add_body(name="paddle", pos=paddle_pos)

        # Rotate paddle so the face points toward the ground (normal down) instead
        # of robot-left in the current wrist frame.
        paddle_quat = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)

        # Thin cylinder (disc shape): radius 7.5cm, half-thickness 4mm
        paddle_geom = paddle_body.add_geom(
            name="paddle_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(0.075, 0.004),  # [radius, half-height]
            rgba=(0.9, 0.1, 0.1, 1.0),
            quat=paddle_quat,
            mass=0.05,
        )

        paddle_geom.solref = (0.002, 0.6)
        paddle_geom.solimp = (0.9, 0.95, 0.001, 0.5, 2)

        paddle_body.add_site(
            name="paddle_face",
            pos=(0.0, 0.004, 0.0),
            size=(0.01,),
            rgba=(1.0, 0.0, 0.0, 0.5),
        )

        # ------------------------------------------------------------------
        # Head camera site for frustum computation
        # ------------------------------------------------------------------
        torso = spec.body("torso_link")

        # 42.4-degree downward tilt from forward-looking
        tilt_rad = math.radians(42.4)
        camera_quat = (
            math.cos(tilt_rad / 2),
            0.0,
            math.sin(tilt_rad / 2),
            0.0,
        )

        torso.add_site(
            name="head_camera",
            pos=(0.0, 0.0, 0.43),
            quat=camera_quat,
            size=(0.01,),
            rgba=(0.0, 1.0, 0.0, 0.5),
        )

        return spec

    # Get the base G1 config, override spec and initial state
    cfg = g1_constants.get_g1_robot_cfg()
    cfg.spec_fn = create_g1_with_paddle_spec
    cfg.init_state = KEEPYUP_INIT_STATE

    # Add collision config for the paddle
    from mjlab.utils.spec_config import CollisionCfg

    paddle_collision = CollisionCfg(
        geom_names_expr=("paddle_geom",),
        contype=1,
        conaffinity=1,
        condim=3,
        friction=(0.5, 0.005, 0.0001),  # Slightly reduced to discourage sticky scoops
        disable_other_geoms=False,  # Don't clobber the robot's own collision geoms
    )

    # Append paddle collision to existing collisions
    cfg.collisions = cfg.collisions + (paddle_collision,)

    return cfg
