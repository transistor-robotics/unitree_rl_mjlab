"""Ping pong ball entity configuration."""

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def get_pingpong_ball_cfg() -> EntityCfg:
    """Create a ping pong ball entity configuration.
    
    Returns a floating non-articulated entity (sphere with freejoint).
    
    Physical properties:
    - Radius: 20mm (regulation ping pong ball is 40mm diameter)
    - Mass: 2.7g (regulation weight)
    - Bounce: tuned for ~0.7-0.8 coefficient of restitution on rubber paddle
    
    Returns:
        EntityCfg: Configuration for the ping pong ball entity
    """
    
    def create_ball_spec() -> mujoco.MjSpec:
        """Build MuJoCo spec for ping pong ball."""
        spec = mujoco.MjSpec()
        
        # Create body with freejoint (6 DOF floating)
        body = spec.worldbody.add_body(name="ball")
        body.add_freejoint(name="ball_freejoint")
        
        # Add sphere geom with ping pong ball properties
        geom = body.add_geom(
            name="ball_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.02,),  # 20mm radius = 40mm diameter
            rgba=(1.0, 1.0, 1.0, 1.0),  # White
            mass=0.0027,  # 2.7g - set mass directly on geom
        )
        
        # Contact properties for realistic bounce
        # solref controls bounce: [timeconst, dampratio]
        # For coefficient of restitution ~0.7-0.8:
        # - timeconst ~0.002 (contact duration)
        # - dampratio ~0.5 (energy dissipation)
        geom.solref = (0.002, 0.5)
        
        # solimp controls contact impedance
        # [dmin, dmax, width, midpoint, power]
        geom.solimp = (0.9, 0.95, 0.001, 0.5, 2)
        
        return spec
    
    return EntityCfg(
        spec_fn=create_ball_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # Origin; reset_ball event sets absolute position
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),  # Zero; reset_ball event sets absolute velocity
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=("ball_geom",),
                contype=1,
                conaffinity=1,
                condim=3,  # Full 3D friction cone
                friction=(0.8, 0.005, 0.0001),  # [sliding, torsional, rolling]
            ),
        ),
    )
