"""Termination functions for the keepy up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Import reusable terminations from base MDP
from mjlab.envs.mdp.terminations import bad_orientation, time_out  # noqa: F401


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


class ball_out_of_frame:
    """Terminate when ball is too far from the robot base frame for too long.
    
    Tracks consecutive steps where the ball is outside a base-frame distance
    threshold. Terminates when this exceeds grace_steps.
    """
    
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        """Initialize termination counter."""
        self.consecutive_invisible = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset invisible counters for specified environments."""
        if env_ids is None:
            self.consecutive_invisible[:] = 0
        else:
            self.consecutive_invisible[env_ids] = 0
        
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        grace_steps: int = 50,
        max_distance: float = 1.2,
        robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
        ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    ) -> torch.Tensor:
        """Check if ball has been out of range for too long.
        
        Args:
            env: The environment
            grace_steps: Number of consecutive out-of-range steps before termination
            max_distance: Max allowed Euclidean distance in robot base frame (meters)
            robot_cfg: Robot entity configuration
            ball_cfg: Ball entity configuration
            
        Returns:
            Boolean tensor indicating which environments should terminate [num_envs]
        """
        # Compute ball position in base frame and threshold by distance.
        from mjlab.tasks.keepyup.mdp.observations import ball_pos_in_base_frame

        ball_pos_b = ball_pos_in_base_frame(env, robot_cfg, ball_cfg)  # [B, 3]
        ball_dist = torch.norm(ball_pos_b, dim=-1)  # [B]
        in_range = ball_dist <= max_distance
        
        # Update consecutive out-of-range counter.
        self.consecutive_invisible = torch.where(
            in_range,
            torch.zeros_like(self.consecutive_invisible),
            self.consecutive_invisible + 1,
        )
        
        # Terminate if out-of-range for too many consecutive steps.
        terminate = self.consecutive_invisible > grace_steps
        
        # Log metrics
        log = env.extras.setdefault("log", {})
        log["Metrics/consecutive_invisible_max"] = torch.max(
            self.consecutive_invisible.float()
        )
        log["Metrics/consecutive_invisible_mean"] = torch.mean(
            self.consecutive_invisible.float()
        )
        
        return terminate
