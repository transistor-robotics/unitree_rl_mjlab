"""Keepy up task-specific action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SlewRateJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with per-step slew-rate limiting.

    The limiter bounds the target change each environment step:
    |q_target[t] - q_target[t-1]| <= max_velocity * step_dt
    """

    max_velocity: float | dict[str, float] = float("inf")
    """Maximum joint velocity [rad/s] used to limit target change per step."""

    clip_to_joint_limits: bool = True
    """Whether to clip limited targets to the entity's joint position limits."""

    def build(self, env: ManagerBasedRlEnv) -> SlewRateJointPositionAction:
        return SlewRateJointPositionAction(self, env)


class SlewRateJointPositionAction(JointPositionAction):
    """Position action that limits target update rate for safer motion."""

    cfg: SlewRateJointPositionActionCfg

    def __init__(self, cfg: SlewRateJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        if isinstance(cfg.max_velocity, (float, int)):
            max_velocity = torch.full(
                (self.num_envs, self.action_dim),
                float(cfg.max_velocity),
                device=self.device,
            )
        elif isinstance(cfg.max_velocity, dict):
            max_velocity = torch.full(
                (self.num_envs, self.action_dim),
                float("inf"),
                device=self.device,
            )
            idx, _, values = resolve_matching_names_values(cfg.max_velocity, self._target_names)
            max_velocity[:, idx] = torch.tensor(values, device=self.device)
        else:
            raise ValueError(
                f"Unsupported max_velocity type: {type(cfg.max_velocity)}. "
                "Supported types are float and dict."
            )

        self._max_delta = max_velocity * self._env.step_dt
        self._prev_target = self._entity.data.joint_pos_biased[:, self._target_ids].clone()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        current = self._entity.data.joint_pos_biased[:, self._target_ids]
        self._prev_target[env_ids] = current[env_ids]

    def apply_actions(self) -> None:
        desired_target = self._processed_actions
        delta = desired_target - self._prev_target
        limited_target = self._prev_target + torch.clamp(
            delta, min=-self._max_delta, max=self._max_delta
        )

        if self.cfg.clip_to_joint_limits:
            joint_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
            limited_target = torch.clip(
                limited_target, joint_limits[..., 0], joint_limits[..., 1]
            )

        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = limited_target - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
        self._prev_target[:] = limited_target
