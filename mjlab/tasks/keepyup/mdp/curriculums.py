"""Curriculum helpers for keepyup."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class VisionNoiseStage(TypedDict):
    """Stage definition for camera-estimator noise curriculum."""

    step: int
    camera_fps: float | None
    update_prob: float | None
    dropout_prob: float | None
    pos_noise_std: float | None
    vel_noise_std: float | None
    outlier_prob: float | None
    outlier_std: float | None
    stale_vel_decay: float | None


def ball_state_noise_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    stages: list[VisionNoiseStage],
    term_name: str = "ball_state",
    groups: tuple[str, ...] = ("policy", "critic"),
) -> dict[str, float]:
    """Stage camera-estimator noise from near-oracle to realistic settings.

    The schedule is keyed by ``env.common_step_counter``. This is intentionally
    global so all environments share the same curriculum phase.
    """
    del env_ids  # Unused. Curriculum is global.

    current_step = env.common_step_counter
    active_stage_idx = 0
    for i, stage in enumerate(stages):
        if current_step >= stage["step"]:
            active_stage_idx = i
        else:
            break
    active = stages[active_stage_idx]

    # Apply active stage to each requested observation group term.
    applied_any = False
    for group in groups:
        try:
            term_cfg = env.observation_manager.get_term_cfg(group, term_name)
        except ValueError:
            continue

        estimator = term_cfg.func

        # Either derive update probability from fps, or use explicit value.
        if active.get("camera_fps") is not None:
            estimator._update_prob = min(1.0, float(active["camera_fps"]) * float(estimator._step_dt))
        if active.get("update_prob") is not None:
            estimator._update_prob = float(active["update_prob"])

        if active.get("dropout_prob") is not None:
            estimator._dropout_prob = float(active["dropout_prob"])
        if active.get("pos_noise_std") is not None:
            estimator._pos_noise_std = float(active["pos_noise_std"])
        if active.get("vel_noise_std") is not None:
            estimator._vel_noise_std = float(active["vel_noise_std"])
        if active.get("outlier_prob") is not None:
            estimator._outlier_prob = float(active["outlier_prob"])
        if active.get("outlier_std") is not None:
            estimator._outlier_std = float(active["outlier_std"])
        if active.get("stale_vel_decay") is not None:
            estimator._stale_vel_decay = float(active["stale_vel_decay"])
        applied_any = True

    if not applied_any:
        return {}

    return {
        "stage_idx": float(active_stage_idx),
        "camera_fps_equiv": float(active["camera_fps"]) if active.get("camera_fps") is not None else -1.0,
        "update_prob": float(active["update_prob"]) if active.get("update_prob") is not None else -1.0,
        "dropout_prob": float(active["dropout_prob"]) if active.get("dropout_prob") is not None else -1.0,
        "pos_noise_std": float(active["pos_noise_std"]) if active.get("pos_noise_std") is not None else -1.0,
        "vel_noise_std": float(active["vel_noise_std"]) if active.get("vel_noise_std") is not None else -1.0,
    }
