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


class BallSpawnStage(TypedDict, total=False):
    """Stage definition for reset-ball spawn difficulty."""

    step: int
    pose_range: dict[str, tuple[float, float]]
    hit_probability: float
    hit_radius_fraction: float
    miss_radius_range: tuple[float, float]
    entry_angle_deg_range: tuple[float, float]
    time_to_impact_range: tuple[float, float]


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


def ball_spawn_difficulty_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    stages: list[BallSpawnStage],
    event_term_name: str = "reset_ball",
) -> dict[str, float]:
    """Curriculum for targeted ball reset difficulty.

    This progressively increases spawn diversity by adjusting:
    - hit probability (how often spawn targets paddle center region),
    - impact entry angle range from vertical,
    - miss radius and spawn volume ranges.
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

    try:
        term_cfg = env.event_manager.get_term_cfg(event_term_name)
    except ValueError:
        return {}

    if active.get("pose_range") is not None:
        term_cfg.params["pose_range"] = dict(active["pose_range"])
    if active.get("hit_probability") is not None:
        term_cfg.params["hit_probability"] = float(active["hit_probability"])
    if active.get("hit_radius_fraction") is not None:
        term_cfg.params["hit_radius_fraction"] = float(active["hit_radius_fraction"])
    if active.get("miss_radius_range") is not None:
        term_cfg.params["miss_radius_range"] = tuple(active["miss_radius_range"])
    if active.get("entry_angle_deg_range") is not None:
        term_cfg.params["entry_angle_deg_range"] = tuple(active["entry_angle_deg_range"])
    if active.get("time_to_impact_range") is not None:
        term_cfg.params["time_to_impact_range"] = tuple(active["time_to_impact_range"])

    pose_range = term_cfg.params.get("pose_range", {})
    entry_rng = term_cfg.params.get("entry_angle_deg_range", (0.0, 0.0))

    return {
        "stage_idx": float(active_stage_idx),
        "hit_probability": float(term_cfg.params.get("hit_probability", -1.0)),
        "entry_angle_min_deg": float(entry_rng[0]),
        "entry_angle_max_deg": float(entry_rng[1]),
        "spawn_x_span": float(pose_range.get("x", (0.0, 0.0))[1] - pose_range.get("x", (0.0, 0.0))[0]),
        "spawn_y_span": float(pose_range.get("y", (0.0, 0.0))[1] - pose_range.get("y", (0.0, 0.0))[0]),
        "spawn_z_span": float(pose_range.get("z", (0.0, 0.0))[1] - pose_range.get("z", (0.0, 0.0))[0]),
    }
