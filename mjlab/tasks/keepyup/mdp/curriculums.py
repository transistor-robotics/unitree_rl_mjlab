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
    spawn_height: float
    hit_probability: float
    hit_radius_fraction: float
    miss_radius_range: tuple[float, float]
    entry_angle_deg_range: tuple[float, float]


class BounceQualityStage(TypedDict, total=False):
    """Stage definition for bounce_quality_reward curriculum."""

    step: int
    apex_std: float
    velocity_std: float
    vel_weight: float
    vert_std: float
    vert_weight: float
    min_upward_velocity: float


class BounceRewardStage(TypedDict, total=False):
    """Stage definition for bounce-discovery reward shaping."""

    step: int
    discovery_weight: float
    lateral_weight: float
    under_ball_weight: float
    strike_plane_weight: float
    min_upward_velocity: float
    min_apex_height: float
    min_apex_gain: float
    target_upward_velocity: float


def bounce_quality_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    stages: list[BounceQualityStage],
    reward_term_name: str = "bounce_quality",
) -> dict[str, float]:
    """Progressively tighten bounce_quality_reward criteria over training.

    Starts very forgiving (any upward rebound near the apex scores well) and
    tightens three knobs over curriculum stages:

    - ``apex_std``         — narrows the apex Gaussian (strictness of height).
    - ``vel_weight``       — blends in velocity scoring (0=ignored, 1=full).
    - ``vert_weight``      — blends in verticality scoring (0=ignored, 1=full).
    - ``min_upward_velocity`` — raises the minimum threshold to reject micro-taps.
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
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    except ValueError:
        return {}

    for key in ("apex_std", "velocity_std", "vel_weight", "vert_std", "vert_weight", "min_upward_velocity"):
        if active.get(key) is not None:
            term_cfg.params[key] = float(active[key])

    return {
        "stage_idx": float(active_stage_idx),
        "apex_std": float(term_cfg.params.get("apex_std", -1.0)),
        "vel_weight": float(term_cfg.params.get("vel_weight", -1.0)),
        "vert_weight": float(term_cfg.params.get("vert_weight", -1.0)),
        "min_upward_velocity": float(term_cfg.params.get("min_upward_velocity", -1.0)),
    }


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

    if active.get("spawn_height") is not None:
        term_cfg.params["spawn_height"] = float(active["spawn_height"])
    if active.get("hit_probability") is not None:
        term_cfg.params["hit_probability"] = float(active["hit_probability"])
    if active.get("hit_radius_fraction") is not None:
        term_cfg.params["hit_radius_fraction"] = float(active["hit_radius_fraction"])
    if active.get("miss_radius_range") is not None:
        term_cfg.params["miss_radius_range"] = tuple(active["miss_radius_range"])
    if active.get("entry_angle_deg_range") is not None:
        term_cfg.params["entry_angle_deg_range"] = tuple(active["entry_angle_deg_range"])

    entry_rng = term_cfg.params.get("entry_angle_deg_range", (0.0, 0.0))

    return {
        "stage_idx": float(active_stage_idx),
        "spawn_height": float(term_cfg.params.get("spawn_height", -1.0)),
        "hit_probability": float(term_cfg.params.get("hit_probability", -1.0)),
        "entry_angle_min_deg": float(entry_rng[0]),
        "entry_angle_max_deg": float(entry_rng[1]),
    }


def bounce_reward_shaping_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    stages: list[BounceRewardStage],
    discovery_term_name: str = "bounce_discovery",
    lateral_term_name: str = "lateral_drift",
    under_ball_term_name: str = "under_ball_alignment",
    strike_plane_term_name: str = "strike_plane_hold",
) -> dict[str, float]:
    """Progressively tighten bounce-discovery criteria and helper-term weights."""

    del env_ids  # Unused. Curriculum is global.

    current_step = env.common_step_counter
    active_stage_idx = 0
    for i, stage in enumerate(stages):
        if current_step >= stage["step"]:
            active_stage_idx = i
        else:
            break
    active = stages[active_stage_idx]

    def _maybe_get_reward_cfg(term_name: str):
        try:
            return env.reward_manager.get_term_cfg(term_name)
        except ValueError:
            return None

    discovery_cfg = _maybe_get_reward_cfg(discovery_term_name)
    if discovery_cfg is not None:
        if active.get("discovery_weight") is not None:
            discovery_cfg.weight = float(active["discovery_weight"])
        if active.get("min_upward_velocity") is not None:
            discovery_cfg.params["min_upward_velocity"] = float(active["min_upward_velocity"])
        if active.get("min_apex_height") is not None:
            discovery_cfg.params["min_apex_height"] = float(active["min_apex_height"])
        if active.get("min_apex_gain") is not None:
            discovery_cfg.params["min_apex_gain"] = float(active["min_apex_gain"])
        if active.get("target_upward_velocity") is not None:
            discovery_cfg.params["target_upward_velocity"] = float(active["target_upward_velocity"])

    lateral_cfg = _maybe_get_reward_cfg(lateral_term_name)
    if lateral_cfg is not None and active.get("lateral_weight") is not None:
        lateral_cfg.weight = float(active["lateral_weight"])

    under_ball_cfg = _maybe_get_reward_cfg(under_ball_term_name)
    if under_ball_cfg is not None and active.get("under_ball_weight") is not None:
        under_ball_cfg.weight = float(active["under_ball_weight"])

    strike_plane_cfg = _maybe_get_reward_cfg(strike_plane_term_name)
    if strike_plane_cfg is not None and active.get("strike_plane_weight") is not None:
        strike_plane_cfg.weight = float(active["strike_plane_weight"])

    return {
        "stage_idx": float(active_stage_idx),
        "discovery_weight": float(discovery_cfg.weight) if discovery_cfg is not None else -1.0,
        "lateral_weight": float(lateral_cfg.weight) if lateral_cfg is not None else -1.0,
        "under_ball_weight": float(under_ball_cfg.weight) if under_ball_cfg is not None else -1.0,
        "strike_plane_weight": float(strike_plane_cfg.weight) if strike_plane_cfg is not None else -1.0,
        "discovery_min_upward_vz": (
            float(discovery_cfg.params.get("min_upward_velocity", -1.0))
            if discovery_cfg is not None
            else -1.0
        ),
        "discovery_min_apex_height": (
            float(discovery_cfg.params.get("min_apex_height", -1.0))
            if discovery_cfg is not None
            else -1.0
        ),
    }
