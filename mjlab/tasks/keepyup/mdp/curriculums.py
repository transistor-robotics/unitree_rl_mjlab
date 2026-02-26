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
    min_spawn_height: float
    lateral_spawn_variance: float
    frontal_spawn_variance: float
    max_throw_origin_distance: float


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


class PerformanceGatedSpawnSchedule:
    """Performance-gated spawn curriculum based on completed-episode bounce counts.

    The stage advances only when recent success is sustained and rolls back when
    low-bounce failure persists. This avoids fixed step-based difficulty cliffs.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        del cfg  # Configuration is passed into __call__ via term params.
        self._env = env
        self._stage_idx = 0

        self._ema_avg_episode_bounces: float | None = None
        self._ema_p_ge_promote: float | None = None
        self._ema_p_ge_rollback: float | None = None

        self._promote_streak = 0
        self._rollback_streak = 0
        self._last_transition = 0  # -1 rollback, +1 promote, 0 no change.
        self._episodes_accum = 0
        self._bounce_sum_accum = 0.0
        self._ge_promote_accum = 0.0
        self._ge_rollback_accum = 0.0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids  # Keep running statistics across episodes.

    @staticmethod
    def _apply_stage_to_event(
        env: "ManagerBasedRlEnv", stage: BallSpawnStage, event_term_name: str
    ) -> dict[str, float]:
        try:
            term_cfg = env.event_manager.get_term_cfg(event_term_name)
        except ValueError:
            return {}

        if stage.get("lateral_spawn_variance") is not None:
            term_cfg.params["lateral_spawn_variance"] = float(
                stage["lateral_spawn_variance"]
            )
        if stage.get("frontal_spawn_variance") is not None:
            term_cfg.params["frontal_spawn_variance"] = float(
                stage["frontal_spawn_variance"]
            )
        if stage.get("max_throw_origin_distance") is not None:
            term_cfg.params["max_throw_origin_distance"] = float(
                stage["max_throw_origin_distance"]
            )
        if stage.get("min_spawn_height") is not None:
            term_cfg.params["min_spawn_height"] = float(stage["min_spawn_height"])

        return {
            "lateral_spawn_variance": float(
                term_cfg.params.get("lateral_spawn_variance", -1.0)
            ),
            "frontal_spawn_variance": float(
                term_cfg.params.get("frontal_spawn_variance", -1.0)
            ),
            "max_throw_origin_distance": float(
                term_cfg.params.get("max_throw_origin_distance", -1.0)
            ),
            "min_spawn_height": float(term_cfg.params.get("min_spawn_height", -1.0)),
        }

    @staticmethod
    def _apply_mixed_stages_to_event(
        env: "ManagerBasedRlEnv",
        event_term_name: str,
        stages: list[BallSpawnStage],
        sampled_stage_idx: torch.Tensor,
    ) -> dict[str, float]:
        try:
            term_cfg = env.event_manager.get_term_cfg(event_term_name)
        except ValueError:
            return {}

        device = env.device
        idx = sampled_stage_idx.long().to(device)
        num = int(idx.numel())
        if num == 0:
            return {}

        def _gather_param(key: str, fallback: float) -> torch.Tensor:
            values = torch.empty((num,), dtype=torch.float32, device=device)
            for i, stage_id in enumerate(idx.tolist()):
                values[i] = float(stages[stage_id].get(key, fallback))
            return values

        term_cfg.params["min_spawn_height"] = _gather_param("min_spawn_height", 1.6)
        term_cfg.params["lateral_spawn_variance"] = _gather_param(
            "lateral_spawn_variance", 0.0
        )
        term_cfg.params["frontal_spawn_variance"] = _gather_param(
            "frontal_spawn_variance", 0.0
        )
        term_cfg.params["max_throw_origin_distance"] = _gather_param(
            "max_throw_origin_distance", 0.0
        )

        return {
            "sampled_stage_min": float(idx.min().item()),
            "sampled_stage_mean": float(idx.float().mean().item()),
            "sampled_stage_max": float(idx.max().item()),
        }

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | slice | None,
        stages: list[BallSpawnStage],
        event_term_name: str = "reset_ball",
        reward_term_name: str = "total_bounces",
        promote_bounces: float = 3.0,
        rollback_bounces: float = 1.0,
        promote_threshold: float = 0.8,
        rollback_threshold: float = 0.3,
        promote_patience: int = 6,
        rollback_patience: int = 3,
        ema_alpha: float = 0.1,
        min_episodes_per_window: int = 64,
        use_stage_mixture: bool = True,
        mix_current_prob: float = 0.5,
        mix_easier_prob: float = 0.3,
        mix_prev_random_prob: float = 0.2,
    ) -> dict[str, float]:
        if len(stages) == 0:
            return {}

        self._last_transition = 0
        reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        episode_sums = env.reward_manager._episode_sums.get(reward_term_name)
        if episode_sums is None:
            stage_values = self._apply_stage_to_event(
                env, stages[self._stage_idx], event_term_name
            )
            return {
                "stage_idx": float(self._stage_idx),
                "last_transition": float(self._last_transition),
                **stage_values,
            }

        if env_ids is None:
            done_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        elif isinstance(env_ids, slice):
            done_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)[
                env_ids
            ]
        else:
            done_env_ids = env_ids
        num_episodes = int(done_env_ids.numel())

        avg_episode_bounces_window = -1.0
        p_ge_promote_window = -1.0
        p_ge_rollback_window = -1.0

        # Skip startup reset to avoid contaminating curriculum statistics with
        # a non-episode initialization pass.
        if num_episodes > 0 and env.common_step_counter > 0:
            weight_scale = reward_cfg.weight
            if getattr(env.reward_manager, "_scale_by_dt", True):
                weight_scale *= float(env.step_dt)
            if abs(weight_scale) < 1e-8:
                weight_scale = 1.0

            bounce_counts = torch.clamp(episode_sums[done_env_ids] / weight_scale, min=0.0)
            self._episodes_accum += num_episodes
            self._bounce_sum_accum += float(bounce_counts.sum().item())
            self._ge_promote_accum += float(
                (bounce_counts >= promote_bounces).float().sum().item()
            )
            self._ge_rollback_accum += float(
                (bounce_counts >= rollback_bounces).float().sum().item()
            )

            if self._episodes_accum >= int(min_episodes_per_window):
                avg_episode_bounces_window = self._bounce_sum_accum / max(
                    1, self._episodes_accum
                )
                p_ge_promote_window = self._ge_promote_accum / max(1, self._episodes_accum)
                p_ge_rollback_window = self._ge_rollback_accum / max(1, self._episodes_accum)

                self._episodes_accum = 0
                self._bounce_sum_accum = 0.0
                self._ge_promote_accum = 0.0
                self._ge_rollback_accum = 0.0

                if self._ema_avg_episode_bounces is None:
                    self._ema_avg_episode_bounces = avg_episode_bounces_window
                    self._ema_p_ge_promote = p_ge_promote_window
                    self._ema_p_ge_rollback = p_ge_rollback_window
                else:
                    self._ema_avg_episode_bounces = (
                        (1.0 - ema_alpha) * self._ema_avg_episode_bounces
                        + ema_alpha * avg_episode_bounces_window
                    )
                    self._ema_p_ge_promote = (
                        (1.0 - ema_alpha) * self._ema_p_ge_promote
                        + ema_alpha * p_ge_promote_window
                    )
                    self._ema_p_ge_rollback = (
                        (1.0 - ema_alpha) * self._ema_p_ge_rollback
                        + ema_alpha * p_ge_rollback_window
                    )

                promote_ready = (
                    self._ema_p_ge_promote is not None
                    and self._ema_p_ge_promote > promote_threshold
                )
                rollback_ready = (
                    self._ema_p_ge_rollback is not None
                    and self._ema_p_ge_rollback < rollback_threshold
                )

                if rollback_ready:
                    self._rollback_streak += 1
                else:
                    self._rollback_streak = 0

                if promote_ready:
                    self._promote_streak += 1
                else:
                    self._promote_streak = 0

                if (
                    self._rollback_streak >= rollback_patience
                    and self._stage_idx > 0
                ):
                    self._stage_idx -= 1
                    self._rollback_streak = 0
                    self._promote_streak = 0
                    self._last_transition = -1
                elif (
                    self._promote_streak >= promote_patience
                    and self._stage_idx < len(stages) - 1
                ):
                    self._stage_idx += 1
                    self._promote_streak = 0
                    self._rollback_streak = 0
                    self._last_transition = 1

        self._stage_idx = max(0, min(self._stage_idx, len(stages) - 1))

        sampled_stage_values: dict[str, float]
        sampled_stage_idx = torch.full(
            (max(1, num_episodes),),
            self._stage_idx,
            dtype=torch.long,
            device=env.device,
        )
        if use_stage_mixture and num_episodes > 0:
            p0 = max(0.0, float(mix_current_prob))
            p1 = max(0.0, float(mix_easier_prob))
            p2 = max(0.0, float(mix_prev_random_prob))
            total = p0 + p1 + p2
            if total <= 0.0:
                p0, p1, p2 = 1.0, 0.0, 0.0
            else:
                p0, p1, p2 = p0 / total, p1 / total, p2 / total

            draw = torch.rand(num_episodes, device=env.device)
            current_mask = draw < p0
            easier_mask = (draw >= p0) & (draw < (p0 + p1))
            random_mask = ~(current_mask | easier_mask)

            sampled_stage_idx = torch.full(
                (num_episodes,),
                self._stage_idx,
                dtype=torch.long,
                device=env.device,
            )
            if self._stage_idx > 0:
                sampled_stage_idx[easier_mask] = self._stage_idx - 1
            if self._stage_idx >= 0 and random_mask.any():
                sampled_stage_idx[random_mask] = torch.randint(
                    low=0,
                    high=self._stage_idx + 1,
                    size=(int(random_mask.sum().item()),),
                    device=env.device,
                )

            sampled_stage_values = self._apply_mixed_stages_to_event(
                env=env,
                event_term_name=event_term_name,
                stages=stages,
                sampled_stage_idx=sampled_stage_idx,
            )
        else:
            sampled_stage_values = self._apply_stage_to_event(
                env, stages[self._stage_idx], event_term_name
            )

        return {
            "max_unlocked_stage_idx": float(self._stage_idx),
            "stage_idx": float(self._stage_idx),
            "episodes_in_window": float(num_episodes),
            "episodes_accum_toward_window": float(self._episodes_accum),
            "avg_episode_bounces_window": float(avg_episode_bounces_window),
            "p_ge_promote_window": float(p_ge_promote_window),
            "p_ge_rollback_window": float(p_ge_rollback_window),
            "ema_avg_episode_bounces": float(self._ema_avg_episode_bounces)
            if self._ema_avg_episode_bounces is not None
            else -1.0,
            "ema_p_ge_promote": float(self._ema_p_ge_promote)
            if self._ema_p_ge_promote is not None
            else -1.0,
            "ema_p_ge_rollback": float(self._ema_p_ge_rollback)
            if self._ema_p_ge_rollback is not None
            else -1.0,
            "promote_streak": float(self._promote_streak),
            "rollback_streak": float(self._rollback_streak),
            "last_transition": float(self._last_transition),
            "mixture_enabled": 1.0 if use_stage_mixture else 0.0,
            "mixture_prob_current": float(mix_current_prob),
            "mixture_prob_easier": float(mix_easier_prob),
            "mixture_prob_prev_random": float(mix_prev_random_prob),
            **sampled_stage_values,
        }


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

    for key in (
        "apex_std",
        "velocity_std",
        "vel_weight",
        "vert_std",
        "vert_weight",
        "min_upward_velocity",
    ):
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
            estimator._update_prob = min(
                1.0, float(active["camera_fps"]) * float(estimator._step_dt)
            )
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
        "camera_fps_equiv": float(active["camera_fps"])
        if active.get("camera_fps") is not None
        else -1.0,
        "update_prob": float(active["update_prob"])
        if active.get("update_prob") is not None
        else -1.0,
        "dropout_prob": float(active["dropout_prob"])
        if active.get("dropout_prob") is not None
        else -1.0,
        "pos_noise_std": float(active["pos_noise_std"])
        if active.get("pos_noise_std") is not None
        else -1.0,
        "vel_noise_std": float(active["vel_noise_std"])
        if active.get("vel_noise_std") is not None
        else -1.0,
    }


def ball_spawn_difficulty_schedule(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    stages: list[BallSpawnStage],
    event_term_name: str = "reset_ball",
) -> dict[str, float]:
    """Curriculum for keepyup ball reset spawn variance."""

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

    if active.get("lateral_spawn_variance") is not None:
        term_cfg.params["lateral_spawn_variance"] = float(
            active["lateral_spawn_variance"]
        )
    if active.get("frontal_spawn_variance") is not None:
        term_cfg.params["frontal_spawn_variance"] = float(
            active["frontal_spawn_variance"]
        )
    if active.get("max_throw_origin_distance") is not None:
        term_cfg.params["max_throw_origin_distance"] = float(
            active["max_throw_origin_distance"]
        )
    if active.get("min_spawn_height") is not None:
        term_cfg.params["min_spawn_height"] = float(active["min_spawn_height"])

    return {
        "stage_idx": float(active_stage_idx),
        "lateral_spawn_variance": float(
            term_cfg.params.get("lateral_spawn_variance", -1.0)
        ),
        "frontal_spawn_variance": float(
            term_cfg.params.get("frontal_spawn_variance", -1.0)
        ),
        "max_throw_origin_distance": float(
            term_cfg.params.get("max_throw_origin_distance", -1.0)
        ),
        "min_spawn_height": float(term_cfg.params.get("min_spawn_height", -1.0)),
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
            discovery_cfg.params["min_upward_velocity"] = float(
                active["min_upward_velocity"]
            )
        if active.get("min_apex_height") is not None:
            discovery_cfg.params["min_apex_height"] = float(active["min_apex_height"])
        if active.get("min_apex_gain") is not None:
            discovery_cfg.params["min_apex_gain"] = float(active["min_apex_gain"])
        if active.get("target_upward_velocity") is not None:
            discovery_cfg.params["target_upward_velocity"] = float(
                active["target_upward_velocity"]
            )

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
        "discovery_weight": float(discovery_cfg.weight)
        if discovery_cfg is not None
        else -1.0,
        "lateral_weight": float(lateral_cfg.weight)
        if lateral_cfg is not None
        else -1.0,
        "under_ball_weight": float(under_ball_cfg.weight)
        if under_ball_cfg is not None
        else -1.0,
        "strike_plane_weight": float(strike_plane_cfg.weight)
        if strike_plane_cfg is not None
        else -1.0,
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
