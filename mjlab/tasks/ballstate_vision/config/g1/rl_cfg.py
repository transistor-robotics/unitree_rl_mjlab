"""Placeholder RL config for ball-state vision task registration."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ballstate_vision_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Minimal RL config placeholder (task is primarily trained supervised)."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=False,
      critic_obs_normalization=False,
      actor_hidden_dims=(128, 64),
      critic_hidden_dims=(128, 64),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=3,
      num_mini_batches=2,
      learning_rate=1.0e-3,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_ballstate_vision",
    save_interval=200,
    num_steps_per_env=16,
    max_iterations=1000,
  )

