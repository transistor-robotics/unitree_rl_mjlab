"""RL configuration for Unitree G1 keepy up task."""

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_g1_keepyup_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree G1 keepy up task.
    
    Uses smaller networks than the velocity task since the problem is
    more constrained (7 DOF arm control vs full-body locomotion).
    """
    return RslRlOnPolicyRunnerCfg(
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=(256, 128, 64),
            critic_hidden_dims=(256, 128, 64),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,  # Encourage exploration
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_keepyup",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=5001,
    )
