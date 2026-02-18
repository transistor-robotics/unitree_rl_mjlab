"""Visual inspector for the keepy up environment.

Launches the environment with a single env and zero/random policy
so you can inspect paddle placement, ball spawn, and overall setup.

Usage:
    python scripts/view_keepyup.py               # Zero policy (robot holds still)
    python scripts/view_keepyup.py --random       # Random actions
    python scripts/view_keepyup.py --viser        # Web-based viewer (headless)
"""

import argparse
import os

import torch

os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    parser = argparse.ArgumentParser(description="View keepy up environment")
    parser.add_argument(
        "--random", action="store_true", help="Use random actions instead of zero"
    )
    parser.add_argument(
        "--viser", action="store_true", help="Use web-based viser viewer"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of environments"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Use play config (very long episodes, no obs corruption)",
    )
    args = parser.parse_args()

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg("Mjlab-KeepyUp-Unitree-G1", play=args.play)
    agent_cfg = load_rl_cfg("Mjlab-KeepyUp-Unitree-G1")
    env_cfg.scene.num_envs = args.num_envs

    print(f"[INFO] Creating environment with {args.num_envs} env(s) on {device}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_shape = env.unwrapped.action_space.shape

    if args.random:
        print("[INFO] Using random policy")

        class PolicyRandom:
            def __call__(self, obs):
                return 2 * torch.rand(action_shape, device=device) - 1

        policy = PolicyRandom()
    else:
        print("[INFO] Using zero policy (robot holds default pose)")

        class PolicyZero:
            def __call__(self, obs):
                return torch.zeros(action_shape, device=device)

        policy = PolicyZero()

    if args.viser:
        from mjlab.viewer import ViserPlayViewer

        print("[INFO] Starting viser viewer (check terminal for URL)")
        ViserPlayViewer(env, policy).run()
    else:
        from mjlab.viewer import NativeMujocoViewer

        print("[INFO] Starting native MuJoCo viewer")
        NativeMujocoViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
