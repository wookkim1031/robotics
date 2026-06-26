"""
Stage 3 — train DeepMimic tracking on a reference clip.

    python train_mimic.py --model mujoco_menagerie/unitree_g1/scene.xml \
        --clip <path-to-walk.csv> --iterations 500

Reuses the SAME PPO you wrote in Stage 2 — only the environment changes. That's
the payoff of keeping PPO agnostic to the env's reward/obs: balancing and
imitation are the same algorithm pointed at different worlds.
"""

import argparse
from motion_lib import MotionLib, load_lafan1_g1_csv
from mimic_env import MimicEnv
from ppo import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="G1 SCENE xml (with a floor)")
    ap.add_argument("--clip", required=True, help="LAFAN1 G1 .csv to imitate")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--out", default="ppo_mimic.pt")
    args = ap.parse_args()

    frames = load_lafan1_g1_csv(args.clip)
    env = MimicEnv(args.model, MotionLib(frames, fps=30.0))
    print(f"DeepMimic: obs={env.obs_dim} act={env.action_dim} "
          f"clip={env.motion.duration:.1f}s  max_steps={env.max_steps}")
    agent = PPO(env, horizon=2048)
    agent.train(args.iterations)
    agent.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()