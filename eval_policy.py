"""
Stage 2 — watch the trained policy recover.

Loads a checkpoint saved by ppo.py, runs the actor's MEAN action (no exploration
noise) so you see the deterministic learned behavior, and renders frames to a
video. Crucially it re-applies the SAME observation normalization used during
training (from the saved obs_mean / obs_var) — a policy fed unnormalized obs at
eval time would behave like it never trained.

Usage (save a video, works under plain python on macOS):
    python eval_policy.py --model mujoco_menagerie/unitree_g1/scene.xml \
        --ckpt ppo_stand.pt --video recovery.mp4

Live viewer instead (needs mjpython on macOS):
    mjpython eval_policy.py --model ... --ckpt ppo_stand.pt --viewer
"""

import argparse
import numpy as np
import torch
import mujoco

from env import StandEnv
from networks import Actor


def load_policy(ckpt_path, obs_dim, act_dim):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own ckpt
    actor = Actor(obs_dim, act_dim)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    mean = ck.get("obs_mean", None)
    var = ck.get("obs_var", None)
    return actor, mean, var


def normalize(obs, mean, var):
    if mean is None or var is None:
        return obs.astype(np.float32)
    return ((obs - mean) / np.sqrt(var + 1e-8)).astype(np.float32)


@torch.no_grad()
def policy_action(actor, obs_n):
    # MEAN action = deterministic; the learned behavior without exploration noise
    return actor.mean_net(torch.as_tensor(obs_n)).numpy()


def run_video(env, actor, mean, var, path, seconds, fps=50):
    import imageio
    renderer = mujoco.Renderer(env.sim.model, height=480, width=640)
    frames, episodes, steps = [], 0, 0
    obs = env.reset()
    for _ in range(int(seconds * fps)):
        a = policy_action(actor, normalize(obs, mean, var))
        obs, _, done, info = env.step(a)
        renderer.update_scene(env.sim.data)
        frames.append(renderer.render().copy())          # .copy() — same gotcha as before
        steps += 1
        if done:
            episodes += 1
            obs = env.reset()                              # reset to a fresh tilted start
    imageio.mimsave(path, frames, fps=fps)
    print(f"wrote {path}: {len(frames)} frames, {episodes} episode resets")


def run_viewer(env, actor, mean, var):
    import time
    import mujoco.viewer
    obs = env.reset()
    with mujoco.viewer.launch_passive(env.sim.model, env.sim.data) as viewer:
        while viewer.is_running():
            a = policy_action(actor, normalize(obs, mean, var))
            obs, _, done, info = env.step(a)
            viewer.sync()
            time.sleep(env.control_dt)
            if done:
                obs = env.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default="ppo_stand.pt")
    ap.add_argument("--video", default="recovery.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--viewer", action="store_true", help="live viewer (needs mjpython)")
    args = ap.parse_args()

    env = StandEnv(args.model)
    actor, mean, var = load_policy(args.ckpt, env.obs_dim, env.action_dim)
    print(f"loaded {args.ckpt}: obs={env.obs_dim} act={env.action_dim} "
          f"(normalization: {'on' if mean is not None else 'off'})")

    if args.viewer:
        run_viewer(env, actor, mean, var)
    else:
        run_video(env, actor, mean, var, args.video, args.seconds)


if __name__ == "__main__":
    main()