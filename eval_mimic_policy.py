"""
Evaluate a trained DeepMimic (mimic) policy.

Usage — save a video:
    python eval_mimic_policy.py \
        --model mujoco_menagerie/unitree_g1/scene.xml \
        --clip lafan1_g1/g1/walk1_subject1.csv \
        --ckpt ppo_mimic.pt \
        --video mimic_out.mp4

Live viewer instead (needs mjpython on macOS):
    mjpython eval_mimic_policy.py \
        --model mujoco_menagerie/unitree_g1/scene.xml \
        --clip lafan1_g1/g1/walk1_subject1.csv \
        --viewer
"""

import argparse
import numpy as np
import torch
import mujoco

from motion_lib import MotionLib, load_lafan1_g1_csv
from mimic_env import MimicEnv
from networks import Actor


def load_policy(ckpt_path, obs_dim, act_dim):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    return actor.mean_net(torch.as_tensor(obs_n)).numpy()


def run_video(env, actor, mean, var, path, seconds, fps=50):
    import imageio
    renderer = mujoco.Renderer(env.sim.model, height=480, width=640)
    frames, episodes = [], 0
    obs = env.reset()
    for _ in range(int(seconds * fps)):
        a = policy_action(actor, normalize(obs, mean, var))
        obs, _, done, info = env.step(a)
        renderer.update_scene(env.sim.data)
        frames.append(renderer.render().copy())
        if done:
            episodes += 1
            obs = env.reset()
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
    ap.add_argument("--clip", required=True)
    ap.add_argument("--ckpt", default="ppo_mimic.pt")
    ap.add_argument("--video", default="mimic_out.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--viewer", action="store_true", help="live viewer (needs mjpython)")
    args = ap.parse_args()

    frames = load_lafan1_g1_csv(args.clip)
    env = MimicEnv(args.model, MotionLib(frames, fps=30.0))
    actor, mean, var = load_policy(args.ckpt, env.obs_dim, env.action_dim)
    print(f"loaded {args.ckpt}: obs={env.obs_dim} act={env.action_dim} "
          f"(normalization: {'on' if mean is not None else 'off'})")

    if args.viewer:
        run_viewer(env, actor, mean, var)
    else:
        run_video(env, actor, mean, var, args.video, args.seconds)


if __name__ == "__main__":
    main()
