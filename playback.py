"""
Stage 1 — Kinematic playback.
 
Ties Stage 0 + Stage 1 together: load the model, build a clip, and drive the
character through the reference motion using FK only (no physics). If this looks
right, your model loads correctly, your body ordering is sane, and your
interpolation works — the foundation everything else sits on.
 
Two ways to watch it:
 
  Interactive (recommended on your machine, needs a display):
      python playback.py --model /path/to/g1.xml
 
  Save a video (needs an OpenGL backend; set MUJOCO_GL=egl on a GPU box):
      MUJOCO_GL=egl python playback.py --model /path/to/g1.xml --video out.mp4
 
The crucial loop is `set_qpos(state.qpos)` then read state back — that's a
reference clip being played 'in the air'. In Stage 3 (DeepMimic) this same
reference becomes the target a *physics* policy has to chase.
"""

import argparse 
import numpy as np 

from simulator import Simulator
from motion_lib import MotionLib, make_synthetic_squat, load_lafan1_g1_csv

def build_clip(sim: Simulator, fps: float, clip_path:None) -> MotionLib: 
    if clip_path: 
        frames = load_lafan1_g1_csv(clip_path)
        issues = sim.check_qpos_in_range(frames)
        if issues:
            print(" joint-range violations (possible format/retarget mismatch)")
            for name, lo, hi, rlo, rhi in issues:
                print(f"   {name}: data [{lo:.2f}, {hi:.2f}] vs model [{rlo:.2f}, {rhi:.2f}]")
        else:
            print("✓ all joints within model range")
        return MotionLib(frames, fps=30.0)  
    import mujoco
    # Swap this for your real retargeted-clip loader. Contract: frames is [T, nq]. 
    base = None
    kid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid >= 0:
        base = sim.model.key_qpos[kid].copy()
    frames = make_synthetic_squat(nq=sim.model.nq, num_dof=sim.num_dof,
                                  base_qpos=base, fps=fps)
    return MotionLib(frames, fps=fps)

def run_viewer(sim: Simulator, lib: MotionLib, fps: float): 
    import time
    import mujoco.viewer
 
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        t = 0.0
        while viewer.is_running():
            state = lib.get_motion_state(t)
            sim.set_qpos(state.qpos)          # FK only — no mj_step here
            viewer.sync()
            t += 1.0 / fps
            time.sleep(1.0 / fps)

def run_video(sim: Simulator, lib: MotionLib, fps: float, path: str, seconds=None):
    import mujoco
    import imageio

    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    frames = []
    t = 0.0
    if seconds is None:
        seconds = min(lib.duration, 8.0) 
    while t < seconds:
        state = lib.get_motion_state(t)
        sim.set_qpos(state.qpos)
        renderer.update_scene(sim.data)
        frames.append(renderer.render().copy())
        t += 1.0 / fps
    imageio.mimsave(path, frames, fps=int(fps))
    print(f"wrote {path} ({len(frames)} frames)")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the G1 MJCF (.xml)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--video", default=None, help="save mp4 here instead of viewer")
    ap.add_argument("--clip", default=None, help="path to a LAFAN1 G1 .csv clip")
    args = ap.parse_args()
 
    sim = Simulator(args.model)
    print(f"loaded {args.model}: {sim.num_bodies} bodies, {sim.num_dof} dof")
    print("body order:", sim.body_names)
 
    lib = build_clip(sim, args.fps, clip_path=args.clip)
    print(f"clip duration: {lib.duration:.2f}s")
 
    if args.video:
        run_video(sim, lib, args.fps, args.video)
    else:
        run_viewer(sim, lib, args.fps)
 
 
if __name__ == "__main__":
    main()
