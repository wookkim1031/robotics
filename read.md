# Proto-from-scratch — Stage 0 & 1

A minimal, honest reimplementation of the bottom two layers of a ProtoMotions-style
framework, built to *understand* the machinery rather than use it as a black box.

- **Stage 0** (`simulator.py`) — a backend-agnostic MuJoCo wrapper. This is your
  `self.env.simulator`.
- **Stage 1** (`motion_lib.py` + `playback.py`) — a motion library with
  interpolation, plus kinematic playback. This is your `motion_lib` /
  `motion_manager`, driven by forward kinematics only (no physics yet).

## Install

```bash
pip install mujoco imageio        # imageio only needed for --video
```

## Run it right now (no dataset needed)

The minimal test model is included logic-wise; to see it move, point it at any
floating-base MJCF. The playback uses a **synthetic squat** clip so it works
before you've wired in real motion data.

```bash
# interactive viewer (needs a display)
python playback.py --model /path/to/g1.xml

# or save a video (on a GPU box):
MUJOCO_GL=egl python playback.py --model /path/to/g1.xml --video squat.mp4
```

## Getting the G1 model

The Unitree G1 MJCF lives in MuJoCo Menagerie:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
python playback.py --model mujoco_menagerie/unitree_g1/g1.xml


python playback.py --model mujoco_menagerie/unitree_g1/g1.xml --video squat.mp4


python playback.py --model mujoco_menagerie/unitree_g1/g1.xml \
  --clip lafan1_g1/g1/walk1_subject1.csv --video walk.mp4
```

The menagerie model has a floating base (`freejoint`) as its first joint, which
is the one assumption `Simulator` makes — so it works out of the box.

## What to verify before moving on (this is the whole point of Stage 1)

1. The character **plays the clip cleanly** — no jitter, no exploding limbs.
   Jitter usually means a bad interpolation or a quaternion handled as `xyzw`
   where MuJoCo wanted `wxyz`.
2. `sim.body_names` prints in an order that makes sense, and **index 0 is the
   pelvis/root** of your tracked bodies (body 0 in MuJoCo is the `world`; we drop
   it, so `body_ids[0]` is your true root). The ProtoMotions code's
   `ref_state_gt[:, 0]` assumes exactly this.
3. The character is the **right scale and orientation** — if it's lying down or
   tiny, your clip's root frame doesn't match the model's frame (a retargeting
   issue, not a code issue).

If all three hold, your foundation is solid and Stage 2 (bare PPO) can begin.

## The three gotchas this code handles (so you don't rediscover them painfully)

| Gotcha | MuJoCo's convention | What downstream expects |
|---|---|---|
| Body indexing | index 0 = `world` | your robot starts at body 1 |
| Quaternion order | scalar-first `[w,x,y,z]` | scalar-last `[x,y,z,w]` (`w_last=True`) |
| Floating base | `qpos[0:3]`=pos, `[3:7]`=quat, `[7:]`=joints | same, but know the layout |

## Swapping in real motion

Replace `make_synthetic_squat` in `playback.py:build_clip` with a loader that
returns a `[T, nq]` array of full-qpos frames. That's the only contract. Each row
is `root_pos(3) + root_quat_wxyz(4) + joint_angles(num_dof)`. Retargeting your
source mocap onto the G1's joints to produce those rows is the genuinely fiddly
part (SOMA / IK territory) — keep it as a separate offline step so it doesn't
block you here.

## Where this connects to the code you were reading

`set_qpos` + `mj_forward` *is* forward kinematics — the same FK that turns a clip
into the `rigid_body_pos` that `compute_humanoid_max_coords_observations`
consumes. In Stage 3, `get_motion_state(t)` becomes the **reference** and
`get_bodies_state()` becomes the **current** pose; their difference is exactly the
`tracking_diff_obs = ref_pose - current_pose` you saw in `MimicADD`.