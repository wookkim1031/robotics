# `wookkim1031/robotics` — Code Structure & Function Reference

A from-scratch humanoid-control stack for the Unitree G1 in MuJoCo, built in four stages. Each stage adds one layer on top of the previous and ships with its own self-test, so a layer can be validated in isolation before the next is built on it.

*Commit `b3e4e62`. This document maps every module, what each one tests, and lists all functions.*

---

## 1. The stage architecture (how it fits together)

```
Stage 0  simulator.py     MuJoCo wrapper: load model, step physics, read state
            │
Stage 1  motion_lib.py    reference clips: load + interpolate (lerp/slerp)
         playback.py      drive the model through a clip by FK only (no physics)
            │
Stage 2  networks.py      Actor (policy) + Critic (value)
         env.py           StandEnv: reward / termination for "stay upright"
         ppo.py           PPO trainer (buffer, GAE, clipped update)
         eval_policy.py   load a trained checkpoint, watch it recover
            │
Stage 3  mimic_env.py     MimicEnv: reward for tracking a reference clip (DeepMimic)
         train_mimic.py   wire MotionLib + MimicEnv + PPO, train
         eval_mimic.py    evaluate tracking (NOTE: redefines its own MimicEnv)
```

**Data flow at runtime (Stage 2/3):** `env.reset()` → observation → `Actor.act` samples an action → `env.step(action)` applies it as PD targets, returns `(obs, reward, done, info)` → `PPO` stores the transition, computes advantages, updates the networks → repeat.

---

## 2. Dependency graph (who imports whom)

```
simulator.py        ── (standalone, only mujoco/numpy)
motion_lib.py       ── (standalone, only numpy/mujoco)
networks.py         ── (standalone, only torch/numpy)

env.py              ──► simulator.py
ppo.py              ──► networks.py, env.py (in __main__)
playback.py         ──► simulator.py, motion_lib.py

mimic_env.py        ──► simulator.py, env.py (quat_rotate_inverse)
train_mimic.py      ──► motion_lib.py, mimic_env.py, ppo.py
eval_policy.py      ──► env.py, networks.py
eval_mimic.py       ──► simulator.py, env.py, motion_lib.py  (defines its OWN MimicEnv)
```

The three leaf modules (`simulator`, `motion_lib`, `networks`) have no internal dependencies — that's deliberate, it's what lets each be unit-tested alone.

---

## 3. What tests what (every entry point)

Every module has an `if __name__ == "__main__"` self-test that validates *that layer only*; the `eval_*`/`playback` scripts are integration + visual checks; `ppo.py`/`train_mimic.py` are the real training runs.

| Run this | Stage | What it validates / produces |
|---|---|---|
| `python simulator.py <g1.xml>` | 0 | Model loads; prints `bodies / dof / dt`, body order, and a body-state shape. *Smoke test that the wrapper works.* ⚠️ currently calls the mis-bound `get_bodies_state` → would raise `AttributeError` until that's fixed. |
| `python motion_lib.py` | 1 | Builds a synthetic squat on a 9-dof toy model and interpolates one frame. *Confirms lerp/slerp math.* (minor: the test's `round(...),5)` print has a misplaced paren.) |
| `python playback.py --model <g1.xml> [--clip <csv>]` | 1 | Drives the G1 through a clip by **FK only**. *If it looks right, the model loads, body ordering is sane, retargeting is in range, and interpolation works.* |
| `python networks.py` | 2 | Pushes a batch of 8 random obs through `act` / `forward` / `evaluate`; prints output shapes + param counts. *Confirms network I/O shapes.* |
| `python env.py <g1.xml>` | 2 | Steps the env with **zero action** for a full episode and reports when it falls. *Confirms physics, observation assembly, and termination fire correctly.* |
| `python ppo.py <g1.xml>` | 2 | **Trains** StandEnv for 200 iterations, saves `ppo_stand.pt`. *The Stage-2 training run.* |
| `python eval_policy.py` | 2 | Loads `ppo_stand.pt`, runs the trained policy, renders recovery. *Produces `recovery.mp4`.* |
| `python train_mimic.py --model <g1.xml> --clip <csv>` | 3 | Loads a LAFAN1 clip → `MotionLib` → `MimicEnv` (from `mimic_env.py`) → `PPO`. *The Stage-3 training run.* |
| `python eval_mimic.py` | 3 | Evaluates motion tracking under physics. *Uses the `MimicEnv` defined inside this file, not `mimic_env.py`.* |

**Artifacts already in the repo:** `ppo_stand.pt` (trained Stage-2 policy), `recovery.mp4`, `walk.mp4`, `squat.mp4` (rendered results).

---

## 4. Full function & class reference

### `simulator.py` — Stage 0: MuJoCo wrapper
Loads an MJCF model and exposes a minimal physics/FK interface. Key attributes set in `__init__`: `model`, `data`, `dt` (physics timestep), `body_ids`/`body_names`/`num_bodies` (bodies excluding the world body 0), `num_dof = model.nq - 7` (actuated joints, dropping the 7-dof free base).

| Symbol | Signature | What it does |
|---|---|---|
| `Bodystate` | dataclass | Container for per-body `rigid_body_pos / rot / vel / ang_vel`. |
| `quat_wxyz_to_xzyw` | `(q)` | Reorders a quaternion from MuJoCo `wxyz` to scipy/ProtoMotions `xyzw`. *(name typo: should be `xyzw`)* |
| `Simulator.__init__` | `(model_path, num_envs=1)` | Loads `MjModel`+`MjData`, computes body/dof counts. |
| `Simulator.reset` | `(self)` | `mj_resetData` + forward — back to the model's initial state. |
| `Simulator.step` | `(self, ctrl)` | Writes `ctrl` (PD targets) and integrates physics one timestep. |
| `Simulator.set_qpos` | `(self, qpos)` | Writes a full pose and runs **FK only** — no dynamics. Used for playback. |
| `Simulator.check_qpos_in_range` | `(self, qpos, tol=1e-3)` | Flags joints whose clip values exceed the model's limits. ⚠️ builds `issues` but never `return`s it. |
| `get_bodies_state` | `(self, w_last=True)` | Reads every body's pos/rot/vel/ang-vel into a `Bodystate`. ⚠️ **defined at module level, not inside `Simulator`**, and its `return` sits inside the velocity loop — both need fixing before body-tracking obs work. |

### `motion_lib.py` — Stage 1: reference clips
Turns a continuous time into a pose by blending the two nearest stored frames.

| Symbol | Signature | What it does |
|---|---|---|
| `slerp` | `(q0, q1, t)` | Spherical-linear interpolation between two `wxyz` quaternions, shortest path. |
| `MotionState` | dataclass | One interpolated frame: `qpos`, `dof_pos`, root pose, etc. |
| `MotionLib.__init__` | `(self, frames, fps)` | Stores `[T, nq]` frames; computes `duration`, `num_frames`. |
| `MotionLib.get_motion_state` | `(self, t, loop=True)` | `fk = t*fps`; bracket frames `i0`/`i1`, blend weight `a`; lerp positions/joints, slerp root quat. `% num_frames` makes it loop. |
| `make_synthetic_squat` | `(nq, num_dof, base_qpos=None, fps, seconds, sink)` | Generates a synthetic squat (pelvis sinks, legs bend) on top of a base pose, via a cosine depth envelope `s`. ⚠️ stray `base_qpos[10]=1` debug line. |
| `load_lafan1_g1_csv` | `(path)` | Loads a retargeted LAFAN1 G1 CSV (36 cols), reorders root quat `xyzw→wxyz`, returns `[T, nq]` qpos frames. |

### `networks.py` — Stage 2: the policy and value networks
| Symbol | Signature | What it does |
|---|---|---|
| `build_mlp` | `(in_dim, out_dim, hidden=(256,256), out_gain)` | Feed-forward net: `Linear→ELU` on hidden layers, linear output, orthogonal init. |
| `Actor.__init__` | `(self, obs_dim, act_dim, hidden, init_std=0.6)` | `mean_net` (97→29, `out_gain=0.01`) + learnable `log_std` vector of size `act_dim`. |
| `Actor.distribution` | `(self, obs)` | Builds `Normal(mean_net(obs), exp(log_std))` — a diagonal Gaussian policy. |
| `Actor.act` | `@torch.no_grad() (self, obs)` | **Collection:** samples an action + returns its `old_logp`. No gradients. |
| `Actor.evaluate` | `(self, obs, action)` | **Update:** log-prob of stored actions under the *current* policy + entropy. Gradients on. |
| `Critic.__init__` | `(self, obs_dim, hidden=(256,256))` | `v_net` mapping obs → 1 (`out_gain=1.0`). |
| `Critic.forward` | `(self, obs)` | Returns `V(s)`, shape `(batch,)`. |

### `env.py` — Stage 2: the "stand / balance" task
Wraps `Simulator` into an RL environment. `__init__` sets `default_qpos` (from a `"stand"` keyframe if present), `default_joint_targets`, `target_height`, `control_dt = sim.dt * decimation` (50 Hz), `max_steps`, `action_dim = nu = 29`, and infers `obs_dim = 97` from one `_get_obs()` call.

| Symbol | Signature | What it does |
|---|---|---|
| `quat_rotate_inverse` | `(q_wxyz, v)` | Expresses a world-frame vector in the body frame (used for the gravity-tilt obs). |
| `StandEnv.__init__` | `(self, model_path, decimation=10, action_scale, episode_seconds, reset_noise, max_tilt, vel_noise, seed)` | Builds the env, defaults, episode clock. |
| `StandEnv.reset` | `(self)` | Resets to the default pose + small noise; returns the first obs. |
| `StandEnv.step` | `(self, action)` | `ctrl = default_targets + action_scale*action`; integrate `decimation` physics steps; return `(obs, reward, done, info)` with `info["fell"]`/`["timeout"]`. |
| `StandEnv._get_obs` | `(self)` | Assembles the 97-dim vector: height err(1)+gravity(3)+lin-vel(3)+ang-vel(3)+joint-pos(29)+joint-vel(29)+last-action(29). |
| `StandEnv._check_done` | `(self)` | Returns `(fell, timeout)` — fell on excess tilt/low height, timeout on the episode clock. |
| `StandEnv._reward` | `(self, action, fell)` | Upright/height/effort-shaped reward. |

### `ppo.py` — Stage 2: the trainer
| Symbol | Signature | What it does |
|---|---|---|
| `RunningMeanStd.__init__` | `(self, shape, eps=1e-4)` | Online (Welford) obs-normalizer state. |
| `RunningMeanStd.update` | `(self, x)` | Merges a batch into the running mean/var. |
| `RunningMeanStd.normalize` | `(self, x)` | Z-scores an obs to ~zero-mean/unit-var. |
| `RolloutBuffer.__init__` | `(self, size, obs_dim, act_dim)` | Pre-allocates `size`-step arrays (obs, actions, logprobs, rewards, values, next_values, dones, fells, advantages, returns). |
| `RolloutBuffer.reset` | `(self)` | Rewinds the write cursor. |
| `RolloutBuffer.add` | `(self, obs, action, logp, reward, value, next_value, done, fell)` | Stores one transition; `done` and `fell` kept **separate**. |
| `RolloutBuffer.compute_gae` | `(self, gamma, lam)` | Backward pass: `bootstrap=1-fell` (value target), `continue_=1-done` (advantage flow); fills `advantages` + `returns`. |
| `PPO.__init__` | `(self, env, horizon=2048, gamma=.99, lam=.95, clip=.2, lr=3e-4, epochs=10, num_minibatches=8, ent_coef=.005, vf_coef=.5, max_grad_norm=1.0, device, normalize_obs=True)` | Builds actor/critic/optimizer/buffer/normalizer + live state. |
| `PPO._norm` | `(self, obs)` | Normalize if enabled, else cast to float32. |
| `PPO._t` | `(self, x)` | numpy → tensor on device. |
| `PPO.collect` | `@torch.no_grad() (self)` | Runs the policy `horizon` steps, fills the buffer, runs GAE. Returns finished-episode lengths/returns. |
| `PPO.update` | `(self)` | epochs × minibatches clipped-surrogate + value + entropy update; returns diagnostics (`pi_loss`, `v_loss`, `entropy`, `approx_kl`). |
| `PPO.train` | `(self, iterations, log_every=1)` | Outer loop: collect → update → log. |
| `PPO.save` | `(self, path)` | Saves actor/critic weights **and** obs mean/var. |

### `mimic_env.py` — Stage 3: DeepMimic tracking (the trained env)
Rewards the robot for matching a reference clip frame-by-frame; the reference clock advances by `control_dt` each step.

| Symbol | Signature | What it does |
|---|---|---|
| `MimicEnv.__init__` | `(self, model_path, motion, decimation=10, action_scale, episode_seconds, reset_noise, seed)` | Holds a `MotionLib`; sets up obs/action dims, episode clock. |
| `MimicEnv.reset` | `(self)` | Reference State Initialization — start at a **random phase** of the clip. (Sets `qvel=0` — see note.) |
| `MimicEnv.step` | `(self, action)` | Residual PD on top of the reference pose; advance physics + reference clock. |
| `MimicEnv._get_obs` | `(self)` | Obs including the **phase** of the clip. |
| `MimicEnv._check_done` | `(self)` | Terminal on fall. |
| `MimicEnv._reward` | `(self, fell)` | `exp(-error)` over **joint angles + root height**. |

### `train_mimic.py` — Stage 3 training entry point
| `main` | `()` | Parse args → load CSV → `MotionLib` → `MimicEnv` → `PPO` → `train`. |

### `eval_policy.py` — Stage 2 evaluation
| Symbol | Signature | What it does |
|---|---|---|
| `load_policy` | `(ckpt_path, obs_dim, act_dim)` | Rebuild `Actor`, load weights. |
| `normalize` | `(obs, mean, var)` | Apply saved normalization stats. |
| `policy_action` | `@torch.no_grad() (actor, obs_n)` | Deterministic action (the mean) for eval. |
| `run_video` | `(env, actor, mean, var, path, seconds, fps)` | Render to mp4. |
| `run_viewer` | `(env, actor, mean, var)` | Interactive viewer. |
| `main` | `()` | Load `ppo_stand.pt`, run the policy. |

### `eval_mimic.py` — Stage 3 evaluation ⚠️ **redefines `MimicEnv`**
This file does **not** import `mimic_env.MimicEnv`; it defines its own, **more complete** version. Differences from the trained one:
- `_ref_state(t)` computes reference **velocities** via `mj_differentiatePos` (instead of `qvel=0`).
- Adds `_ref_joint_targets(t)` and a reward that also penalizes the **action**.

| Symbol | Signature | What it does |
|---|---|---|
| `MimicEnv._ref_state` | `(self, t)` | Reference `(qpos, qvel)` at time `t`, qvel via finite difference. |
| `MimicEnv._ref_joint_targets` | `(self, t)` | Reference joint targets for PD. |
| `MimicEnv.reset / step / _get_obs / _check_done` | — | As in Stage 3, but using the reference-velocity state. |
| `MimicEnv._reward` | `(self, action, lost)` | Tracking reward including an action term. |

---

## 5. Two things the structure makes visible

**A. The `MimicEnv` divergence.** You train with `mimic_env.py`'s version (joint-angle + height tracking, `qvel=0` at reset) but evaluate with `eval_mimic.py`'s richer version (reference velocities + action-aware reward). The eval env effectively contains the fixes the training env is missing. Decide which is canonical and have **one** `MimicEnv` that both `train_mimic.py` and `eval_mimic.py` import — otherwise you're evaluating a policy under a different contract than it trained on.

**B. `get_bodies_state` is the next blocker.** It's the one broken function (mis-bound + early return), and it's exactly the body-level state readout that the *next* things you build — tracking observations, and anything ProtoMotions-style — depend on. Worth fixing before Isaac Lab, not after.

---

## 6. What isn't here yet (maps to the proposal)

The repo currently covers Stages 0–3 in MuJoCo (single-environment). Still to come, per the thesis plan:

- **Parallel/vectorized training** — the MuJoCo single-env limit is why the next move is Isaac Lab (where MaskedMimic/ProtoMotions live), or MJX for GPU-batched MuJoCo.
- **Phase 3–4 (proposal):** HIRO manager/worker hierarchy, the frozen **Motion-VAE** latent action space, the **AMP** discriminator — to be reused from ProtoMotions, not reimplemented here.
- **Phase 5:** torque-bounded safe control + sim-to-real export on the physical G1.
- **MaskedMimic baseline** — the head-to-head comparison your contribution is measured against.
