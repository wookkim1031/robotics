"""
Stage 3 — DeepMimic imitation environment (track a reference motion under physics).

Stage 2 taught the G1 to stay upright. Stage 3 teaches it to IMITATE a clip:
the physical robot must reproduce the LAFAN1 walk validated in Stage 1. It fuses
everything built so far — Stage 1 MotionLib (the reference) + Stage 0 Simulator
(the physics) + Stage 2 PPO (the learner) — into a physics-based imitation
controller.

This is the single, canonical MimicEnv. Both `train_mimic.py` and `eval_mimic.py`
import it, so the policy is always trained and evaluated under the SAME contract.

The four ideas that turn "stay upright" into "track a motion":

1. REFERENCE STATE INITIALIZATION (RSI). Reset to a RANDOM frame of the clip,
   matching both pose (qpos) AND velocity (qvel), instead of always the stand
   pose. This is the trick that makes DeepMimic work: the policy experiences
   every part of the motion (mid-stride, mid-turn) from the first episode —
   states it could never reach on its own early in training. The velocity is
   recovered from two nearby frames via mj_differentiatePos, since the clip
   stores positions only. (Resetting qvel to 0 instead would inject a velocity
   discontinuity at every reset that fights RSI.)

2. TRACKING REWARD. A weighted sum of exp(-error) terms — joint pose, joint
   velocity, root position, root orientation. Each term is 1.0 at a perfect
   match and decays as the robot drifts. Maximizing it == imitating.

3. FEEDFORWARD + RESIDUAL CONTROL. The PD target is the REFERENCE joint angles
   at the current motion time plus a small learned offset, so action == 0
   already tries to hold the reference pose; the policy only learns the
   corrections that keep the whole body balanced and on-track.

4. EARLY TERMINATION on lost tracking. If the robot drifts too far from the
   reference root (or falls), end the episode — focusing learning near the
   reference.

The observation includes (ref_pose - current_pose): the `tracking_diff` signal
mirrored from the ProtoMotions MaskedMimic interface.
"""

import numpy as np
import mujoco
from simulator import Simulator
from motion_lib import MotionLib
from env import quat_rotate_inverse


class MimicEnv:
    def __init__(self, model_path, motion: MotionLib, decimation=10,
                 action_scale=0.25, episode_seconds=None, reset_noise=0.0, seed=0):
        self.sim = Simulator(model_path)
        self.motion = motion
        self.rng = np.random.default_rng(seed)

        self.decimation = decimation
        self.action_scale = action_scale
        self.reset_noise = reset_noise
        self.control_dt = self.sim.dt * decimation

        # default: one full pass over the clip; override with episode_seconds
        secs = episode_seconds if episode_seconds is not None else motion.duration
        self.max_steps = int(round(secs / self.control_dt))

        self.nv = self.sim.model.nv
        self.action_dim = self.sim.model.nu
        self.last_action = np.zeros(self.action_dim)
        self.motion_time = 0.0
        self.step_count = 0
        self.obs_dim = self._get_obs().shape[0]   # inferred from one call

    # ---------------------- reference helpers ---------------------- #
    def _ref_state(self, t):
        """Reference (qpos, qvel) at motion time t. qvel via mj_differentiatePos
        on two nearby frames (the clip stores positions only)."""
        q0 = self.motion.get_motion_state(t).qpos
        q1 = self.motion.get_motion_state(t + self.control_dt).qpos
        qvel = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.sim.model, qvel, self.control_dt, q0, q1)
        return q0, qvel

    def _ref_joint_targets(self, t):
        return self.motion.get_motion_state(t).qpos[7:]

    # ----------------------- RL interface -------------------------- #
    def reset(self):
        # RSI: start at a random time in the clip, matching pose AND velocity
        self.motion_time = float(self.rng.uniform(0.0, self.motion.duration))
        ref_qpos, ref_qvel = self._ref_state(self.motion_time)
        qpos = ref_qpos.copy()
        if self.reset_noise > 0.0:
            qpos[7:] += self.rng.uniform(-self.reset_noise, self.reset_noise,
                                         size=self.action_dim)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = ref_qvel
        mujoco.mj_forward(self.sim.model, self.sim.data)

        self.last_action = np.zeros(self.action_dim)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # feedforward reference pose + learned residual
        ref_targets = self._ref_joint_targets(self.motion_time)
        ctrl = ref_targets + self.action_scale * action
        for _ in range(self.decimation):
            self.sim.step(ctrl)

        self.last_action = action
        self.motion_time += self.control_dt       # advance the reference clock
        self.step_count += 1

        obs = self._get_obs()
        done, lost = self._check_done()
        reward = self._reward(action, lost)
        # 'fell' is the true-terminal key PPO reads (here = lost tracking)
        info = {"fell": lost, "lost": lost,
                "timeout": self.step_count >= self.max_steps}
        return obs, reward, done, info

    # ------------------------- internals --------------------------- #
    def _get_obs(self):
        d = self.sim.data
        root_quat = d.qpos[3:7]
        ref_qpos, _ = self._ref_state(self.motion_time)

        proj_grav  = quat_rotate_inverse(root_quat, [0, 0, -1.0])          # (3)
        root_h     = np.array([d.qpos[2] - ref_qpos[2]])                  # (1) height vs ref
        lin_vel    = quat_rotate_inverse(root_quat, d.qvel[0:3])          # (3)
        ang_vel    = d.qvel[3:6]                                          # (3)
        joint_pos  = d.qpos[7:]                                           # (nu)
        joint_vel  = d.qvel[6:]                                           # (nu)
        track_diff = ref_qpos[7:] - d.qpos[7:]                            # (nu) the imitation target
        phase = self.motion_time / self.motion.duration
        phase_obs = np.array([np.sin(2 * np.pi * phase),
                              np.cos(2 * np.pi * phase)])                 # (2)
        return np.concatenate([root_h, proj_grav, lin_vel, ang_vel,
                               joint_pos, joint_vel, track_diff,
                               phase_obs]).astype(np.float32)

    def _check_done(self):
        d = self.sim.data
        ref_qpos, _ = self._ref_state(self.motion_time)
        root_drift = np.linalg.norm(d.qpos[0:3] - ref_qpos[0:3])
        proj_grav = quat_rotate_inverse(d.qpos[3:7], [0, 0, -1.0])
        # lost tracking if drifted far from reference root OR fell over
        lost = (root_drift > 0.5) or (d.qpos[2] < 0.3) or (proj_grav[2] > -0.5)
        timeout = self.step_count >= self.max_steps
        return (lost or timeout), lost

    def _reward(self, action, lost):
        d = self.sim.data
        ref_qpos, ref_qvel = self._ref_state(self.motion_time)

        pose_err = np.sum((d.qpos[7:] - ref_qpos[7:]) ** 2)    # joint angles
        vel_err  = np.sum((d.qvel[6:] - ref_qvel[6:]) ** 2)    # joint velocities
        root_err = np.sum((d.qpos[0:3] - ref_qpos[0:3]) ** 2)  # root position
        dq = np.zeros(3)
        mujoco.mju_subQuat(dq, d.qpos[3:7], ref_qpos[3:7])     # root orientation
        rot_err = np.sum(dq ** 2)

        # DeepMimic-style: each term is 1.0 at a perfect match, decays with error
        r = (0.55 * np.exp(-2.0 * pose_err)
             + 0.10 * np.exp(-0.1 * vel_err)
             + 0.20 * np.exp(-10.0 * root_err)
             + 0.15 * np.exp(-2.0 * rot_err))
        if lost:
            r -= 0.5
        return float(r)