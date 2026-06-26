"""
Stage 3 - DeepMimic imitation environment.

The robot is rewarded for matching a reference motion clip frame-by-frame.
At each policy step the reference clock advances by control_dt; the reward
is an exponential of the joint-angle and height tracking errors.
"""

import numpy as np
import mujoco
from simulator import Simulator
from motion_lib import MotionLib
from env import quat_rotate_inverse


class MimicEnv:
    def __init__(self, model_path, motion: MotionLib,
                 decimation=10, action_scale=0.25,
                 episode_seconds=None, reset_noise=0.02, seed=0):
        self.sim = Simulator(model_path)
        self.motion = motion
        self.rng = np.random.default_rng(seed)

        kid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0:
            self.default_qpos = self.sim.model.key_qpos[kid].copy()
        else:
            self.default_qpos = self.sim.model.qpos0.copy()
        self.default_joint_targets = self.default_qpos[7:].copy()

        self.decimation = decimation
        self.action_scale = action_scale
        self.reset_noise = reset_noise
        self.control_dt = self.sim.dt * decimation

        secs = episode_seconds if episode_seconds is not None else motion.duration
        self.max_steps = int(round(secs / self.control_dt))

        self.action_dim = self.sim.model.nu
        self.last_action = np.zeros(self.action_dim)
        self.ref_time = 0.0
        self.step_count = 0
        self.obs_dim = self._get_obs().shape[0]

    # ----------------------- interface ------------------------------ #

    def reset(self):
        # start at a random phase in the clip
        self.ref_time = self.rng.uniform(0.0, self.motion.duration)
        ref = self.motion.get_motion_state(self.ref_time)

        qpos = ref.qpos.copy()
        qpos[7:] += self.rng.uniform(-self.reset_noise, self.reset_noise,
                                     size=self.action_dim)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(self.sim.model, self.sim.data)

        self.last_action = np.zeros(self.action_dim)
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # residual PD on top of the reference pose at the current clock
        ref = self.motion.get_motion_state(self.ref_time)
        ctrl = ref.dof_pos + self.action_scale * action
        for _ in range(self.decimation):
            self.sim.step(ctrl)

        self.ref_time += self.control_dt
        self.last_action = action
        self.step_count += 1

        obs = self._get_obs()
        done, fell = self._check_done()
        reward = self._reward(fell)
        info = {"fell": fell, "timeout": self.step_count >= self.max_steps}
        return obs, reward, done, info

    # ------------------------ internal ------------------------------- #

    def _get_obs(self):
        d = self.sim.data
        root_quat = d.qpos[3:7]
        proj_grav = quat_rotate_inverse(root_quat, [0, 0, -1.0])
        root_h = np.array([d.qpos[2]])
        lin_vel = quat_rotate_inverse(root_quat, d.qvel[0:3])
        ang_vel = d.qvel[3:6]
        joint_pos = d.qpos[7:]
        joint_vel = d.qvel[6:]

        ref = self.motion.get_motion_state(self.ref_time)
        ref_dof_pos = ref.dof_pos
        ref_root_h = np.array([ref.root_pos[2]])
        phase = np.array([self.ref_time / self.motion.duration])

        return np.concatenate([
            root_h, proj_grav, lin_vel, ang_vel,
            joint_pos, joint_vel,
            ref_dof_pos, ref_root_h, phase,
            self.last_action,
        ]).astype(np.float32)

    def _check_done(self):
        d = self.sim.data
        height = d.qpos[2]
        proj_grav = quat_rotate_inverse(d.qpos[3:7], [0, 0, -1.0])
        fell = (height < 0.3) or (proj_grav[2] > -0.5)
        timeout = self.step_count >= self.max_steps
        return (fell or timeout), fell

    def _reward(self, fell):
        d = self.sim.data
        ref = self.motion.get_motion_state(self.ref_time)

        joint_err = np.sum((d.qpos[7:] - ref.dof_pos) ** 2)
        pose_r = float(np.exp(-2.0 * joint_err))

        h_err = (d.qpos[2] - ref.root_pos[2]) ** 2
        height_r = float(np.exp(-10.0 * h_err))

        r = 0.6 * pose_r + 0.4 * height_r
        if fell:
            r -= 1.0
        return r