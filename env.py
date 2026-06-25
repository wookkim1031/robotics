"""
Stage 2 - RL Environemnt

This is the first time PHYSICS enters. Stage 1 posed the robot kinematically (set_qpos + mj_forward, no forces).
Here: policy outputs an action, turn it into actuator targets, and sim.step() integrates real dynamics. 
"""

import numpy as np
import mujoco
from simulator import Simulator

def quat_rotate_inverse(q_wxyz, v):
    """Express world-frame vector v in the body frame given by quaternion q."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    R = R.reshape(3,3) # body -> world
    return R.T @ np.asarray(v, dtype=np.float64) # world -> body

class StandEnv: 
    def __init__(self, model_path, decimation=10, action_scale=0.25,
                 episode_seconds=10.0, reset_noise=0.02, seed=0):
        self.sim = Simulator(model_path)
        # exact same sequence of random numbers come out every run for a given seed.
        # np.random.uniform draws from one global generator shared by entire program
        # PPO uses global NumPy randomness and PPO's sampling would interfere with each other's sequence
        self.rng = np.random.default_rng(seed)

        # Default pose = "stand" keyframe if present, else the model's qpos0. 
        kid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0: 
            self.default_qpos = self.sim.model.key_qpos[kid].copy()
        else: 
            self.default_qpos = self.sim.model.qpos0.copy()
        self.default_joint_targets = self.default_qpos[7:].copy()
        self.target_height = float(self.default_qpos[2])

        self.decimation = decimation
        self.action_scale = action_scale
        self.reset_noise = reset_noise
        self.control_dt = self.sim.dt * decimation
        self.max_steps = int(round(episode_seconds / self.control_dt))

        self.action_dim = self.sim.model.nu # 29 for G1
        self.last_action = np.zeros(self.action_dim) 
        self.obs_dim = self._get_obs().shape[0] # infer from one call
        self.step_count = 0

    # ----------------------- interface ------------------------------ #
    def reset(self):
        qpos = self.default_qpos.copy()
        
        # small noise on joints + height so the policy sees varied starts
        qpos[7:] += self.rng.uniform(-self.reset_noise, self.reset_noise, size=self.action_dim)
        qpos[2] += self.rng.uniform(-0.01,0.01)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.last_action = np.zeros(self.action_dim)
        self.step_count = 0
        return self._get_obs()
    
    def step(self, action):
        action = np.clip(action, -1.0, -1.0)
        # residual PD: target = standing pose + scaled offset
        ctrl = self.default_joint_targets + self.action_scale * action
        for _ in range(self.decimation): 
            self.sim.step(ctrl)

