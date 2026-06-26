"""
Stage 2 - RL Environemnt

This is the first time PHYSICS enters. Stage 1 posed the robot kinematically (set_qpos + mj_forward, no forces).
Here: policy outputs an action, turn it into actuator targets, and sim.step() integrates real dynamics. 
"""

import numpy as np
import mujoco
from simulator import Simulator

def quat_rotate_inverse(q_wxyz, v):
    """Express world-frame vector v in the body frame given by quaternion q.
    
    A quaternion is a compact encoding of a 3D rotation. 

    A world vector (gravity) and express it in the body frame (world -> body). That's the inverse of R. 

    Example usage: used for gravity direction
    - proj_grav = quat_rotate_inverse(root_quat, [0, 0, -1.0])  

    Root linear velocity 
    - lin_vel   = quat_rotate_inverse(root_quat, d.qvel[0:3])
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    # reshape flat 9 into an actual 3x3 grid 
    R = R.reshape(3,3) # body -> world
    # R.T is the transpose = inverse rotation
    return R.T @ np.asarray(v, dtype=np.float64) # world -> body


class StandEnv: 
    def __init__(self, model_path, decimation=10, action_scale=0.25,
                 episode_seconds=10.0, reset_noise=0.10, max_tilt=0.30,
                 vel_noise=0.5, seed=0):
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

        self.max_tilt = max_tilt
        self.vel_noise = vel_noise

    # ----------------------- interface ------------------------------ #
    def reset(self):
        qpos = self.default_qpos.copy()
        
        # small noise on joints + height so the policy sees varied starts
        qpos[7:] += self.rng.uniform(-self.reset_noise, self.reset_noise, size=self.action_dim)
        qpos[2] += self.rng.uniform(-0.01,0.01)

        # random initial lean (up to max_tilt) about a mostly-horizontal axis
        axis = self.rng.normal(size=3); axis[2] *= 0.2
        axis /= np.linalg.norm(axis)
        angle = self.rng.uniform(0, self.max_tilt)
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, axis, angle)
        qpos[3:7] = q

        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = 0.0
        self.sim.data.qvel[0:3] = self.rng.uniform(-0.2, 0.2, 3)            # root linear
        self.sim.data.qvel[3:6] = self.rng.uniform(-0.3, 0.3, 3)            # root angular
        self.sim.data.qvel[6:]  = self.rng.uniform(-self.vel_noise,
                                                   self.vel_noise, size=self.action_dim)

        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.last_action = np.zeros(self.action_dim)
        self.step_count = 0
        return self._get_obs()
    
    # How do I advance one step? 
    def step(self, action):
        """Advance the simulation by one policy decision"""
        # policy's raw output gets clamped to [-1,1]
        action = np.clip(action, -1.0, 1.0)
        # residual PD: target = standing pose + scaled offset
        ctrl = self.default_joint_targets + self.action_scale * action
        # Apply same target for 10 physics steps. Physics runs at 500Hz; policy decides at ~50Hz. 
        for _ in range(self.decimation): 
            self.sim.step(ctrl)
        # goes into the next observation
        self.last_action = action
        # tick the episode length counter
        self.step_count += 1

        obs = self._get_obs()
        done, fell =self._check_done()
        reward = self._reward(action, fell)
        info = {"fell": fell, "timeout": self.step_count >= self.max_steps}
        return obs, reward, done, info

    # ------------------------ internal ------------------------------- # 
    """
    Builds the observation i.e. everything the robot "senses" about itself each step. 
    Policy network sees this to decide what to do. 
    """
    def _get_obs(self): 
        # handles MuJoCo's live state
        d = self.sim.data
        root_quat = d.qpos[3:7] # wxyz
        # gravity expressed in the body frame, the tilt/balance signal
        proj_grav = quat_rotate_inverse(root_quat, [0,0,-1.0]) # up signal (3)
        # pelvis height minus the target standing height 
        root_h = np.array([d.qpos[2] - self.target_height]) # height err
        # base-frame
        # root's linear velocity, converted into the body frame 
        lin_vel = quat_rotate_inverse(root_quat, d.qvel[0:3])
        # ang vel
        # the root's angular velocity (already body-frame in MuJoCo, so no conversion)
        ang_vel = d.qvel[3:6]
        # rel to stand
        # every joint's angle, but relative to the stand pose
        joint_pos = d.qpos[7:] - self.default_joint_targets
        # (nu)
        # every joint's angular speed. 
        joint_vel = d.qvel[6:]
        return np.concatenate([root_h, proj_grav, lin_vel, ang_vel, 
                               joint_pos, joint_vel, self.last_action]).astype(np.float32)
    
    """
    Did the episode end, and did it end badly? 
    """
    def _check_done(self):
        d = self.sim.data
        # current pelvis height
        height = d.qpos[2]
        proj_grav = quat_rotate_inverse(d.qpos[3:7], [0,0,-1.0])
        # fell if pelvis dropped a lot OR torso tilted past -50 deg
        # (upright => proj_grav[2] - -1; tilt makes it rise toward 0)
        # if eigther its pelvis dropped more than 20cm below standing height or it tilted past ~50 degree
        fell = (height < self.target_height - 0.25) or (proj_grav[2] > -0.6)
        # the robot survived the full episode 10s without falling 
        timeout = self.step_count >= self.max_steps
        return (fell or timeout), fell
    
    def _reward(self, action, fell):
        d = self.sim.data
        proj_grav = quat_rotate_inverse(d.qpos[3:7], [0,0,-1.0])
        alive = 1.0 # the core signal 
        upright = -1.0 * proj_grav[2] # 1 when perfectly upright
        height_pen = (d.qpos[2] - self.target_height) ** 2
        ctrl_pen = np.sum(action ** 2)
        r = alive + 0.3 * upright - 2.0 * height_pen - 0.005 * ctrl_pen
        if fell: 
            r -= 1.0 # small terminal penalty
        return float(r)

if __name__ == "__main__":
    import sys 
    env = StandEnv(sys.argv[1] if len(sys.argv) > 1 else "_test_min.xml")
    """
    env.obs_dim: observation vector is expected 97 numbers (1 height + 3 gravity + 3 lin-vel + 3 ang-vel + 29 joint-pos + 29 joint-vel + 29 last-action)
    action_dim = 29: 29 joint targets, one per actuator. 
    control_dt = 0.020s: the policy acts every 0.02s = 50Hz
    max_steps = 500: an episode is 500 policy steps = 500 * 0.02s = 10 seconds

    """
    print(f"obs_dim={env.obs_dim} action_dim={env.action_dim}   " 
          f"control_dt={env.control_dt:.3f}s max_steps={env.max_steps}")
    obs = env.reset()
    print("reset obs shape", obs.shape)
    """
    The test drove nev with zero action for the whole episode and it never fell
    """
    # zero action = no correction -> it should eventually fall
    for t in range(env.max_steps): 
        obs, r, done, info = env.step(np.zeros(env.action_dim))
        if done: 
            print(f"episode ended at step {t}: fell={info['fell']} "
                  f"timeout={info['timeout']}, last_reward={r:.3f}")
            break 

