"""
Version of self.env.simulator

Three MuJoCo gotchas this file handles explicitly 
1. Body index 0 is ALWAYS the "world" body. My body start at 1. 
2. MuJoCo quaternioins are scalar FIRST: [w,x,y,z]. 
   Promotions featurizers want scalar-LAST: [x,y,z,w]. We convert.
3. floating base is a "freejoint": qpose[0:3]=root pos, qpos[3:7]=root quat(wxyz), 
   qpos[7:]=actuated joint angles. qvel[0:6]=root (lin3, ang3).
"""

"""
What is ndarray? 
- n-dimensional array 
"""

from dataclasses import dataclass
import numpy as np
import mujoco

@dataclass
class Bodystate: 
    """
    Per-body kinematic state, shaped [num_envs, num_bodies, k].

    Naming mirrors ProtoMotions ('rigid_body_pos', etc.) on purpose, so when you get to the 
    RL stages the interface already lines up. 
    """
    rigid_body_pos: np.ndarray # [E,B,3] world-frame positions
    rigid_body_rot: np.ndarray # [E,B,4] quaternions 
    rigid_body_vel: np.ndarray # [E,B,3] world-frame linear velocity
    rigid_body_ang_vel: np.ndarray # [E,B,3] world-frame angular velocity 

def quat_wxyz_to_xzyw(q: np.ndarray):
    """MuJoCo (w,x,y,z) -> ProtoMotions/scipy (x,y,z,w). Last-axis is the quat."""
    return q[..., [1, 2, 3, 0]]

class Simulator: 
    def __init__(self, model_path: str, num_envs: int = 1):
        """
        What does single-dev (num_envs == 1) mean? 
        - your running one copy of that world. 
        """
        assert num_envs == 1, "Stage 0 is a single-env, batch later via MJX."
        self.num_envs = num_envs
        # Load MjModel and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # Sanity-check our floating-based assumption: first joint is a freejoint 
        # occupying qpos[0:7] / qpos[0:6]. 
        assert self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, (
            "Expected a floating base (freejoint) as the first joint"
        )
        
        # Track every body EXCEPT world (index 0). These indices are what we read state for, 
        # in a stable order you can rely on for observations later.
        self.body_ids = list(range(1, self.model.nbody))
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in self.body_ids
        ]
        self.num_bodies = len(self.body_ids)
        self.num_dof = self.model.nq - 7 # actuated joints (qpos minus free joint)
        
        mujoco.mj_forward(self.model, self.data)

    # Physics stepping (used in the RL stages, not in kinematic playback)
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray): 
        """Apply controls (PD targets for 'position' actuators) and integrate"""
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    # Kinematic interface (Stage 1: drive the character from a clip)
    def set_qpos(self, qpos: np.ndarray): 
        """
        Write a full pose and run FK only - NO dynamics integration. 

        'mj_forward' is forward kinematics

        Takes the current state (qpos, qvel, ctrl) and fills in every derived 
        quantity in d that follows from it,but stops short of advancing time. 
        Nothing about qpos/qvel changes 
        """
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def check_qpos_in_range(self, qpos, tol=1e-3): 
        """Report joints whose loaded values exceed the model's limit
        Out-of-range values usually signal a wrong column order, unit, or 
        quaternion convention - i.e. a retargeting/format bug.

        qpos: loaded clip -- a [T, 36] array of all the frames you want to validate
        tol=1e-3: small tolerance so a value that sits a hair outside a limit due to floating point rounding doesn't get falsely flagged
        """
        m = self.model 
        issues = [] # to accumulate one entry per violating joint and returns this
        for jid in range(1, m.njnt): # skip joint 0 (free base) (m.njnt # number of joints)
            if not m.jnt_limited[jid]: # If the joint has enforce limit 
                continue
            # key bridge between "joint index" and "qpos location"
            adr = m.jnt_qposadr[jid] 
            # 2-element array [lower, upper]- this joint's allowed angle limit
            # Unpacking gives the lower bound lo and upper bound hi for the joint 
            lo, hi = m.jnt_range[jid]
            # qpos[:, adr] takes every row (: = all T frames) at column adr 
            # so col is 1D array of this joint's angle at every timestep in the clip. 
            col = qpos[:, adr]
            # if joint's maximum value over the whole clips below the lower limit
            # its maximum rises above the upper limit 
            if col.min() < lo - tol or col.max() > hi + tol: 
                # translate the numeric joint index back into a human-readable name
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                # record violation: the joint name, the acutal min/max values seen in the clip, the model's allowed lo/hi. 
                issues.append((name, float(col.min()), float(col.max()), float(lo), float(hi)))

# State readout
def get_bodies_state(self, w_last: bool = True) -> Bodystate:
    d, ids = self.data, self.body_ids

    # Where a body's frame origin sits in the world
    pos = d.xpos[ids].copy() # [B,3]
    # Which way the body is oriented. Contains no position information at all
    rot = d.xquat[ids].copy() # [B,4] wxyz
    if w_last: 
        rot = quat_wxyz_to_xzyw(rot) # -> xyzw to match ProtoMotions
    
    lin = np.zeros((self.num_bodies, 3))
    ang = np.zeros((self.num_bodies, 3))
    res = np.zeros(6)
    for k, bid in enumerate(ids): 
        # 6D spatial velocity in world frame 
        mujoco.mj_objectVelocity(
            self.model, d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0 
        )
        ang[k] = res[:3]
        lin[k] = res[3:]

        # add the leading env dim (==1 for now)
        return Bodystate(
            rigid_body_pos=pos[None],
            rigid_body_rot=rot[None],
            rigid_body_vel=lin[None],
            rigid_body_ang_vel=ang[None]
        )
    
if __name__ == "__main__": 
    import sys
    sim = Simulator(sys.argv[1] if len(sys.argv) > 1 else "_test_min.xml")
    print(f"bodies={sim.num_bodies} dof={sim.num_dof} dt={sim.dt}")
    print("body order:", sim.body_names)
    s = sim.get_bodies_state()
    print("rigid_body_pos shape:", s.rigid_body_pos.shape)