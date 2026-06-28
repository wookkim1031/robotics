"""
Stage 1 -- Motion Library. 

"Motion_lib" a clip is just a time-series of poses (full qpos frames) plus a frame rate. 
The one job that matters: given a continuous time t, return the pose at that exact moment by interpolating between stored frames. 

    - root position & joint angles: linear interpolation (lerp)
    - root orientation: spherical linear interpolation (slerp), because you can't lerp quaternions and stay on the unit sphere. 

'get_motion_state(motion_id, t)' is the same call you saw in the ProtoMotions code, minus the batch dimension for now. 

For Stage 1 we generate a SYNTHETIC clip so this runs against your real G1 model with no dataset wiredup. 
Swapping in real retargeted motion = replacing 'make_synthetic_squat' with a loader that fills the same 'frames' array.
"""

from dataclasses import dataclass
import numpy as np

def slerp(q0, q1, t):
    """Slerp between two wxyz quaternions, shortest path"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0,q1))
    if dot < 0.0: 
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(dot)
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th0 * t) + q2 * np.sin(th0 * t)

@dataclass
class MotionState: 
    qpos: np.ndarray
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray

class MotionLib: 
    def __init__(self, frames: np.ndarray, fps:float):
        """frames: [T, nq] full-qpos per frame. fps: frames per second"""
        self.frames = np.asarray(frames, dtype=np.float64)
        self.fps = float(fps)
        self.num_frames = self.frames.shape[0]
        self.duration = self.num_frames / self.fps
    
    def get_motion_state(self, t: float, loop: bool = True) -> MotionState: 
        if loop: 
            t = t % self.duration
        else: 
            t = min(t, self.duration - 1e-6)
        
        fk = t * self.fps  # how many frames have elapsed at time t. 
        i0 = int(np.floor(fk)) % self.num_frames # the frame just before t
        i1 = (i0 + 1) % self.num_frames # the next frame
        a = fk - np.floor(fk) # how far between them

        f0, f1 = self.frames[i0], self.frames[i1]
        q = f0.copy()
        q[0:3] = (1 - a) * f0[0:3] + a * f1[0:3] # root pos : lerp
        q[3:7] = slerp(f0[3:7], f1[3:7], a)      # root rot : slerp
        q[7:] = (1 - a) * f0[7:] + a * f1[7:]    # joints   : lerp

        return MotionState(qpos=q, root_pos=q[0:3], root_rot=q[3:7], dof_pos=q[7:])
    
# nq: full length of qpos 
# num_dof: Number of actuated joints 
def make_synthetic_squat(nq: int, num_dof: int, base_qpos=None,
                         fps: float = 30.0, seconds: float = 2.0,
                         sink: float = 0.20) -> np.ndarray:
    """
    Symmetric squat: pelvis sinks while both legs bend. The pelvis drop makes
    motion visible from any camera angle, independent of joint-index mapping.

    Replace this with your real loader. Contract unchanged: returns [T, nq],
    each row = root_pos(3) + root_quat_wxyz(4) + joint_angles(num_dof).
    """
    T = int(round(seconds * fps))
    if base_qpos is None:
        base_qpos = np.zeros(nq)
        base_qpos[2] = 0.78
        base_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    base_qpos = np.asarray(base_qpos, dtype=np.float64)
    frames = np.tile(base_qpos, (T, 1))           # every frame starts from stand
    for k in range(T):
        ph = 2 * np.pi * k / T
        s = 0.5 - 0.5 * np.cos(ph)                # smooth 0 -> 1 -> 0
        frames[k, 2] = base_qpos[2] - sink * s    # pelvis sinks as it squats

        def addj(i, v):
            if i < num_dof:
                frames[k, 7 + i] += v             # delta on top of the stand pose
        # addj adds v * s to joint i, so the hips/knees/ankles bend proportionally to how deep the squat currently is.
        addj(0, 0.6 * s); addj(3, 1.2 * s); addj(4, -0.6 * s)   # left  hip, knee, ankle
        addj(6, 0.6 * s); addj(9, 1.2 * s); addj(10, -0.6 * s)  # right hip, knee, ankle
    return frames

def load_lafan1_g1_csv(path): 
    """Load a Unitree LAFAN1_Retargeting_Dataset G1 CSV into MuJoCo qpos frames.
        CSV: one row per frame, 36 columns, 30 FPS:
            [0:3]  root position    x y z
            [3:7]  root quaternion  qx qy qz qw   (scalar-LAST / xyzw)
            [7:36] 29 joint angles in G1 order (matches the g1_29dof model)
        MuJoCo qpos needs the root quat scalar-FIRST (wxyz), so we reorder it.
    """
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim == 1:
        raw = raw[None]
    assert raw.shape[1] == 36, f"expected 36 columns, got {raw.shape[1]}"
    T = raw.shape[0]
    qpos = np.empty((T,36))
    qpos[:, 0:3] = raw[:, 0:3] # root position
    qpos[:, 3] = raw[:, 6] # qw (scalar-first)
    qpos[:, 4:7] = raw[:, 3:6] # qx qy qz
    qpos[:, 7:] = raw[:, 7:36] # 29 joints, same order as the model
    return qpos


if __name__ == "__main__": 
    # Tiny self-test against the minimal model's layout (nq=9, dof=2)
    frames = make_synthetic_squat(nq=9, num_dof=2)
    lib = MotionLib(frames, fps=30.0)
    s = lib.get_motion_state(0.5 / 30.0)
    print("duration:", lib.duration, "s")
    print("interpolated joint[0] @ t=0.0166:", round(float(s.dof_pos[0])),5)