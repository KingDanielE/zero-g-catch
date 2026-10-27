"""Panda arm kinematics on a private MjData: tool-point FK, damped least-squares IK,
self-collision check. Never touches the simulation's own data."""

import mujoco
import numpy as np

from sim.env import ARM_JOINTS

TCP_OFFSET = np.array([0.0, 0.0, 0.1034])  # hand frame -> point between the fingertip pads
QD_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s, Franka spec


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    if q[0] < 0:
        q = -q
    angle = 2 * np.arccos(min(1.0, q[0]))
    s = np.sqrt(max(1e-12, 1 - q[0] ** 2))
    return q[1:] / s * angle if angle > 1e-9 else np.zeros(3)


class ArmKinematics:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.hand = model.body("hand").id
        jids = [model.joint(j).id for j in ARM_JOINTS]
        self.qadr = model.jnt_qposadr[jids]
        self.dadr = model.jnt_dofadr[jids]
        self.lo, self.hi = model.jnt_range[jids].T
        # Park the target far away so collision checks only see the arm.
        tq = model.joint("target_free").qposadr[0]
        self.d.qpos[tq : tq + 3] = [10.0, 10.0, 10.0]
        self.d.qpos[tq + 3] = 1.0
        # Fingers open.
        for f in ("finger_joint1", "finger_joint2"):
            self.d.qpos[model.joint(f).qposadr[0]] = 0.04

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool point position and hand rotation matrix for arm joint angles q."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.hand].reshape(3, 3).copy()
        return self.d.xpos[self.hand] + R @ TCP_OFFSET, R

    def jacobian(self, point: np.ndarray) -> np.ndarray:
        """6x7 [linear; angular] Jacobian of `point` on the hand (after fk)."""
        mujoco.mj_comPos(self.m, self.d)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jacp, jacr, point, self.hand)
        return np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])

    def ik(self, q0, pos, R, rest=None, iters=200, pos_tol=0.003, rot_tol_deg=2.0,
           damping=0.02, max_step=0.2, rest_gain=0.1):
        """Damped least squares to (pos, R), starting from q0. Steps are capped at `max_step`
        rad, and the null space pulls toward `rest` (default q0) so the posture stays close to
        where the arm already is. Returns (q, converged)."""
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        rest = q.copy() if rest is None else np.asarray(rest, float)
        for _ in range(iters):
            p, Rc = self.fk(q)
            err = np.concatenate([pos - p, rotvec_from_matrix(R @ Rc.T)])
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < np.deg2rad(rot_tol_deg):
                return q, True
            J = self.jacobian(p)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (rest_gain * (rest - q))
            dq *= min(1.0, max_step / max(np.abs(dq).max(), 1e-12))
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False

    def ik_axis(self, q0, pos, axis, iters=200, pos_tol=0.003, align_tol_deg=2.0,
                damping=0.02, max_step=0.2, rest_gain=0.1):
        """Like ik(), but only puts the hand's x axis along +/-`axis` (the bar): the rotation
        about the bar is left free and the null space spends it staying near q0, which gives
        the closest grasp posture. Returns (q, converged)."""
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        rest = q.copy()
        a = axis / np.linalg.norm(axis)
        if np.dot(self.fk(q)[1][:, 0], a) < 0:
            a = -a  # align with whichever direction is nearer
        P = np.eye(3) - np.outer(a, a)  # rotations about the bar don't count
        for _ in range(iters):
            p, Rc = self.fk(q)
            x = Rc[:, 0]
            e_rot = np.cross(x, a)
            align = np.rad2deg(np.arctan2(np.linalg.norm(e_rot), np.dot(x, a)))
            if np.linalg.norm(pos - p) < pos_tol and align < align_tol_deg:
                return q, True
            J6 = self.jacobian(p)
            J = np.vstack([J6[:3], P @ J6[3:]])
            err = np.concatenate([pos - p, P @ e_rot])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (rest_gain * (rest - q))
            dq *= min(1.0, max_step / max(np.abs(dq).max(), 1e-12))
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False

    def ik_multi(self, q0, pos, R, n_seeds=8):
        """IK from q0 first, then from fixed random postures; returns the converged,
        collision-free solution closest to q0, or (q0, False)."""
        if not hasattr(self, "_seeds"):
            rng = np.random.default_rng(0)
            self._seeds = rng.uniform(self.lo, self.hi, size=(16, 7))
        best = None
        for seed in [q0, *self._seeds[:n_seeds]]:
            q, ok = self.ik(seed, pos, R)
            if ok and not self.self_collision(q):
                dist = np.abs(q - q0).max()
                if best is None or dist < best[1]:
                    best = (q, dist)
                if seed is q0:
                    break  # warm start worked: keep the nearby solution
        return (best[0], True) if best else (np.asarray(q0, float), False)

    def self_collision(self, q: np.ndarray) -> bool:
        self.d.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(self.m, self.d)
        return self.d.ncon > 0
