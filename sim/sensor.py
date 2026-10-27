"""Simulated pose sensor: what a camera + AprilTag pipeline would report about the target."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Measurement:
    t: float  # s, sim time
    pos: np.ndarray  # m, world frame (noisy)
    quat: np.ndarray  # w, x, y, z (noisy)


def small_rotation_quat(rotvec: np.ndarray) -> np.ndarray:
    """Quaternion for a rotation vector (axis * angle, rad)."""
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, rotvec / max(np.linalg.norm(rotvec), 1e-12), np.linalg.norm(rotvec))
    return q


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle between two orientations, degrees."""
    return float(np.rad2deg(2 * np.arccos(min(1.0, abs(np.dot(q1, q2))))))


class PoseSensor:
    """Samples the target pose at `rate_hz`, adds Gaussian position noise (`pos_std`, m per axis)
    and orientation noise (`rot_std_deg` per axis, as a small random rotation), and drops each
    reading with probability `dropout`."""

    def __init__(self, rate_hz=30.0, pos_std=0.005, rot_std_deg=2.0, dropout=0.05, seed=None):
        self.period = 1.0 / rate_hz
        self.pos_std = pos_std
        self.rot_std = np.deg2rad(rot_std_deg)
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.next_t = 0.0

    def reset(self, t: float = 0.0):
        self.next_t = t

    def poll(self, t: float, true_pos: np.ndarray, true_quat: np.ndarray) -> Measurement | None:
        """Call every physics step; returns a measurement when one is due (and not dropped)."""
        if t + 1e-9 < self.next_t:
            return None
        self.next_t += self.period
        if self.rng.random() < self.dropout:
            return None
        pos = true_pos + self.rng.normal(0.0, self.pos_std, 3)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, true_quat, small_rotation_quat(self.rng.normal(0.0, self.rot_std, 3)))
        return Measurement(t, pos, quat / np.linalg.norm(quat))
