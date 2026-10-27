"""Target state estimation from noisy pose readings.

- TranslationKF: linear Kalman filter on [position, velocity] with a constant-velocity model
  (zero-g, no drag: exact apart from contacts). Handles irregular dt from dropped readings.
- SpinEstimator: world-frame angular velocity from a sliding window of orientations, by a
  least-squares line fit of each reading's rotation vector relative to the window's first one.
  Torque-free spin is not constant for a general body, so the window is kept short.
"""

from collections import deque

import mujoco
import numpy as np


class TranslationKF:
    def __init__(self, pos_std=0.005, accel_std=0.01, init_vel_std=0.3):
        self.r = pos_std**2  # measurement noise variance per axis
        self.q = accel_std**2  # white-acceleration process noise (absorbs model error)
        self.init_vel_var = init_vel_std**2
        self.x = None  # [p(3), v(3)]
        self.P = None
        self.t = None

    @property
    def ready(self) -> bool:
        return self.x is not None

    def predict_to(self, t: float):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # Discrete white-noise acceleration model, per axis.
        q = self.q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        Q = np.kron(q, np.eye(3))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, t: float, pos: np.ndarray):
        if self.x is None:
            self.x = np.concatenate([pos, np.zeros(3)])
            self.P = np.diag([self.r] * 3 + [self.init_vel_var] * 3)
            self.t = t
            return
        self.predict_to(t)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        S = H @ self.P @ H.T + np.eye(3) * self.r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (pos - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    def predict(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and 3x3 position covariance at future time t (state unchanged)."""
        dt = t - self.t
        p = self.x[:3] + self.x[3:] * dt
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        P = F @ self.P @ F.T
        return p, self.x[3:].copy(), P[:3, :3]

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def vel_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P)[3:])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q if q[0] >= 0 else -q  # shortest rotation
    angle = 2 * np.arccos(min(1.0, q[0]))
    s = np.sqrt(max(1e-12, 1 - q[0] ** 2))
    return q[1:] / s * angle if angle > 1e-9 else np.zeros(3)


class SpinEstimator:
    def __init__(self, window_s=0.6, min_readings=6):
        self.window = window_s
        self.min_readings = min_readings
        self.buf: deque[tuple[float, np.ndarray]] = deque()
        self.omega = None  # rad/s, world frame
        self.quat = None  # latest measured orientation

    @property
    def ready(self) -> bool:
        return self.omega is not None

    def update(self, t: float, quat: np.ndarray):
        self.quat = quat
        self.buf.append((t, quat))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()
        if len(self.buf) < self.min_readings:
            return
        t0, q0 = self.buf[0]
        q0_inv = np.zeros(4)
        mujoco.mju_negQuat(q0_inv, q0)
        ts, rv = [], []
        for ti, qi in self.buf:
            dq = np.zeros(4)
            mujoco.mju_mulQuat(dq, qi, q0_inv)  # world-frame rotation from q0 to qi
            ts.append(ti - t0)
            rv.append(quat_to_rotvec(dq))
        ts, rv = np.array(ts), np.array(rv)
        # Slope of rotation vector vs time = world angular velocity (small-angle window).
        A = np.vstack([ts, np.ones_like(ts)]).T
        coef, *_ = np.linalg.lstsq(A, rv, rcond=None)
        self.omega = coef[0]
        # Smoothed orientation at the latest reading: the fitted line evaluated at t.
        self.t = t
        self.quat_smooth = rotate(q0, coef[1] + coef[0] * (t - t0))

    def predict(self, t: float) -> np.ndarray:
        """Orientation at future time t, assuming constant world-frame spin."""
        return rotate(self.quat_smooth, self.omega * (t - self.t))


def rotate(q: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Apply a world-frame rotation vector to orientation q."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return q.copy()
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, rotvec / angle, angle)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, q)
    return out / np.linalg.norm(out)
