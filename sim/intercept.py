"""Intercept selection: the earliest predicted target pose the arm can reach in time.

Sweep capture times forward from now. At each one, predict the target's position (Kalman filter)
and orientation (constant spin), work out the grasp, and accept the first that
  - has a converged, collision-free IK solution within joint limits,
  - stays reachable, well clear of singularities, for `hold` seconds after capture (the bar
    keeps drifting while the gripper closes and brakes it),
  - the arm can get to from its current posture in time (at most `MAX_REACH_TIME`), along a
    minimum-jerk move whose peak joint speed stays under `speed_frac` of Franka's limits, plus
    `settle` seconds to match velocity before closing.

Grasp geometry: the hand's x axis lies along the bar's long axis, and the approach axis (hand z)
is square to one of the bar's four long faces, so the fingers close across its 5 cm width (not
the 7 cm diagonal, which nearly fills the 8 cm opening and squeezes corners) and the palm faces a
flat side. Of the faces, only those within `MAX_FACE_ANGLE` of the direction from the tool point
to the bar are used (the hand reaches out at the bar from its own side, so the pre-grasp point,
`STANDOFF` back along the approach, lies between the hand and the bar); the quickest to reach
wins.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from sim.estimator import SpinEstimator, TranslationKF
from sim.kinematics import QD_MAX, ArmKinematics

SHOULDER = np.array([0.0, 0.0, 0.333])
REACH = 0.82  # m from the shoulder; Panda nominal 0.855
MIN_HEIGHT = 0.15  # m, keep the catch above the deck/base clutter
MIN_JERK_PEAK = 1.875
ELBOW_MAX = -0.75  # rad; the Panda's arm is fully stretched (singular) at joint 4 = -0.46
WRIST_MIN = 0.3  # rad; joint 6 = 0 lines up joints 5 and 7 (wrist singularity)
STANDOFF = 0.10  # m, pre-grasp distance behind the bar (matches sim.catcher)
MAX_REACH_TIME = 2.5  # s; longer moves (wrist flips) can't track the Cartesian path, and the
# spin prediction has drifted by the time they arrive
MAX_FACE_ANGLE = 70.0  # deg between the approach axis and the hand-to-bar direction


@dataclass
class Intercept:
    t: float  # capture time, s (sim clock)
    pos: np.ndarray  # predicted target centre
    vel: np.ndarray  # predicted target velocity
    quat: np.ndarray  # predicted target orientation
    hand_R: np.ndarray  # desired hand rotation at capture
    q: np.ndarray  # arm joint angles at capture
    reach_time: float  # s the arm needs to get there
    pos_std: float  # predicted position 1-sigma (norm), m
    planned_at: float  # s


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def plan_intercept(
    kin: ArmKinematics,
    kf: TranslationKF,
    spin: SpinEstimator,
    t_now: float,
    q_now: np.ndarray,
    horizon=(0.4, 4.0),
    dt=0.1,
    speed_frac=0.5,
    settle=0.3,
    hold=0.5,
) -> Intercept | None:
    q_now = np.asarray(q_now, float)
    tcp_now, R_now = kin.fk(q_now)
    for tc in np.arange(t_now + horizon[0], t_now + horizon[1], dt):
        pos, vel, P = kf.predict(tc)
        if np.linalg.norm(pos - SHOULDER) > REACH or pos[2] < MIN_HEIGHT:
            continue
        quat = spin.predict(tc)
        # Every candidate grasp, IK from the current posture, quickest to reach first. The
        # quickest can still fail the hold check (e.g. from above, a high bar's pre-grasp is out
        # of reach) when a grasp from the side would not, so try them in turn.
        candidates = []
        for R in grasp_rotations(quat_to_mat(quat), pos - tcp_now, R_now[:, 0]):
            q_try, ok = kin.ik(q_now, pos, R, iters=80)
            if ok and posture_ok(kin, q_try):
                candidates.append((np.max(np.abs(q_try - q_now) / QD_MAX), q_try))
        for dist, q in sorted(candidates, key=lambda c: c[0]):
            # A minimum-jerk move peaks at 1.875x its average speed.
            reach_time = float(MIN_JERK_PEAK * dist / speed_frac)
            if reach_time > MAX_REACH_TIME or t_now + reach_time + settle > tc:
                break  # the rest are slower still
            # Most expensive check last.
            if stays_reachable(kin, kf, spin, q, tc, hold, settle):
                hand_R = kin.fk(q)[1]
                return Intercept(tc, pos, vel, quat, hand_R, q, reach_time,
                                 float(np.sqrt(np.trace(P))), t_now)
    return None


def grasp_rotations(R_bar, reach_dir, hand_x_now):
    """Hand rotations for the grasp: x along the bar (either way, the one nearer the hand's
    current x first), z square to a long face of the bar and within `MAX_FACE_ANGLE` of
    `reach_dir`, nearest first."""
    axis = R_bar[:, 2]  # long axis (box half-size 0.075 along z)
    faces = [R_bar[:, 0], -R_bar[:, 0], R_bar[:, 1], -R_bar[:, 1]]
    r = reach_dir - np.dot(reach_dir, axis) * axis
    if np.linalg.norm(r) < 1e-6:
        return
    r /= np.linalg.norm(r)
    cos_max = np.cos(np.deg2rad(MAX_FACE_ANGLE))
    first = axis if np.dot(axis, hand_x_now) >= 0 else -axis
    for z in sorted((f for f in faces if np.dot(f, r) >= cos_max), key=lambda f: -np.dot(f, r)):
        for x in (first, -first):
            yield np.column_stack([x, np.cross(z, x), z])


def posture_ok(kin: ArmKinematics, q: np.ndarray) -> bool:
    """Clear of the elbow and wrist singularities, and not in self-collision."""
    return q[3] <= ELBOW_MAX and q[5] >= WRIST_MIN and not kin.self_collision(q)


def stays_reachable(kin, kf, spin, q, tc, hold, settle=0.3, steps=3) -> bool:
    """From the pre-grasp (`settle` s before tc) until `hold` s after tc, the grasp and the
    pre-grasp postures stay reachable and clear of singularities. The hand turns with the bar
    (fixed grasp in the bar's frame), as the controller will make it."""
    if not posture_ok(kin, q):
        return False
    R_rel = quat_to_mat(spin.predict(tc)).T @ kin.fk(q)[1]
    q_grasp = q
    for t in np.linspace(tc - settle, tc + hold, steps + 2):
        pos, _, _ = kf.predict(t)
        if np.linalg.norm(pos - SHOULDER) > REACH:
            return False
        R_hand = quat_to_mat(spin.predict(t)) @ R_rel
        q_grasp, ok = kin.ik(q_grasp, pos, R_hand)
        if not ok or not posture_ok(kin, q_grasp):
            return False
        q_pre, ok = kin.ik(q_grasp, pos - STANDOFF * R_hand[:, 2], R_hand)
        if not ok or not posture_ok(kin, q_pre):
            return False
    return True
