"""Catch controller, phases so far:

  WAIT      observe until the velocity estimate is good, then plan an intercept (sim.intercept);
            if none is reachable yet, try again every `replan_every` s until the bar has left
            the workspace (FAILED)
  APPROACH  move the tool point to the pre-grasp pose by `t_meet` (= capture time - settle)
  RIDE      hold the pre-grasp pose relative to the moving, spinning bar for `ride` s
  CLOSE_IN  slide in along the approach axis from `standoff` to 0, arriving at the capture time
            with zero relative velocity (minimum-jerk), fingers straddling the bar
  GRASP     close the gripper, but only if the estimates say the tool point is within
            `grasp_dist` of the bar centre and moving with it to within `grasp_speed`; otherwise
            wait up to `grasp_wait` s, then ABORT (back off to the standoff, fingers open, ride
            along) and plan a new catch from there, up to `max_aborts` times.
            The fingers close gently: quickly to just short of the bar, then at `finger_speed`
            through contact and on until the grip force has built up. From the moment they
            start closing, the hand stops following the estimates and carries on with the bar's
            motion as estimated then (constant velocity and spin): once the fingers touch, the
            hand moves the bar, and following the estimates would mean following itself (the
            spin ran away, 20 -> 44 deg/s, before this)
  BRAKE     bar held: bring hand and bar to rest along a minimum-jerk velocity profile, both the
            drift and the spin, with the deceleration capped at `max_decel` / `max_ang_decel`,
            so the grip only has to supply a small, smooth force
  STOPPED   at rest, holding the bar

Everything is expressed relative to the bar. The grasp is fixed in the bar's frame when the plan
is made (`R_rel`: hand orientation relative to the bar); the pre-grasp point sits `standoff` back
along the gripper's approach axis. Each step the target pose is rebuilt from the latest estimates
(filter position/velocity, spin orientation), so the arm keeps correcting as the estimates
improve, without re-running the planner.

Approach path, in the grasp frame (moves and spins with the bar): for the first `split` of the
approach time, line up behind the bar at `standoff + clear` along the approach axis while
turning the hand to the grasp orientation; then close straight in to `standoff`. Each stage is a
minimum-jerk profile, so the arm arrives with zero relative velocity and is already riding along.
Lining up first keeps the hand from sweeping sideways past the ends of the bar.

Control: Cartesian velocity command (feedforward + P on pose error) -> damped pseudo-inverse ->
joint velocities, plus a null-space pull towards the planned capture posture (the Panda has one
joint more than the task needs; left alone, the elbow drifts to full stretch) -> capped ->
integrated joint targets for the Panda's position servos, with the servo's velocity lag
compensated (ctrl = q_cmd + (kv/kp) * qd_cmd).
"""

from enum import Enum

import mujoco
import numpy as np

from sim.estimator import SpinEstimator, TranslationKF
from sim.intercept import REACH, SHOULDER, Intercept, plan_intercept, quat_to_mat
from sim.kinematics import QD_MAX, ArmKinematics, rotvec_from_matrix
from sim.sensor import PoseSensor

GRIPPER_OPEN = 255.0
FINGER_MAX = 0.04  # m per finger at ctrl 255


class Phase(Enum):
    WAIT = "wait"
    APPROACH = "approach"
    RIDE = "ride"
    CLOSE_IN = "close_in"
    GRASP = "grasp"
    BRAKE = "brake"
    STOPPED = "stopped"
    ABORT = "abort"
    FAILED = "failed"


def min_jerk(s: float) -> tuple[float, float]:
    """Position and d/ds of the minimum-jerk profile on s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


def braking_distance(s: float) -> float:
    """Distance covered, in units of (initial speed x duration), while the speed falls from
    full to zero along a minimum-jerk profile; s in [0, 1] (0.5 at the end)."""
    s = min(max(s, 0.0), 1.0)
    return s - 2.5 * s**4 + 3 * s**5 - s**6


class Catcher:
    def __init__(self, env, sensor_seed=None, standoff=0.10, clear=0.10, split=0.6, settle=0.6,
                 ride=0.1, grasp_depth=0.01, grasp_dist=0.01, grasp_speed=0.02, grasp_wait=0.15,
                 finger_fast=0.03, finger_speed=0.05, max_decel=0.3, max_ang_decel=1.0,
                 min_brake=0.5, hold=1.0, kp_pos=8.0, kp_rot=6.0, speed_frac=0.8,
                 replan_every=0.5, max_aborts=2):
        self.env, self.m, self.d = env, env.model, env.data
        self.kin = ArmKinematics(self.m)
        self.sensor = PoseSensor(seed=sensor_seed)
        self.kf, self.spin = TranslationKF(), SpinEstimator()
        self.standoff, self.clear, self.split, self.settle = standoff, clear, split, settle
        self.ride, self.hold = ride, hold
        # The fingertip pads stop `grasp_depth` short of the bar's centre: more room between the
        # bar's near face and the palm (only ~2 cm with the pads centred).
        self.grasp_depth = grasp_depth
        self.grasp_dist, self.grasp_speed, self.grasp_wait = grasp_dist, grasp_speed, grasp_wait
        self.fingers = [self.m.joint(f).qposadr[0] for f in ("finger_joint1", "finger_joint2")]
        # Fingers: fast to `finger_fast` per finger (the bar's faces are at 0.025, so 5 mm short),
        # then `finger_speed` m/s down to 0 (fully closed command: ~2.5 N grip on the bar).
        self.finger_fast, self.finger_speed = finger_fast, finger_speed
        self.max_decel, self.max_ang_decel, self.min_brake = max_decel, max_ang_decel, min_brake
        self.t_grasp = self.t_abort = self.t_brake = None
        self.grasp_check = None  # (distance, relative speed) estimated when the gripper closed
        self.replan_every, self.max_aborts = replan_every, max_aborts
        self.t_replan, self.n_plans, self.aborts = 0.0, 0, 0
        self.gave_up = False  # FAILED, or aborted with no retries left
        self.track_err = 0.0  # tool point to its target (m), last step
        self.kp_pos, self.kp_rot = kp_pos, kp_rot
        self.qd_max = QD_MAX * speed_frac
        self.qadr, self.dadr = self.kin.qadr, self.kin.dadr
        kp = self.m.actuator_gainprm[:7, 0]
        self.lag = -self.m.actuator_biasprm[:7, 2] / kp  # kv / kp per joint
        self.phase = Phase.WAIT
        self.plan: Intercept | None = None
        self.q_cmd = self.d.qpos[self.qadr].copy()
        self.qd_cmd, self.qd_tau = np.zeros(7), 0.03
        self.unwind = 0.05
        self.k_null = 2.0
        self.d.ctrl[:7] = self.q_cmd
        self.d.ctrl[7] = GRIPPER_OPEN

    # --- target pose, relative to the bar --------------------------------------------------

    def bar_pose(self, t: float):
        """Estimated bar centre, velocity and rotation matrix at time t."""
        pos, vel, _ = self.kf.predict(t)
        return pos, vel, quat_to_mat(self.spin.predict(t))

    def grasp_offset(self, R_bar: np.ndarray, standoff: float):
        """Hand rotation and tool-point offset from the bar centre for a given bar rotation."""
        R_hand = R_bar @ self.R_rel
        return R_hand, -standoff * R_hand[:, 2]

    def standoff_at(self, t: float) -> float:
        """Distance of the tool point behind the bar centre, along the approach axis."""
        if self.phase is Phase.CLOSE_IN:
            f, _ = min_jerk((t - self.t_close) / (self.plan.t - self.t_close))
            return self.standoff + (self.grasp_depth - self.standoff) * f
        if self.phase in (Phase.GRASP, Phase.BRAKE, Phase.STOPPED):
            return self.grasp_depth
        if self.phase is Phase.ABORT:
            f, _ = min_jerk((t - self.t_abort) / 0.3)
            return self.abort_from + (self.standoff - self.abort_from) * f
        return self.standoff

    def target(self, t: float):
        """Desired tool-point position and hand rotation at time t (estimates only)."""
        if self.phase is Phase.GRASP:
            g = t - self.t_grasp
            return self.grasp_p0 + self.grasp_v0 * g, self.rotate(self.grasp_R0, self.grasp_w0 * g)
        if self.phase in (Phase.BRAKE, Phase.STOPPED):
            g = braking_distance((t - self.t_brake) / self.brake_T) * self.brake_T
            return self.brake_p0 + self.brake_v0 * g, self.rotate(self.brake_R0, self.brake_w0 * g)
        p_bar, _, R_bar = self.bar_pose(t)
        R_hand, offset = self.grasp_offset(R_bar, self.standoff_at(t))
        if self.phase is Phase.APPROACH:
            R_grasp = R_hand
            s = (t - self.t_start) / (self.t_meet - self.t_start)
            back = -(self.standoff + self.clear)
            if s < self.split:
                f, _ = min_jerk(s / self.split)
                c = (1 - f) * self.c0 + f * np.array([0.0, 0.0, back])
                R_hand = self.slerp(self.R0, R_grasp, f)
            else:
                f, _ = min_jerk((s - self.split) / (1 - self.split))
                c = np.array([0.0, 0.0, back + f * self.clear])
            offset = R_grasp @ c
        return p_bar + offset, R_hand

    @staticmethod
    def rotate(R0, rotvec):
        """Apply a world-frame rotation vector to rotation matrix R0."""
        angle = np.linalg.norm(rotvec)
        if angle < 1e-12:
            return R0
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, rotvec / angle, angle)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3) @ R0

    @staticmethod
    def slerp(R0, R1, f):
        rv = rotvec_from_matrix(R1 @ R0.T) * f
        q = np.zeros(4)
        angle = np.linalg.norm(rv)
        if angle < 1e-9:
            return R1 if f >= 1 else R0
        mujoco.mju_axisAngle2Quat(q, rv / angle, angle)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3) @ R0

    # --- main loop -------------------------------------------------------------------------

    def step(self):
        """Call once per physics step, before env.step()."""
        t = self.d.time
        p, q = self.env.target_pose()
        z = self.sensor.poll(t, p, q)
        if z is not None:
            self.kf.update(z.t, z.pos)
            self.spin.update(z.t, z.quat)

        if self.phase is Phase.WAIT:
            if (self.spin.ready and np.linalg.norm(self.kf.vel_std) < 0.01
                    and t >= self.t_replan):
                self.start_approach(t)
            return
        if self.phase is Phase.FAILED:
            return
        if self.phase is Phase.ABORT and not self.gave_up and t >= self.t_replan:
            self.start_approach(t)
        if self.phase is Phase.APPROACH and t >= self.t_meet:
            self.phase = Phase.RIDE
        if self.phase is Phase.RIDE and t >= self.t_close:
            self.phase = Phase.CLOSE_IN
        if self.phase is Phase.CLOSE_IN and t >= self.plan.t:
            self.try_grasp(t)
        if self.phase is Phase.GRASP:
            w = self.finger_target(t)
            self.d.ctrl[7] = GRIPPER_OPEN * w / FINGER_MAX
            stalled = np.abs(self.d.qvel[self.fingers]).max() < 0.005
            if w == 0.0 and stalled:
                self.finger_gap = float(self.d.qpos[self.fingers].sum())
                self.start_brake(t)
        if self.phase is Phase.BRAKE and t >= self.t_brake + self.brake_T:
            self.phase = Phase.STOPPED
        self.track(t)

    def finger_target(self, t: float) -> float:
        """Commanded opening per finger (m) while closing."""
        dt = t - self.t_grasp
        fast_time = (FINGER_MAX - self.finger_fast) / 0.2
        if dt < fast_time:
            return FINGER_MAX - 0.2 * dt
        return max(0.0, self.finger_fast - self.finger_speed * (dt - fast_time))

    def current_motion(self, t: float):
        """Target pose at t and its linear / angular velocity (finite difference)."""
        dt = self.m.opt.timestep
        p0, R0 = self.target(t)
        p1, R1 = self.target(t + dt)
        return p0, R0, (p1 - p0) / dt, rotvec_from_matrix(R1 @ R0.T) / dt

    def start_brake(self, t: float):
        """Decelerate the current target motion to rest."""
        self.brake_p0, self.brake_R0, self.brake_v0, self.brake_w0 = self.current_motion(t)
        # A minimum-jerk speed profile peaks at 1.875x the average deceleration.
        self.brake_T = max(self.min_brake,
                           1.875 * np.linalg.norm(self.brake_v0) / self.max_decel,
                           1.875 * np.linalg.norm(self.brake_w0) / self.max_ang_decel)
        self.phase, self.t_brake = Phase.BRAKE, t

    def estimated_mismatch(self, t: float) -> tuple[float, float]:
        """Distance from the tool point to the grasp point on the bar, and their relative speed,
        as far as the estimates can tell (no ground truth)."""
        p_bar, v_bar, _ = self.bar_pose(t)
        p_des, _ = self.target(t)
        q = self.d.qpos[self.qadr].copy()
        p, _ = self.kin.fk(q)
        v_tcp = self.kin.jacobian(p)[:3] @ self.d.qvel[self.dadr]
        v_point = v_bar + np.cross(self.spin.omega, p_des - p_bar)
        return float(np.linalg.norm(p - p_des)), float(np.linalg.norm(v_tcp - v_point))

    def try_grasp(self, t: float):
        dist, speed = self.estimated_mismatch(t)
        if dist < self.grasp_dist and speed < self.grasp_speed:
            self.grasp_p0, self.grasp_R0, self.grasp_v0, self.grasp_w0 = self.current_motion(t)
            self.phase, self.t_grasp, self.grasp_check = Phase.GRASP, t, (dist, speed)
        elif t - self.plan.t > self.grasp_wait:
            self.phase, self.t_abort = Phase.ABORT, t
            self.aborts += 1
            self.gave_up = self.aborts > self.max_aborts
            self.t_replan = t + 0.3 + self.replan_every / 2  # after backing off
            self.abort_from = float(np.linalg.norm(self.kin.fk(self.d.qpos[self.qadr])[0]
                                                   - self.bar_pose(t)[0]))

    def start_approach(self, t: float):
        q_now = self.d.qpos[self.qadr].copy()
        # Plan for peaks at 60% of Franka's joint speeds; the controller may use up to 80%
        # (the Cartesian path needs more joint travel than the straight joint-space estimate).
        # `hold`: the bar must stay reachable while the fingers close (~0.65 s) and the arm
        # brakes (travelling about half the bar's speed x the braking time).
        plan = plan_intercept(self.kin, self.kf, self.spin, t, q_now, speed_frac=0.6,
                              settle=self.settle, hold=self.hold)
        self.n_plans += 1
        if plan is None:
            # Keep trying while the bar may still come within reach (when riding along after an
            # abort, keep the old plan: tracking uses its posture).
            self.t_replan = t + self.replan_every
            r = self.kf.pos - SHOULDER
            if np.linalg.norm(r) > REACH and np.dot(r, self.kf.vel) > 0:
                self.gave_up = True
                if self.phase is Phase.WAIT:
                    self.phase = Phase.FAILED
            return
        self.plan = plan
        # Grasp fixed in the bar's frame, from the planned bar orientation and hand rotation.
        self.R_rel = quat_to_mat(self.plan.quat).T @ self.plan.hand_R
        p_tcp, self.R0 = self.kin.fk(q_now)
        p_bar, _, R_bar = self.bar_pose(t)
        self.c0 = (R_bar @ self.R_rel).T @ (p_tcp - p_bar)  # start offset in the grasp frame
        self.t_start, self.t_meet = t, self.plan.t - self.settle
        self.t_close = self.t_meet + self.ride
        self.phase = Phase.APPROACH

    def track(self, t: float):
        dt = self.m.opt.timestep
        p_des, R_des = self.target(t)
        p_next, R_next = self.target(t + dt)
        v_ff = (p_next - p_des) / dt
        w_ff = rotvec_from_matrix(R_next @ R_des.T) / dt

        q_meas = self.d.qpos[self.qadr].copy()
        p, R = self.kin.fk(q_meas)
        self.track_err = float(np.linalg.norm(p_des - p))
        v = v_ff + self.kp_pos * (p_des - p)
        w = w_ff + self.kp_rot * rotvec_from_matrix(R_des @ R.T)
        J = self.kin.jacobian(p)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.02**2 * np.eye(6))
        qd = J_pinv @ np.concatenate([v, w])
        qd += (np.eye(7) - J_pinv @ J) @ (self.k_null * (self.plan.q - q_meas))
        qd *= min(1.0, np.min(self.qd_max / np.maximum(np.abs(qd), 1e-12)))

        self.q_cmd = np.clip(self.q_cmd + qd * dt, self.kin.lo, self.kin.hi)
        # Pull the integrated command back towards the measured joints, so servo error
        # can't wind up.
        self.q_cmd += self.unwind * (q_meas - self.q_cmd)
        # The lag term multiplies speed by kv (up to 450 Nm per rad/s), so the small jumps in
        # qd at each new 30 Hz reading would become torque spikes that saturate the 12 Nm
        # wrist motors. Low-pass it first.
        self.qd_cmd += dt / (self.qd_tau + dt) * (qd - self.qd_cmd)
        self.d.ctrl[:7] = np.clip(self.q_cmd + self.lag * self.qd_cmd,
                                  *self.m.actuator_ctrlrange[:7].T)

    # --- ground truth, for checks ----------------------------------------------------------

    def true_errors(self):
        """Tool-point position error vs the true target point (pre-grasp or grasp), its velocity
        relative to that
        point (which moves with the bar and swings with its spin), and hand orientation error
        (deg), all from ground truth."""
        p_bar, q_bar = self.env.target_pose()
        v_bar, w_body = self.env.target_velocity()
        R_bar = quat_to_mat(q_bar)
        R_hand_des, offset = self.grasp_offset(R_bar, self.standoff_at(self.d.time))
        v_point = v_bar + np.cross(R_bar @ w_body, offset)
        q = self.d.qpos[self.qadr].copy()
        p, R = self.kin.fk(q)
        v_tcp = self.kin.jacobian(p)[:3] @ self.d.qvel[self.dadr]
        rot_err = np.rad2deg(np.linalg.norm(rotvec_from_matrix(R_hand_des @ R.T)))
        return np.linalg.norm(p_bar + offset - p), np.linalg.norm(v_tcp - v_point), rot_err
