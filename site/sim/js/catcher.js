// Port of sim/env.py and sim/catcher.py: the zero-g environment and the catch controller
// (wait -> approach -> ride -> close in -> grasp -> brake -> stopped, or abort and re-plan).

import { PoseSensor, SpinEstimator, TranslationKF } from "./estimation.js";
import {
  add, col, cross, dampedPinv, matMul, matTVec, matVec, mulVec, norm, nullProject, quatToMat,
  rotateMat, rotvecFromMatrix, scale, sub, transpose,
} from "./math.js";
import { ArmKinematics, QD_MAX, READY_Q, REACH, SHOULDER, planIntercept } from "./planning.js";

export const GRIPPER_OPEN = 255;
export const FINGER_MAX = 0.04;

export const Phase = {
  WAIT: "wait", APPROACH: "approach", RIDE: "ride", CLOSE_IN: "close_in", GRASP: "grasp",
  BRAKE: "brake", STOPPED: "stopped", ABORT: "abort", FAILED: "failed",
};

export function minJerk(s) {
  s = Math.min(Math.max(s, 0), 1);
  return [10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4];
}
// Distance covered while speed falls along a min-jerk profile, in units of (v0 x duration).
export function brakingDistance(s) {
  s = Math.min(Math.max(s, 0), 1);
  return s - 2.5 * s ** 4 + 3 * s ** 5 - s ** 6;
}

// --- environment -------------------------------------------------------------------------

export class ZeroGEnv {
  constructor(mj, model) {
    this.mj = mj;
    this.m = model;
    this.d = new mj.MjData(model);
    const j = model.jnt("target_free").id;
    this.tq = model.jnt_qposadr[j];
    this.tv = model.jnt_dofadr[j];
    this.targetBody = model.body("target").id;
    this.targetGeom = model.geom("target_geom").id;
    this.home = model.key("home").id;
  }

  // target: {pos, quat [w,x,y,z], vel (world), omega (body frame, rad/s)}
  reset(target) {
    const { mj, m, d } = this;
    mj.mj_resetDataKeyframe(m, d, this.home);
    const qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl;
    for (let i = 0; i < 7; i++) { qpos[i] = READY_Q[i]; ctrl[i] = READY_Q[i]; }
    for (let i = 0; i < 3; i++) qpos[this.tq + i] = target.pos[i];
    for (let i = 0; i < 4; i++) qpos[this.tq + 3 + i] = target.quat[i];
    for (let i = 0; i < 3; i++) { qvel[this.tv + i] = target.vel[i]; qvel[this.tv + 3 + i] = target.omega[i]; }
    mj.mj_forward(m, d);
  }

  step() {
    this.mj.mj_step(this.m, this.d);
  }

  targetPose() {
    const xp = this.d.xpos, xq = this.d.xquat, b = this.targetBody;
    return [[xp[3 * b], xp[3 * b + 1], xp[3 * b + 2]], [xq[4 * b], xq[4 * b + 1], xq[4 * b + 2], xq[4 * b + 3]]];
  }

  // Linear velocity (world) and angular velocity (body frame), like the Python env.
  targetVelocity() {
    const v = this.d.qvel, a = this.tv;
    return [[v[a], v[a + 1], v[a + 2]], [v[a + 3], v[a + 4], v[a + 5]]];
  }
}

// --- the catch controller ----------------------------------------------------------------

export class Catcher {
  constructor(env, {
    sensor = {}, sensorSeed = 1, standoff = 0.1, clear = 0.1, split = 0.6, settle = 0.6, ride = 0.1,
    graspDepth = 0.01, graspDist = 0.01, graspSpeed = 0.02, graspWait = 0.15, fingerFast = 0.03,
    fingerSpeed = 0.05, maxDecel = 0.3, maxAngDecel = 1.0, minBrake = 0.5, hold = 1.0, kpPos = 8,
    kpRot = 6, speedFrac = 0.8, replanEvery = 0.5, maxAborts = 2,
  } = {}) {
    this.env = env;
    this.mj = env.mj;
    this.m = env.m;
    this.d = env.d;
    this.kin = new ArmKinematics(env.mj, env.m);
    this.sensor = new PoseSensor({ seed: sensorSeed, ...sensor });
    this.kf = new TranslationKF();
    this.spin = new SpinEstimator();
    Object.assign(this, {
      standoff, clear, split, settle, ride, hold, graspDepth, graspDist, graspSpeed, graspWait,
      fingerFast, fingerSpeed, maxDecel, maxAngDecel, minBrake, kpPos, kpRot, replanEvery, maxAborts,
    });
    const m = this.m;
    this.fingers = ["finger_joint1", "finger_joint2"].map((f) => m.jnt_qposadr[m.jnt(f).id]);
    this.fingerDofs = ["finger_joint1", "finger_joint2"].map((f) => m.jnt_dofadr[m.jnt(f).id]);
    this.qadr = this.kin.qadr;
    this.dadr = this.kin.dadr;
    this.qdMax = QD_MAX.map((x) => x * speedFrac);
    // Position servos: torque = kp (ctrl - q) - kv qdot. Lag term kv / kp per joint.
    this.lag = [...Array(7)].map((_, i) => -m.actuator_biasprm[10 * i + 2] / m.actuator_gainprm[10 * i]);
    this.ctrlLo = [...Array(7)].map((_, i) => m.actuator_ctrlrange[2 * i]);
    this.ctrlHi = [...Array(7)].map((_, i) => m.actuator_ctrlrange[2 * i + 1]);
    this.tGrasp = this.tAbort = this.tBrake = null;
    this.graspCheck = null;
    this.tReplan = 0;
    this.nPlans = 0;
    this.aborts = 0;
    this.gaveUp = false;
    this.trackErr = 0;
    this.phase = Phase.WAIT;
    this.plan = null;
    this.qCmd = this.qMeas();
    this.qdCmd = new Array(7).fill(0);
    this.qdTau = 0.03;
    this.unwind = 0.05;
    this.kNull = 2.0;
    const ctrl = this.d.ctrl;
    for (let i = 0; i < 7; i++) ctrl[i] = this.qCmd[i];
    ctrl[7] = GRIPPER_OPEN;
    this.events = []; // [time, text] for the UI
  }

  qMeas() {
    const qpos = this.d.qpos;
    return this.qadr.map((a) => qpos[a]);
  }

  log(text) {
    this.events.push([this.d.time, text]);
  }

  // --- target pose, relative to the bar --------------------------------------------------

  barPose(t) {
    const { p, v } = this.kf.predict(t);
    return [p, v, quatToMat(this.spin.predict(t))];
  }

  graspOffset(Rbar, standoff) {
    const Rh = matMul(Rbar, this.Rrel);
    return [Rh, scale(col(Rh, 2), -standoff)];
  }

  standoffAt(t) {
    if (this.phase === Phase.CLOSE_IN) {
      const [f] = minJerk((t - this.tClose) / (this.plan.t - this.tClose));
      return this.standoff + (this.graspDepth - this.standoff) * f;
    }
    if (this.phase === Phase.GRASP || this.phase === Phase.BRAKE || this.phase === Phase.STOPPED) return this.graspDepth;
    if (this.phase === Phase.ABORT) {
      const [f] = minJerk((t - this.tAbort) / 0.3);
      return this.abortFrom + (this.standoff - this.abortFrom) * f;
    }
    return this.standoff;
  }

  // Desired tool-point position and hand rotation at time t (estimates only).
  target(t) {
    if (this.phase === Phase.GRASP) {
      const g = t - this.tGrasp;
      return [add(this.graspP0, scale(this.graspV0, g)), rotateMat(this.graspR0, scale(this.graspW0, g))];
    }
    if (this.phase === Phase.BRAKE || this.phase === Phase.STOPPED) {
      const g = brakingDistance((t - this.tBrake) / this.brakeT) * this.brakeT;
      return [add(this.brakeP0, scale(this.brakeV0, g)), rotateMat(this.brakeR0, scale(this.brakeW0, g))];
    }
    const [pBar, , Rbar] = this.barPose(t);
    let [Rh, offset] = this.graspOffset(Rbar, this.standoffAt(t));
    if (this.phase === Phase.APPROACH) {
      const Rgrasp = Rh;
      const s = (t - this.tStart) / (this.tMeet - this.tStart);
      const back = -(this.standoff + this.clear);
      let c;
      if (s < this.split) {
        const [f] = minJerk(s / this.split);
        c = add(scale(this.c0, 1 - f), scale([0, 0, back], f));
        Rh = slerp(this.R0, Rgrasp, f);
      } else {
        const [f] = minJerk((s - this.split) / (1 - this.split));
        c = [0, 0, back + f * this.clear];
      }
      offset = matVec(Rgrasp, c);
    }
    return [add(pBar, offset), Rh];
  }

  // --- main loop -------------------------------------------------------------------------

  // Call once per physics step, before env.step().
  step() {
    const t = this.d.time;
    const [p, q] = this.env.targetPose();
    const z = this.sensor.poll(t, p, q);
    if (z) {
      this.kf.update(z.t, z.pos);
      this.spin.update(z.t, z.quat);
    }
    if (this.phase === Phase.WAIT) {
      if (this.spin.ready && norm(this.kf.velStd) < 0.01 && t >= this.tReplan) this.startApproach(t);
      return;
    }
    if (this.phase === Phase.FAILED) return;
    if (this.phase === Phase.ABORT && !this.gaveUp && t >= this.tReplan) this.startApproach(t);
    if (this.phase === Phase.APPROACH && t >= this.tMeet) this.setPhase(Phase.RIDE, t);
    if (this.phase === Phase.RIDE && t >= this.tClose) this.setPhase(Phase.CLOSE_IN, t);
    if (this.phase === Phase.CLOSE_IN && t >= this.plan.t) this.tryGrasp(t);
    if (this.phase === Phase.GRASP) {
      const w = this.fingerTarget(t);
      this.d.ctrl[7] = (GRIPPER_OPEN * w) / FINGER_MAX;
      const qv = this.d.qvel;
      const stalled = Math.max(Math.abs(qv[this.fingerDofs[0]]), Math.abs(qv[this.fingerDofs[1]])) < 0.005;
      if (w === 0 && stalled) {
        this.fingerGap = this.d.qpos[this.fingers[0]] + this.d.qpos[this.fingers[1]];
        this.startBrake(t);
      }
    }
    if (this.phase === Phase.BRAKE && t >= this.tBrake + this.brakeT) this.setPhase(Phase.STOPPED, t);
    this.track(t);
  }

  setPhase(phase, t) {
    this.phase = phase;
    this.log(phase);
  }

  fingerTarget(t) {
    const dt = t - this.tGrasp;
    const fastTime = (FINGER_MAX - this.fingerFast) / 0.2;
    if (dt < fastTime) return FINGER_MAX - 0.2 * dt;
    return Math.max(0, this.fingerFast - this.fingerSpeed * (dt - fastTime));
  }

  currentMotion(t) {
    const dt = this.m.opt.timestep;
    const [p0, R0] = this.target(t);
    const [p1, R1] = this.target(t + dt);
    return [p0, R0, scale(sub(p1, p0), 1 / dt), scale(rotvecFromMatrix(matMul(R1, transpose(R0))), 1 / dt)];
  }

  startBrake(t) {
    [this.brakeP0, this.brakeR0, this.brakeV0, this.brakeW0] = this.currentMotion(t);
    this.brakeT = Math.max(this.minBrake, (1.875 * norm(this.brakeV0)) / this.maxDecel, (1.875 * norm(this.brakeW0)) / this.maxAngDecel);
    this.tBrake = t;
    this.setPhase(Phase.BRAKE, t);
  }

  // Distance from the tool point to the grasp point and their relative speed, from estimates.
  estimatedMismatch(t) {
    const [pBar, vBar] = this.barPose(t);
    const [pDes] = this.target(t);
    const q = this.qMeas();
    const [p] = this.kin.fk(q);
    const J = this.kin.jacobian(p);
    const qv = this.d.qvel;
    const qd = this.dadr.map((a) => qv[a]);
    const vTcp = mulVec(J, 3, 7, qd);
    const vPoint = add(vBar, cross(this.spin.omega, sub(pDes, pBar)));
    return [norm(sub(p, pDes)), norm(sub(vTcp, vPoint))];
  }

  tryGrasp(t) {
    const [dist, speed] = this.estimatedMismatch(t);
    if (dist < this.graspDist && speed < this.graspSpeed) {
      [this.graspP0, this.graspR0, this.graspV0, this.graspW0] = this.currentMotion(t);
      this.tGrasp = t;
      this.graspCheck = [dist, speed];
      this.setPhase(Phase.GRASP, t);
    } else if (t - this.plan.t > this.graspWait) {
      this.tAbort = t;
      this.aborts++;
      this.gaveUp = this.aborts > this.maxAborts;
      this.tReplan = t + 0.3 + this.replanEvery / 2;
      this.abortFrom = norm(sub(this.kin.fk(this.qMeas())[0], this.barPose(t)[0]));
      this.setPhase(Phase.ABORT, t);
      this.log(this.gaveUp ? "gave up" : "backing off to re-plan");
    }
  }

  startApproach(t) {
    const qNow = this.qMeas();
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const plan = planIntercept(this.kin, this.kf, this.spin, t, qNow, { speedFrac: 0.6, settle: this.settle, hold: this.hold });
    this.planMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
    this.nPlans++;
    if (!plan) {
      this.tReplan = t + this.replanEvery;
      const r = sub(this.kf.pos, SHOULDER);
      if (norm(r) > REACH && r[0] * this.kf.vel[0] + r[1] * this.kf.vel[1] + r[2] * this.kf.vel[2] > 0) {
        this.gaveUp = true;
        if (this.phase === Phase.WAIT) this.setPhase(Phase.FAILED, t);
      } else if (this.nPlans === 1) {
        this.log("no reachable catch yet, re-planning");
      }
      return;
    }
    this.plan = plan;
    this.Rrel = matMul(transpose(quatToMat(plan.quat)), plan.handR);
    const [pTcp, R0] = this.kin.fk(qNow);
    this.R0 = R0;
    const [pBar, , Rbar] = this.barPose(t);
    this.c0 = matTVec(matMul(Rbar, this.Rrel), sub(pTcp, pBar));
    this.tStart = t;
    this.tMeet = plan.t - this.settle;
    this.tClose = this.tMeet + this.ride;
    this.setPhase(Phase.APPROACH, t);
  }

  track(t) {
    const dt = this.m.opt.timestep;
    const [pDes, Rdes] = this.target(t);
    const [pNext, Rnext] = this.target(t + dt);
    const vff = scale(sub(pNext, pDes), 1 / dt);
    const wff = scale(rotvecFromMatrix(matMul(Rnext, transpose(Rdes))), 1 / dt);
    const q = this.qMeas();
    const [p, R] = this.kin.fk(q);
    this.trackErr = norm(sub(pDes, p));
    const v = add(vff, scale(sub(pDes, p), this.kpPos));
    const w = add(wff, scale(rotvecFromMatrix(matMul(Rdes, transpose(R))), this.kpRot));
    const J = this.kin.jacobian(p);
    const P = dampedPinv(J, 7, 0.02);
    let qd = mulVec(P, 7, 6, [...v, ...w]);
    const nul = nullProject(P, J, 7, this.plan.q.map((x, i) => this.kNull * (x - q[i])));
    qd = qd.map((x, i) => x + nul[i]);
    const s = Math.min(1, ...qd.map((x, i) => this.qdMax[i] / Math.max(Math.abs(x), 1e-12)));
    qd = qd.map((x) => x * s);
    const lo = this.kin.lo, hi = this.kin.hi;
    for (let i = 0; i < 7; i++) {
      this.qCmd[i] = Math.min(Math.max(this.qCmd[i] + qd[i] * dt, lo[i]), hi[i]);
      this.qCmd[i] += this.unwind * (q[i] - this.qCmd[i]);
      this.qdCmd[i] += (dt / (this.qdTau + dt)) * (qd[i] - this.qdCmd[i]);
    }
    const ctrl = this.d.ctrl;
    for (let i = 0; i < 7; i++)
      ctrl[i] = Math.min(Math.max(this.qCmd[i] + this.lag[i] * this.qdCmd[i], this.ctrlLo[i]), this.ctrlHi[i]);
  }

  // --- ground truth, for scoring and the readouts ------------------------------------------

  // Tool point to the true target point, their relative speed, hand orientation error (deg).
  trueErrors() {
    const [pBar, qBar] = this.env.targetPose();
    const [vBar, wBody] = this.env.targetVelocity();
    const Rbar = quatToMat(qBar);
    const [RhDes, offset] = this.graspOffset(Rbar, this.standoffAt(this.d.time));
    const vPoint = add(vBar, cross(matVec(Rbar, wBody), offset));
    const q = this.qMeas();
    const [p, R] = this.kin.fk(q);
    const J = this.kin.jacobian(p);
    const qv = this.d.qvel;
    const vTcp = mulVec(J, 3, 7, this.dadr.map((a) => qv[a]));
    const rot = (rotvecFromMatrix(matMul(RhDes, transpose(R))).reduce((s, x) => s + x * x, 0) ** 0.5 * 180) / Math.PI;
    return [norm(sub(add(pBar, offset), p)), norm(sub(vTcp, vPoint)), rot];
  }
}

function slerp(R0, R1, f) {
  const rv = scale(rotvecFromMatrix(matMul(R1, transpose(R0))), f);
  if (norm(rv) < 1e-9) return f >= 1 ? R1 : R0;
  return rotateMat(R0, rv);
}
