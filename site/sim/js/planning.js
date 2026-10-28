// Port of sim/kinematics.py and sim/intercept.py: tool-point kinematics on a private MjData,
// damped least-squares IK, self-collision checks, and the intercept planner.

import {
  add, col, cross, dampedPinv, dot, fromCols, matMul, matVec, mulVec, norm, nullProject, quatToMat,
  rotvecFromMatrix, scale, sub, transpose,
} from "./math.js";

export const ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"];
export const READY_Q = [0.0, -0.169, 0.0, -1.188, 0.0, 1.019, -0.785];
export const TCP_OFFSET = [0, 0, 0.1034];
export const QD_MAX = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61];

export class ArmKinematics {
  constructor(mj, model) {
    this.mj = mj;
    this.m = model;
    this.d = new mj.MjData(model);
    this.hand = model.body("hand").id;
    const jids = ARM_JOINTS.map((j) => model.jnt(j).id);
    this.qadr = jids.map((j) => model.jnt_qposadr[j]);
    this.dadr = jids.map((j) => model.jnt_dofadr[j]);
    this.lo = jids.map((j) => model.jnt_range[2 * j]);
    this.hi = jids.map((j) => model.jnt_range[2 * j + 1]);
    this.nv = model.nv;
    // Park the target far away so collision checks only see the arm; fingers open.
    const tq = model.jnt_qposadr[model.jnt("target_free").id];
    const qpos = this.d.qpos;
    qpos[tq] = 10; qpos[tq + 1] = 10; qpos[tq + 2] = 10; qpos[tq + 3] = 1;
    for (const f of ["finger_joint1", "finger_joint2"]) qpos[model.jnt_qposadr[model.jnt(f).id]] = 0.04;
    this.jacp = new mj.DoubleBuffer(3 * this.nv);
    this.jacr = new mj.DoubleBuffer(3 * this.nv);
  }

  clip(q) {
    return q.map((x, i) => Math.min(Math.max(x, this.lo[i]), this.hi[i]));
  }

  // Tool point position and hand rotation matrix for arm joint angles q.
  fk(q) {
    const qpos = this.d.qpos;
    for (let i = 0; i < 7; i++) qpos[this.qadr[i]] = q[i];
    this.mj.mj_kinematics(this.m, this.d);
    const xm = this.d.xmat, xp = this.d.xpos, h = this.hand;
    const R = Float64Array.from(xm.subarray(9 * h, 9 * h + 9));
    const p = add([xp[3 * h], xp[3 * h + 1], xp[3 * h + 2]], matVec(R, TCP_OFFSET));
    return [p, R];
  }

  // 6x7 [linear; angular] Jacobian of `point` on the hand (call after fk).
  jacobian(point) {
    this.mj.mj_comPos(this.m, this.d);
    this.mj.mj_jac(this.m, this.d, this.jacp, this.jacr, point, this.hand);
    const P = this.jacp.GetView(), Rr = this.jacr.GetView(), nv = this.nv;
    const J = new Float64Array(42);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 7; c++) {
        J[r * 7 + c] = P[r * nv + this.dadr[c]];
        J[(3 + r) * 7 + c] = Rr[r * nv + this.dadr[c]];
      }
    return J;
  }

  // Damped least squares to (pos, R) from q0, steps capped at maxStep rad, null space pulled
  // toward `rest` (default q0). Returns [q, converged].
  ik(q0, pos, R, { rest = null, iters = 200, posTol = 0.003, rotTolDeg = 2, damping = 0.02, maxStep = 0.2, restGain = 0.1 } = {}) {
    let q = this.clip(q0);
    const r = rest ?? q.slice();
    const rotTol = (rotTolDeg * Math.PI) / 180;
    for (let it = 0; it < iters; it++) {
      const [p, Rc] = this.fk(q);
      const ep = sub(pos, p);
      const er = rotvecFromMatrix(matMul(R, transpose(Rc)));
      if (norm(ep) < posTol && norm(er) < rotTol) return [q, true];
      const J = this.jacobian(p);
      const Pinv = dampedPinv(J, 7, damping);
      const err = [...ep, ...er];
      let dq = mulVec(Pinv, 7, 6, err);
      const nul = nullProject(Pinv, J, 7, r.map((x, i) => restGain * (x - q[i])));
      dq = dq.map((x, i) => x + nul[i]);
      const mx = Math.max(...dq.map(Math.abs), 1e-12);
      const s = Math.min(1, maxStep / mx);
      q = this.clip(q.map((x, i) => x + dq[i] * s));
    }
    return [q, false];
  }

  selfCollision(q) {
    const qpos = this.d.qpos;
    for (let i = 0; i < 7; i++) qpos[this.qadr[i]] = q[i];
    this.mj.mj_fwdPosition(this.m, this.d);
    return this.d.ncon > 0;
  }
}

// --- intercept planning ------------------------------------------------------------------

export const SHOULDER = [0, 0, 0.333];
export const REACH = 0.82;
export const MIN_HEIGHT = 0.15;
export const MIN_JERK_PEAK = 1.875;
export const ELBOW_MAX = -0.75;
export const WRIST_MIN = 0.3;
export const STANDOFF = 0.1;
export const MAX_REACH_TIME = 2.5;
export const MAX_FACE_ANGLE = 70;

// Hand rotations: x along the bar (either way, nearer the hand's current x first), z square to a
// long face within MAX_FACE_ANGLE of the hand-to-bar direction, nearest first.
export function graspRotations(Rbar, reachDir, handXNow) {
  const axis = col(Rbar, 2);
  const faces = [col(Rbar, 0), scale(col(Rbar, 0), -1), col(Rbar, 1), scale(col(Rbar, 1), -1)];
  let r = sub(reachDir, scale(axis, dot(reachDir, axis)));
  if (norm(r) < 1e-6) return [];
  r = scale(r, 1 / norm(r));
  const cosMax = Math.cos((MAX_FACE_ANGLE * Math.PI) / 180);
  const first = dot(axis, handXNow) >= 0 ? axis : scale(axis, -1);
  const out = [];
  const ok = faces.filter((f) => dot(f, r) >= cosMax).sort((a, b) => dot(b, r) - dot(a, r));
  for (const z of ok) for (const x of [first, scale(first, -1)]) out.push(fromCols(x, cross(z, x), z));
  return out;
}

export function postureOk(kin, q) {
  return q[3] <= ELBOW_MAX && q[5] >= WRIST_MIN && !kin.selfCollision(q);
}

function staysReachable(kin, kf, spin, q, tc, hold, settle, steps = 3) {
  if (!postureOk(kin, q)) return false;
  const Rrel = matMul(transpose(quatToMat(spin.predict(tc))), kin.fk(q)[1]);
  let qg = q;
  const n = steps + 2;
  for (let i = 0; i < n; i++) {
    const t = tc - settle + ((hold + settle) * i) / (n - 1);
    const pos = kf.predict(t).p;
    if (norm(sub(pos, SHOULDER)) > REACH) return false;
    const Rh = matMul(quatToMat(spin.predict(t)), Rrel);
    let ok;
    [qg, ok] = kin.ik(qg, pos, Rh);
    if (!ok || !postureOk(kin, qg)) return false;
    const [qp, okp] = kin.ik(qg, sub(pos, scale(col(Rh, 2), STANDOFF)), Rh);
    if (!okp || !postureOk(kin, qp)) return false;
  }
  return true;
}

// Earliest reachable, safe capture. Returns {t, pos, vel, quat, handR, q, reachTime, posStd,
// plannedAt} or null.
export function planIntercept(kin, kf, spin, tNow, qNow, { horizon = [0.4, 4.0], dt = 0.1, speedFrac = 0.5, settle = 0.3, hold = 0.5 } = {}) {
  const [tcpNow, RNow] = kin.fk(qNow);
  const nSteps = Math.ceil((horizon[1] - horizon[0]) / dt - 1e-9);
  for (let k = 0; k < nSteps; k++) {
    const tc = tNow + horizon[0] + k * dt;
    const { p: pos, v: vel, varp } = kf.predict(tc);
    if (norm(sub(pos, SHOULDER)) > REACH || pos[2] < MIN_HEIGHT) continue;
    const quat = spin.predict(tc);
    const candidates = [];
    for (const R of graspRotations(quatToMat(quat), sub(pos, tcpNow), col(RNow, 0))) {
      const [qTry, ok] = kin.ik(qNow, pos, R, { iters: 80 });
      if (ok && postureOk(kin, qTry))
        candidates.push([Math.max(...qTry.map((x, i) => Math.abs(x - qNow[i]) / QD_MAX[i])), qTry]);
    }
    candidates.sort((a, b) => a[0] - b[0]);
    for (const [dist, q] of candidates) {
      const reachTime = (MIN_JERK_PEAK * dist) / speedFrac;
      if (reachTime > MAX_REACH_TIME || tNow + reachTime + settle > tc) break;
      if (staysReachable(kin, kf, spin, q, tc, hold, settle)) {
        return {
          t: tc, pos, vel, quat, handR: kin.fk(q)[1], q, reachTime,
          posStd: Math.sqrt(varp[0] + varp[1] + varp[2]), plannedAt: tNow,
        };
      }
    }
  }
  return null;
}
