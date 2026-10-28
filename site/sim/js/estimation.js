// Port of sim/sensor.py and sim/estimator.py: the simulated 30 Hz pose sensor, the Kalman filter
// on [position, velocity], and the sliding-window spin estimator.

import { Rng, axisAngleQuat, quatConj, quatMul, quatNormalize, quatToRotvec, rotateQuat, scale } from "./math.js";

export class PoseSensor {
  // Gaussian position noise (pos_std m per axis), orientation noise as a small random rotation
  // (rot_std_deg per axis), each reading dropped with probability `dropout`.
  constructor({ rateHz = 30, posStd = 0.005, rotStdDeg = 2, dropout = 0.05, seed = 1 } = {}) {
    this.period = 1 / rateHz;
    this.posStd = posStd;
    this.rotStd = (rotStdDeg * Math.PI) / 180;
    this.dropout = dropout;
    this.rng = new Rng(seed);
    this.nextT = 0;
  }

  poll(t, pos, quat) {
    if (t + 1e-9 < this.nextT) return null;
    this.nextT += this.period;
    if (this.rng.random() < this.dropout) return null;
    const p = pos.map((x) => x + this.rng.normal(0, this.posStd));
    const rv = [0, 1, 2].map(() => this.rng.normal(0, this.rotStd));
    const a = Math.hypot(...rv);
    const dq = a > 0 ? axisAngleQuat(scale(rv, 1 / a), a) : [1, 0, 0, 0];
    return { t, pos: p, quat: quatNormalize(quatMul(quat, dq)) };
  }
}

export class TranslationKF {
  constructor({ posStd = 0.005, accelStd = 0.01, initVelStd = 0.3 } = {}) {
    this.r = posStd ** 2;
    this.q = accelStd ** 2;
    this.initVelVar = initVelStd ** 2;
    this.x = null; // [p(3), v(3)]
    this.P = null; // 6x6 row-major
    this.t = null;
  }

  get ready() {
    return this.x !== null;
  }

  // The state is block-diagonal per axis (x, y, z each have [p, v]): filter axes independently.
  // P is stored as three 2x2 blocks [Ppp, Ppv, Pvv] per axis.
  predictTo(t) {
    const dt = t - this.t;
    if (dt <= 0) return;
    const q = this.q;
    for (let a = 0; a < 3; a++) {
      this.x[a] += this.x[3 + a] * dt;
      const [pp, pv, vv] = this.P[a];
      this.P[a] = [
        pp + 2 * dt * pv + dt * dt * vv + (q * dt ** 4) / 4,
        pv + dt * vv + (q * dt ** 3) / 2,
        vv + q * dt * dt,
      ];
    }
    this.t = t;
  }

  update(t, pos) {
    if (this.x === null) {
      this.x = [...pos, 0, 0, 0];
      this.P = [0, 1, 2].map(() => [this.r, 0, this.initVelVar]);
      this.t = t;
      return;
    }
    this.predictTo(t);
    for (let a = 0; a < 3; a++) {
      const [pp, pv, vv] = this.P[a];
      const s = pp + this.r;
      const kp = pp / s;
      const kv = pv / s;
      const innov = pos[a] - this.x[a];
      this.x[a] += kp * innov;
      this.x[3 + a] += kv * innov;
      this.P[a] = [(1 - kp) * pp, (1 - kp) * pv, vv - kv * pv];
    }
  }

  // Position, velocity and per-axis position variance at future time t (state unchanged).
  predict(t) {
    const dt = t - this.t;
    const p = [0, 1, 2].map((a) => this.x[a] + this.x[3 + a] * dt);
    const v = this.x.slice(3);
    const varp = this.P.map(([pp, pv, vv]) => pp + 2 * dt * pv + dt * dt * vv);
    return { p, v, varp };
  }

  get pos() {
    return this.x.slice(0, 3);
  }
  get vel() {
    return this.x.slice(3);
  }
  get velStd() {
    return this.P.map((b) => Math.sqrt(b[2]));
  }
}

export class SpinEstimator {
  // World-frame angular velocity from a least-squares line fit of rotation vectors over a short
  // window (torque-free spin precesses, so the window stays short).
  constructor({ windowS = 0.6, minReadings = 6 } = {}) {
    this.window = windowS;
    this.minReadings = minReadings;
    this.buf = [];
    this.omega = null;
    this.quatSmooth = null;
    this.t = null;
  }

  get ready() {
    return this.omega !== null;
  }

  update(t, quat) {
    this.buf.push([t, quat]);
    while (this.buf.length && t - this.buf[0][0] > this.window) this.buf.shift();
    if (this.buf.length < this.minReadings) return;
    const [t0, q0] = this.buf[0];
    const q0inv = quatConj(q0);
    // Fit rv(t) = a + b t per component.
    let n = 0, st = 0, stt = 0;
    const sr = [0, 0, 0], str = [0, 0, 0];
    for (const [ti, qi] of this.buf) {
      const rv = quatToRotvec(quatMul(qi, q0inv));
      const tt = ti - t0;
      n++;
      st += tt;
      stt += tt * tt;
      for (let k = 0; k < 3; k++) {
        sr[k] += rv[k];
        str[k] += tt * rv[k];
      }
    }
    const den = n * stt - st * st;
    const slope = sr.map((_, k) => (n * str[k] - st * sr[k]) / den);
    const icpt = sr.map((_, k) => (sr[k] - slope[k] * st) / n);
    this.omega = slope;
    this.t = t;
    this.quatSmooth = rotateQuat(q0, icpt.map((c, k) => c + slope[k] * (t - t0)));
  }

  predict(t) {
    return rotateQuat(this.quatSmooth, scale(this.omega, t - this.t));
  }
}
