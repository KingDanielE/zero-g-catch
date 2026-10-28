// Port of sim/evaluate.py and the Monte Carlo sampler in scripts/monte_carlo.py: build throws,
// run one end to end, and score it against ground truth.

import { Catcher, Phase, ZeroGEnv } from "./catcher.js";
import { Rng, add, matTVec, matMul, norm, quatNormalize, quatToMat, rotvecFromMatrix, sub, transpose } from "./math.js";

export const SPEED = [0.02, 0.15];
export const SPIN_DEG = [0, 30];
export const CONE_DEG = 30;
export const START = [0.6, 0.5, 0.5];
export const START_JITTER = 0.05;
export const REST_SPEED = 0.005;
export const REST_SPIN = 2.0;
export const SETTLE_S = 0.5;
export const T_MAX = 12;

// A throw from explicit parameters. side +1 starts at -y and drifts toward +y. Direction: yaw
// (horizontal) and pitch (vertical) away from straight across, in degrees.
export function makeThrow({ speed, spinDeg, yawDeg = 0, pitchDeg = 0, side = 1, jitter = [0, 0, 0], spinAxis = [0, 0, 1], quat = [1, 0, 0, 0] }) {
  const y = (yawDeg * Math.PI) / 180, p = (pitchDeg * Math.PI) / 180;
  const dir = [Math.sin(y) * Math.cos(p), side * Math.cos(y) * Math.cos(p), Math.sin(p)];
  const a = norm(spinAxis);
  return {
    pos: add([START[0], -side * START[1], START[2]], jitter),
    quat: quatNormalize(quat),
    vel: dir.map((x) => x * speed),
    omega: spinAxis.map((x) => (x / a) * ((spinDeg * Math.PI) / 180)),
    params: { side, speed, spinDeg, coneDeg: (Math.acos(Math.cos(y) * Math.cos(p)) * 180) / Math.PI, yawDeg, pitchDeg },
  };
}

// Same distribution as scripts/monte_carlo.py (different random numbers).
export function sampleThrow(seed) {
  const rng = new Rng(Math.imul(seed, 2654435761) >>> 0); // 32-bit hash of the seed
  const side = rng.random() < 0.5 ? 1 : -1;
  const speed = rng.uniform(...SPEED);
  const th = Math.acos(rng.uniform(Math.cos((CONE_DEG * Math.PI) / 180), 1));
  const az = rng.uniform(0, 2 * Math.PI);
  const dir = [Math.sin(th) * Math.cos(az), side * Math.cos(th), Math.sin(th) * Math.sin(az)];
  const spinDeg = rng.uniform(...SPIN_DEG);
  const axis = [rng.normal(), rng.normal(), rng.normal()];
  const jitter = [0, 1, 2].map(() => rng.uniform(-START_JITTER, START_JITTER));
  const quat = quatNormalize([rng.normal(), rng.normal(), rng.normal(), rng.normal()]);
  const a = norm(axis);
  return {
    pos: add([START[0], -side * START[1], START[2]], jitter),
    quat,
    vel: dir.map((x) => x * speed),
    omega: axis.map((x) => (x / a) * ((spinDeg * Math.PI) / 180)),
    params: { side, speed, spinDeg, coneDeg: (th * 180) / Math.PI },
  };
}

// Bodies touching the bar and the total normal force on it (N).
export function barContacts(env, forceBuf) {
  const { mj, m, d } = env;
  const bodies = new Set();
  let force = 0;
  const n = d.ncon;
  if (!n) return [bodies, 0];
  const contacts = d.contact;
  try {
    for (let i = 0; i < n; i++) {
      const c = contacts.get(i);
      const g1 = c.geom1, g2 = c.geom2;
      c.delete();
      if (g1 !== env.targetGeom && g2 !== env.targetGeom) continue;
      const other = g1 === env.targetGeom ? g2 : g1;
      bodies.add(m.body(m.geom_bodyid[other]).name);
      mj.mj_contactForce(m, d, i, forceBuf);
      force += Math.abs(forceBuf.GetView()[0]);
    }
  } finally {
    contacts.delete();
  }
  return [bodies, force];
}

function barInHand(env, c) {
  const [pBar, qBar] = env.targetPose();
  const [p, R] = c.kin.fk(c.qMeas());
  return [matTVec(R, sub(pBar, p)), matMul(transpose(R), quatToMat(qBar))];
}

// One throw, advanced a physics step at a time and scored like sim/evaluate.py. The page calls
// step() a few times per animation frame; runThrow() loops it to the end.
export class ThrowRun {
  constructor(mj, model, target, { sensorSeed = 1, sensor = {} } = {}) {
    this.mj = mj;
    this.env = new ZeroGEnv(mj, model);
    this.env.reset(target);
    this.target = target;
    this.c = new Catcher(this.env, { sensorSeed, sensor });
    this.forceBuf = new mj.DoubleBuffer(6);
    this.vGrasp = null; this.kick = null; this.atGrasp = null; this.earlyTouch = false;
    this.peakForce = 0; this.poseBrake = null; this.slip = null; this.tEnd = null;
    this.bodies = new Set(); this.force = 0;
    this.done = false;
    this.result = null;
  }

  // Advance one physics step (2 ms). Returns true while the throw is still running.
  step() {
    if (this.done) return false;
    const { env, c } = this;
    const d = env.d;
    c.step();
    env.step();
    const [v] = env.targetVelocity();
    const [bodies, force] = barContacts(env, this.forceBuf);
    this.bodies = bodies; this.force = force;
    if ([Phase.APPROACH, Phase.RIDE, Phase.CLOSE_IN].includes(c.phase)) this.earlyTouch ||= bodies.size > 0;
    if (c.phase === Phase.GRASP) {
      if (!this.vGrasp) {
        this.vGrasp = v;
        this.atGrasp = c.trueErrors();
      }
      this.kick = Math.max(this.kick ?? 0, norm(sub(v, this.vGrasp)));
    }
    if ([Phase.GRASP, Phase.BRAKE, Phase.STOPPED].includes(c.phase)) this.peakForce = Math.max(this.peakForce, force);
    if (c.phase === Phase.BRAKE && !this.poseBrake) this.poseBrake = barInHand(env, c);
    if (c.phase === Phase.STOPPED && this.tEnd === null) {
      this.tEnd = d.time + SETTLE_S;
      if (this.poseBrake) {
        const [p1, R1] = barInHand(env, c);
        const dR = rotvecFromMatrix(matMul(R1, transpose(this.poseBrake[1])));
        this.slip = [norm(sub(p1, this.poseBrake[0])), (norm(dR) * 180) / Math.PI];
      }
    }
    if (c.phase === Phase.FAILED || (c.gaveUp && c.tAbort !== null && d.time > c.tAbort + 0.3) ||
        (this.tEnd !== null && d.time >= this.tEnd) || d.time >= T_MAX) this.finish();
    return !this.done;
  }

  finish() {
    const { env, c } = this;
    const [v, wBody] = env.targetVelocity();
    const [bodies] = barContacts(env, this.forceBuf);
    const finalSpeed = norm(v), finalSpin = (norm(wBody) * 180) / Math.PI;
    let outcome;
    if (c.phase === Phase.FAILED) outcome = "no_plan";
    else if (c.phase === Phase.ABORT) outcome = "aborted";
    else if (c.phase === Phase.STOPPED && bodies.has("left_finger") && bodies.has("right_finger") && finalSpeed < REST_SPEED && finalSpin < REST_SPIN) outcome = "caught";
    else if ([Phase.GRASP, Phase.BRAKE, Phase.STOPPED].includes(c.phase)) outcome = "not_at_rest";
    else outcome = "timeout";
    const { atGrasp, slip } = this;
    this.result = {
      outcome, phase: c.phase, earlyTouch: this.earlyTouch, nPlans: c.nPlans, aborts: c.aborts,
      tPlan: c.plan ? c.tStart : null, tCatch: c.plan ? c.plan.t : null,
      graspDist: atGrasp ? atGrasp[0] : null, graspRelSpeed: atGrasp ? atGrasp[1] : null,
      kick: this.kick, slipMm: slip ? slip[0] * 1000 : null, slipDeg: slip ? slip[1] : null,
      peakForce: this.vGrasp ? this.peakForce : null, brakeT: c.brakeT ?? null,
      finalSpeed, finalSpin, simTime: env.d.time, events: c.events,
    };
    this.done = true;
  }

  // Free the WebAssembly-side objects (not garbage-collected).
  dispose() {
    this.forceBuf.delete();
    this.c.kin.jacp.delete();
    this.c.kin.jacr.delete();
    this.c.kin.d.delete();
    this.env.d.delete();
  }
}

export function runThrow(mj, model, target, opts = {}) {
  const run = new ThrowRun(mj, model, target, opts);
  try {
    while (run.step());
    return run.result;
  } finally {
    run.dispose();
  }
}
