// Check the browser port (site/sim/js) against the Python implementation.
//
//   node scripts/validate_web.mjs reference      replay out/reference.json (noiseless Python runs,
//                                                from scripts/export_reference.py) and compare
//   node scripts/validate_web.mjs mc [N]         N randomized throws with sensor noise, summary
//                                                to compare with the Python Monte Carlo

import fs from "node:fs";
import loadMujoco from "../site/sim/vendor/mujoco.js";
import { runThrow, sampleThrow } from "../site/sim/js/evaluate.js";

const mj = await loadMujoco();
const vfs = new mj.MjVFS();
vfs.addBuffer("scene.mjb", new Uint8Array(fs.readFileSync(new URL("../site/sim/scene.mjb", import.meta.url))));
const model = mj.MjModel.from_binary_path("scene.mjb", vfs);

const mode = process.argv[2] ?? "reference";
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const fmt = (x, k = 3) => (x === null || x === undefined ? "-" : Number(x).toFixed(k));

if (mode === "reference") {
  const ref = JSON.parse(fs.readFileSync("out/reference.json", "utf8"));
  const noiseless = { posStd: 0, rotStdDeg: 0, dropout: 0 };
  let same = 0;
  const dPlan = [], dCatch = [], dGrasp = [], dRel = [];
  const t0 = performance.now();
  for (const row of ref) {
    const r = runThrow(mj, model, row.target, { sensor: noiseless });
    const p = row.result;
    const match = r.outcome === p.outcome;
    same += match;
    if (p.t_plan !== null && r.tPlan !== null) dPlan.push(Math.abs(r.tPlan - p.t_plan));
    if (p.t_catch !== null && r.tCatch !== null) dCatch.push(Math.abs(r.tCatch - p.t_catch));
    if (p.grasp_dist !== null && r.graspDist !== null) dGrasp.push(Math.abs(r.graspDist - p.grasp_dist) * 100);
    if (p.grasp_rel_speed !== null && r.graspRelSpeed !== null) dRel.push(Math.abs(r.graspRelSpeed - p.grasp_rel_speed) * 100);
    console.log(
      `throw ${String(row.throw).padStart(2)}  py ${p.outcome.padEnd(11)} js ${r.outcome.padEnd(11)}${match ? "" : "  <-- differs"}` +
      `  plan ${fmt(p.t_plan, 2)}/${fmt(r.tPlan, 2)}  catch ${fmt(p.t_catch, 2)}/${fmt(r.tCatch, 2)}` +
      `  rel ${fmt(p.grasp_rel_speed && p.grasp_rel_speed * 100, 2)}/${fmt(r.graspRelSpeed && r.graspRelSpeed * 100, 2)} cm/s` +
      `  plans ${p.n_plans}/${r.nPlans} aborts ${p.aborts}/${r.aborts}`,
    );
  }
  console.log(`\n${same}/${ref.length} same outcome; ${((performance.now() - t0) / ref.length).toFixed(0)} ms per throw`);
  console.log(`|plan time diff| median ${fmt(median(dPlan), 4)} s, max ${fmt(Math.max(...dPlan), 4)}`);
  console.log(`|catch time diff| median ${fmt(median(dCatch), 4)} s, max ${fmt(Math.max(...dCatch), 4)}`);
  console.log(`|grasp distance diff| median ${fmt(median(dGrasp), 3)} cm; |contact speed diff| median ${fmt(median(dRel), 3)} cm/s`);
} else {
  const n = Number(process.argv[3] ?? 100);
  const rows = [];
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    const target = sampleThrow(i + 1);
    const r = runThrow(mj, model, target, { sensorSeed: 1000 + i });
    rows.push({ ...target.params, ...r });
    if ((i + 1) % 20 === 0) console.log(`${i + 1} throws...`);
  }
  const count = (o) => rows.filter((r) => r.outcome === o).length;
  const caught = rows.filter((r) => r.outcome === "caught");
  console.log(`\n${n} throws: ${["caught", "aborted", "no_plan", "not_at_rest", "timeout"].map((o) => `${count(o)} ${o}`).join(", ")}`);
  console.log(`caught ${(100 * caught.length / n).toFixed(1)}%; ${((performance.now() - t0) / n).toFixed(0)} ms per throw`);
  for (const [lo, hi] of [[2, 5], [5, 8], [8, 11], [11, 15.01]]) {
    const s = rows.filter((r) => r.speed * 100 >= lo && r.speed * 100 < hi);
    console.log(`  ${lo}-${Math.round(hi)} cm/s: ${s.filter((r) => r.outcome === "caught").length}/${s.length}`);
  }
  console.log(`contact speed median ${fmt(median(caught.map((r) => r.graspRelSpeed * 100)), 2)} cm/s; kick median ${fmt(median(caught.map((r) => r.kick * 100)), 2)} cm/s; ` +
    `peak force median ${fmt(median(caught.map((r) => r.peakForce)), 2)} N; braking median ${fmt(median(caught.map((r) => r.brakeT)), 2)} s`);
}
