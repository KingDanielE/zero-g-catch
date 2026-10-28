// The live page: loads MuJoCo (WebAssembly) and the compiled scene, runs a throw step by step with
// the ported controller, and draws it with three.js straight from MuJoCo's geometry.

import * as THREE from "three";
import { OrbitControls } from "../vendor/three/OrbitControls.js";
import { toCreasedNormals } from "../vendor/three/BufferGeometryUtils.js";
import loadMujoco from "../vendor/mujoco.js";
import { Phase } from "./catcher.js";
import { ThrowRun, makeThrow, sampleThrow } from "./evaluate.js";
import { Rng, norm, quatNormalize, sub } from "./math.js";

const $ = (id) => document.getElementById(id);
const DT = 0.002;
const PHASE_TEXT = {
  [Phase.WAIT]: "Watching: filter converging",
  [Phase.APPROACH]: "Approach: lining up behind the bar",
  [Phase.RIDE]: "Riding along, 10 cm back",
  [Phase.CLOSE_IN]: "Closing in, matching drift and spin",
  [Phase.GRASP]: "Fingers closing gently",
  [Phase.BRAKE]: "Braking drift and spin",
  [Phase.STOPPED]: "At rest, held",
  [Phase.ABORT]: "Abort: estimate too far off, backing off",
  [Phase.FAILED]: "No reachable catch",
};
const TIMELINE_TEXT = {
  approach: "Plan, approach", ride: "Ride along", close_in: "Close in", grasp: "Grasp", brake: "Brake",
  stopped: "At rest", abort: "Abort", failed: "No catch", "gave up": "Gave up",
  "backing off to re-plan": "Re-plan", "no reachable catch yet, re-planning": "Waiting for reach",
};
const OUTCOME = {
  caught: ["Caught", "Brought to rest in the gripper"],
  no_plan: ["Missed", "No reachable catch: the bar never came within reach in time"],
  aborted: ["Aborted", "The estimate never got close enough to close the gripper safely"],
  not_at_rest: ["Held", "Grasped, but still turning in the grip"],
  timeout: ["Out of time", "Still re-planning at the 12 s limit"],
};

let mj, model, run = null, rate = 1, acc = 0, lastFrame = null, throwCount = 0, caughtCount = 0;
const sessionRng = new Rng((Date.now() ^ 0x5bd1e995) >>> 0);

// --- loading -----------------------------------------------------------------------------

// expected: uncompressed size, for progress when the server sends compressed data without a length.
async function fetchWithProgress(url, expected, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = (res.headers.get("content-encoding") ? 0 : Number(res.headers.get("content-length"))) || expected;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress(Math.min(1, got / total));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function init() {
  const bar = $("load-bar");
  try {
    let wasmFrac = 0, mjbFrac = 0;
    const show = () => (bar.style.width = `${Math.round(100 * (0.8 * wasmFrac + 0.2 * mjbFrac))}%`);
    const wasmUrl = new URL("../vendor/mujoco.wasm", import.meta.url).href;
    const [wasmBinary, mjb] = await Promise.all([
      fetchWithProgress(wasmUrl, 10.9e6, (f) => { wasmFrac = f; show(); }),
      fetchWithProgress(new URL("../scene.mjb", import.meta.url).href, 3.2e6, (f) => { mjbFrac = f; show(); }),
    ]);
    mj = await loadMujoco({ wasmBinary });
    const vfs = new mj.MjVFS();
    vfs.addBuffer("scene.mjb", mjb);
    model = mj.MjModel.from_binary_path("scene.mjb", vfs);
    vfs.delete();
  } catch (err) {
    $("loading").classList.add("error");
    $("loading").querySelector(".loading-note").textContent =
      "Couldn't start the simulation in this browser (WebAssembly failed to load). Try a recent Chrome, Edge, Firefox or Safari.";
    console.error(err);
    return;
  }
  buildScene();
  $("loading").hidden = true;
  $("go").disabled = false;
  $("random").disabled = false;
  startThrow(throwFromControls());
  requestAnimationFrame(frame);
}

// --- three.js scene from the MuJoCo model ---------------------------------------------------

const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.02, 20);
const controls = new OrbitControls(camera, canvas);
// MuJoCo is z-up; three is y-up. Everything lives under `world`, rotated so MuJoCo z points up.
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
const geomMeshes = [];
let planBall, pathPoints;

function setView() {
  // Looking at the workspace from the front and slightly above (MuJoCo coordinates).
  const target = new THREE.Vector3(0.42, 0.0, 0.45).applyMatrix4(world.matrixWorld);
  // Back off on narrow (portrait) views so the whole workspace stays in frame.
  const az = THREE.MathUtils.degToRad(200), el = THREE.MathUtils.degToRad(18);
  const dist = 1.75 * Math.min(2.1, Math.max(1, 1.25 / camera.aspect));
  const offset = new THREE.Vector3(-Math.cos(el) * Math.cos(az), -Math.cos(el) * Math.sin(az), Math.sin(el)).multiplyScalar(dist);
  camera.position.copy(target).add(new THREE.Vector3(offset.x, offset.z, -offset.y));
  controls.target.copy(target);
  controls.update();
}

function buildScene() {
  scene.add(new THREE.HemisphereLight(0xdfe3ff, 0x30281c, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(1.5, 3, 2);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x6f86ff, 0.8);
  rim.position.set(-2, 1.2, -1.5);
  scene.add(rim);

  // Deck: a faint grid (the scene's plane is visual only; nothing rests on it).
  const grid = new THREE.GridHelper(4, 32, 0x3a3f5e, 0x2a2d40);
  grid.position.y = -0.01;
  scene.add(grid);

  const types = mj.mjtGeom;
  const T = (k) => types[k].value;
  for (let g = 0; g < model.ngeom; g++) {
    const group = model.geom_group[g];
    const type = model.geom_type[g];
    if (group === 3 || type === T("mjGEOM_PLANE")) continue; // collision shapes, deck plane
    const s = model.geom_size.subarray(3 * g, 3 * g + 3);
    let geo;
    if (type === T("mjGEOM_MESH")) geo = meshGeometry(model.geom_dataid[g]);
    else if (type === T("mjGEOM_BOX")) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    else if (type === T("mjGEOM_SPHERE")) geo = new THREE.SphereGeometry(s[0], 24, 16);
    else if (type === T("mjGEOM_CAPSULE")) geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
    else if (type === T("mjGEOM_CYLINDER")) geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
    else continue;
    const matId = model.geom_matid[g];
    const rgba = matId >= 0 ? model.mat_rgba.subarray(4 * matId, 4 * matId + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4);
    if (rgba[3] === 0) continue;
    const isTarget = model.geom_bodyid[g] === model.body("target").id;
    const material = new THREE.MeshStandardMaterial({
      color: isTarget ? 0xf79915 : new THREE.Color(rgba[0], rgba[1], rgba[2]),
      roughness: isTarget ? 0.55 : 0.42,
      metalness: 0.05,
      emissive: isTarget ? 0x3a1f00 : 0x000000,
      flatShading: false,
    });
    const mesh = new THREE.Mesh(geo, material);
    mesh.matrixAutoUpdate = false;
    world.add(mesh);
    geomMeshes.push([g, mesh]);
  }

  planBall = new THREE.Mesh(
    new THREE.SphereGeometry(0.018, 24, 16),
    new THREE.MeshBasicMaterial({ color: 0x4f6af0, transparent: true, opacity: 0.55, depthWrite: false }),
  );
  planBall.visible = false;
  world.add(planBall);
  pathPoints = new THREE.Points(
    new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(12 * 3), 3)),
    new THREE.PointsMaterial({ color: 0xf79915, size: 0.009, transparent: true, opacity: 0.8 }),
  );
  pathPoints.visible = false;
  world.add(pathPoints);

  world.updateMatrixWorld(true);
  resize();
  setView();
}

function meshGeometry(id) {
  const va = model.mesh_vertadr[id], vn = model.mesh_vertnum[id];
  const fa = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
  const verts = Float32Array.from(model.mesh_vert.subarray(3 * va, 3 * (va + vn)));
  const faces = Uint32Array.from(model.mesh_face.subarray(3 * fa, 3 * (fa + fn)));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  // Smooth across gentle curves, keep the casting's hard edges sharp.
  return toCreasedNormals(geo, THREE.MathUtils.degToRad(35));
}

function syncScene() {
  const d = run.env.d;
  const xp = d.geom_xpos, xm = d.geom_xmat;
  for (const [g, mesh] of geomMeshes) {
    const p = 3 * g, r = 9 * g;
    mesh.matrix.set(
      xm[r], xm[r + 1], xm[r + 2], xp[p],
      xm[r + 3], xm[r + 4], xm[r + 5], xp[p + 1],
      xm[r + 6], xm[r + 7], xm[r + 8], xp[p + 2],
      0, 0, 0, 1,
    );
  }
  const c = run.c;
  const showPlan = c.plan && [Phase.APPROACH, Phase.RIDE, Phase.CLOSE_IN].includes(c.phase);
  planBall.visible = !!showPlan;
  if (showPlan) planBall.position.set(...c.plan.pos);
  const showPath = c.kf.ready && norm(c.kf.velStd) < 0.03 && [Phase.WAIT, Phase.APPROACH, Phase.RIDE, Phase.ABORT].includes(c.phase);
  pathPoints.visible = showPath;
  if (showPath) {
    const arr = pathPoints.geometry.attributes.position.array;
    for (let i = 0; i < 12; i++) {
      const p = c.kf.predict(d.time + 0.25 * (i + 1)).p;
      arr.set(p, 3 * i);
    }
    pathPoints.geometry.attributes.position.needsUpdate = true;
  }
}

function resize() {
  const vp = $("viewport");
  const w = vp.clientWidth, h = vp.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// --- throws ------------------------------------------------------------------------------

const ctl = {
  speed: $("speed"), spin: $("spin"), yaw: $("yaw"), pitch: $("pitch"),
};
const radio = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
function showControls() {
  $("o-speed").textContent = `${Number(ctl.speed.value).toFixed(1)} cm/s`;
  $("o-spin").textContent = `${ctl.spin.value}°/s`;
  $("o-yaw").textContent = `${ctl.yaw.value > 0 ? "+" : ""}${ctl.yaw.value}°`;
  $("o-pitch").textContent = `${ctl.pitch.value > 0 ? "+" : ""}${ctl.pitch.value}°`;
}
for (const el of Object.values(ctl)) el.addEventListener("input", showControls);
showControls();

const NOISE = { 0: { posStd: 0, rotStdDeg: 0, dropout: 0 }, 1: {}, 2: { posStd: 0.01, rotStdDeg: 4, dropout: 0.1 } };

function randomAxis() {
  return [sessionRng.normal(), sessionRng.normal(), sessionRng.normal()];
}

function throwFromControls() {
  return makeThrow({
    speed: Number(ctl.speed.value) / 100,
    spinDeg: Number(ctl.spin.value),
    yawDeg: Number(ctl.yaw.value),
    pitchDeg: Number(ctl.pitch.value),
    side: Number(radio("side")),
    spinAxis: randomAxis(),
    quat: quatNormalize([sessionRng.normal(), sessionRng.normal(), sessionRng.normal(), sessionRng.normal()]),
  });
}

function startThrow(target) {
  if (run) run.dispose();
  run = new ThrowRun(mj, model, target, { sensorSeed: (sessionRng.random() * 2 ** 31) >>> 0, sensor: NOISE[radio("noise")] });
  acc = 0;
  shownEvents = 0;
  $("timeline").replaceChildren();
  $("outcome").hidden = true;
  recorded = false;
}

$("setup").addEventListener("submit", (e) => {
  e.preventDefault();
  startThrow(throwFromControls());
});
$("random").addEventListener("click", () => {
  const target = sampleThrow((sessionRng.random() * 2 ** 31) >>> 0);
  // Show the sampled throw on the controls.
  const p = target.params, v = target.vel, side = p.side;
  ctl.speed.value = (p.speed * 100).toFixed(1);
  ctl.spin.value = Math.round(p.spinDeg);
  ctl.yaw.value = Math.round((Math.atan2(v[0], side * v[1]) * 180) / Math.PI);
  ctl.pitch.value = Math.round((Math.asin(v[2] / norm(v)) * 180) / Math.PI);
  document.querySelector(`input[name="side"][value="${side}"]`).checked = true;
  showControls();
  startThrow(target);
});
for (const r of document.querySelectorAll('input[name="rate"]'))
  r.addEventListener("change", () => {
    rate = Number(radio("rate"));
    $("hud-rate").textContent = rate === 1 ? "1× real time" : rate < 1 ? "¼ speed" : "2× speed";
    $("hud-rate").parentElement.classList.toggle("not-real-time", rate !== 1);
  });

// --- HUD ---------------------------------------------------------------------------------

let shownEvents = 0, recorded = false;

function updateHud() {
  const c = run.c, env = run.env, d = env.d;
  $("hud-phase").textContent = run.done ? "Throw finished" : PHASE_TEXT[c.phase];
  $("r-time").textContent = `${d.time.toFixed(1)} s`;
  const [pBar] = env.targetPose();
  const [p] = c.kin.fk(c.qMeas());
  $("r-dist").textContent = `${(norm(sub(pBar, p)) * 100).toFixed(1)} cm`;
  const [v, w] = env.targetVelocity();
  $("r-speed").textContent = `${(norm(v) * 100).toFixed(1)} cm/s`;
  $("r-spin").textContent = `${((norm(w) * 180) / Math.PI).toFixed(1)}°/s`;
  $("r-force").textContent = `${run.force.toFixed(2)} N`;

  const ev = c.events;
  if (ev.length > shownEvents) {
    const tl = $("timeline");
    for (; shownEvents < ev.length; shownEvents++) {
      const [t, text] = ev[shownEvents];
      if (!TIMELINE_TEXT[text]) continue;
      tl.querySelector(".now")?.classList.remove("now");
      const li = document.createElement("li");
      li.className = "now";
      li.innerHTML = `<b>${t.toFixed(2)} s</b>`;
      li.append(TIMELINE_TEXT[text]);
      tl.append(li);
    }
  }

  if (run.done && !recorded) {
    recorded = true;
    const r = run.result;
    const [k, v2] = OUTCOME[r.outcome];
    const box = $("outcome");
    box.classList.toggle("miss", r.outcome !== "caught");
    $("outcome-k").textContent = k;
    $("outcome-v").textContent = v2;
    $("outcome-d").textContent = r.outcome === "caught"
      ? `Contact at ${(r.graspRelSpeed * 100).toFixed(1)} cm/s, ${r.peakForce.toFixed(1)} N grip, stopped in ${r.brakeT.toFixed(2)} s`
      : r.aborts ? `${r.aborts} abort${r.aborts > 1 ? "s" : ""}, ${r.nPlans} plans tried` : `${r.nPlans} plan${r.nPlans > 1 ? "s" : ""} tried`;
    box.hidden = false;
    throwCount++;
    if (r.outcome === "caught") caughtCount++;
    $("tally").textContent = `${caughtCount} of ${throwCount} caught`;
    const p = run.target.params;
    const li = document.createElement("li");
    li.innerHTML = `<span>${(p.speed * 100).toFixed(1)} cm/s · ${Math.round(p.spinDeg)}°/s · ${Math.round(p.coneDeg)}° off</span>`;
    const s = document.createElement("span");
    s.textContent = k;
    if (r.outcome === "caught") s.className = "ok";
    li.append(s);
    $("history").prepend(li);
  }
}

// --- loop --------------------------------------------------------------------------------

function frame(now) {
  requestAnimationFrame(frame);
  if (!run) return;
  if (lastFrame === null) lastFrame = now;
  const elapsed = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  if (!document.hidden && !run.done) {
    acc += elapsed * rate;
    const t0 = performance.now();
    while (acc >= DT && !run.done) {
      run.step();
      acc -= DT;
      if (performance.now() - t0 > 40) { acc = 0; break; } // fall behind rather than freeze
    }
  }
  syncScene();
  updateHud();
  controls.update();
  renderer.render(scene, camera);
}

// Read-only handle for debugging from the console.
window.zeroG = { get run() { return run; }, Phase };

init();
