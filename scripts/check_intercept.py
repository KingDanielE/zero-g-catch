"""Hour 4 check: plan an intercept once the velocity estimate is good, then let the sim run on to
the capture time and compare the prediction with where the bar really is.

The arm doesn't move yet (hour 5), so contacts are switched off here: otherwise the bar would hit
the parked arm and the prediction couldn't be checked. Saves out/intercept.png."""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sim.env import READY_Q, ZeroGEnv
from sim.estimator import SpinEstimator, TranslationKF
from sim.intercept import REACH, SHOULDER, plan_intercept, quat_to_mat
from sim.kinematics import ArmKinematics
from sim.sensor import PoseSensor

VEL_READY = 0.01  # plan once the filter's own velocity 1-sigma (norm) is under 1 cm/s
N = 30


def run(seed: int):
    env = ZeroGEnv(seed=seed)
    env.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    env.reset()
    m, d = env.model, env.data
    kin = ArmKinematics(m)
    sensor, kf, spin = PoseSensor(seed=seed + 1000), TranslationKF(), SpinEstimator()
    q_now = np.array(READY_Q)
    plan, compute, path = None, None, []
    while d.time < 6.0:
        p, q = env.target_pose()
        path.append(p)
        z = sensor.poll(d.time, p, q)
        if z is not None:
            kf.update(z.t, z.pos)
            spin.update(z.t, z.quat)
            if plan is None and spin.ready and np.linalg.norm(kf.vel_std) < VEL_READY:
                t0 = time.perf_counter()
                plan = plan_intercept(kin, kf, spin, d.time, q_now, speed_frac=0.6, settle=0.6)
                compute = time.perf_counter() - t0
                if plan is None:
                    return dict(seed=seed, plan=None, planned_at=d.time, compute=compute,
                                path=np.array(path))
        if plan is not None and d.time >= plan.t:
            true_axis = quat_to_mat(q)[:, 2]
            pred_axis = quat_to_mat(plan.quat)[:, 2]
            axis_err = np.rad2deg(np.arccos(min(1.0, abs(np.dot(true_axis, pred_axis)))))
            return dict(seed=seed, plan=plan, compute=compute, true_pos=p, path=np.array(path),
                        pos_err=np.linalg.norm(p - plan.pos), axis_err=axis_err)
        env.step()
    return dict(seed=seed, plan=None, planned_at=None, compute=compute, path=np.array(path))


results = [run(s) for s in range(N)]
ok = [r for r in results if r["plan"] is not None]
print(f"{len(ok)}/{N} throws got a feasible intercept")
for r in results:
    if r["plan"] is None:
        print(f"  seed {r['seed']}: no intercept (planned at {r.get('planned_at')})")
pos_err = np.array([r["pos_err"] for r in ok]) * 100
axis_err = np.array([r["axis_err"] for r in ok])
lead = np.array([r["plan"].t - r["plan"].planned_at for r in ok])
planned_at = np.array([r["plan"].planned_at for r in ok])
reach = np.array([r["plan"].reach_time for r in ok])
compute = np.array([r["compute"] for r in ok]) * 1000
sigma = np.array([r["plan"].pos_std for r in ok]) * 100
print(f"planned at median {np.median(planned_at):.2f} s; capture {np.median(lead):.2f} s later "
      f"(arm needs {np.median(reach):.2f} s)")
print(f"position error at capture: median {np.median(pos_err):.2f} cm, "
      f"95th pct {np.percentile(pos_err, 95):.2f} cm (filter's own 1-sigma median {np.median(sigma):.2f})")
print(f"bar-axis error at capture: median {np.median(axis_err):.1f} deg, "
      f"95th pct {np.percentile(axis_err, 95):.1f} deg")
print(f"planner compute: median {np.median(compute):.0f} ms, worst {compute.max():.0f} ms")
r0 = ok[0]
p0 = r0["plan"]
print(f"\nexample (seed {r0['seed']}): planned at {p0.planned_at:.2f} s -> catch at {p0.t:.2f} s, "
      f"point {p0.pos.round(3)}, error {r0['pos_err'] * 100:.2f} cm")
print(f"  arm joints at catch (rad): {p0.q.round(2)}")

# Plot: top view of the paths and capture points, and the error distribution.
out = Path("out")
out.mkdir(exist_ok=True)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
circle = plt.Circle(SHOULDER[:2], REACH, fill=False, ls="--", color="gray", label="reach limit")
ax0.add_patch(circle)
ax0.plot(0, 0, "ks", ms=8, label="arm base")
for i, r in enumerate(results):
    ax0.plot(r["path"][:, 0], r["path"][:, 1], color="lightgray", lw=0.8, zorder=1)
for i, r in enumerate(ok):
    ax0.plot(*r["plan"].pos[:2], "o", color="tab:blue", ms=5, label="predicted catch" if i == 0 else None)
    ax0.plot(*r["true_pos"][:2], "x", color="tab:red", ms=5, label="actual" if i == 0 else None)
ax0.set_aspect("equal")
ax0.set_xlim(-0.3, 1.0)
ax0.set_ylim(-0.75, 0.9)
ax0.set_xlabel("x (m)")
ax0.set_ylabel("y (m)")
ax0.set_title("Top view: drift paths and capture points")
ax0.legend(loc="upper left", fontsize=8)
ax1.hist(pos_err, bins=12, color="tab:blue", alpha=0.8)
ax1.set_xlabel("position error at capture time (cm)")
ax1.set_ylabel("throws")
ax1.set_title(f"Prediction error, {len(ok)} throws (median {np.median(pos_err):.2f} cm)")
fig.tight_layout()
fig.savefig(out / "intercept.png", dpi=120)
print(f"saved {out / 'intercept.png'}")

assert len(ok) >= 0.8 * N, "too few feasible intercepts"
# One-shot prediction over ~2 s is only as good as the filter says it is; the catcher keeps
# correcting from the live estimates, so what matters here is that the filter is honest.
assert np.median(pos_err) < 1.5 * np.median(sigma), "prediction worse than the filter claims"
print("OK: intercept planning works")
