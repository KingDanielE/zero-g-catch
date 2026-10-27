"""Hour 5 check: the arm moves to the pre-grasp pose (10 cm behind the bar along the approach
axis) by the planned time and then rides along with the drifting, spinning bar. Contacts are on:
touching the bar before the grasp counts as a failure. Saves out/approach.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.catcher import Catcher, Phase
from sim.env import ZeroGEnv

RIDE_S = 0.6  # ride along this long (the catcher's default is 0.1 s before closing in)
N = 30


def bar_touched(env) -> bool:
    gid = env.model.geom("target_geom").id
    con = env.data.contact[: env.data.ncon]
    return any(c.geom1 == gid or c.geom2 == gid for c in con)


def run(seed: int):
    env = ZeroGEnv(seed=seed)
    env.reset()
    c = Catcher(env, sensor_seed=seed + 1000, ride=RIDE_S, settle=RIDE_S + 0.5)
    log = {k: [] for k in ("t", "pos_err", "rel_vel", "rot_err", "phase")}
    touched = None
    while env.data.time < 8.0:
        c.step()
        env.step()
        if touched is None and bar_touched(env):
            touched = env.data.time
        if c.phase in (Phase.APPROACH, Phase.RIDE):
            e = c.true_errors()
            log["t"].append(env.data.time)
            log["pos_err"].append(e[0])
            log["rel_vel"].append(e[1])
            log["rot_err"].append(e[2])
            log["phase"].append(c.phase)
        if c.phase in (Phase.FAILED, Phase.CLOSE_IN):
            break
    out = {k: np.array(v) for k, v in log.items() if k != "phase"}
    out.update(seed=seed, catcher=c, touched=touched, phase=log["phase"])
    return out


results = [run(s) for s in range(N)]
failed = [r for r in results if r["catcher"].phase is Phase.FAILED]
touched = [r for r in results if r["touched"] is not None]
bad = {r["seed"] for r in failed + touched}
good = [r for r in results if r["seed"] not in bad]
print(f"{N} throws: {len(failed)} found no intercept, {len(touched)} touched the bar early, "
      f"{len(good)} rode along cleanly")
for r in touched:
    print(f"  seed {r['seed']}: touched at t={r['touched']:.2f} s "
          f"(meet at {r['catcher'].t_meet:.2f} s)")


def ride(r):
    return r["t"] >= r["catcher"].t_meet


arrive = np.array([r["pos_err"][ride(r)][0] for r in good]) * 100
pos = np.concatenate([r["pos_err"][ride(r)] for r in good]) * 100
vel = np.concatenate([r["rel_vel"][ride(r)] for r in good]) * 100
rot = np.concatenate([r["rot_err"][ride(r)] for r in good])
print(f"at the meet time: tool point {np.median(arrive):.2f} cm from the pre-grasp point "
      f"(worst {arrive.max():.2f})")
print(f"while riding: position error median {np.median(pos):.2f} cm (95th {np.percentile(pos, 95):.2f}), "
      f"relative speed median {np.median(vel):.2f} cm/s (95th {np.percentile(vel, 95):.2f}), "
      f"orientation error median {np.median(rot):.1f} deg (95th {np.percentile(rot, 95):.1f})")

# Plot one throw.
r = good[0]
out = Path("out")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
t_meet = r["catcher"].t_meet
for ax, key, scale, label in ((axes[0], "pos_err", 100, "distance to pre-grasp point (cm)"),
                              (axes[1], "rel_vel", 100, "speed relative to it (cm/s)"),
                              (axes[2], "rot_err", 1, "hand orientation error (deg)")):
    ax.plot(r["t"], r[key] * scale, color="tab:blue")
    ax.axvspan(t_meet, r["t"][-1], color="tab:green", alpha=0.12)
    ax.set_ylabel(label)
    ax.set_yscale("log")
axes[0].text(t_meet, axes[0].get_ylim()[1] * 0.5, " riding along", color="tab:green")
axes[2].set_xlabel("time (s)")
fig.suptitle(f"Approach and ride-along, seed {r['seed']} (ground-truth errors)")
fig.tight_layout()
fig.savefig(out / "approach.png", dpi=120)
print(f"saved {out / 'approach.png'}")

# Throws with no reachable catch are the planner's result (check_intercept); here, of the
# throws that got a plan, nearly all should approach and ride along without touching the bar.
assert len(good) >= 0.9 * (N - len(failed)), "too many failed approaches"
assert np.median(pos) < 1.0 and np.median(vel) < 2.0, "ride-along not tight enough"
print("OK: approach and ride-along work")
