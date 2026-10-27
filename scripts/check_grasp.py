"""Hour 6 check: close in, match the bar's motion, close the gripper and keep hold of it.

Per throw: did it grasp (or abort / find no intercept), the true distance and relative speed when
the gripper started closing, whether anything touched the bar before that, how much the grasp
disturbed the bar's velocity, the peak contact force, and whether the bar is still held between
both fingers 0.5 s later. Saves out/grasp.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sim.catcher import Catcher, Phase
from sim.env import ZeroGEnv

HOLD_S = 0.5
N = 30


def bar_contacts(env):
    """(touching anything, set of bodies touching it, total normal force N)."""
    m, d = env.model, env.data
    gid = m.geom("target_geom").id
    bodies, force = set(), 0.0
    f6 = np.zeros(6)
    for i, c in enumerate(d.contact[: d.ncon]):
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            bodies.add(m.body(m.geom_bodyid[other]).name)
            mujoco.mj_contactForce(m, d, i, f6)
            force += abs(f6[0])
    return bool(bodies), bodies, force


def run(seed: int):
    env = ZeroGEnv(seed=seed)
    env.reset()
    c = Catcher(env, sensor_seed=seed + 1000)
    d = env.data
    log = {k: [] for k in ("t", "dist", "rel_vel", "bar_speed", "force", "gap")}
    early_touch, at_grasp, v_before, peak_force, t_end, dv = None, None, None, 0.0, None, None
    while d.time < 8.0:
        c.step()
        env.step()
        touching, bodies, force = bar_contacts(env)
        if c.phase in (Phase.APPROACH, Phase.RIDE, Phase.CLOSE_IN) and touching and early_touch is None:
            early_touch = (d.time, sorted(bodies))
        if c.phase is Phase.GRASP and at_grasp is None:
            e = c.true_errors()
            at_grasp = dict(dist=e[0], rel_vel=e[1], est=c.grasp_check)
            v_before = env.target_velocity()[0]
        if c.phase in (Phase.GRASP, Phase.BRAKE, Phase.STOPPED):
            peak_force = max(peak_force, force)
        if c.phase is not Phase.WAIT and c.plan is not None:
            p_bar = env.target_pose()[0]
            p_tcp = c.kin.fk(d.qpos[c.qadr])[0]
            log["t"].append(d.time)
            log["dist"].append(np.linalg.norm(p_bar - p_tcp))
            log["rel_vel"].append(c.true_errors()[1])
            log["bar_speed"].append(np.linalg.norm(env.target_velocity()[0]))
            log["force"].append(force)
            log["gap"].append(d.qpos[c.fingers].sum())
        if c.phase is Phase.BRAKE and t_end is None:
            t_end = d.time + HOLD_S
            # Net change the grasp made, before the (intentional) braking changes it further.
            dv = np.linalg.norm(env.target_velocity()[0] - v_before)
        if c.phase is Phase.ABORT and t_end is None:
            t_end = d.time + 0.3
        if c.phase is Phase.FAILED or (t_end is not None and d.time >= t_end):
            break
    _, bodies, _ = bar_contacts(env)
    p_bar = env.target_pose()[0]
    p_tcp = c.kin.fk(d.qpos[c.qadr])[0]
    held = (c.phase in (Phase.BRAKE, Phase.STOPPED) and {"left_finger", "right_finger"} <= bodies
            and np.linalg.norm(p_bar - p_tcp) < 0.02)
    return dict(seed=seed, phase=c.phase, catcher=c, early_touch=early_touch, at_grasp=at_grasp,
                held=held, dv=dv, peak_force=peak_force, bodies=bodies,
                log={k: np.array(v) for k, v in log.items()})


results = [run(s) for s in range(N)]
held = [r for r in results if r["held"]]
no_plan = [r for r in results if r["phase"] is Phase.FAILED]
aborted = [r for r in results if r["phase"] is Phase.ABORT]
print(f"{N} throws: {len(held)} caught and held, {len(aborted)} aborted (estimates not good "
      f"enough to close), {len(no_plan)} no reachable catch, "
      f"{N - len(held) - len(aborted) - len(no_plan)} grasped but lost")
for r in results:
    if r["early_touch"]:
        print(f"  seed {r['seed']}: touched before closing at t={r['early_touch'][0]:.2f} s "
              f"by {r['early_touch'][1]}")
    if r["phase"] in (Phase.BRAKE, Phase.STOPPED) and not r["held"]:
        print(f"  seed {r['seed']}: grasped but not held; touching {sorted(r['bodies'])}")
g = [r for r in results if r["at_grasp"] is not None]
dist = np.array([r["at_grasp"]["dist"] for r in g]) * 100
relv = np.array([r["at_grasp"]["rel_vel"] for r in g]) * 100
dv = np.array([r["dv"] for r in g if r["dv"] is not None]) * 100
force = np.array([r["peak_force"] for r in g])
print(f"when the gripper closed (truth): distance median {np.median(dist):.2f} cm "
      f"(worst {dist.max():.2f}), relative speed median {np.median(relv):.2f} cm/s "
      f"(worst {relv.max():.2f})")
print(f"bar velocity change from the grasp: median {np.median(dv):.2f} cm/s (worst {dv.max():.2f})")
print(f"peak contact force: median {np.median(force):.1f} N (worst {force.max():.1f})")

# Plot one caught throw.
r = held[0]
L, c = r["log"], r["catcher"]
out = Path("out")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
axes[0].plot(L["t"], L["dist"] * 100)
axes[0].set_ylabel("tool point to bar centre (cm)")
axes[1].plot(L["t"], L["rel_vel"] * 100)
axes[1].set_ylabel("relative speed (cm/s)")
axes[1].set_yscale("log")
axes[2].plot(L["t"], L["bar_speed"] * 100, label="bar speed")
axes[2].set_ylabel("bar speed (cm/s)")
ax2 = axes[2].twinx()
ax2.plot(L["t"], L["gap"] * 100, color="tab:gray", ls="--")
ax2.set_ylabel("finger opening (cm)", color="tab:gray")
axes[3].plot(L["t"], L["force"], color="tab:red")
axes[3].set_ylabel("contact force on bar (N)")
axes[3].set_xlabel("time (s)")
for ax in axes:
    ax.axvspan(c.t_close, c.plan.t, color="tab:blue", alpha=0.1)
    ax.axvline(c.t_grasp, color="black", lw=0.8)
axes[0].text(c.t_close, axes[0].get_ylim()[1] * 0.85, " close in", color="tab:blue")
axes[0].text(c.t_grasp, axes[0].get_ylim()[1] * 0.6, " gripper closes")
axes[0].set_xlim(c.t_meet - 0.3, L["t"][-1])
fig.suptitle(f"Close-in and grasp, seed {r['seed']} (ground truth)")
fig.tight_layout()
fig.savefig(out / "grasp.png", dpi=120)
print(f"saved {out / 'grasp.png'}")

assert len(held) >= 0.7 * N, "too few catches"
print("OK: grasping works")
