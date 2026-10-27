"""Hour 7 check: the full catch, ending with the bar at rest in the gripper.

Per throw: caught and stopped (both fingers on the bar, bar speed < 0.5 cm/s and spin < 2 deg/s,
0.5 s after the arm stops), how hard the closing fingers kicked the bar, how far the bar slipped
in the grip while braking, the braking time and peak deceleration, and the peak contact force.
Saves out/stop.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.evaluate import run_throw

N = 30

results = [run_throw(s, keep=True) for s in range(N)]
stopped = [r for r in results if r["outcome"] == "caught"]
no_plan = [r for r in results if r["outcome"] == "no_plan"]
aborted = [r for r in results if r["outcome"] == "aborted"]
print(f"{N} throws: {len(stopped)} caught and brought to rest, {len(aborted)} aborted, "
      f"{len(no_plan)} no reachable catch, {N - len(stopped) - len(aborted) - len(no_plan)} "
      "grasped but not at rest in the gripper")
for r in results:
    if r["outcome"] in ("not_at_rest", "timeout"):
        print(f"  seed {r['seed']}: ended {r['phase']}, bar speed "
              f"{r['final_speed'] * 100:.2f} cm/s, spin {r['final_spin']:.1f} deg/s")
g = [r for r in results if r["kick"] is not None]
kick = np.array([r["kick"] for r in g]) * 100
s = [r for r in stopped if r["slip_mm"] is not None]
slip_mm = np.array([r["slip_mm"] for r in s])
slip_deg = np.array([r["slip_deg"] for r in s])
force = np.array([r["peak_force"] for r in g])
brake_T = np.array([r["brake_T"] for r in stopped])
v0 = np.array([np.linalg.norm(r["v_grasp"]) for r in stopped]) * 100
print(f"finger close kick on the bar: median {np.median(kick):.2f} cm/s (worst {kick.max():.2f})")
print(f"slip in the grip while braking: median {np.median(slip_mm):.2f} mm / "
      f"{np.median(slip_deg):.2f} deg (worst {slip_mm.max():.2f} mm / {slip_deg.max():.2f} deg)")
print(f"braking: {np.median(v0):.1f} cm/s to rest in median {np.median(brake_T):.2f} s "
      f"(longest {brake_T.max():.2f})")
print(f"peak contact force: median {np.median(force):.1f} N (worst {force.max():.1f})")
print(f"at rest: bar speed worst {max(r['final_speed'] for r in stopped) * 100:.2f} cm/s, "
      f"spin worst {max(r['final_spin'] for r in stopped):.2f} deg/s")

# Plot one throw.
r = stopped[0]
L, c = r["log"], r["catcher"]
out = Path("out")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axes[0].plot(L["t"], L["speed"] * 100, label="drift speed (cm/s)")
axes[0].plot(L["t"], L["spin"], label="spin (deg/s)")
axes[0].set_ylabel("bar motion")
axes[0].legend(loc="upper right")
axes[1].plot(L["t"], L["force"], color="tab:red")
axes[1].set_ylabel("contact force on bar (N)")
axes[2].plot(L["t"], L["gap"] * 100, color="tab:gray")
axes[2].set_ylabel("finger opening (cm)")
axes[2].set_xlabel("time (s)")
for ax in axes:
    ax.axvspan(c.t_grasp, c.t_brake, color="tab:orange", alpha=0.12)
    ax.axvspan(c.t_brake, c.t_brake + c.brake_T, color="tab:green", alpha=0.12)
axes[0].text(c.t_grasp, axes[0].get_ylim()[1] * 0.9, " fingers close", color="tab:orange")
axes[0].text(c.t_brake, axes[0].get_ylim()[1] * 0.75, " braking", color="tab:green")
axes[0].set_xlim(c.plan.t - 0.4, L["t"][-1])
fig.suptitle(f"Grasp and compliant stop, seed {r['seed']} (ground truth)")
fig.tight_layout()
fig.savefig(out / "stop.png", dpi=120)
print(f"saved {out / 'stop.png'}")

assert len(stopped) >= 0.7 * N, "too few catches brought to rest"
print("OK: compliant stop works")
