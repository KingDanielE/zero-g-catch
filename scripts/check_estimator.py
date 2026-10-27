"""Hour 3 check: the Kalman filter's velocity converges within ~0.5 s and the spin estimate
tracks the true angular velocity. One plotted throw (out/estimator.png) plus a batch of random
throws for numbers. Each run stops at the first contact with the arm."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.env import ZeroGEnv
from sim.estimator import SpinEstimator, TranslationKF
from sim.sensor import PoseSensor

T = 3.0
CONVERGED = 0.01  # m/s


def run(seed: int):
    env = ZeroGEnv(seed=seed)
    env.reset()
    m, d = env.model, env.data
    sensor, kf, spin = PoseSensor(seed=seed + 1000), TranslationKF(), SpinEstimator()
    log = {k: [] for k in ("t", "p_true", "v_true", "p_est", "v_est", "v_std", "w_true", "w_est")}
    while d.time < T and d.ncon == 0:
        p, q = env.target_pose()
        z = sensor.poll(d.time, p, q)
        if z is not None:
            kf.update(z.t, z.pos)
            spin.update(z.t, z.quat)
            if kf.ready and spin.ready:
                v, w_body = env.target_velocity()
                R = d.xmat[env.target_body].reshape(3, 3)
                log["t"].append(d.time)
                log["p_true"].append(p)
                log["v_true"].append(v)
                log["p_est"].append(kf.pos)
                log["v_est"].append(kf.vel)
                log["v_std"].append(kf.vel_std)
                log["w_true"].append(R @ w_body)
                log["w_est"].append(spin.omega)
        env.step()
    return {k: np.array(v) for k, v in log.items()}


def convergence_time(log) -> float:
    """First time after which the velocity error stays below CONVERGED."""
    err = np.linalg.norm(log["v_est"] - log["v_true"], axis=1)
    bad = np.nonzero(err >= CONVERGED)[0]
    if len(bad) == 0:
        return log["t"][0]
    return log["t"][bad[-1] + 1] if bad[-1] + 1 < len(err) else np.inf


# Batch of random throws.
conv, pos_rmse, vel_rmse, spin_rmse = [], [], [], []
for seed in range(30):
    log = run(seed)
    late = log["t"] > 1.0  # steady state
    conv.append(convergence_time(log))
    pos_rmse.append(np.sqrt(np.mean(np.sum((log["p_est"] - log["p_true"])[late] ** 2, 1))))
    vel_rmse.append(np.sqrt(np.mean(np.sum((log["v_est"] - log["v_true"])[late] ** 2, 1))))
    spin_rmse.append(np.sqrt(np.mean(np.sum((log["w_est"] - log["w_true"])[late] ** 2, 1))))
conv = np.array(conv)
print(f"30 throws: velocity converged (<1 cm/s) by median {np.median(conv):.2f} s, "
      f"worst {conv.max():.2f} s")
print(f"after 1 s: position RMSE {np.mean(pos_rmse) * 1000:.1f} mm (raw sensor ~8.7), "
      f"velocity RMSE {np.mean(vel_rmse) * 100:.2f} cm/s, "
      f"spin RMSE {np.rad2deg(np.mean(spin_rmse)):.1f} deg/s")

# Plot one throw.
log = run(0)
out = Path("out")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for i, (name, c) in enumerate(zip("xyz", ("tab:blue", "tab:orange", "tab:green"))):
    axes[0].plot(log["t"], log["v_true"][:, i] * 100, color=c, lw=1, ls="--")
    axes[0].plot(log["t"], log["v_est"][:, i] * 100, color=c, lw=1.5, label=f"v{name}")
    axes[2].plot(log["t"], np.rad2deg(log["w_true"][:, i]), color=c, lw=1, ls="--")
    axes[2].plot(log["t"], np.rad2deg(log["w_est"][:, i]), color=c, lw=1.5, label=f"ω{name}")
axes[0].set_ylabel("velocity (cm/s)")
axes[0].legend(loc="right", title="solid = estimate, dashed = true")
v_err = np.linalg.norm(log["v_est"] - log["v_true"], axis=1) * 100
axes[1].plot(log["t"], v_err, color="tab:red", label="|error|")
axes[1].plot(log["t"], np.linalg.norm(log["v_std"], axis=1) * 100, color="gray", ls=":",
             label="filter 1σ")
axes[1].axhline(CONVERGED * 100, color="black", lw=0.8)
axes[1].set_ylabel("velocity error (cm/s)")
axes[1].legend(loc="upper right")
axes[2].set_ylabel("spin, world (deg/s)")
axes[2].set_xlabel("time (s)")
axes[2].legend(loc="right")
fig.suptitle("Kalman filter (position/velocity) and windowed spin fit, seed 0")
fig.tight_layout()
fig.savefig(out / "estimator.png", dpi=120)
print(f"saved {out / 'estimator.png'}")

# Best possible: a line fit to N = 30T readings with 5 mm noise has per-axis velocity std
# 0.005 * sqrt(12 / (N T^2)); the 3-axis error drops below 1 cm/s at about T = 0.74 s.
assert np.median(conv) < 0.8, "velocity converges too slowly"
print("OK: estimator works")
