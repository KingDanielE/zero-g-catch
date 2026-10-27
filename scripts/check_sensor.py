"""Hour 2 check: noisy 30 Hz pose readings scatter around the true path with the expected spread.
Saves out/sensor.png."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.env import ZeroGEnv
from sim.sensor import PoseSensor, quat_angle_deg

T = 3.0  # seed 0's throw hits the arm at ~3.3 s; stop before contact
env = ZeroGEnv(seed=0)
env.reset()
sensor = PoseSensor(seed=1)
m, d = env.model, env.data

truth_t, truth_p = [], []
meas_t, meas_p, pos_err, rot_err = [], [], [], []
expected = int(T * 30)
for _ in range(int(T / m.opt.timestep)):
    p, q = env.target_pose()
    truth_t.append(d.time)
    truth_p.append(p)
    z = sensor.poll(d.time, p, q)
    if z is not None:
        meas_t.append(z.t)
        meas_p.append(z.pos)
        pos_err.append(z.pos - p)
        rot_err.append(quat_angle_deg(z.quat, q))
    env.step()

truth_t, truth_p, meas_t, meas_p = map(np.array, (truth_t, truth_p, meas_t, meas_p))
pos_err, rot_err = np.array(pos_err), np.array(rot_err)
std_mm = pos_err.std(axis=0) * 1000
print(f"{len(meas_t)} readings of {expected} expected ({1 - len(meas_t) / expected:.0%} dropped)")
print(f"position error std per axis (mm): {std_mm.round(2)}  (target 5)")
print(f"orientation error: mean {rot_err.mean():.2f} deg, 95th pct {np.percentile(rot_err, 95):.2f} deg")

out = Path("out")
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
for i, name in enumerate("xyz"):
    ax = axes[i]
    ax.plot(truth_t, truth_p[:, i], color="black", lw=1.2, label="true")
    ax.scatter(meas_t, meas_p[:, i], s=8, color="tab:orange", label="measured")
    ax.set_ylabel(f"{name} (m)")
axes[0].legend(loc="best")
axes[3].plot(meas_t, np.linalg.norm(pos_err, axis=1) * 1000, ".", color="tab:red", ms=4)
axes[3].set_ylabel("|pos error| (mm)")
axes[3].set_xlabel("time (s)")
fig.suptitle("Simulated pose sensor: 30 Hz, 5 mm / 2° noise, 5% dropout")
fig.tight_layout()
fig.savefig(out / "sensor.png", dpi=120)
print(f"saved {out / 'sensor.png'}")

assert np.all(np.abs(std_mm - 5) < 1.5), "position noise spread off target"
print("OK: sensor works")
