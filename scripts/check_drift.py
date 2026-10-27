"""Hour 1 check: with gravity off, the bar's centre moves in a straight line at constant speed,
its angular momentum is conserved, and the arm holds its home pose."""

import mujoco
import numpy as np

from sim.env import ZeroGEnv

env = ZeroGEnv(seed=0)
init = env.reset()
m, d = env.model, env.data
print(f"start {init.pos.round(3)}  speed {np.linalg.norm(init.vel) * 100:.1f} cm/s  "
      f"spin {np.rad2deg(np.linalg.norm(init.omega)):.1f} deg/s")


def angular_momentum():
    # World-frame L = R I w_body (box: body and inertial frames coincide).
    R = d.ximat[env.target_body].reshape(3, 3)
    inertia = np.diag(m.body_inertia[env.target_body])
    _, w_body = env.target_velocity()
    return R @ inertia @ w_body


T = 3.0
L0, hand0 = angular_momentum(), env.hand_pos()
pos = []
for _ in range(int(T / m.opt.timestep)):
    env.step()
    pos.append(env.target_pose()[0])
pos = np.array(pos)

expected = init.pos + init.vel * T
line_err = np.linalg.norm(pos[-1] - expected)
L_err = np.linalg.norm(angular_momentum() - L0) / max(np.linalg.norm(L0), 1e-9)
hand_drift = np.linalg.norm(env.hand_pos() - hand0)
contacts = d.ncon

print(f"after {T:.0f} s: position error vs straight line {line_err * 1000:.3f} mm")
print(f"angular momentum relative change {L_err:.2e}")
print(f"hand moved {hand_drift * 1000:.3f} mm, contacts now {contacts}")
assert line_err < 1e-3 and L_err < 1e-2 and hand_drift < 1e-3, "zero-g check failed"
print("OK: zero-g drift works")
