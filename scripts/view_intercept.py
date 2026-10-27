"""Watch the planner: each throw, once the velocity estimate settles, the planned catch is drawn
(blue ball = predicted bar centre at capture, orange line = predicted bar axis, green line =
gripper approach direction, grey dots = predicted path). The arm doesn't move yet and contacts
are off, so the bar flies through it. A new throw starts every 4 s after the catch time.

    .venv\\Scripts\\python -m scripts.view_intercept [--seed N]
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim.env import READY_Q, ZeroGEnv
from sim.estimator import SpinEstimator, TranslationKF
from sim.intercept import plan_intercept, quat_to_mat
from sim.kinematics import ArmKinematics
from sim.sensor import PoseSensor

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = ZeroGEnv(seed=args.seed)
env.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
m, d = env.model, env.data
kin = ArmKinematics(m)
q_home = np.array(READY_Q)


def add_geom(scn, gtype, size, pos, rgba, mat=np.eye(3)):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float),
                        np.asarray(pos, float), np.asarray(mat, float).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_line(scn, a, b, rgba, width=0.006):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float),
                         np.asarray(b, float))
    scn.ngeom += 1


def draw(scn, plan, kf):
    scn.ngeom = 0
    if plan is None:
        return
    for t in np.linspace(d.time, plan.t, 12):
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], kf.predict(t)[0],
                 [0.7, 0.7, 0.7, 0.6])
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0], plan.pos, [0.2, 0.5, 1.0, 0.5])
    axis = quat_to_mat(plan.quat)[:, 2]
    add_line(scn, plan.pos - 0.1 * axis, plan.pos + 0.1 * axis, [1.0, 0.55, 0.1, 0.9])
    approach = plan.hand_R[:, 2]
    add_line(scn, plan.pos - 0.18 * approach, plan.pos, [0.2, 0.9, 0.3, 0.9])


def new_throw():
    env.reset()
    return PoseSensor(seed=int(env.rng.integers(1 << 30))), TranslationKF(), SpinEstimator()


sensor, kf, spin = new_throw()
plan, planned = None, False
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        tick = time.time()
        p, q = env.target_pose()
        z = sensor.poll(d.time, p, q)
        if z is not None:
            kf.update(z.t, z.pos)
            spin.update(z.t, z.quat)
            if not planned and spin.ready and np.linalg.norm(kf.vel_std) < 0.01:
                planned = True
                plan = plan_intercept(kin, kf, spin, d.time, q_home)
                if plan is None:
                    print(f"t={d.time:.2f} s: no reachable intercept")
                else:
                    print(f"t={d.time:.2f} s: catch at t={plan.t:.2f} s at {plan.pos.round(3)} "
                          f"(arm needs {plan.reach_time:.2f} s)")
        if plan is not None and abs(d.time - plan.t) < m.opt.timestep / 2:
            print(f"   at t={plan.t:.2f} s the bar is {np.linalg.norm(p - plan.pos) * 100:.1f} cm "
                  f"from the prediction")
        if d.time > (plan.t if plan else 3.0) + 4.0:
            sensor, kf, spin = new_throw()
            plan, planned = None, False
        env.step()
        with viewer.lock():
            draw(viewer.user_scn, plan, kf)
        viewer.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - tick)))
