"""Watch the arm catch the bar: it waits in the ready pose, plans once the velocity estimate
settles, lines up behind the bar, rides along 10 cm back, slides in, closes the gripper gently
and brings the bar to rest.
Blue ball = planned capture point. A new throw starts 1.5 s after it stops (or aborts).

    .venv\\Scripts\\python -m scripts.view_catch [--seed N]
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim.catcher import Catcher, Phase
from sim.env import ZeroGEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = ZeroGEnv(seed=args.seed)
m, d = env.model, env.data


def new_throw():
    env.reset()
    return Catcher(env, sensor_seed=int(env.rng.integers(1 << 30)))


def draw(scn, c):
    scn.ngeom = 0
    if c.plan is None:
        return
    mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]),
                        c.plan.pos, np.eye(3).flatten(), np.array([0.2, 0.5, 1.0, 0.4], np.float32))
    scn.ngeom = 1


c = new_throw()
last_phase, done_at = None, None
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        tick = time.time()
        c.step()
        env.step()
        if c.phase is not last_phase:
            last_phase = c.phase
            extra = ""
            if c.phase is Phase.APPROACH:
                extra = f" (catch planned for t={c.plan.t:.2f} s, meet at {c.t_meet:.2f} s)"
            print(f"t={d.time:.2f} s: {c.phase.value}{extra}")
            done_at = d.time + 1.5 if c.phase is Phase.STOPPED or c.gave_up else None
        if done_at is not None and d.time > done_at or d.time > 12.0:
            c = new_throw()
            last_phase = None
        with viewer.lock():
            draw(viewer.user_scn, c)
        viewer.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - tick)))
