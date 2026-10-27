"""Watch the bar drift past the arm in real time. A new random throw starts every 8 s;
close the window to quit.

    .venv/Scripts/python -m scripts.view [--seed N]
"""

import argparse
import time

import mujoco.viewer

from sim.env import ZeroGEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--period", type=float, default=8.0, help="seconds per throw")
args = parser.parse_args()

env = ZeroGEnv(seed=args.seed)
env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    throw_start = time.time()
    while viewer.is_running():
        tick = time.time()
        if tick - throw_start > args.period:
            env.reset()
            throw_start = tick
        env.step()
        viewer.sync()
        time.sleep(max(0.0, env.model.opt.timestep - (time.time() - tick)))
