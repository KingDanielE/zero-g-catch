"""Reference runs for checking the browser port (site/sim/js) against this Python code.

Runs Monte Carlo throws with the sensor noise switched off, so both implementations are
deterministic and should agree closely, and writes each throw's exact starting state and results
to out/reference.json. scripts/validate_web.mjs replays them in JavaScript and compares.

    python -m scripts.export_reference [--n 40]
"""

import argparse
import json
from multiprocessing import Pool

import sim.catcher
from scripts.monte_carlo import SEED0, sample_throw
from sim.evaluate import run_throw
from sim.sensor import PoseSensor

KEYS = ["outcome", "phase", "n_plans", "aborts", "t_plan", "t_catch", "grasp_dist", "grasp_rel_speed",
        "kick", "slip_mm", "peak_force", "brake_T", "final_speed", "final_spin", "early_touch"]


def noiseless(seed=None):
    return PoseSensor(pos_std=0.0, rot_std_deg=0.0, dropout=0.0, seed=seed)


def one(i):
    sim.catcher.PoseSensor = noiseless
    target, params = sample_throw(SEED0 + i)
    r = run_throw(SEED0 + i, target)
    as_float = lambda x: None if x is None else float(x)  # noqa: E731
    return {
        "throw": i,
        "target": {"pos": target.pos.tolist(), "quat": target.quat.tolist(),
                   "vel": target.vel.tolist(), "omega": target.omega.tolist()},
        "params": {k: as_float(v) for k, v in params.items()},
        "result": {k: (r[k] if isinstance(r[k], (str, bool)) or r[k] is None else as_float(r[k]))
                   for k in KEYS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        rows = pool.map(one, range(args.n))
    with open("out/reference.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    caught = sum(r["result"]["outcome"] == "caught" for r in rows)
    print(f"{args.n} noiseless throws: {caught} caught; saved out/reference.json")


if __name__ == "__main__":
    main()
