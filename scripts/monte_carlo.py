"""Hour 8: Monte Carlo over randomized throws, end to end.

Each throw draws drift speed (2-15 cm/s), drift direction (within 30 deg of straight across the
workspace, from either side), spin rate (0-30 deg/s) about a random axis, start position (+-5 cm)
and orientation, and runs the full catch with its own sensor noise. Writes one row per throw to
out/monte_carlo.csv (hour 9 plots read it) and prints success rates by speed, spin and direction.

    python -m scripts.monte_carlo [--n 100] [--workers 4] [--out out/monte_carlo.csv]
"""

import argparse
import csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sim.env import TargetInit
from sim.evaluate import run_throw

SPEED = (0.02, 0.15)  # m/s
SPIN_DEG = (0.0, 30.0)  # deg/s
CONE_DEG = 30.0  # drift direction within this angle of straight across (+-y)
START = np.array([0.6, 0.5, 0.5])  # y sign flipped by side
START_JITTER = 0.05  # m
SEED0 = 10_000  # throw i uses seed SEED0 + i (sampler, sensor noise)
OUTCOMES = ("caught", "aborted", "no_plan", "not_at_rest", "timeout")


def sample_throw(seed: int) -> tuple[TargetInit, dict]:
    rng = np.random.default_rng(seed)
    side = rng.choice([1, -1])  # +1: starts at -y, drifts toward +y
    speed = rng.uniform(*SPEED)
    # Uniform over the spherical cap around the nominal direction.
    cone = np.rad2deg(np.arccos(rng.uniform(np.cos(np.deg2rad(CONE_DEG)), 1.0)))
    azim = rng.uniform(0, 2 * np.pi)
    th = np.deg2rad(cone)
    direction = np.array([np.sin(th) * np.cos(azim), side * np.cos(th), np.sin(th) * np.sin(azim)])
    spin = rng.uniform(*SPIN_DEG)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    pos = START * [1, -side, 1] + rng.uniform(-START_JITTER, START_JITTER, 3)
    quat = rng.normal(size=4)
    target = TargetInit(pos, quat / np.linalg.norm(quat), direction * speed,
                        axis * np.deg2rad(spin))
    params = dict(side=int(side), speed=speed, spin_deg=spin, cone_deg=cone,
                  dir_x=direction[0], dir_y=direction[1], dir_z=direction[2],
                  start_x=pos[0], start_y=pos[1], start_z=pos[2])
    return target, params


def one(i: int) -> dict:
    seed = SEED0 + i
    target, params = sample_throw(seed)
    r = run_throw(seed, target)
    r.pop("v_grasp")
    est = r.pop("grasp_check")
    r["est_grasp_dist"], r["est_grasp_rel_speed"] = est if est else (None, None)
    return {"throw": i, **params, **r}


def rate(k: int, n: int) -> str:
    """Success rate with a 95% Wilson interval."""
    if n == 0:
        return "   -"
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = 100 * (centre - half), 100 * (centre + half)
    return f"{k:3d}/{n:<3d} {100 * p:5.1f}%  [{lo:4.0f}-{hi:3.0f}%]"


def by(rows, key, edges, unit):
    print(f"\ncaught by {key}:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        k = sum(r["outcome"] == "caught" for r in sel)
        print(f"  {lo:5.1f}-{hi:<5.1f} {unit:6s} {rate(k, len(sel))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="out/monte_carlo.csv")
    args = ap.parse_args()

    with Pool(args.workers) as pool:
        rows = []
        for r in pool.imap(one, range(args.n)):
            rows.append(r)
            print(f"  throw {r['throw']:3d}: {r['speed'] * 100:5.1f} cm/s, "
                  f"{r['spin_deg']:4.1f} deg/s, {r['cone_deg']:4.1f} deg off, "
                  f"side {r['side']:+d} -> {r['outcome']}", flush=True)
    for r in rows:
        r["speed_cms"] = r["speed"] * 100

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    print(f"\n{n} throws: " + ", ".join(f"{counts[o]} {o}" for o in OUTCOMES))
    print(f"caught overall:          {rate(counts['caught'], n)}")
    planned = [r for r in rows if r["outcome"] != "no_plan"]
    print(f"caught, given a plan:    {rate(counts['caught'], len(planned))}")
    closed = [r for r in rows if r["kick"] is not None]
    print(f"caught, given a grasp:   {rate(counts['caught'], len(closed))}")
    by(rows, "speed_cms", [2, 5, 8, 11, 15.01], "cm/s")
    by(rows, "spin_deg", [0, 10, 20, 30.01], "deg/s")
    by(rows, "cone_deg", [0, 10, 20, 30.01], "deg off")
    print("\ncaught by side:")
    for s, name in ((1, "from -y"), (-1, "from +y")):
        sel = [r for r in rows if r["side"] == s]
        print(f"  {name:12s}      {rate(sum(r['outcome'] == 'caught' for r in sel), len(sel))}")

    caught = [r for r in rows if r["outcome"] == "caught"]
    if caught:
        def stat(key, scale=1.0, unit=""):
            a = np.array([r[key] for r in caught if r[key] is not None]) * scale
            return (f"median {np.median(a):.2f}{unit}, 95th pct {np.percentile(a, 95):.2f}{unit}, "
                    f"worst {a.max():.2f}{unit}")
        print("\nover caught throws (ground truth):")
        print(f"  gripper-to-grasp-point distance at close: {stat('grasp_dist', 100, ' cm')}")
        print("  relative speed at close (contact speed):  "
              f"{stat('grasp_rel_speed', 100, ' cm/s')}")
        print(f"  finger kick on the bar:                   {stat('kick', 100, ' cm/s')}")
        print(f"  slip in the grip while braking:           {stat('slip_mm', 1, ' mm')}")
        print(f"  peak contact force:                       {stat('peak_force', 1, ' N')}")
        print(f"  braking time:                             {stat('brake_T', 1, ' s')}")
        print(f"  catch time after release:                 {stat('t_catch', 1, ' s')}")
    early = [r for r in rows if r["early_touch"]]
    print(f"\ntouched the bar before closing: {len(early)} throws")
    print("not caught:")
    for r in rows:
        if r["outcome"] != "caught":
            extra = ""
            if r["outcome"] == "aborted" and r["abort_from"] is not None:
                extra = f" (abort {r['abort_from'] * 100:.1f} cm from the bar)"
            print(f"  throw {r['throw']:3d} {r['outcome']:8s} {r['speed_cms']:5.1f} cm/s "
                  f"{r['spin_deg']:4.1f} deg/s {r['cone_deg']:4.1f} deg off side {r['side']:+d}"
                  + extra)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
