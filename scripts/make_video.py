"""Hour 9: render the demo video (out/catch_demo.mp4, 1280x720, 30 fps, ~45 s).

Four throws from the Monte Carlo set in real time (slow, medium and fast, all spinning ~28 deg/s,
then one that aborts and catches on the retry), a quarter-speed close-up of the fast throw's
grasp, and title / results cards. Overlays: planned capture point (blue ball), the bar's
predicted path from the filter (small dots), phase and live ground-truth readouts.

    python -m scripts.make_video [--out out/catch_demo.mp4]
"""

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.monte_carlo import SEED0, sample_throw
from sim.catcher import Catcher, Phase
from sim.env import ZeroGEnv

W, H, FPS = 1280, 720, 30
SLOW = 4  # close-up slow-motion factor
THROWS = [(94, "slow"), (8, "medium"), (40, "fast"), (13, "abort, back off, retry")]
CLOSEUP = 40  # throw whose grasp gets the slow-motion close-up
PHASE_TEXT = {
    Phase.WAIT: "watching: filter converging",
    Phase.APPROACH: "approach: line up behind the bar",
    Phase.RIDE: "ride along, 10 cm back",
    Phase.CLOSE_IN: "close in, matching drift and spin",
    Phase.GRASP: "fingers closing gently",
    Phase.BRAKE: "braking drift and spin",
    Phase.STOPPED: "at rest, held",
    Phase.ABORT: "abort: estimate too far off, back off",
    Phase.FAILED: "no reachable catch",
}
INK, INK2, ACCENT = (240, 239, 236), (195, 194, 183), (57, 135, 229)


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


F_BIG, F_MED, F_SMALL = font(46, True), font(26, True), font(22)


def add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, float), np.eye(3).flatten(),
                        np.array(rgba, np.float32))
    scn.ngeom += 1


def overlays(scn, c):
    if c.plan is not None and c.phase in (Phase.APPROACH, Phase.RIDE, Phase.CLOSE_IN):
        add_sphere(scn, c.plan.pos, 0.018, (0.22, 0.53, 0.9, 0.45))
    if (c.phase in (Phase.WAIT, Phase.APPROACH, Phase.RIDE, Phase.ABORT) and c.kf.x is not None
            and np.linalg.norm(c.kf.vel_std) < 0.03):
        t = c.d.time
        for dt in np.arange(0.25, 3.01, 0.25):
            add_sphere(scn, c.kf.predict(t + dt)[0], 0.004, (0.95, 0.75, 0.4, 0.6))


def caption(frame, lines_top, lines_bottom):
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img, "RGBA")
    if lines_top:
        dr.rectangle((0, 0, W, 54), fill=(0, 0, 0, 140))
        dr.text((24, 13), lines_top, font=F_MED, fill=INK)
    if lines_bottom:
        h = 34 * len(lines_bottom) + 18
        dr.rectangle((0, H - h, W, H), fill=(0, 0, 0, 140))
        for i, (txt, color) in enumerate(lines_bottom):
            dr.text((24, H - h + 10 + 34 * i), txt, font=F_SMALL, fill=color)
    return np.asarray(img)


def card(title, lines, seconds):
    img = Image.new("RGB", (W, H), (14, 17, 26))
    dr = ImageDraw.Draw(img)
    dr.text((90, 170), title, font=F_BIG, fill=INK)
    dr.rectangle((90, 240, 210, 244), fill=ACCENT)
    for i, txt in enumerate(lines):
        dr.text((90, 280 + 46 * i), txt, font=F_MED if i == 0 else F_SMALL,
                fill=INK if i == 0 else INK2)
    return [np.asarray(img)] * int(seconds * FPS)


def readout(env, c):
    v, w = env.target_velocity()
    p_bar = env.target_pose()[0]
    gap = np.linalg.norm(c.kin.fk(env.data.qpos[c.qadr])[0] - p_bar) * 100
    color = INK if c.phase is not Phase.ABORT else (229, 103, 103)
    return [(f"{PHASE_TEXT[c.phase]}", color),
            (f"t = {env.data.time:4.1f} s    gripper to bar {gap:5.1f} cm    bar drift "
             f"{np.linalg.norm(v) * 100:4.1f} cm/s    spin {np.rad2deg(np.linalg.norm(w)):4.1f} deg/s",
             INK2)]


def run(env, throw, label, k, renderer, cam):
    """Real-time frames, plus slow-motion close-up frames around the grasp."""
    seed = SEED0 + throw
    target, params = sample_throw(seed)
    env.reset(target)
    c = Catcher(env, sensor_seed=seed + 1000)
    m, d = env.model, env.data
    top = (f"Throw {k}/{len(THROWS)} ({label}): {params['speed'] * 100:.1f} cm/s drift, "
           f"{params['spin_deg']:.0f} deg/s spin, {params['cone_deg']:.0f} deg off straight across"
           "  |  real time")
    steps = round(1 / (FPS * m.opt.timestep))
    slow_steps = max(1, steps // SLOW)
    close_cam = mujoco.MjvCamera()
    close_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    close_cam.distance, close_cam.azimuth, close_cam.elevation = 0.75, cam.azimuth, -20
    frames, slow, t_end, i = [], [], None, 0
    while d.time < 12.0:
        c.step()
        env.step()
        i += 1
        want_slow = (throw == CLOSEUP and c.plan is not None
                     and c.phase in (Phase.CLOSE_IN, Phase.GRASP, Phase.BRAKE)
                     and d.time >= c.plan.t - 0.35)
        if want_slow and i % slow_steps == 0:
            close_cam.lookat[:] = env.target_pose()[0]
            renderer.update_scene(d, camera=close_cam)
            img = renderer.render()
            slow.append(caption(img, f"Close-up of throw {k}, 1/{SLOW} speed",
                                readout(env, c)))
        if i % steps == 0:
            renderer.update_scene(d, camera=cam)
            overlays(renderer.scene, c)
            frames.append(caption(renderer.render(), top, readout(env, c)))
        if t_end is None and (c.phase is Phase.STOPPED or c.gave_up):
            t_end = d.time + (1.0 if c.phase is Phase.STOPPED else 1.5)
        if t_end is not None and d.time >= t_end:
            break
    print(f"throw {throw}: {c.phase.value} at {d.time:.1f} s, {len(frames)} frames"
          + (f", {len(slow)} slow-motion frames" if slow else ""))
    return frames, slow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/catch_demo.mp4")
    ap.add_argument("--csv", default="out/monte_carlo_300.csv")
    args = ap.parse_args()

    env = ZeroGEnv()
    m = env.model
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.45, 0.0, 0.42)
    cam.distance, cam.azimuth, cam.elevation = 1.55, 200, -22

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    n, k = len(rows), sum(r["outcome"] == "caught" for r in rows)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    total = 0

    def put(frames):
        nonlocal total
        for f in frames:
            writer.append_data(f)
        total += len(frames)

    put(card("Catching a drifting object in zero-g",
             ["Franka Panda, MuJoCo, zero gravity",
              "30 Hz pose sensor (5 mm / 2 deg noise) -> Kalman filter + spin fit",
              "plan an intercept -> ride along -> close in -> grasp gently -> brake to rest",
              "blue ball: planned catch point   dots: predicted path"], 4.0))
    for j, (throw, label) in enumerate(THROWS, 1):
        frames, slow = run(env, throw, label, j, renderer, cam)
        put(frames)
        put(slow)
    put(card("Monte Carlo",
             [f"{k} of {n} randomized throws caught and brought to rest ({100 * k / n:.0f}%)",
              "speed 2-15 cm/s, spin 0-30 deg/s about a random axis, direction within 30 deg",
              "contact speed when closing: median 1.9 cm/s; peak grip force 2.5 N",
              "misses: no reachable catch (fast / far throws) or abort before closing"], 5.0))
    writer.close()
    print(f"saved {out}: {total / FPS:.1f} s")


if __name__ == "__main__":
    main()
