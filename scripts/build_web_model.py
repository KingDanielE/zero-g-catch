"""Build the compiled model for the in-browser simulation (site/sim/scene.mjb).

The Menagerie Panda's visual meshes are ~400K vertices (40 MB compiled): far too much for a web
page. Visual geoms don't collide (contype/conaffinity 0) and every body sets its mass and inertia
explicitly, so simplifying the visual meshes leaves the physics unchanged. Collision meshes,
inertias, actuators and the scene are kept as they are. The script checks that: the physical
model arrays must match exactly, and a scripted run must give identical states.

    python -m scripts.build_web_model [--keep 0.06]

Needs trimesh and fast-simplification (development only, not in requirements.txt).
"""

import argparse
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from sim.env import READY_Q, SCENE

PANDA = Path("third_party/menagerie/franka_emika_panda")
BUILD = Path("out/web_model")
OUT = Path("site/sim/scene.mjb")


def simplify(src: Path, dst: Path, keep: float, min_faces: int = 60):
    # The OBJ files repeat vertices per triangle: weld them first, or the decimation collapses the
    # triangle soup (it kept 6% of the surface area before this).
    mesh = trimesh.load(src, force="mesh", process=False)
    area = mesh.area
    mesh.merge_vertices()
    # Keep more faces until the simplified mesh keeps at least 90% of the surface area.
    out = mesh
    while keep < 1:
        target = max(min_faces, int(len(mesh.faces) * keep))
        if len(mesh.faces) <= target:
            out = mesh
            break
        out = mesh.simplify_quadric_decimation(face_count=target)
        if out.area > 0.9 * area:
            break
        keep *= 2
    else:
        out = mesh
    out.export(dst)
    return len(out.faces)


def build(keep: float) -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    assets = BUILD / "assets"
    assets.mkdir(parents=True)
    xml = (PANDA / "panda.xml").read_text(encoding="utf-8")
    # Only meshes used purely for looks (finger_0 is also the fingers' collision shape).
    used = [(re.search(r'mesh="([^"]+)"', g).group(1), g)
            for g in re.findall(r"<geom [^>]*/>", xml) if "mesh=" in g]
    visual = {m for m, g in used if 'class="visual"' in g}
    visual -= {m for m, g in used if 'class="visual"' not in g}
    n_before = n_after = 0
    for line in re.findall(r"<mesh [^>]*file=\"[^\"]+\"[^>]*/>", xml):
        file = re.search(r'file="([^"]+)"', line).group(1)
        name = re.search(r'name="([^"]+)"', line)
        name = name.group(1) if name else Path(file).stem
        src = PANDA / "assets" / file
        if name in visual:
            n_before += len(trimesh.load(src, force="mesh", process=False).faces)
            n_after += simplify(src, assets / file, keep)
        else:
            shutil.copy(src, assets / file)  # collision meshes: unchanged
    (BUILD / "panda.xml").write_text(xml, encoding="utf-8")
    scene = SCENE.read_text(encoding="utf-8")
    scene = scene.replace('file="../third_party/menagerie/franka_emika_panda/panda.xml"', 'file="panda.xml"')
    scene = scene.replace('meshdir="../third_party/menagerie/franka_emika_panda/assets"', 'meshdir="assets"')
    # The page draws its own background: drop the 5 MB skybox texture (visual only).
    scene = re.sub(r'\s*<texture type="skybox"[^>]*/>', "", scene)
    (BUILD / "scene.xml").write_text(scene, encoding="utf-8")
    print(f"visual mesh faces: {n_before} -> {n_after}")
    return BUILD / "scene.xml"


PHYSICAL = ["body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
            "jnt_range", "jnt_axis", "dof_damping", "dof_armature", "actuator_gainprm",
            "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "geom_contype",
            "geom_conaffinity", "geom_size", "geom_friction", "geom_pos", "geom_quat"]


def check(full: mujoco.MjModel, web: mujoco.MjModel):
    collide = lambda m: (m.geom_contype | m.geom_conaffinity) > 0  # noqa: E731
    for name in PHYSICAL:
        a, b = getattr(full, name), getattr(web, name)
        if name.startswith("geom_"):
            a, b = a[collide(full)], b[collide(web)]
        assert a.shape == b.shape and np.array_equal(a, b), f"{name} differs"
    assert full.opt.timestep == web.opt.timestep and np.array_equal(full.opt.gravity, web.opt.gravity)
    # Same collision meshes, vertex for vertex (geoms are in the same order in both models).
    verts = lambda m, i: m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]  # noqa: E731
    for g in np.flatnonzero(collide(full) & (full.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
        assert np.array_equal(verts(full, full.geom_dataid[g]), verts(web, web.geom_dataid[g]))
    # A scripted run: arm swinging through a sine on every joint, bar drifting and spinning into it.
    states = []
    for m in (full, web):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[:7] = READY_Q
        t = m.joint("target_free")
        d.qpos[t.qposadr[0]:t.qposadr[0] + 3] = [0.45, -0.25, 0.6]
        d.qvel[t.dofadr[0]:t.dofadr[0] + 6] = [0, 0.12, 0, 0.3, 0.2, 0.1]
        out = []
        for i in range(2500):
            d.ctrl[:7] = np.array(READY_Q) + 0.4 * np.sin(np.arange(7) + i * 0.004)
            d.ctrl[7] = 255 if i < 1500 else 0
            mujoco.mj_step(m, d)
            out.append(d.qpos.copy())
        states.append(np.array(out))
    diff = np.abs(states[0] - states[1]).max()
    print(f"scripted 5 s run: max state difference {diff:.2e}")
    assert diff == 0.0, "physics differs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, default=0.06, help="fraction of visual faces kept")
    args = ap.parse_args()
    web_xml = build(args.keep)
    full = mujoco.MjModel.from_xml_path(str(SCENE))
    web = mujoco.MjModel.from_xml_path(str(web_xml))
    check(full, web)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(web, str(OUT), None)
    print(f"saved {OUT}: {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
