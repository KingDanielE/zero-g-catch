"""Zero-g catch environment: a fixed-base Panda and one free-drifting, spinning bar."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).with_name("scene.xml")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
# Ready posture: tool point at (0.40, 0, 0.72) m, gripper pointing down, above and behind where
# the bar drifts past (the Menagerie "home" pose puts the hand right in its path).
READY_Q = [0.0, -0.169, 0.0, -1.188, 0.0, 1.019, -0.785]


@dataclass
class TargetInit:
    pos: np.ndarray  # m, world frame
    quat: np.ndarray  # w, x, y, z
    vel: np.ndarray  # m/s, world frame
    omega: np.ndarray  # rad/s, body frame (MuJoCo free joint convention)


class ZeroGEnv:
    def __init__(self, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        jnt = self.model.joint("target_free")
        self.target_qadr = jnt.qposadr[0]  # 7 values: pos(3) + quat(4)
        self.target_vadr = jnt.dofadr[0]  # 6 values: lin vel(3) + ang vel(3)
        self.target_body = self.model.body("target").id
        self.hand_body = self.model.body("hand").id

    def random_target(
        self,
        speed=(0.02, 0.15),
        spin_deg=(0.0, 30.0),
        start=(0.6, -0.5, 0.5),
    ) -> TargetInit:
        """Object starts near `start`, drifts roughly across the arm's workspace (+y)."""
        pos = np.array(start) + self.rng.uniform(-0.05, 0.05, 3)
        heading = np.array([self.rng.uniform(-0.3, 0.3), 1.0, self.rng.uniform(-0.2, 0.2)])
        vel = heading / np.linalg.norm(heading) * self.rng.uniform(*speed)
        axis = self.rng.normal(size=3)
        omega = axis / np.linalg.norm(axis) * np.deg2rad(self.rng.uniform(*spin_deg))
        quat = self.rng.normal(size=4)
        return TargetInit(pos, quat / np.linalg.norm(quat), vel, omega)

    def reset(self, target: TargetInit | None = None) -> TargetInit:
        target = target or self.random_target()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[:7] = READY_Q
        self.data.ctrl[:7] = READY_Q
        q, v = self.target_qadr, self.target_vadr
        self.data.qpos[q : q + 3] = target.pos
        self.data.qpos[q + 3 : q + 7] = target.quat
        self.data.qvel[v : v + 3] = target.vel
        self.data.qvel[v + 3 : v + 6] = target.omega
        mujoco.mj_forward(self.model, self.data)
        return target

    def step(self, n: int = 1):
        mujoco.mj_step(self.model, self.data, nstep=n)

    # Ground truth (the simulated sensor in hour 2 adds noise on top of these).
    def target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.target_body].copy(), self.data.xquat[self.target_body].copy()

    def target_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.target_vadr
        return self.data.qvel[v : v + 3].copy(), self.data.qvel[v + 3 : v + 6].copy()

    def hand_pos(self) -> np.ndarray:
        return self.data.xpos[self.hand_body].copy()
