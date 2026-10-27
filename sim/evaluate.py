"""Run one throw end to end and measure it against ground truth (used by check_stop and the
Monte Carlo)."""

import time

import mujoco
import numpy as np

from sim.catcher import Catcher, Phase
from sim.env import TargetInit, ZeroGEnv
from sim.intercept import quat_to_mat
from sim.kinematics import rotvec_from_matrix

SETTLE_S = 0.5  # keep watching this long after the arm stops
T_MAX = 12.0
REST_SPEED = 0.005  # m/s
REST_SPIN = 2.0  # deg/s


def bar_contacts(env):
    """Bodies touching the bar and the total normal force on it (N)."""
    m, d = env.model, env.data
    gid = m.geom("target_geom").id
    bodies, force, f6 = set(), 0.0, np.zeros(6)
    for i, c in enumerate(d.contact[: d.ncon]):
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            bodies.add(m.body(m.geom_bodyid[other]).name)
            mujoco.mj_contactForce(m, d, i, f6)
            force += abs(f6[0])
    return bodies, force


def bar_in_hand(env, c):
    """Bar pose in the hand frame: position (m) and rotation matrix."""
    p_bar, q_bar = env.target_pose()
    p, R = c.kin.fk(env.data.qpos[c.qadr])
    return R.T @ (p_bar - p), R.T @ quat_to_mat(q_bar)


def run_throw(seed: int, target: TargetInit | None = None, keep: bool = False) -> dict:
    """One throw (random from `seed` unless `target` is given). Outcome is one of caught (bar at
    rest between both fingers), aborted, no_plan, not_at_rest (grasped but still
    moving or turning in the grip) or timeout (still trying at T_MAX).
    `keep` adds the Catcher and a time log, for plots."""
    wall0 = time.perf_counter()
    env = ZeroGEnv(seed=seed)
    env.reset(target)
    c = Catcher(env, sensor_seed=seed + 1000)
    d = env.data
    log = {k: [] for k in ("t", "speed", "spin", "force", "gap", "dist", "rel_speed", "phase")}
    v_grasp = kick = at_grasp = None
    early_touch = False
    max_track = {}  # largest tracking error per phase (m), before the grasp
    peak_force = peak_decel = 0.0
    pose_brake = slip = None
    v_prev, t_end = None, None
    while d.time < T_MAX:
        c.step()
        env.step()
        v, w_body = env.target_velocity()
        bodies, force = bar_contacts(env)
        if c.phase in (Phase.APPROACH, Phase.RIDE, Phase.CLOSE_IN):
            early_touch |= bool(bodies)
            max_track[c.phase.value] = max(max_track.get(c.phase.value, 0.0), c.track_err)
        if c.phase is Phase.GRASP:
            if v_grasp is None:
                v_grasp = v.copy()
                at_grasp = c.true_errors()
            kick = max(kick or 0.0, np.linalg.norm(v - v_grasp))
        if c.phase in (Phase.GRASP, Phase.BRAKE, Phase.STOPPED):
            peak_force = max(peak_force, force)
        if c.phase is Phase.BRAKE:
            if pose_brake is None:
                pose_brake = bar_in_hand(env, c)
            if v_prev is not None:
                peak_decel = max(peak_decel, np.linalg.norm(v - v_prev) / env.model.opt.timestep)
        v_prev = v.copy()
        if keep and c.plan is not None and c.phase is not Phase.WAIT:
            log["t"].append(d.time)
            log["speed"].append(np.linalg.norm(v))
            log["spin"].append(np.rad2deg(np.linalg.norm(w_body)))
            log["force"].append(force)
            log["gap"].append(d.qpos[c.fingers].sum())
            p_bar = env.target_pose()[0]
            log["dist"].append(np.linalg.norm(c.kin.fk(d.qpos[c.qadr])[0] - p_bar))
            log["rel_speed"].append(c.true_errors()[1])
            log["phase"].append(c.phase.value)
        if c.phase is Phase.STOPPED and t_end is None:
            t_end = d.time + SETTLE_S
            if pose_brake is not None:
                p1, R1 = bar_in_hand(env, c)
                slip = (float(np.linalg.norm(p1 - pose_brake[0])),
                        float(np.rad2deg(np.linalg.norm(rotvec_from_matrix(R1 @ pose_brake[1].T)))))
        if c.phase is Phase.FAILED or (c.gave_up and d.time > c.t_abort + 0.3):
            break
        if t_end is not None and d.time >= t_end:
            break
    v, w_body = env.target_velocity()
    bodies, _ = bar_contacts(env)
    final_speed, final_spin = float(np.linalg.norm(v)), float(np.rad2deg(np.linalg.norm(w_body)))
    if c.phase is Phase.FAILED:
        outcome = "no_plan"
    elif c.phase is Phase.ABORT:
        outcome = "aborted"
    elif (c.phase is Phase.STOPPED and {"left_finger", "right_finger"} <= bodies
          and final_speed < REST_SPEED and final_spin < REST_SPIN):
        outcome = "caught"
    elif c.phase in (Phase.GRASP, Phase.BRAKE, Phase.STOPPED):
        outcome = "not_at_rest"
    else:
        outcome = "timeout"
    r = dict(
        seed=seed, outcome=outcome, phase=c.phase.value, early_touch=early_touch,
        t_plan=c.t_start if c.plan is not None else None,
        t_catch=c.plan.t if c.plan is not None else None,
        reach_time=c.plan.reach_time if c.plan is not None else None,
        grasp_dist=at_grasp[0] if at_grasp else None,  # true, when the gripper started closing
        grasp_rel_speed=at_grasp[1] if at_grasp else None,
        grasp_check=c.grasp_check,  # estimated (distance, relative speed)
        abort_from=getattr(c, "abort_from", None),
        kick=kick, slip_mm=slip[0] * 1000 if slip else None, slip_deg=slip[1] if slip else None,
        peak_force=peak_force if v_grasp is not None else None,
        peak_decel=peak_decel if pose_brake is not None else None,
        brake_T=getattr(c, "brake_T", None), v_grasp=v_grasp, n_plans=c.n_plans,
        aborts=c.aborts,
        **{f"track_{k}": max_track.get(k) for k in ("approach", "ride", "close_in")},
        final_speed=final_speed, final_spin=final_spin,
        sim_time=float(d.time), wall_time=time.perf_counter() - wall0,
    )
    if keep:
        r["catcher"] = c
        r["log"] = {k: np.array(v) for k, v in log.items()}
    return r
