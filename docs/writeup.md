# Catching a drifting object in zero-g

**Simulation study in MuJoCo · [code](../README.md) · [video](media/catch_demo.mp4)**

## Problem

A fixed-base 7-DoF arm (Franka Panda) has to catch a 5 × 5 × 15 cm, 0.5 kg bar that drifts past
it in zero gravity at 2–15 cm/s while tumbling at up to 30°/s, seen only through a noisy pose
sensor, and bring it to rest without knocking it away. In orbit a missed or bounced object
doesn't fall: it leaves. So the goal is not just contact, but a soft capture that ends with the
object held and still.

## Approach

1. **Sensing and estimation.** A simulated 30 Hz pose sensor (5 mm, 2° noise, 5% dropout) feeds a
   constant-velocity Kalman filter for position and velocity, and a sliding-window (0.6 s)
   least-squares fit of rotation vectors for spin. Velocity is good to 1 cm/s after a median
   0.70 s; after 1 s position error is 2.3 mm RMS (raw sensor 8.7 mm).
2. **Intercept planning.** Sweep capture times along the predicted path; for each, try every grasp
   square to a long face of the bar, solve IK from the current posture, and accept the first one
   that the arm can reach in time (minimum-jerk timing at 60% of joint speed limits), that stays
   clear of elbow and wrist singularities, and that stays reachable through the approach and a
   1 s hold while the hand turns with the bar. If none fits yet, plan again every 0.5 s.
3. **Approach and ride-along.** Everything is expressed in the bar's frame: line up behind the bar,
   close to 10 cm, ride along matching its drift and spin, then slide in (minimum-jerk profiles).
   Control: Cartesian feedforward + P → damped pseudo-inverse with a null-space pull towards
   the planned posture → integrated targets for the position servos with lag compensation.
4. **Grasp gate.** Close only if the *estimated* miss is under 1 cm and 2 cm/s; otherwise abort,
   back off, ride along and re-plan (up to twice). Fingers close fast to 5 mm short of the bar,
   then at 5 cm/s through contact.
5. **Compliant stop.** Once held, decelerate drift and spin together along a minimum-jerk
   profile capped at 0.3 m/s² and 1 rad/s², so the grip only supplies a small, smooth force.

## Results (300 randomized throws)

Speed, direction (within 30° of straight across, either side), spin axis and rate, start pose
and sensor noise are all random. "Caught" means held between both fingers at < 0.5 cm/s and
< 2°/s, from ground truth.

| | |
|---|---|
| Caught and brought to rest | **85%** (95% CI 80–88) |
| … when a reachable plan existed | 92% |
| … when the gripper started closing | 99% |
| Contact speed when closing (median / worst) | 1.9 / 2.3 cm/s |
| Kick from the closing fingers (median) | 3.3 cm/s |
| Slip in the grip while braking (median) | 0.75 mm |
| Peak grip force / braking time (median) | 2.5 N / 0.68 s |

![Catch rate vs speed, spin and direction](media/fig_success.png)

Spin barely matters (83–86% across 0–30°/s): matching it is handled by the bar-frame
formulation. Speed does: 93% at 5–8 cm/s, 76% at 11–15 cm/s, where most misses are throws whose
window is shorter than the arm needs (18% of that band have no reachable catch). The other losses: 5% aborted
at the gate (the hand still several cm off; in the cases traced, a wrist joint at its limit or
the arm at full stretch),
1% held but still turning, 1% still retrying at the 12 s limit.

## What the testing changed

- **Self-referencing spin.** Once the fingers touched, the estimator saw the hand moving the bar
  and the hand followed the estimate: spin ran away from 20 to 44°/s. Fix: freeze the target
  motion at the moment closing starts.
- **Gentle close.** Closing in two speeds instead of at once cut the kick from 5.6 to 3.8 cm/s
  median (10.6 → 5.3 worst).
- **Monte Carlo before tuning.** With wider directions the first run caught 65%. The planner
  committed to the quickest grasp (from above), which fails for high bars, and gave up after one
  attempt. Trying every grasp and re-planning after a miss lifted it to 82–85% with no change to
  the grasp itself.

## Limits and next steps

The stop is position control with bounded deceleration, not force control; a wrist
force/torque sensor with admittance control is the real next step, and could let the grasp
gate be less conservative. The approach clips at joint limits rather than avoiding them
(behind several aborts). Planning takes ~0.5 s in pure-Python IK, during which the simulation
waits: a compiled IK or a precomputed reachability map would make it real-time. The object is a
known rigid box seen by an ideal pose sensor; a vision pipeline, unknown inertia and a
free-floating base are the obvious extensions.
