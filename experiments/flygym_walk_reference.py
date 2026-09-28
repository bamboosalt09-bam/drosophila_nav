"""Baseline: does flygym's own walking controller walk?  No connectome here.

This is the step that should have come first.  flygym ships a complete
locomotion layer -- CPG oscillators, preprogrammed step kinematics, retraction
and stumbling corrections, leg adhesion -- and NeuroMechFly v2's descending
interface is two numbers, [delta_L, delta_R], not 24 joint angles.  Building a
joint-level drive instead was the error; a rate model converging to a fixed
point can never produce the phase-shifted rhythm walking needs, but it does
not have to, because the rhythm is the body's job.

`HybridTurningController.step(descending_signal, obs)`:
    amplitude  |delta| per side -> CPG intrinsic_amps
    sign       delta < 0 -> that side's CPG runs backwards

So the whole motor mapping the connectome has to supply is TWO NUMBERS.

Conditions here are the sanity checks that make the later connectome runs
readable: stand still, walk straight, turn each way.

Usage:
    python experiments/flygym_walk_reference.py --duration 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reference"))

import numpy as np

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D

from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)

# [left, right] descending drive.  1.0 is the controller's intrinsic amplitude.
CONDITIONS = {
    "still":       np.array([0.0, 0.0]),
    "forward":     np.array([1.0, 1.0]),
    "turn_right":  np.array([1.0, 0.2]),
    "turn_left":   np.array([0.2, 1.0]),
}


def run(descending: np.ndarray, duration_s: float, steps_obj) -> dict:
    """One fly, one constant descending command, for `duration_s` of fly time."""
    fly = make_locomotion_fly()
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=[0.0, 0.0, 1.0],
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    sim.reset()

    controller = HybridTurningController(timestep=sim.mj_model.opt.timestep,
                                         preprogrammed_steps=steps_obj)
    controller.reset(seed=0)

    body_order = fly.get_bodysegs_order()
    thorax = body_order.index(type(fly).BODY_SEGMENT_CLASS("c_thorax"))
    n = int(round(duration_s / sim.mj_model.opt.timestep))

    p0 = np.asarray(sim.get_body_positions(fly.name))[thorax].copy()
    h0 = None
    phases = []
    for i in range(n):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        if h0 is None:
            h0 = obs.fly_heading.copy()
        action = controller.step(descending, obs)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        if i % 100 == 0:
            phases.append(controller.cpg_network.curr_phases.copy())

    obs = HybridControllerObservation.from_sim(sim, fly.name)
    p1 = np.asarray(sim.get_body_positions(fly.name))[thorax].copy()
    d = p1 - p0
    # heading change about the vertical axis, signed
    yaw = float(np.degrees(np.arctan2(obs.fly_heading[1], obs.fly_heading[0])
                           - np.arctan2(h0[1], h0[0])))
    yaw = (yaw + 180.0) % 360.0 - 180.0
    ph = np.array(phases)
    # tripod: legs 0,2,4 in phase with each other and antiphase to 1,3,5
    tripod = np.abs(np.cos(ph[:, 0] - ph[:, 2])).mean() if len(ph) else np.nan
    return {"forward_mm": float(d[0]), "lateral_mm": float(d[1]),
            "z_mm": float(p1[2]), "yaw_deg": yaw,
            "speed_mm_s": float(np.hypot(d[0], d[1]) / duration_s),
            "tripod_sync": float(tripod)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=0.5, help="seconds of fly time")
    args = ap.parse_args(argv)

    steps_obj = PreprogrammedSteps()
    print("%-11s %10s %10s %9s %9s %8s %7s"
          % ("condition", "forward", "lateral", "speed", "yaw", "z", "tripod"))
    res = {}
    for name, d in CONDITIONS.items():
        t0 = time.perf_counter()
        r = run(d, args.duration, steps_obj)
        res[name] = r
        print("%-11s %9.3f %10.3f %8.2f %9.2f %8.3f %7.3f   (%.0f s)"
              % (name, r["forward_mm"], r["lateral_mm"], r["speed_mm_s"],
                 r["yaw_deg"], r["z_mm"], r["tripod_sync"],
                 time.perf_counter() - t0), flush=True)

    print()
    # the three claims that make this a usable baseline
    assert res["forward"]["speed_mm_s"] > 2.0, \
        "forward walking is slower than 2 mm/s: %r" % res["forward"]
    assert res["forward"]["speed_mm_s"] > 5 * res["still"]["speed_mm_s"], \
        "standing still is not distinguishable from walking"
    assert res["turn_right"]["yaw_deg"] < res["turn_left"]["yaw_deg"], \
        "the two turn commands do not separate: %.2f vs %.2f" % (
            res["turn_right"]["yaw_deg"], res["turn_left"]["yaw_deg"])
    print("baseline ok: the body walks, stands, and turns both ways")
    print("descending interface is 2 numbers; that is all the connectome "
          "must supply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
