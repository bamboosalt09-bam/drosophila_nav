"""Ten closed rooms, fisheye sensing, rho 0.50.

Everything fixed today in one run: one fisheye image per step feeding both
the lamina and a blob tracker, the cue held as an allocentric goal, rho at
0.50 so the circuit settles inside its control cycle, and the drone yaw
limits actually enforced with speed scaled to turn demand.

References on the identical rooms and the identical cue: the greedy planner
(full information, the ceiling) and P control on the cue alone.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch

from core.flywire_rate import activity
from environment.target_world import Agent, TargetWorld, room_world
from sensors.connectome_input import ConnectomeInput
from sim.episode import DT

from sensors.fisheye import FisheyeCamera, contrast_centroid

from cx_room import ORNS, R_MAX, START, cruise_for, planner_for
from sensors.fisheye import clearance_ahead
from cx_vision import VisionSub, build

MAX_STEPS = 600


def connectome_gain(vs, net, w, p0):
    """Gain AND zero point, measured in the room the arm will fly.

    The gain alone is not enough.  Measured on this readout: offset/range
    0.59, so the command sits on one side of zero and the body keeps turning
    the same way even when the target crosses to the other side -- a spiral
    into a wall, which is what 0/5 with corr +0.86 looks like.

    The zero is taken as the sweep value at zero bearing, interpolated from
    the standing sweep that already runs here.  It is one constant, measured
    once per room the same way the gain is, not fitted to the flight.
    """
    vals = []
    for hd in range(-180, 180, 20):
        vs.reset()
        vs.inp.reset()
        d, _ = vs.inp.drive(w, p0, math.radians(hd))
        with torch.no_grad():
            for _ in range(vs.steps):
                vs.v = net.step(vs.v, d, vs.dt_ms)
            r = activity(vs.v).numpy().ravel()
        cue = vs.inp.tracker.update(
            vs.inp.cam.blobs(vs.inp.cam.render(w, p0, math.radians(hd))),
            vs.inp.cam.az_span)
        vals.append((cue.bearing_deg if cue.valid else np.nan,
                     float(r[vs.out_r].mean() - r[vs.out_l].mean())))
    g = np.array([v for v in vals if not np.isnan(v[0])])
    if len(g) < 4 or g[:, 1].std() == 0:
        return 0.0, float("nan")
    cc = float(np.corrcoef(g[:, 0], g[:, 1])[0, 1])
    rng = g[:, 1].max() - g[:, 1].min()
    order = np.argsort(g[:, 0])
    zero = float(np.interp(0.0, g[order, 0], g[order, 1]))
    return math.copysign(1.2 / (rng / 2), cc), cc, zero


def hunt_connectome(vs, net, w, heading, gain, zero=0.0):
    vs.reset()
    vs.inp.reset()
    ag = Agent(world=w, start=START, heading=heading)
    rem = list(w.all_targets())
    found, first, bs, us, seen = 0, None, [], [], 0
    for k in range(MAX_STEPS):
        d, trace = vs.inp.drive(ag.world, ag.p, ag.heading)
        with torch.no_grad():
            for _ in range(vs.steps):
                vs.v = net.step(vs.v, d, vs.dt_ms)
            r = activity(vs.v).numpy().ravel()
        u = gain * (float(r[vs.out_r].mean() - r[vs.out_l].mean())
                    - zero)
        if trace.cue_valid:
            seen += 1
            bs.append(vs.inp._goal_world - ag.heading)
            us.append(u)
        # `eye_dark_fraction` now carries the frontal blockage, the same
        # number the references get: identical information, different
        # processing, which is the only comparison worth making
        # the standoff needs a RANGE, not just a blockage fraction; this
        # loop used to pass only the fraction and so flew with rng=inf
        sc = vs.inp.cam.sense(ag.world, ag.p, ag.heading)
        rng, _ = clearance_ahead(vs.inp.cam, sc["range"], 0.0, 60.0)
        # urgency -> brake multiplier; passing the urgency itself made the
        # arm speed UP as the frontal field darkened
        ag.step(u, cruise_for(u, 1.0 / max(1.0, trace.eye_dark_fraction),
                              rng, sc["rear"]), 0.0, DT)
        if ag.collided:
            break
        i = ag.world.reached_any(ag.p)
        if i is not None:
            found += 1
            if first is None:
                first = k
            rem.pop(i)
            if not rem:
                break
            ag.world = TargetWorld(target=rem[0], extra_targets=rem[1:],
                                   obstacles=w.obstacles, bounds=w.bounds)
    corr = (float(np.corrcoef(bs, us)[0, 1])
            if len(bs) > 3 and np.std(us) > 0 else float("nan"))
    return {"found": found, "first": first, "steps": k + 1,
            "collided": bool(ag.collided), "cue_frac": seen / (k + 1),
            "corr": corr}


def hunt_simple(cmd_for, w, heading, k_yaw=0.05):
    """Reference arm: the SAME image, no target detection.

    It used to be handed `cue.bearing_deg` -- a bearing produced by my own
    detector, tracker and target-selection logic -- and all it did was
    multiply.  That is not a baseline, it is my algorithm with a gain on it.
    Here it gets what the connectome gets, the image, and turns the whole
    weighted centroid into a command with one constant.
    """
    cam = FisheyeCamera()
    ag = Agent(world=w, start=START, heading=heading)
    rem = list(w.all_targets())
    cmd = cmd_for(ag.world) if cmd_for is not None else None
    found, first, seen = 0, None, 0
    for k in range(MAX_STEPS):
        sc = cam.sense(ag.world, ag.p, ag.heading)
        bearing, blocked, weight = sc["bearing"], sc["blocked"], sc["weight"]
        seen += int(weight > 1e-6)
        if cmd is None:
            u = float(np.clip(k_yaw * bearing, -R_MAX, R_MAX))
        else:
            u = float(cmd(None, ag.heading, ag))
        rng, _ = clearance_ahead(cam, sc["range"], 0.0, 60.0)
        ag.step(u, cruise_for(u, 1.0 / max(1.0, blocked), rng, sc["rear"]),
                0.0, DT)
        if ag.collided:
            break
        i = ag.world.reached_any(ag.p)
        if i is not None:
            found += 1
            if first is None:
                first = k
            rem.pop(i)
            if not rem:
                break
            ag.world = TargetWorld(target=rem[0], extra_targets=rem[1:],
                                   obstacles=w.obstacles, bounds=w.bounds)
            if cmd_for is not None:
                cmd = cmd_for(ag.world)
    return {"found": found, "first": first, "steps": k + 1,
            "collided": bool(ag.collided), "cue_frac": seen / (k + 1),
            "corr": float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=10)
    args = ap.parse_args(argv)

    net, ann, sub, rho, nl, nc = build()
    vs = VisionSub(net, ann)
    vs.inp = ConnectomeInput(net, ann, cue_types=ORNS, orn_gain=200.0)
    vs.inp.fixed_strength = 0.1
    p0 = np.array(START)

    rows = []
    print("%5s %-12s %6s %4s %9s %7s %8s %7s"
          % ("room", "arm", "found", "of", "collided", "steps", "first", "seen"))
    for ri in range(args.rooms):
        w, h, reach = room_world(seed=ri)
        n_ok = sum(reach)
        gain, cc, zero = connectome_gain(vs, net, w, p0)
        t = time.perf_counter()
        r = hunt_connectome(vs, net, w, h, gain, zero)
        r.update({"room": ri, "arm": "connectome", "reachable": n_ok,
                  "gain": gain, "sweep_corr": cc, "zero": zero})
        rows.append(r)
        print("%5d %-12s %6d %4d %9s %7d %8s %6.0f%%  (%.0f s, gain %.1e, "
              "sweep %+.2f, flight %+.2f, zero %.1e)"
              % (ri, "connectome", r["found"], n_ok, r["collided"], r["steps"],
                 r["first"] if r["first"] is not None else "-",
                 100*r["cue_frac"], time.perf_counter()-t, gain, cc,
                 r["corr"], zero), flush=True)

        for nm, fac in (("planner", planner_for), ("centroid", None)):
            r = hunt_simple(fac, w, h)
            r.update({"room": ri, "arm": nm, "reachable": n_ok})
            rows.append(r)
            print("%5d %-12s %6d %4d %9s %7d %8s %6.0f%%"
                  % (ri, nm, r["found"], n_ok, r["collided"], r["steps"],
                     r["first"] if r["first"] is not None else "-",
                     100*r["cue_frac"]), flush=True)
        print("", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/cx_room10.csv", index=False)
    print("--- totals over %d rooms " % args.rooms + "-" * 28)
    print("%-12s %8s %10s %10s %9s %9s"
          % ("arm", "found", "of reach", "collided", "first@", "seen"))
    for nm in ("planner", "connectome", "centroid"):
        s = df[df.arm == nm]
        print("%-12s %8d %10d %10.2f %9.0f %8.0f%%"
              % (nm, s.found.sum(), s.reachable.sum(), s.collided.mean(),
                 s.first.dropna().mean() if s.first.notna().any() else -1,
                 100*s.cue_frac.mean()))
    print("wrote results/cx_room10.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
