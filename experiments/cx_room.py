"""Search a closed room for as many beacons as possible.

Every earlier task was an open field, and the connectome's signature failure
there was to leave: it would end 60 m from a target it had started 27 m from.
That says nothing about navigation.  A sealed room removes the escape, so the
only failures left are hitting something and running out of time.

Six beacons, three to five of them reachable and the rest walled off, which
turns a 0-or-1 outcome into a score and separates an arm that fixates on an
unreachable beacon from one that gives up and finds another.

A beacon is removed from the world once reached, so the arm has to go looking
for the next one rather than circling what it already has.

    planner     greedy: flyable-shortest path to the nearest reachable
                beacon, then the next.  The ceiling.
    P blind     steer at the cue.
    connectome  eyes and the held ORN goal, frozen 66k network.
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

from environment.target_world import (Agent, TargetWorld, room_world,
                                      shortest_path)
from sensors.attraction_cue import AttractionCueSensor, WIDE_RIG
from sensors.connectome_input import ConnectomeInput
from sensors.fisheye import clearance_ahead
from sensors.vp_input import TAU_CRIT
from sim.episode import DT, V_CRUISE

from cx_vision import VisionSub, build

ORNS = ("ORN_DA1", "ORN_VM2", "ORN_VA1v", "ORN_DC3", "ORN_VA6")
MAX_STEPS = 600
START = (0.0, 0.0, 2.0)
R_MAX = math.radians(90.0)       # the drone's yaw rate limit
# ABSOLUTE standoff, in metres, independent of speed.
D_STOP = 0.8                     # nothing allowed nearer than this
# Where braking has to BEGIN, derived rather than picked: stopping distance
# at V_CRUISE is v^2/2a = 2^2/8 = 0.50 m, plus one control cycle of travel
# at 2 m/s = 0.20 m, so 0.70 m of margin on top of D_STOP.  The first value
# tried was 2.5 m, which in a 161-obstacle room fires almost always: it
# removed every collision and replaced them with all nine arms crawling into
# the 600-step cap.
D_FREE = 1.5
# REVERSE, now that the range-sensor ring watches the rear sector.  It was
# removed while the only sensing faced forward.  Without it the accurate
# sensor exposed a trap the inflated lamp estimate had hidden: the drone
# stopped at the standoff, the goal behind the wall kept its steering aimed
# into the wall, and the centroid arm sat at zero speed for 82% of a
# 600-step flight.  Backing off is a drone-side safety reflex, in the same
# layer as the stop itself -- it does not choose the path, the circuit does.
V_BACK = -0.4


def cruise_for(u, brake, rng, rear):
    """Forward speed given the yaw the controller is asking for.

    At 2 m/s and 90 deg/s the minimum turning radius is 1.27 m, so a 4.5 m
    doorway has to be entered nearly straight -- and a fixed-speed follower
    simply cannot make the corner.  That is why the full-information planner
    kept crashing (16 steps in room 0) and why its failures were being read
    as the task being hard.  Slowing into a turn is what a real vehicle does
    and it is applied to EVERY arm here, in the runner, so no arm gets a
    manoeuvre the others do not.

    At quarter speed the radius is 0.32 m, which fits any doorway the
    flyability check admits.

    `blocked` is the second term and it is a different thing entirely: how
    much of the frontal field is darker than sky.  A wall straight ahead is
    left-right symmetric, so it cannot appear in a steering signal at all --
    measured, R-L moves by 2.6e-09 as a wall closes from 12 m to 3 m while
    R+L moves by 5x.  Without a speed channel there is no way to represent
    "stop and turn", which is what a fly does and what the descending
    interface (delta_L, delta_R) has room for.  Same law for every arm.

    A THIRD limit, `rng`, is the absolute one, and it exists because the
    other two have a fixed point at contact.  `blocked` is tau_crit/tau with
    tau = range/speed, so the distance that triggers braking is
    `tau_crit * speed` -- which SHRINKS as the vehicle slows.  Slowing down
    lowers the bar that caused the slowing.  Measured in room 0, steps 37-52:

        step  bearing  u cmd   v now   range
          37     22.5  -10.1    1.08    1.21
          43     19.9  -13.7    0.80    0.70
          47     13.0  -26.6    0.61    0.54
          52     -5.2  -49.7    0.45    0.62   -> contact

    Range sits at 0.5-1.2 m for sixteen steps while speed decays 1.08 ->
    0.45, the vehicle turning hard the whole time.  It is not failing to
    avoid; it is converging on an equilibrium `range = tau_crit * v` whose
    limit as v -> 0 is zero clearance.  The fixed point IS the collision.

    D_STOP breaks it: below 0.8 m the speed is zero no matter what the
    ratio says, and yaw keeps working at zero speed, so the vehicle stops
    and turns -- which is what a fly does at a wall.

    NO DEFAULTS on `blocked` or `rng`.  They had them, and three call sites
    (cx_room's episode loop and both of cx_room10's) went on calling
    `cruise_for(u)` and `cruise_for(u, blocked)` after the standoff was
    added, silently getting rng=inf and therefore no standoff at all, while
    the arm they were being compared against had it.  A required argument
    makes that a crash instead of a quiet asymmetry.
    """
    # Two independent limits, both physical:
    #   turn   a tight turn has to be taken slowly or the radius will not fit
    #   close  speed must not exceed range / tau_safe, or there is no room to
    #          stop or turn out -- and `blocked` carries exactly that ratio
    #          as urgency = tau_safe / tau, so dividing by it IS the limit
    turn = max(0.15, 1.0 - abs(u) / R_MAX)
    # `brake` is a speed multiplier in [0, 1] and it is the only part of
    # this law that is a DECISION rather than a physical limit.  `turn` and
    # the D_STOP gate are the envelope: speed should be maximal unless the
    # turning radius or the stopping distance says otherwise.  Who computes
    # `brake` is what the arms differ in -- the connectome arm reads it off
    # the descending common mode, the references compute it from tau.
    close = max(min(brake, 1.0), 0.0)
    # absolute clearance gate: a pure function of distance, so it cannot be
    # talked down by going slower
    if rng < D_STOP:
        # Blocked ahead: back off if the rear sector is clear, otherwise
        # stop.  Yaw keeps working either way, so the circuit's turn away
        # continues while the vehicle opens up room to complete it.
        return V_BACK if rear > D_STOP else 0.0
    gate = min(max((rng - D_STOP) / (D_FREE - D_STOP), 0.0), 1.0)
    return V_CRUISE * turn * max(close, 0.05) * gate


def sub_world(base, remaining):
    return TargetWorld(target=remaining[0], extra_targets=remaining[1:],
                       obstacles=base.obstacles, bounds=base.bounds)


def hunt(cmd_for, base, heading, sensor, max_steps=MAX_STEPS):
    """One episode.  `cmd_for(world)` is rebuilt whenever a beacon is taken."""
    remaining = list(base.all_targets())
    cur = sub_world(base, remaining)
    cmd = cmd_for(cur)
    ag = Agent(world=cur, start=START, heading=heading)
    found, first, seen, path = 0, None, 0, [ag.p.copy()]
    found_pts = []          # where each beacon was taken, for the picture
    for k in range(max_steps):
        cue = sensor.sense_many(ag.p, ag.heading, cur.all_targets(),
                                cur.obstacles)
        seen += int(cue.valid)
        u = float(cmd(cue, ag.heading, ag))
        # the same clearance the other arms get, from the same compiled
        # sweep -- this loop used to call cruise_for(u) alone and so flew
        # with no standoff while being compared against arms that had one
        sc = sensor.cam.sense(cur, ag.p, ag.heading)
        rng, _ = clearance_ahead(sensor.cam, sc["range"], 0.0, 60.0)
        # `brake` is a speed MULTIPLIER in [0, 1].  This line passed the
        # raw urgency instead -- which is >= 0 and grows as the wall nears --
        # so after `cruise_for` switched from urgency to brake the lamina arm
        # would have flown FASTER the closer it got.
        urg = TAU_CRIT / max(rng / max(ag.v, 0.2), 1e-3)
        ag.step(u, cruise_for(u, 1.0 / max(1.0, urg), rng, sc["rear"]),
                0.0, DT)
        path.append(ag.p.copy())
        if ag.collided:
            break
        i = cur.reached_any(ag.p)
        if i is not None:
            found += 1
            if first is None:
                first = k
            found_pts.append((k, remaining[i].copy()))
            remaining.pop(i)
            if not remaining:
                break
            cur = sub_world(base, remaining)
            ag.world = cur
            cmd = cmd_for(cur)
    return {"found": found, "first": first, "collided": bool(ag.collided),
            "steps": k + 1, "cue_frac": seen / (k + 1),
            "path": np.array(path), "found_pts": found_pts}


def planner_for(world):
    """Greedy: head for the nearest beacon that is actually flyable to."""
    best = None
    for t in world.all_targets():
        probe = TargetWorld(target=t, obstacles=world.obstacles,
                            bounds=world.bounds)
        sp = shortest_path(probe, START, cell=0.3, clearance=1.4)
        if sp is None:
            continue
        L = float(np.linalg.norm(np.diff(np.array(sp)[:, :2], axis=0),
                                 axis=1).sum())
        if best is None or L < best[0]:
            best = (L, np.array(sp))
    pts = best[1] if best else np.array([world.target])

    def f(cue, h, ag):
        # lookahead scales with current speed: far ahead when running, close
        # when crawling through a doorway.  A fixed lookahead aims across
        # walls at exactly the moment it must not.
        la = max(0.7, 0.8 * ag.v)
        d = np.linalg.norm(pts[:, :2] - ag.p[:2], axis=1)
        j = int(np.argmin(d))
        while j < len(pts) - 1 and np.linalg.norm(pts[j, :2] - ag.p[:2]) < la:
            j += 1
        v = pts[j, :2] - ag.p[:2]
        err = (math.atan2(v[1], v[0]) - h + math.pi) % (2 * math.pi) - math.pi
        return float(np.clip(3.0 * err, -R_MAX, R_MAX))
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=6)
    args = ap.parse_args(argv)

    net, ann, sub, rho, nl, nc = build()
    sen = AttractionCueSensor(rig=WIDE_RIG)
    vs = VisionSub(net, ann)
    vs.inp = ConnectomeInput(net, ann, cue_types=ORNS, orn_gain=200.0)
    vs.inp.cue_sensor = sen
    vs.inp.fixed_strength = 0.1

    # normalise the connectome gain the same way as every other run
    vals = []
    for b in (-40., -20., 0., 20., 40.):
        th = np.radians(b)
        wt = TargetWorld(target=np.array([8*np.cos(th), 8*np.sin(th), 2.]),
                         obstacles=[])
        vs.reset()
        for _ in range(40):
            x = vs.turn(wt, np.array(START), 0.)
        vals.append(x)
    gain = -1.2 / ((max(vals) - min(vals)) / 2)
    print("connectome gain %.2e\n" % gain, flush=True)

    rooms = [room_world(seed=s) for s in range(args.rooms)]
    rows = []
    print("%5s %-12s %7s %8s %9s %7s %8s %8s"
          % ("room", "arm", "found", "of", "collided", "steps", "first", "seen"))
    for ri, (w, h, reach) in enumerate(rooms):
        n_ok = sum(reach)
        arms = [("planner", lambda ww: planner_for(ww)),
                ("P blind", lambda ww: (lambda c, hh, a:
                                        2.0 * math.radians(c.bearing_deg)
                                        if c.valid else 0.0))]
        for nm, fac in arms:
            r = hunt(fac, w, h, sen)
            r.update({"room": ri, "arm": nm, "reachable": n_ok})
            rows.append(r)
            print("%5d %-12s %7d %8d %9s %7d %8s %7.0f%%"
                  % (ri, nm, r["found"], n_ok, r["collided"], r["steps"],
                     r["first"] if r["first"] is not None else "-",
                     100*r["cue_frac"]), flush=True)
        vs.reset()
        t = time.perf_counter()
        r = hunt(lambda ww: (lambda c, hh, a: gain * vs.turn(ww, a.p, hh)),
                 w, h, sen)
        r.update({"room": ri, "arm": "connectome", "reachable": n_ok})
        rows.append(r)
        print("%5d %-12s %7d %8d %9s %7d %8s %7.0f%%  (%.0f s)"
              % (ri, "connectome", r["found"], n_ok, r["collided"], r["steps"],
                 r["first"] if r["first"] is not None else "-",
                 100*r["cue_frac"], time.perf_counter()-t), flush=True)
        print("", flush=True)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "path"}
                       for r in rows])
    df.to_csv(REPO / "results/cx_room.csv", index=False)
    print("--- totals over %d rooms " % args.rooms + "-" * 26)
    print("%-12s %8s %10s %10s %9s"
          % ("arm", "found", "of reach", "collided", "first@"))
    for nm in ("planner", "P blind", "connectome"):
        s = df[df.arm == nm]
        print("%-12s %8d %10d %10.2f %9.0f"
              % (nm, s.found.sum(), s.reachable.sum(), s.collided.mean(),
                 s.first.dropna().mean() if s.first.notna().any() else -1))
    np.save(REPO / "results/cx_room_paths.npy",
            np.array([r["path"] for r in rows], dtype=object),
            allow_pickle=True)
    print("wrote results/cx_room.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
