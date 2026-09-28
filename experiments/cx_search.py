"""The task as it was actually specified: find the target in dense clutter.

"복잡한 3D 환경에서 직접 시뮬레이션" and "스스로 목표물을 찾는 것".  What
had been measured until now was neither.  `corridor_world` put 2-14 pillars
in front of a target 12 m dead ahead, and the numbers say plainly what that
was: the target sat inside the camera frame at t=0 in 94% of episodes, the
median start bearing was 14 deg, and with no pillar on the line all three
arms drew the same straight path.  Nothing searched for anything.

    task                     old          now
    obstacles                2-14         45
    target distance          12 m         27.7 m
    |start bearing| median   14 deg       74 deg   (up to 170)
    target visible at t=0    94%          27%

Every map is proven solvable by grid BFS before it is used, so a failure is
always the controller's and never the generator's.

Four arms, and the two new ones exist because the old references stop being
references the moment the target is not already visible:

    planner      full information: grid-shortest path, followed.  This is the
                 ceiling.  Not a controller, a yardstick.
    P + oracle   given obstacle positions, but must still SEE the target.
    P blind      cue only.
    connectome   eyes and cue, the same frozen 66k network.

None of these four has a search behaviour.  A policy that cannot see the
target commands zero and flies straight, which in a 27 m field is a good way
to see nothing for 400 steps.  That is the expected result, it is worth
having on record, and it names the next question precisely: where does
searching come from?
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

from environment.target_world import Agent, cluttered_world, shortest_path
from sensors.attraction_cue import AttractionCueSensor, WIDE_RIG
from sim.episode import DT, V_CRUISE

from cx_vision import VisionSub, build

GAIN = -3e6
MAX_STEPS = 400          # 27.7 m at 2 m/s is 139 steps; the rest is for search


def planner_cmd(world, start=(0.0, 0.0, 2.0), lookahead=2.0):
    """Follow the grid-shortest path.  Full information, the ceiling."""
    path = shortest_path(world, start)
    pts = np.array(path) if path else np.array([world.target])

    def f(cue, h, ag):
        d = np.linalg.norm(pts[:, :2] - ag.p[:2], axis=1)
        i = int(np.argmin(d))
        j = i
        while j < len(pts) - 1 and np.linalg.norm(pts[j, :2] - ag.p[:2]) < lookahead:
            j += 1
        v = pts[j, :2] - ag.p[:2]
        err = (math.atan2(v[1], v[0]) - h + math.pi) % (2 * math.pi) - math.pi
        return 2.0 * err
    return f


def oracle_cmd(world):
    def f(cue, h, ag):
        u = 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0
        for o in world.obstacles:
            d = np.array([o.x, o.y]) - ag.p[:2]
            r = float(np.linalg.norm(d))
            if r > 4.0 or r < 1e-6:
                continue
            rel = (math.atan2(d[1], d[0]) - h + math.pi) % (2*math.pi) - math.pi
            if abs(rel) > math.radians(70):
                continue
            u -= math.copysign(1.0, rel) * 3.0 * (4.0 - r) / 4.0
        return u
    return f


def run(cmd_fn, world, heading, sensor, max_steps=MAX_STEPS):
    agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=heading)
    n_seen = 0
    for k in range(max_steps):
        cue = sensor.sense(agent.p, agent.heading, world.target,
                           world.obstacles)
        n_seen += int(cue.valid)
        agent.step(float(cmd_fn(cue, agent.heading, agent)), V_CRUISE, 0.0, DT)
        if agent.collided or world.reached(agent.p):
            break
    ok = bool(world.reached(agent.p)) and not agent.collided
    p = agent.path_array()
    sp = shortest_path(world)
    opt = (float(np.linalg.norm(np.diff(np.array(sp)[:, :2], axis=0), axis=1).sum())
           if sp else np.nan)
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return {"reached": ok, "collided": bool(agent.collided),
            "timeout": (not ok) and (not agent.collided) and k + 1 >= max_steps,
            "dist": float(np.linalg.norm(world.target - agent.p)),
            "steps": k + 1, "cue_frac": n_seen / (k + 1),
            # against the grid-shortest path, not the straight line: in
            # clutter the straight line is usually through a pillar
            "detour": length / opt if (ok and opt and opt > 0) else np.nan,
            "rate_sat": float(agent.plant.rate_saturation_fraction),
            "accel_sat": float(agent.plant.accel_saturation_fraction)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=16)
    ap.add_argument("--first", type=int, default=0)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, ann, sub, rho, n_lam, n_cx = build()
    vs = VisionSub(net, ann)
    vs_b = VisionSub(net, ann)
    vs_b.inp.drive_gain = 0.0          # eyes off, same network, same cue
    print("%d neurons; %d maps, up to %d steps  (%.0f s)\n"
          % (net.n, args.maps, MAX_STEPS, time.perf_counter() - t0), flush=True)
    # EVERY arm gets the compound eye's field.  With a +-45 deg camera the
    # connectome was being handed a target the references could not see at
    # all -- on map 14, 24 lamina cells had it at -74 deg while the camera
    # called it invisible -- so the comparison was measuring the lens.
    sensor = AttractionCueSensor(rig=WIDE_RIG)
    vs.inp.cue_sensor = sensor
    vs_b.inp.cue_sensor = sensor

    tasks = [cluttered_world(seed=s)
             for s in range(args.first, args.first + args.maps)]
    d0 = np.mean([np.linalg.norm(w.target - np.array([0., 0., 2.]))
                  for w, _ in tasks])
    vis0 = np.mean([sensor.sense(np.array([0., 0., 2.]), h, w.target,
                                 w.obstacles).valid for w, h in tasks])
    print("target %.1f m away, visible at t=0 in %.0f%% of maps\n"
          % (d0, 100 * vis0), flush=True)

    def conn(w):
        vs.reset()
        return lambda cue, h, ag: GAIN * vs.turn(w, ag.p, h)

    def conn_blind(w):
        # THE control for the emergent-search claim.  Identical network,
        # identical cue, eyes switched off.  If the search survives this it
        # was never visual; if it disappears, it was.
        vs_b.reset()
        return lambda cue, h, ag: GAIN * vs_b.turn(w, ag.p, h)

    arms = [("planner (full info)", planner_cmd),
            ("P + oracle", lambda w: oracle_cmd(w)),
            ("P blind", lambda w: (lambda cue, h, ag:
                                   2.0 * math.radians(cue.bearing_deg)
                                   if cue.valid else 0.0)),
            ("connectome", conn),
            ("connectome no eyes", conn_blind)]

    rows, per_map = [], []
    print("%-20s %8s %9s %8s %8s %8s %8s"
          % ("arm", "reached", "collided", "timeout", "dist", "detour",
             "cue_frac"))
    for name, fac in arms:
        t = time.perf_counter()
        res = [run(fac(w), w, h, sensor) for w, h in tasks]
        for mi, x in enumerate(res):
            per_map.append(dict(x, arm=name, map=args.first + mi))
        r = {k: float(np.nanmean([x[k] for x in res]))
             for k in ("reached", "collided", "timeout", "dist", "detour",
                       "cue_frac", "accel_sat")}
        r["arm"] = name
        rows.append(r)
        print("%-20s %8.3f %9.3f %8.3f %8.2f %8.3f %8.3f   (%.0f s)"
              % (name, r["reached"], r["collided"], r["timeout"], r["dist"],
                 r["detour"], r["cue_frac"], time.perf_counter() - t),
              flush=True)

    pd.DataFrame(rows).to_csv(REPO / "results/cx_search.csv", index=False)
    print("\nwrote results/cx_search.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
