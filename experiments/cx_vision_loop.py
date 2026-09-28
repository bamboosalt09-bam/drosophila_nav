"""Closed loop for the 66k lamina-to-CX network.

The blind 532 stalls at reach 0.80 with every failure a collision.  This is
the same task, same seeds, same references -- the only change is that the
controller can now see the obstacles, through 6,199 lamina cells and four
synapses of real wiring rather than through anything hand-written.

References in every table, because a miss is uninterpretable without them:

    P control         cue only, blind.  What the 532 was beaten against.
    P + oracle        ground-truth obstacle positions.  The ceiling vision
                      could buy: measured 0.938 reach, 0.000 collisions.
    straight          never turns.

The gain is NEGATIVE.  The readout is right-minus-left, and a target to the
left (positive bearing) drives the left descending neurons harder, so the
raw readout goes negative exactly when the command must go positive.

ponytail: gain is swept rather than solved for.  The readout scale differs
from the 532 by ~300x, so the 532's gain of 3000 is meaningless here.
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

from sensors.attraction_cue import AttractionCueSensor

from cx_subcircuit import episode
from cx_validate import tasks_for
from cx_vision import VisionSub, build


def oracle_cmd(agent_sink, world):
    """P on the cue, plus repulsion from obstacles handed to it for free."""
    def f(cue, h):
        u = 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0
        p = agent_sink[0].p
        for o in world.obstacles:
            d = np.array([o.x, o.y]) - p[:2]
            r = float(np.linalg.norm(d))
            if r > 4.0 or r < 1e-6:
                continue
            rel = (math.atan2(d[1], d[0]) - h + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) > math.radians(70):
                continue
            u -= math.copysign(1.0, rel) * 3.0 * (4.0 - r) / 4.0
        return u
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--gains", type=float, nargs="*",
                    default=[-3e5, -1e6, -3e6, -1e7])
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, ann, sub, rho, n_lam, n_cx = build()
    vs = VisionSub(net, ann)
    vs.calibrate()
    print("%d neurons, %d edges, rho %.0f -> 0.9; lamina %d, CX %d  (%.0f s)"
          % (net.n, sub.nnz, rho, n_lam, n_cx, time.perf_counter() - t0),
          flush=True)

    sensor = AttractionCueSensor()
    sink = []

    def p_cmd(cue, h):
        return 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0

    rows = []
    print("\n%-5s %-20s %8s %9s %8s %8s"
          % ("seed", "arm", "reached", "collided", "timeout", "dist"))
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        tk = tasks_for(seed, args.episodes)
        arms = [("P control (blind)", lambda w: p_cmd),
                ("P + oracle obstacles", lambda w: oracle_cmd(sink, w)),
                ("straight", lambda w: (lambda cue, h: 0.0))]
        for g in args.gains:
            def mk(w, g=g):
                vs.reset()
                return lambda cue, h: g * vs.turn(w, sink[0].p, h)
            arms.append(("connectome gain %.0e" % g, mk))

        for name, factory in arms:
            t = time.perf_counter()
            res = [episode(factory(w), w, h0, sensor, agent_sink=sink)
                   for w, h0 in tk]
            r = {"seed": seed, "arm": name,
                 "reached": float(np.mean([x["reached"] for x in res])),
                 "collided": float(np.mean([x["collided"] for x in res])),
                 "timeout": float(np.mean([x["timeout"] for x in res])),
                 "dist": float(np.mean([x["dist"] for x in res])),
                 "rate_sat": float(np.mean([x["rate_sat"] for x in res])),
                 "accel_sat": float(np.mean([x["accel_sat"] for x in res])),
                 "secs": time.perf_counter() - t}
            rows.append(r)
            print("%-5d %-20s %8.2f %9.2f %8.2f %8.2f %7.3f %7.3f  (%.0f s)"
                  % (seed, name, r["reached"], r["collided"], r["timeout"],
                     r["dist"], r["rate_sat"], r["accel_sat"], r["secs"]),
                  flush=True)
        print("", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/cx_vision_loop.csv", index=False)
    print("--- pooled " + "-" * 48)
    print("%-20s %8s %9s %8s %8s %7s %7s"
          % ("arm", "reached", "collided", "timeout", "dist", "r_sat", "a_sat"))
    for a in df.arm.unique():
        s = df[df.arm == a]
        print("%-20s %8.3f %9.3f %8.3f %8.2f %7.3f %7.3f"
              % (a, s["reached"].mean(), s["collided"].mean(),
                 s["timeout"].mean(), s["dist"].mean(),
                 s["rate_sat"].mean(), s["accel_sat"].mean()))
    conn = df[df.arm.str.startswith("connectome")]
    b = conn.loc[conn["reached"].idxmax()]
    p = df[df.arm == "P control (blind)"]["collided"].mean()
    print("\nbest connectome: %s -> reached %.3f, collided %.3f "
          "(blind P reference collides %.3f)"
          % (b["arm"], b["reached"], b["collided"], p))
    print("wrote results/cx_vision_loop.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
