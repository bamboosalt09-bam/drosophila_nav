"""How tight does the drone have to be before the connectome stops winning?

"드론 자체 거동 허용 범위 내에서만 이동하게 할 때 얼마나 기존 인공지능에
비해서 잘 이동하냐" is not answered by one body.  It is a curve.

Two facts make the sweep necessary rather than decorative:

  - Until today NO limit was enforced at all.  `YawPlantParams` defaults to
    r_max = alpha_max = inf, tau_r = 0, and every experiment built its Agent
    without passing anything.
  - Putting a real small-quadrotor limit on (90 deg/s, 5 rad/s^2, 0.08 s)
    changed the outcome by nothing: reach 0.875 -> 0.879, distance 1.32 ->
    1.33 m.  Yet the connectome sits at the yaw ACCELERATION limit 26.9% of
    the time against 1.1% for the P reference.  Saturating a quarter of the
    time and losing no performance means the limit is still loose -- the
    clipped part of the command was not carrying anything.

So tighten one axis at a time and find where it starts to cost something,
and whether it costs the two arms the same.  If the connectome degrades
faster, its advantage was bought with body it does not have; if slower, the
inductive bias is robust to a worse body, which is the interesting result.

ponytail: one axis at a time, not a grid.  A 2D sweep is 25 cells at ~2 min
each; the axes are close to independent here and the marginal ones locate
the knee well enough to decide whether the full grid is worth it.
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

from body.yaw_plant import YawPlantParams
from sensors.attraction_cue import AttractionCueSensor

from cx_subcircuit import episode
from cx_validate import tasks_for
from cx_vision import VisionSub, build
from cx_vision_loop import oracle_cmd

GAIN = -3e6          # frozen, chosen before any limit existed


def bodies():
    """(label, params).  The first is the unconstrained baseline."""
    out = [("unconstrained", YawPlantParams())]
    for a in (5.0, 2.0, 1.0, 0.5):
        out.append(("alpha %.2f" % a,
                    YawPlantParams(r_max=math.radians(90.0), alpha_max=a,
                                   tau_r=0.08)))
    for r in (45.0, 25.0):
        out.append(("r_max %.0f deg/s" % r,
                    YawPlantParams(r_max=math.radians(r), alpha_max=5.0,
                                   tau_r=0.08)))
    for t in (0.2, 0.4):
        out.append(("tau %.2f s" % t,
                    YawPlantParams(r_max=math.radians(90.0), alpha_max=5.0,
                                   tau_r=t)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, ann, sub, rho, n_lam, n_cx = build()
    vs = VisionSub(net, ann)
    print("%d neurons; gain %.0e frozen  (%.0f s)\n"
          % (net.n, GAIN, time.perf_counter() - t0), flush=True)

    sensor = AttractionCueSensor()
    sink = []

    def p_cmd(cue, h):
        return 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0

    rows = []
    print("%-16s %-14s %8s %9s %8s %7s %7s"
          % ("body", "arm", "reached", "collided", "dist", "r_sat", "a_sat"))
    for blabel, bp in bodies():
        for seed in range(2, 2 + args.seeds):
            tk = tasks_for(seed, args.episodes)
            arms = [("connectome", lambda w: (vs.reset() or
                                              (lambda cue, h:
                                               GAIN * vs.turn(w, sink[0].p, h)))),
                    ("P blind", lambda w: p_cmd),
                    ("P + oracle", lambda w: oracle_cmd(sink, w))]
            for name, factory in arms:
                res = [episode(factory(w), w, h0, sensor, agent_sink=sink,
                               yaw_params=bp) for w, h0 in tk]
                rows.append({
                    "body": blabel, "arm": name, "seed": seed,
                    "reached": float(np.mean([x["reached"] for x in res])),
                    "collided": float(np.mean([x["collided"] for x in res])),
                    "dist": float(np.mean([x["dist"] for x in res])),
                    "rate_sat": float(np.mean([x["rate_sat"] for x in res])),
                    "accel_sat": float(np.mean([x["accel_sat"] for x in res]))})
        for name in ("connectome", "P blind", "P + oracle"):
            s = pd.DataFrame(rows)
            s = s[(s.body == blabel) & (s.arm == name)]
            print("%-16s %-14s %8.3f %9.3f %8.2f %7.3f %7.3f"
                  % (blabel, name, s.reached.mean(), s.collided.mean(),
                     s.dist.mean(), s.rate_sat.mean(), s.accel_sat.mean()),
                  flush=True)
        print("", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/cx_body_sweep.csv", index=False)
    print("--- connectome advantage over the blind P reference, by body " + "-" * 6)
    print("%-16s %10s %10s %10s" % ("body", "d_reached", "d_dist", "a_sat"))
    for blabel, _ in bodies():
        c = df[(df.body == blabel) & (df.arm == "connectome")]
        p = df[(df.body == blabel) & (df.arm == "P blind")]
        print("%-16s %+10.3f %+10.2f %10.3f"
              % (blabel, c.reached.mean() - p.reached.mean(),
                 c.dist.mean() - p.dist.mean(), c.accel_sat.mean()))
    print("wrote results/cx_body_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
