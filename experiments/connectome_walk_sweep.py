"""Does the connectome walk the fly, and does its own asymmetry steer it?

The descending readout comes out at ~7e-3 while the CPG's intrinsic amplitude
is 1.0, so a scale constant is needed.  That constant is a calibration knob,
not a free parameter to fit against the answer: it is set once so that the
typical descending drive lands near the controller's own working amplitude,
and then the SAME value is used for every condition.

Three questions, and they need different controls:

  1. does it walk?          gain 0 against gain g.  Gain 0 is the standing
                            baseline, and it is not the same as "no brain" --
                            the brain still runs, it just cannot reach the CPG.
  2. does it steer?         is yaw different from the straight-walking
                            reference, and does it depend on what is seen?
  3. is the steering real?  `swap_sides` exchanges the two descending
                            populations.  If yaw is unchanged by the swap, the
                            turn is not coming from the left-right difference
                            at all and is an artifact of something common to
                            both sides.

The swap is the important one.  The left-right difference has been one-sided
in every measurement so far, so a fly that turns consistently is exactly what
a BIAS would produce as well as what steering would produce.  Only the swap
separates those two.

Usage:
    python experiments/connectome_walk_sweep.py --duration 1.0
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
import pandas as pd

from flygym.vision.retina import Retina
from flygym_demo.complex_terrain import PreprogrammedSteps

import sensors.flywire_eye as eye
import sensors.flygym_bridge as bridge
from core.flywire_rate import FlyWireRate
from core.malecns import load_malecns
from sim.flygym_walk_loop import (
    BRAIN_DT_MS, ConnectomeWalkLoop, WalkTrace, make_sim_and_controller)


def run(net, ann, lat, omm, steps_obj, duration_s: float,
        descending_gain: float, swap_sides: bool, seed: int = 0) -> dict:
    sim, fly, ctrl = make_sim_and_controller(steps_obj, seed=seed)
    loop = ConnectomeWalkLoop(net, ann, lat, omm, sim, fly, ctrl,
                              descending_gain=descending_gain,
                              swap_sides=swap_sides)
    trace = WalkTrace()
    n = int(round(duration_s * 1000.0 / BRAIN_DT_MS))
    for _ in range(n):
        loop.step(trace)
    a = trace.as_arrays()
    p, d = a["position"], a["descending"]
    disp = p[-1][:2] - p[0][:2]
    return {"gain": descending_gain, "swap": swap_sides,
            "speed_mm_s": float(np.hypot(*disp) / duration_s),
            "forward_mm": float(disp[0]), "lateral_mm": float(disp[1]),
            "yaw_deg": float(trace.yaw_deg()[-1]),
            "z_mm": float(p[-1][2]),
            "delta_l_mean": float(d[:, 0].mean()),
            "delta_r_mean": float(d[:, 1].mean()),
            "delta_diff_mean": float((d[:, 1] - d[:, 0]).mean()),
            "delta_diff_min": float((d[:, 1] - d[:, 0]).min()),
            "delta_diff_max": float((d[:, 1] - d[:, 0]).max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--gains", type=float, nargs="+",
                    default=[0.0, 50.0, 150.0, 400.0])
    ap.add_argument("--out", default="results/connectome_walk.csv")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)
    omm, _ = bridge.join(lat, Retina())
    steps_obj = PreprogrammedSteps()
    print("ready (%.1f s)" % (time.perf_counter() - t0), flush=True)

    print("%-7s %-5s %8s %9s %9s %8s %11s"
          % ("gain", "swap", "speed", "forward", "lateral", "yaw", "dR-dL"))
    recs = []
    for g in args.gains:
        for swap in ((False,) if g == 0 else (False, True)):
            t0 = time.perf_counter()
            r = run(net, ann, lat, omm, steps_obj, args.duration, g, swap)
            recs.append(r)
            print("%-7g %-5s %8.2f %9.3f %9.3f %8.2f %11.3e   (%.0f s)"
                  % (g, swap, r["speed_mm_s"], r["forward_mm"],
                     r["lateral_mm"], r["yaw_deg"], r["delta_diff_mean"],
                     time.perf_counter() - t0), flush=True)

    df = pd.DataFrame(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nwrote %s" % args.out)

    print("--- verdict " + "-" * 50)
    still = df[df["gain"] == 0].iloc[0]
    moving = df[(df["gain"] > 0) & (~df["swap"])]
    best = moving.loc[moving["speed_mm_s"].idxmax()]
    print("gain 0 (cannot reach CPG): speed %.2f mm/s, yaw %+.1f deg"
          % (still["speed_mm_s"], still["yaw_deg"]))
    print("best gain %g:              speed %.2f mm/s, yaw %+.1f deg"
          % (best["gain"], best["speed_mm_s"], best["yaw_deg"]))
    if best["speed_mm_s"] > 2.0 * max(still["speed_mm_s"], 1e-9):
        print("=> the connectome DOES drive walking")
    else:
        print("=> walking is NOT distinguishable from standing")

    for g in sorted(set(df[df["gain"] > 0]["gain"])):
        a = df[(df["gain"] == g) & (~df["swap"])].iloc[0]
        b = df[(df["gain"] == g) & (df["swap"])].iloc[0]
        print("gain %-6g yaw %+8.2f (normal) vs %+8.2f (sides swapped), "
              "delta %.2f deg" % (g, a["yaw_deg"], b["yaw_deg"],
                                  abs(a["yaw_deg"] - b["yaw_deg"])))
    print("a swap that does not change yaw means the turn is a bias, "
          "not steering")
    assert len(df) == len(args.gains) * 2 - 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
