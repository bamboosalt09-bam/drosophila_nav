"""What is the 20% that does not arrive?

Reach sits at 0.796 and stops there, for the connectome AND for the P
reference, which is the clue: a ceiling both arms hit at the same height is a
property of the TASK or the SENSING, not of either controller.

Three candidates, and they need different fixes, so they get separated rather
than guessed at:

    collision   flew into an obstacle.  Neither arm can see obstacles -- the
                532-neuron subcircuit contains zero visual neurons (checked:
                all 532 are cb_intrinsic, 0 of the 6,199 lamina cells) and the
                P reference is given only the cue.  This is the cost of a
                blind controller in a cluttered world.
    blind       the target was occluded for much of the run, so there was no
                cue to steer on.  `AttractionCueSensor` reports invalid when
                the centroid is hidden, and a blind agent flies straight.
    timeout     never arrived, never hit anything -- ran out of steps.  That
                is a speed or a circling problem, not a sensing one.

Also reports how far the target sits outside the camera's field of view at the
start, because a target that is never in frame cannot be steered toward by any
of these arms.
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

from environment.target_world import corridor_world
from sensors.attraction_cue import AttractionCueSensor

from cx_subcircuit import Sub, build, episode
from cx_validate import FIXED, N_OBSTACLES, SCALE, tasks_for


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=24)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, cue_a, head_a, out_a, rho, sub = build()
    print("%d neurons, %d edges  (%.0f s)\n"
          % (net.n, sub.nnz, time.perf_counter() - t0), flush=True)

    sensor = AttractionCueSensor()
    conn = Sub(net, cue_a, head_a, out_a, head_gain=0.0, fixed_strength=FIXED)
    conn.calibrate()

    arms = [("connectome", conn, lambda c, h: SCALE * conn.turn(c, h)),
            ("P control", None,
             lambda c, h: 2.0 * math.radians(c.bearing_deg) if c.valid else 0.0)]

    rows, starts = [], []
    for seed in range(2, 2 + args.seeds):
        tk = tasks_for(seed, args.episodes)
        for w, h0 in tk:
            # where is the target, relative to where the agent points, at t=0?
            v = w.target - np.array([0.0, 0.0, 2.0])
            b = math.degrees(math.atan2(v[1], v[0])) - math.degrees(h0)
            c0 = sensor.sense(np.array([0.0, 0.0, 2.0]), h0, w.target,
                              w.obstacles)
            starts.append({"bearing0": (b + 180) % 360 - 180,
                           "visible0": bool(c0.valid)})
        for name, state, cmd in arms:
            for w, h0 in tk:
                if state is not None:
                    state.reset()
                r = episode(cmd, w, h0, sensor)
                r.update({"seed": seed, "arm": name})
                rows.append(r)
        print("seed %d done" % seed, flush=True)

    df = pd.DataFrame(rows)
    st = pd.DataFrame(starts)
    df.to_csv(REPO / "results/cx_failures.csv", index=False)

    hfov = sensor.rig.hfov_deg / 2
    print("\n--- the task, before any controller runs " + "-" * 20)
    print("camera half-FOV %.1f deg" % hfov)
    print("target outside the frame at t=0 : %.1f%% of episodes"
          % (100 * (st["bearing0"].abs() > hfov).mean()))
    print("cue invalid at t=0 (occluded or out of frame): %.1f%%"
          % (100 * (~st["visible0"]).mean()))
    print("|start bearing| median %.1f deg, max %.1f deg"
          % (st["bearing0"].abs().median(), st["bearing0"].abs().max()))

    print("\n--- outcomes, %d episodes per arm " % len(df[df.arm == "P control"])
          + "-" * 14)
    print("%-12s %8s %9s %8s %9s %9s"
          % ("arm", "reached", "collided", "timeout", "cue_frac", "cue_frac"))
    print("%-12s %8s %9s %8s %9s %9s"
          % ("", "", "", "", "reached", "failed"))
    for name in ("connectome", "P control"):
        s = df[df.arm == name]
        f = s[~s["reached"]]
        print("%-12s %8.3f %9.3f %8.3f %9.3f %9.3f"
              % (name, s["reached"].mean(), s["collided"].mean(),
                 s["timeout"].mean(), s[s["reached"]]["cue_frac"].mean(),
                 f["cue_frac"].mean()))

    print("\n--- does losing sight of the target explain the failures? " + "-" * 4)
    for name in ("connectome", "P control"):
        s = df[df.arm == name]
        lo = s[s["cue_frac"] < 0.5]
        hi = s[s["cue_frac"] >= 0.5]
        print("%-12s cue visible <50%% of steps: %3d episodes, reach %.3f"
              % (name, len(lo), lo["reached"].mean() if len(lo) else float("nan")))
        print("%-12s cue visible >=50%%        : %3d episodes, reach %.3f"
              % ("", len(hi), hi["reached"].mean() if len(hi) else float("nan")))
    print("\nwrote results/cx_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
