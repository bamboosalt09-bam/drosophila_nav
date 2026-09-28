"""Does the 2026-09-19 result hold on seeds it was not chosen on?

The operating point (fixed cue magnitude, gain 3000) was picked by looking at
seeds 0 and 1.  Re-reporting those seeds would measure the choice, not the
circuit.  So: freeze every setting, run seeds 2..11, and report each seed
separately rather than one pooled average that a couple of lucky seeds could
carry.

The P reference and the never-turn baseline run on the IDENTICAL tasks in the
same seed, so the comparison is paired -- which is the comparison that means
something when per-seed variance is this large.

ponytail: one Sub per condition, reset per episode, instead of rebuilding and
recalibrating it 240 times.  The baseline is deterministic, so this is the
same number for half the work.
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

SCALE = 3000.0        # frozen, chosen on seeds 0-1
FIXED = 0.1           # cue magnitude pinned; only the bilateral ratio varies
N_OBSTACLES = 2


def tasks_for(seed: int, n: int):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        w = corridor_world(seed=int(rng.randint(1 << 30)),
                           n_obstacles=N_OBSTACLES, spread=6.0)
        w.target = np.array([12.0, float(rng.uniform(-5, 5)), 2.0])
        out.append((w, float(rng.uniform(-0.6, 0.6))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=24)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    net, cue_a, head_a, out_a, rho, sub = build()
    print("%d neurons, %d edges, rho %.1f -> 0.9  (%.0f s)"
          % (net.n, sub.nnz, rho, time.perf_counter() - t0))
    print("frozen: fixed cue magnitude %.2f, gain %.0f, %d obstacles, "
          "%d episodes x %d held-out seeds\n"
          % (FIXED, SCALE, N_OBSTACLES, args.episodes, args.seeds), flush=True)

    sensor = AttractionCueSensor()
    conn = Sub(net, cue_a, head_a, out_a, head_gain=0.0, fixed_strength=FIXED)
    conn.calibrate()
    head = Sub(net, cue_a, head_a, out_a, head_gain=1.0, fixed_strength=FIXED)
    head.calibrate()

    def conn_cmd(cue, h):
        return SCALE * conn.turn(cue, h)

    def head_cmd(cue, h):
        return SCALE * head.turn(cue, h)

    def p_cmd(cue, h):
        return 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0

    arms = [("connectome", conn, conn_cmd),
            ("conn+heading", head, head_cmd),
            ("P control", None, p_cmd),
            ("straight", None, lambda cue, h: 0.0)]

    rows = []
    print("%-5s %-13s %8s %9s %8s" % ("seed", "arm", "reached", "collided",
                                      "dist"))
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        tk = tasks_for(seed, args.episodes)
        for name, state, cmd in arms:
            res = []
            for w, h0 in tk:
                if state is not None:
                    state.reset()
                res.append(episode(cmd, w, h0, sensor))
            r = {"seed": seed, "arm": name,
                 "reached": float(np.mean([x["reached"] for x in res])),
                 "collided": float(np.mean([x["collided"] for x in res])),
                 "dist": float(np.mean([x["dist"] for x in res]))}
            rows.append(r)
            print("%-5d %-13s %8.2f %9.2f %8.2f"
                  % (seed, name, r["reached"], r["collided"], r["dist"]),
                  flush=True)
        print("", flush=True)

    df = pd.DataFrame(rows)
    out = REPO / "results/cx_validate.csv"
    df.to_csv(out, index=False)

    print("--- pooled over %d held-out seeds " % args.seeds + "-" * 24)
    print("%-13s %8s %8s %9s %8s %8s"
          % ("arm", "reached", "sd", "collided", "dist", "sd"))
    for name, _, _ in [(a[0], None, None) for a in arms]:
        s = df[df.arm == name]
        print("%-13s %8.3f %8.3f %9.3f %8.2f %8.2f"
              % (name, s["reached"].mean(), s["reached"].std(),
                 s["collided"].mean(), s["dist"].mean(), s["dist"].std()))

    # paired, per seed: the only honest read when seed variance is this big
    c = df[df.arm == "connectome"].set_index("seed")
    p = df[df.arm == "P control"].set_index("seed")
    d_reach = c["reached"] - p["reached"]
    d_dist = c["dist"] - p["dist"]
    print("\nconnectome minus P reference, paired by seed:")
    print("  reached  mean %+.3f, wins %d / losses %d / ties %d"
          % (d_reach.mean(), int((d_reach > 0).sum()),
             int((d_reach < 0).sum()), int((d_reach == 0).sum())))
    print("  distance mean %+.2f m, better in %d of %d seeds"
          % (d_dist.mean(), int((d_dist < 0).sum()), len(d_dist)))
    s = df[df.arm == "straight"]
    print("\nnever-turn baseline: reached %.3f, dist %.2f m"
          % (s["reached"].mean(), s["dist"].mean()))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
