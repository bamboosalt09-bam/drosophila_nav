"""Does the steering signal survive the closed loop, or does it drift?

Open loop, standing still, the readout is clean: monotonic in target bearing,
crosses zero, offset/range 0.036.  Closed loop it is worse than useless -- a
policy that never turns ends 6.79 m from the target while every steering gain
ends 11-17 m away, and the strongest one sweeps 1,063 full turns in ten
seconds.

One structural difference between the two measurements: the open-loop probe
calls `reset()` before each reading, so the membrane state starts from zero
every time.  The closed loop carries `self.v` across the whole episode.  If
the network integrates itself into a corner, the readout stops tracking the
target and a standing turn is exactly what comes out.

So: log the raw descending readout every control step, against the target
bearing the cue reports at that same step, under two conditions --

    carried   state persists across the episode (what training used)
    reset     state cleared every control step (what the probe measured)

If `carried` decorrelates from bearing over time while `reset` does not, the
signal is being destroyed by state accumulation, not by gain.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch

from environment.target_world import Agent
from sim.episode import DT, V_CRUISE


def trace(policy, inp, world, heading, reset_each: bool, steps: int = 100,
          leak: float = 1.0, fixed_strength=None):
    """One deterministic episode, logging the readout and the cue each step.

    `leak` multiplies the membrane state once per control cycle.  1.0 is the
    closed loop as trained; below 1.0 bleeds off the accumulation that drove
    v_mean from 8.0e-3 to 1.26e-2 over an episode, WITHOUT erasing the
    circuit's memory the way reset does -- erasing it would remove the one
    thing a recurrent core is for.
    """
    from core.flywire_rate import activity

    inp.fixed_strength = fixed_strength
    agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=heading)
    policy.reset()
    rows = []
    with torch.no_grad():
        for t in range(steps):
            if reset_each:
                policy.reset()
            elif leak != 1.0:
                policy.v = policy.v * leak
            drive, tr = inp.drive(world, agent.p, agent.heading)
            for _ in range(policy.steps_per_cycle):
                policy.v = policy.net.step(policy.v, drive, policy.dt_ms)
            r = activity(policy.v).numpy().ravel()
            turn = float(r[policy.desc_r].mean() - r[policy.desc_l].mean())
            cue = inp.cue_sensor.sense(agent.p, agent.heading, world.target,
                                       world.obstacles)
            rows.append({"t": t, "turn": turn,
                         "bearing": cue.bearing_deg if cue.valid else np.nan,
                         "cue_valid": cue.valid,
                         "mean_rate": float(r.mean()),
                         "v_mean": float(policy.v.mean()),
                         "dist": float(np.linalg.norm(world.target - agent.p))})
            agent.step(float(policy.scale.item()) * turn
                       + float(policy.bias.item()), V_CRUISE, 0.0, DT)
            if agent.collided or world.reached(agent.p):
                break
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> dict:
    ok = df.dropna(subset=["bearing"])
    # does the readout still track where the target is?
    corr = (float(np.corrcoef(ok["bearing"], ok["turn"])[0, 1])
            if len(ok) > 3 and ok["turn"].std() > 0 else float("nan"))
    early, late = ok.head(len(ok) // 3), ok.tail(len(ok) // 3)

    def c(d):
        return (float(np.corrcoef(d["bearing"], d["turn"])[0, 1])
                if len(d) > 3 and d["turn"].std() > 0 else float("nan"))
    rng = df["turn"].max() - df["turn"].min()
    out = {"condition": label, "steps": len(df), "corr_all": corr,
           "corr_early": c(early), "corr_late": c(late),
           "turn_mean": df["turn"].mean(), "turn_range": rng,
           "offset_over_range": abs(df["turn"].mean()) / rng if rng > 0 else np.nan,
           "v_mean_first": df["v_mean"].iloc[0],
           "v_mean_last": df["v_mean"].iloc[-1],
           "rate_first": df["mean_rate"].iloc[0],
           "rate_last": df["mean_rate"].iloc[-1]}
    print("%-9s steps %3d  corr all %+.3f (early %+.3f -> late %+.3f)  "
          "off/rng %6.2f  v_mean %.3e -> %.3e"
          % (label, out["steps"], corr, out["corr_early"], out["corr_late"],
             out["offset_over_range"], out["v_mean_first"], out["v_mean_last"]),
          flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--scale", type=float, default=6e4)
    args = ap.parse_args(argv)

    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns
    from sensors.connectome_input import ConnectomeInput
    from sim.policies import ConnectomeFlyPolicy
    from train_compare import make_task

    t0 = time.perf_counter()
    ids, csr, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=csr, ids=ids, ann=ann)
    inp = ConnectomeInput(net, ann)
    pol = ConnectomeFlyPolicy(net, ann, inp, kappa_scale=args.scale)
    print("loaded in %.0f s; scale %.0e" % (time.perf_counter() - t0,
                                            args.scale), flush=True)

    rng = np.random.RandomState(0)
    tasks = [make_task(rng) for _ in range(args.episodes)]

    conds = [("carried", False, 1.0, None),
             ("reset", True, 1.0, None),
             ("leak0.5", False, 0.5, None),
             ("leak0.9", False, 0.9, None),
             ("fixedstr", False, 1.0, 0.08),
             ("leak0.5+fix", False, 0.5, 0.08)]

    rows, traces = [], []
    for i, (world, heading) in enumerate(tasks):
        for lbl, reset_each, leak, fx in conds:
            df = trace(pol, inp, world, heading, reset_each, leak=leak,
                       fixed_strength=fx)
            df["episode"], df["condition"] = i, lbl
            traces.append(df)
            r = report(df, "ep%d %s" % (i, lbl))
            r["episode"] = i
            rows.append(r)

    res = pd.DataFrame(rows)
    pd.concat(traces).to_csv(REPO / "results/closed_loop_trace.csv", index=False)
    res.to_csv(REPO / "results/closed_loop_probe.csv", index=False)

    print("\n--- verdict " + "-" * 50)
    for lbl in [c[0] for c in conds]:
        s = res[res["condition"].str.endswith(lbl)]
        print("%-8s corr with bearing: all %+.3f, early %+.3f, late %+.3f; "
              "offset/range %.2f"
              % (lbl, s["corr_all"].mean(), s["corr_early"].mean(),
                 s["corr_late"].mean(), s["offset_over_range"].mean()))
    # offset/range is the number that matters: below 0.5 the steering signal
    # crosses zero and is therefore signed.  corr alone hid this last time.
    best = min([c[0] for c in conds],
               key=lambda l: res[res["condition"].str.endswith(l)]
               ["offset_over_range"].mean())
    print("best offset/range: %s" % best)
    print("wrote results/closed_loop_probe.csv and closed_loop_trace.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
