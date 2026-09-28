"""Why is the connectome arm not learning?  Measure before changing anything.

After 320 episodes the reward oscillates between -0.34 and +0.05 with no
trend, 2 of 320 episodes reach the target, and the adapter has run to
scale +1.7e5, bias -0.167.  Three candidate causes, and they call for
different fixes, so they have to be separated rather than guessed at:

  A. no scale works      the readout cannot steer this task at all, and the
                         optimiser is not the problem
  B. a scale works       the circuit is fine and the OPTIMISER is at fault --
                         SCALE_LR 5e4 was picked by hand and may be far too
                         large, and the learned bias adds a standing turn
  C. exploration swamps  sigma 0.3 rad/s against a policy output of the same
                         size means REINFORCE sees 1:1 noise

This sweeps scale and bias with NO training and NO exploration noise, so what
comes out is the circuit's own competence at each operating point.  If some
scale reaches the target reliably, the search is broken, not the circuit.
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


def evaluate(policy, inp, tasks, max_steps=100) -> dict:
    """Deterministic rollout of a batch of tasks: no noise, no learning."""
    worlds = [t[0] for t in tasks]
    agents = [Agent(world=w, start=(0.0, 0.0, 2.0), heading=t[1])
              for w, t in zip(worlds, tasks)]
    policy.reset_batch(len(worlds))
    done = np.zeros(len(worlds), dtype=bool)
    yaws = [[] for _ in worlds]

    with torch.no_grad():
        for _ in range(max_steps):
            pos = [a.p for a in agents]
            hds = [a.heading for a in agents]
            mu = policy.act_batch(worlds, pos, hds)
            for i in range(len(worlds)):
                if done[i]:
                    continue
                agents[i].step(float(mu[i].item()), V_CRUISE, 0.0, DT)
                yaws[i].append(agents[i].plant.r)
                if agents[i].collided or worlds[i].reached(agents[i].p):
                    done[i] = True
            if done.all():
                break

    reached, collided, dist, turn_tot, sat = [], [], [], [], []
    for i, (w, a) in enumerate(zip(worlds, agents)):
        reached.append(bool(w.reached(a.p)) and not a.collided)
        collided.append(bool(a.collided))
        dist.append(float(np.linalg.norm(w.target - a.p)))
        # total heading swept: a circling agent racks this up without progress
        turn_tot.append(float(np.abs(yaws[i]).sum() * DT))
        sat.append(float(a.plant.rate_saturation_fraction))
    d0 = float(np.linalg.norm(worlds[0].target - agents[0].start))
    return {"reached": np.mean(reached), "collided": np.mean(collided),
            "final_dist": np.mean(dist), "start_dist": d0,
            "turn_rad": np.mean(turn_tot), "saturation": np.mean(sat)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
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
    print("loaded in %.0f s" % (time.perf_counter() - t0), flush=True)

    # the SAME tasks for every operating point, so differences are the policy
    rng = np.random.RandomState(args.seed)
    tasks = [make_task(rng) for _ in range(args.episodes)]

    trained = torch.load(REPO / "results/train_compare/fly.pt",
                         weights_only=True)
    conds = [("trained (320 ep)", float(trained["scale"]),
              float(trained["bias"]))]
    for s in (1e4, 3e4, 6e4, 1e5, 3e5, 1e6):
        conds.append(("scale %.0e" % s, s, 0.0))
    conds.append(("scale -6e4 (flipped)", -6e4, 0.0))
    conds.append(("zero (straight)", 0.0, 0.0))

    rows = []
    print("\n%-22s %8s %9s %10s %9s %8s  (%.0f s)"
          % ("condition", "reached", "collided", "final_dist", "turn_rad",
             "sat", 0.0))
    for label, s, b in conds:
        pol = ConnectomeFlyPolicy(net, ann, inp, kappa_scale=s)
        with torch.no_grad():
            pol.bias.fill_(b)
        t = time.perf_counter()
        r = evaluate(pol, inp, tasks)
        r.update({"condition": label, "scale": s, "bias": b})
        rows.append(r)
        print("%-22s %8.2f %9.2f %10.2f %9.2f %8.2f  (%.0f s)"
              % (label, r["reached"], r["collided"], r["final_dist"],
                 r["turn_rad"], r["saturation"], time.perf_counter() - t),
              flush=True)

    df = pd.DataFrame(rows)
    out = REPO / "results/train_diagnose.csv"
    df.to_csv(out, index=False)
    print("\nstart distance %.2f m; a policy that never turns ends at %.2f m"
          % (rows[0]["start_dist"],
             df[df.condition == "zero (straight)"]["final_dist"].iloc[0]))
    best = df.loc[df["reached"].idxmax()]
    print("best: %s -> reached %.2f" % (best["condition"], best["reached"]))
    if best["reached"] == 0:
        print("=> NO operating point reaches the target: the readout cannot "
              "steer this task, and the optimiser is not the problem (A)")
    else:
        print("=> some operating point works, so the circuit can do it and "
              "the SEARCH is what failed (B)")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
