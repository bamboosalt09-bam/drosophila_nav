"""Same task, same episodes, same optimiser: connectome core against a GRU.

"동일 학습" is identical observation, action space, objective, episode stream
and training exposure -- NOT identical parameter counts.  The asymmetry is the
result:

    connectome arm   166,700 fixed neurons + 2 trainable numbers
    rnn arm          3,969 trainable weights

REINFORCE, because the environment is numpy and not differentiable, so the
only way both arms can run the SAME algorithm is a score-function estimator.
Gaussian exploration on the action, one baseline, one learning rate, one seed
stream.  Nothing is tuned per arm.

Reward, per episode:
    fraction of the start distance closed, +1 for reaching, -1 for colliding.
Shaped only by distance, so it says nothing about how to steer.

Training happens ONCE, on the reference body.  Transfer to other bodies is a
separate run against frozen parameters -- no per-body retraining.

Usage:
    python experiments/train_compare.py --episodes 1000
    python experiments/train_compare.py --episodes 20 --arms rnn   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch

from environment.target_world import Agent, TargetWorld, corridor_world
from sim.episode import DT, V_CRUISE

SIGMA = 0.30            # exploration noise on the yaw command, rad/s
LR = 0.05
SCALE_LR = 5.0e4   # scale multiplies a ~1e-5 readout, so its gradient is
                   # ~1e-5 x smaller than the bias gradient.  One learning
                   # rate for both leaves scale frozen -- the same parameter-
                   # scale trap that broke the earlier connectome training.
MAX_STEPS = 100
BATCH = 16          # measured: 6.2x per-episode speedup on the sparse matvec


def make_task(rng: np.random.RandomState) -> tuple:
    """One episode's world and start pose.  Obstacles between start and goal.

    The target sits off-axis and behind clutter, so it goes in and out of view
    -- which is the point.  A task where the target is always visible is
    solved by 'turn toward the cue' and measures nothing.
    """
    world = corridor_world(seed=int(rng.randint(1 << 30)), n_obstacles=8,
                           spread=6.0)
    world.target = np.array([12.0, float(rng.uniform(-5, 5)), 2.0])
    heading = float(rng.uniform(-0.6, 0.6))
    return world, heading


def reward(res, world, start) -> float:
    d0 = float(np.linalg.norm(world.target - start))
    d1 = float(np.linalg.norm(world.target - res.path[-1]))
    r = (d0 - d1) / max(d0, 1e-9)
    if res.reached:
        r += 1.0
    if res.collided:
        r -= 1.0
    return r


def rollout_batch(arm, policy, inp, worlds, headings, rng):
    """B episodes in lockstep, so the sparse matvec amortises over columns.

    The brain always runs the full batch even after some episodes have ended;
    masking its columns would save nothing (the matvec cost is the matrix, not
    the columns) and would complicate the one thing that has to stay simple.
    Finished agents are simply frozen, and their log-probs stop accumulating.
    """
    from sim.episode import EpisodeResult

    b = len(worlds)
    agents = [Agent(world=w, start=(0.0, 0.0, 2.0), heading=h)
              for w, h in zip(worlds, headings)]
    policy.reset_batch(b)
    logps = [[] for _ in range(b)]
    rates = [[] for _ in range(b)]
    done = np.zeros(b, dtype=bool)
    steps = np.zeros(b, dtype=int)

    for _ in range(MAX_STEPS):
        pos = [a.p for a in agents]
        hds = [a.heading for a in agents]
        if arm == "fly":
            mu = policy.act_batch(worlds, pos, hds)
        else:
            obs = [inp.observation(w, p, h, n_bins=policy.n_bins)
                   for w, p, h in zip(worlds, pos, hds)]
            mu = policy.act_batch([o[0] for o in obs], [o[1] for o in obs], hds)
        noise = rng.randn(b)
        for i in range(b):
            if done[i]:
                continue
            a = float(mu[i].item()) + SIGMA * noise[i]
            logps[i].append(-((a - mu[i]) ** 2) / (2 * SIGMA ** 2))
            agents[i].step(a, V_CRUISE, 0.0, DT)
            rates[i].append(agents[i].plant.r)
            steps[i] += 1
            if agents[i].collided or worlds[i].reached(agents[i].p):
                done[i] = True
        if done.all():
            break

    out = []
    for i in range(b):
        res = EpisodeResult(
            path=agents[i].path_array(),
            reached=bool(worlds[i].reached(agents[i].p)) and not agents[i].collided,
            collided=bool(agents[i].collided), steps=int(steps[i]),
            straight_line=worlds[i].straight_line_length(agents[i].start),
            reach_radius=float(worlds[i].reach_radius),
            yaw_rates=np.array(rates[i]),
            rate_saturation=float(agents[i].plant.rate_saturation_fraction),
            accel_saturation=float(agents[i].plant.accel_saturation_fraction),
            cue_valid_fraction=float("nan"))
        total = torch.stack(logps[i]).sum() if logps[i] else None
        out.append((reward(res, worlds[i], agents[i].start), total, res))
    return out


def rollout(arm, policy, inp, world, heading, rng, train: bool):
    """One episode.  Returns (reward, log-prob sum, EpisodeResult)."""
    from sim.episode import EpisodeResult

    agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=heading)
    policy.reset()
    logp = []
    rates = []
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        if arm == "fly":
            mu = policy.act_t(world, agent.p, agent.heading)
        else:
            profile, cue = inp.observation(world, agent.p, agent.heading,
                                           n_bins=policy.n_bins)
            mu = policy.act_t(profile, cue, agent.heading)
        if train:
            a = float(mu.item()) + SIGMA * rng.randn()
            logp.append(-((a - mu) ** 2) / (2 * SIGMA ** 2))
        else:
            a = float(mu.item())
        agent.step(a, V_CRUISE, 0.0, DT)
        rates.append(agent.plant.r)
        if agent.collided or world.reached(agent.p):
            break

    res = EpisodeResult(
        path=agent.path_array(),
        reached=bool(world.reached(agent.p)) and not agent.collided,
        collided=bool(agent.collided), steps=steps,
        straight_line=world.straight_line_length(agent.start),
        reach_radius=float(world.reach_radius),
        yaw_rates=np.array(rates),
        rate_saturation=float(agent.plant.rate_saturation_fraction),
        accel_saturation=float(agent.plant.accel_saturation_fraction),
        cue_valid_fraction=float("nan"))
    total = torch.stack(logp).sum() if logp else None
    return reward(res, world, agent.start), total, res


def train_arm(arm: str, policy, inp, episodes: int, seed: int,
              out_csv: Path, resume: bool = False) -> pd.DataFrame:
    # Per-parameter-group learning rates.  The scale multiplies a ~1e-5
    # descending readout while the bias is in rad/s directly, so their
    # gradients differ by five orders of magnitude and one shared lr moves
    # only the bias.  This is the same parameter-scale trap that destroyed an
    # earlier connectome training run in a single Adam step.
    groups = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        groups.append({"params": [p],
                       "lr": SCALE_LR if name == "scale" else LR})
    opt = torch.optim.Adam(groups)
    rng = np.random.RandomState(seed)          # SAME stream for every arm
    baseline = 0.0
    rows = []
    t0 = time.perf_counter()
    # resume: replay the finished episodes from the checkpoint so the curve
    # stays one continuous record, and skip that many tasks from the SAME rng
    # stream so the episode sequence is identical to an uninterrupted run
    if resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        pt = out_csv.with_suffix(".pt")
        if pt.exists():
            policy.load_state_dict(torch.load(pt, weights_only=True))
        baseline = float(prev["baseline"].iloc[-1])
        for _ in range(len(prev)):
            make_task(rng)                      # burn the same draws
        print("  [%-3s] resuming at episode %d" % (arm, len(prev)), flush=True)

    ep = len(rows)
    while ep < episodes:
        b = min(BATCH, episodes - ep)
        tasks = [make_task(rng) for _ in range(b)]
        outs = rollout_batch(arm, policy, inp, [t[0] for t in tasks],
                             [t[1] for t in tasks], rng)
        # one optimiser step per batch, advantage against the running baseline
        loss = None
        for r, logp, res in outs:
            baseline = 0.95 * baseline + 0.05 * r
            if logp is not None:
                term = -(logp * (r - baseline))
                loss = term if loss is None else loss + term
            ep += 1
            rows.append({"episode": ep, "reward": r, "baseline": baseline,
                         "reached": res.reached, "collided": res.collided,
                         "steps": res.steps,
                         "detour": res.detour if res.reached else np.nan})
        if loss is not None and policy.trainable():
            opt.zero_grad()
            (loss / b).backward()
            opt.step()
        # Checkpoint every batch, BOTH the curve and the parameters.  Two long
        # background runs have been killed mid-flight by the harness, at 423 s
        # and at 29 min, with no error in either log -- so the run has to be
        # resumable rather than trusted to survive.  The files are tiny.
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        torch.save(policy.state_dict(), out_csv.with_suffix(".pt"))
        if ep % max(BATCH, episodes // 20) < BATCH or ep >= episodes:
            w = pd.DataFrame(rows[-max(BATCH, episodes // 20):])
            print("  [%-3s] ep %5d  reward %+.3f  reached %.2f  collided %.2f"
                  "  (%.0f s)"
                  % (arm, ep, w["reward"].mean(), w["reached"].mean(),
                     w["collided"].mean(), time.perf_counter() - t0),
                  flush=True)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--arms", nargs="+", default=["fly", "rnn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/train_compare")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint instead of restarting")
    args = ap.parse_args(argv)
    out = Path(args.outdir)

    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns
    from sensors.connectome_input import ConnectomeInput
    from sim.policies import ConnectomeFlyPolicy, RNNPolicy

    t0 = time.perf_counter()
    ids, csr, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=csr, ids=ids, ann=ann)
    inp = ConnectomeInput(net, ann)
    print("loaded in %.0f s; %s" % (time.perf_counter() - t0, inp.counts()),
          flush=True)

    summary = {}
    for arm in args.arms:
        torch.manual_seed(args.seed)
        if arm == "fly":
            # scale starts at ZERO so learning decides both the sign and the
            # magnitude of the descending readout.  Measured open loop, the
            # connectome turns AWAY from the cue under the source convention
            # (target left -> descending R-L negative -> rightward).  Baking in
            # a flip would be fitting the answer; a learned negative scale is a
            # result about the circuit polarity instead.
            policy = ConnectomeFlyPolicy(net, ann, inp, kappa_scale=0.0,
                                         trainable_adapter=True)
        else:
            policy = RNNPolicy()
        n = sum(p.numel() for p in policy.trainable())
        print("--- %s: %d trainable parameters ---" % (arm, n), flush=True)
        # the connectome must never appear in the optimiser: 166,700 neurons
        # with 141 per-type parameters would otherwise ride along as a
        # submodule and quietly un-freeze the core
        if arm == "fly":
            assert n == 2, "fly arm must train exactly 2 numbers, got %d" % n
        df = train_arm(arm, policy, inp, args.episodes, args.seed,
                       out / ("%s.csv" % arm), resume=args.resume)
        last = df.tail(max(1, args.episodes // 10))
        summary[arm] = {"trainable": n,
                        "final_reward": float(last["reward"].mean()),
                        "final_reached": float(last["reached"].mean()),
                        "final_collided": float(last["collided"].mean())}
        torch.save(policy.state_dict(), out / ("%s.pt" % arm))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n--- summary " + "-" * 50)
    for arm, s in summary.items():
        print("  %-4s params %6d   reward %+.3f   reached %.2f   collided %.2f"
              % (arm, s["trainable"], s["final_reward"], s["final_reached"],
                 s["final_collided"]))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
