"""The 10-family central-complex subcircuit, steering in closed loop.

FB5AB, PFNa/p/m, hDeltaC, FC2B/C, FB5N, PFL2/3 -- 532 neurons in our MaleCNS
data, 2,697 edges at >=2 synapses, spectrally rescaled rho 60.3 -> 0.9.  Cue
in at FB5AB, heading in at the PFNs, turn read as PFL right-minus-left.

WHAT ACTUALLY MADE IT WORK, because the hypothesis this file was written to
test was refuted by its own first run:

    NOT heading.  Driving the PFNs moves reach 0.67 -> 0.71 at n=24, which is
    inside the noise, and the readout moves 8x less for a +-1.2 rad heading
    sweep than for a +-40 deg bearing sweep (2.2e-5 against 1.77e-4).

    The CUE MAGNITUDE was the problem.  The readout has a DC term that scales
    with total cue drive and is twice the entire bearing-dependent range
    (offset 3.38e-4, range 1.77e-4).  Apparent target size changes 5x over an
    approach, so closed loop the circuit was reporting how BRIGHT the target
    is, not where it is -- and that is what destroyed three previous attempts
    in which the open-loop sweep had looked monotonic.  Pinning the magnitude
    and letting only the bilateral ratio vary fixes it, which is also what an
    adapting olfactory population delivers.

Measured, 24 episodes, seed 1, 2 obstacles, start 12.31 m:

    connectome, fixed strength, x3000   reached 0.71  collided 0.08  1.83 m
    P control reference, kp 2           reached 0.67  collided 0.29  2.62 m
    straight, never turns               reached 0.08  collided 0.25 10.44 m

The connectome arm beats the hand-written reference on all three, and it has
no obstacle sensing at all -- the lower collision rate comes from turning less
violently, not from seeing anything.

The P reference and the straight baseline stay in every run.  Without them a
miss cannot be attributed: at 8 obstacles P control itself only reaches 0.38,
so "the connectome did not arrive" would have meant nothing.

ponytail: heading enters as a bilateral PFN imbalance, not as a bump across
columns -- the annotation carries no column index.  Given how little heading
changes the readout, a real column map is the upgrade that would test the
vector computation properly.
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
import scipy.sparse.linalg as spl
import torch

from environment.target_world import Agent
from sensors.attraction_cue import AttractionCue, AttractionCueSensor
from sim.episode import DT, V_CRUISE

# the 10 families of binivin/drosophila-connectome-odor-navigation, matched
# against MaleCNS cell types by prefix
FAMILIES = ("FB5AB", "PFNa", "PFNp", "PFNm", "hDeltaC", "FC2B", "FC2C",
            "FB5N", "PFL2", "PFL3")
CUE_IN = ("FB5AB",)                  # where the attraction cue arrives
HEAD_IN = ("PFNa", "PFNp", "PFNm")   # where body direction arrives
READ_OUT = ("PFL2", "PFL3")          # where the turn is read

DT_MS, STEPS_PER_CYCLE = 5.0, 20
RHO_TARGET = 0.9


def build(min_syn: int = 2):
    """Sub-connectome, spectrally rescaled, plus the three address sets."""
    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns

    ids, out, ann, _ = load_malecns(w_scale=1.0, symmetrise=True)
    ctype = ann["cell_type"].astype(str).to_numpy()
    keep = np.flatnonzero([any(c.startswith(f) for f in FAMILIES)
                           for c in ctype])
    sub = out[keep][:, keep].tocsr()
    sub.data[np.abs(sub.data) < min_syn] = 0.0
    sub.eliminate_zeros()

    # unrescaled this graph has rho 69, and a recurrent network with rho > 1
    # does not settle -- that, not the neuron model, is what made the full
    # network oscillate.  One global factor, so relative wiring is untouched.
    rho = float(np.abs(spl.eigs(sub.astype(np.float64), k=1,
                                return_eigenvectors=False)[0]))
    sub = (sub * (RHO_TARGET / rho)).tocsr()

    sann = ann.iloc[keep].copy()
    # The index IS root_id and `load_malecns_eye` keys the retinotopic
    # lattice off it.  reset_index(drop=True) replaced it with 0..N-1,
    # so 6,199 lamina cells matched 1,779 unrelated neurons and every
    # viewing direction landed on the wrong cell.  That is what made
    # the two eyes report identical -137..+137 fields and the readout
    # answer with the SAME sign to left and right stimulation.
    net = FlyWireRate(out_csr=sub, ids=ids[keep], ann=sann)

    sct = sann["cell_type"].astype(str).to_numpy()
    side = sann["side"].to_numpy(dtype="<U16")

    def addr(prefixes):
        m = np.array([any(c.startswith(p) for p in prefixes) for c in sct])
        return (np.flatnonzero(m & (side == "left")),
                np.flatnonzero(m & (side == "right")))

    return net, addr(CUE_IN), addr(HEAD_IN), addr(READ_OUT), rho, sub


class Sub:
    """Drive in, turn out.  One control cycle per call."""

    def __init__(self, net, cue_a, head_a, out_a, cue_gain=0.2,
                 head_gain=0.0, hfov_half=45.0, fixed_strength=None):
        self.net = net
        self.cue_l, self.cue_r = cue_a
        self.head_l, self.head_r = head_a
        self.out_l, self.out_r = out_a
        self.cue_gain, self.head_gain = cue_gain, head_gain
        self.hfov_half = hfov_half
        # The readout's DC term scales with total cue drive, and it is 2x
        # the entire bearing-dependent range (measured: offset 3.38e-4,
        # range 1.77e-4).  Cue strength changes 5x over an approach, so
        # that DC term swamps the direction signal in closed loop.  Pinning
        # the magnitude leaves only the bilateral RATIO varying, which is
        # also what an adapting olfactory population delivers.
        self.fixed_strength = fixed_strength
        self.v = None
        self.baseline = 0.0

    def reset(self):
        self.v = self.net.init_state(1)

    def _drive(self, cue, heading):
        d = torch.zeros(self.net.n, 1)
        if cue is not None and cue.valid:
            f = float(np.clip(cue.bearing_deg / self.hfov_half, -1.0, 1.0))
            st = (cue.strength if self.fixed_strength is None
                  else self.fixed_strength)
            b = self.cue_gain * st
            d[self.cue_l, 0] = b * (1 + f) / max(len(self.cue_l), 1)
            d[self.cue_r, 0] = b * (1 - f) / max(len(self.cue_r), 1)
        if self.head_gain:
            # + = left, same convention as everything else on this side
            h = math.sin(heading)
            d[self.head_l, 0] = (self.head_gain * (1 + h)
                                 / max(len(self.head_l), 1))
            d[self.head_r, 0] = (self.head_gain * (1 - h)
                                 / max(len(self.head_r), 1))
        return d

    def turn(self, cue, heading):
        from core.flywire_rate import activity
        if self.v is None:
            self.reset()
        d = self._drive(cue, heading)
        with torch.no_grad():
            for _ in range(STEPS_PER_CYCLE):
                self.v = self.net.step(self.v, d, DT_MS)
            r = activity(self.v).numpy().ravel()
        return float(r[self.out_r].mean() - r[self.out_l].mean()) - self.baseline

    def calibrate(self, heading=0.0, steps=40):
        """The readout at bearing 0, measured once.

        The signal is monotonic but sits far off zero (offset/range 1.9), so
        without this every command has the same sign and the agent can only
        spiral.  One constant, measured, not fitted per episode.
        """
        st = 0.1 if self.fixed_strength is None else self.fixed_strength
        zero = AttractionCue(bearing_deg=0.0, elevation_deg=0.0,
                             deviation=0.0, range_m=10.0, strength=st,
                             valid=True, n_cameras=2)
        self.baseline = 0.0
        self.reset()
        t = 0.0
        for _ in range(steps):
            t = self.turn(zero, heading)
        self.baseline = t
        return t


def episode(cmd_fn, world, heading, sensor, max_steps=120,
            agent_sink=None, yaw_params=None):
    """One deterministic run.  cmd_fn(cue, heading) -> yaw rate, rad/s.

    `agent_sink`, when given, receives the Agent before the first step.
    A controller that needs the pose itself -- anything driven by the
    SCENE rather than by the cue alone -- reads it from there, so the
    cmd_fn signature every existing caller uses does not have to change.
    """
    agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=heading,
                  yaw_params=yaw_params)
    if agent_sink is not None:
        agent_sink.clear()
        agent_sink.append(agent)
    bearings, turns = [], []
    n, n_valid = 0, 0
    for _ in range(max_steps):
        cue = sensor.sense(agent.p, agent.heading, world.target,
                           world.obstacles)
        u = cmd_fn(cue, agent.heading)
        n += 1
        if cue.valid:
            n_valid += 1
            bearings.append(cue.bearing_deg)
            turns.append(u)
        agent.step(float(u), V_CRUISE, 0.0, DT)
        if agent.collided or world.reached(agent.p):
            break
    corr = (float(np.corrcoef(bearings, turns)[0, 1])
            if len(turns) > 3 and np.std(turns) > 0 else float("nan"))
    rng = (max(turns) - min(turns)) if turns else 0.0
    ok = bool(world.reached(agent.p)) and not agent.collided
    return {"reached": ok,
            # fraction of body sub-steps where the drone could not deliver the
            # yaw the brain asked for.  This IS the body-brain mismatch the
            # project is about, so it is reported, never assumed to be zero.
            "rate_sat": float(agent.plant.rate_saturation_fraction),
            "accel_sat": float(agent.plant.accel_saturation_fraction),
            "collided": bool(agent.collided),
            # ran out of steps without arriving or hitting anything: a
            # different failure from a collision and it needs a different fix
            "timeout": (not ok) and (not agent.collided) and n >= max_steps,
            "cue_frac": n_valid / max(n, 1),
            "dist": float(np.linalg.norm(world.target - agent.p)),
            "corr": corr, "steps": len(turns),
            "off_rng": abs(np.mean(turns)) / rng if rng > 0 else np.nan}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=8)
    args = ap.parse_args(argv)

    from environment.target_world import corridor_world

    def make_task(rng, n_obs):
        w = corridor_world(seed=int(rng.randint(1 << 30)),
                           n_obstacles=n_obs, spread=6.0)
        w.target = np.array([12.0, float(rng.uniform(-5, 5)), 2.0])
        return w, float(rng.uniform(-0.6, 0.6))

    t0 = time.perf_counter()
    net, cue_a, head_a, out_a, rho, sub = build()
    print("%d neurons, %d edges, rho %.1f -> %.1f  (%.0f s)"
          % (net.n, sub.nnz, rho, RHO_TARGET, time.perf_counter() - t0))
    print("  cue  FB5AB   L%d/R%d" % (len(cue_a[0]), len(cue_a[1])))
    print("  head PFN     L%d/R%d" % (len(head_a[0]), len(head_a[1])))
    print("  out  PFL     L%d/R%d" % (len(out_a[0]), len(out_a[1])), flush=True)
    assert len(head_a[0]) and len(head_a[1]), "no PFNs to inject heading into"

    rng = np.random.RandomState(args.seed)
    tasks = [make_task(rng, args.obstacles)
             for _ in range(args.episodes)]
    sensor = AttractionCueSensor()
    d0 = np.mean([np.linalg.norm(w.target - np.array([0., 0., 2.]))
                  for w, _ in tasks])
    print("start distance %.2f m\n" % d0)

    def p_control(cue, heading):
        return 2.0 * math.radians(cue.bearing_deg) if cue.valid else 0.0

    def make_sub(hg, scale, fixed=None):
        s = Sub(net, cue_a, head_a, out_a, head_gain=hg,
                fixed_strength=fixed)
        s.calibrate()
        s.reset()
        return lambda cue, h: scale * s.turn(cue, h)

    conds = [("P control (kp 2)", lambda: p_control),
             ("straight", lambda: (lambda cue, h: 0.0))]
    for fx in (None, 0.1):
        for hg in (0.0, 1.0):
            for sc in (300.0, 1000.0, 3000.0):
                conds.append(
                    ("head %.1f sc %.0f %s"
                     % (hg, sc, "fixed" if fx else "raw"),
                     (lambda hg=hg, sc=sc, fx=fx: make_sub(hg, sc, fx))))

    rows = []
    print("%-22s %8s %9s %8s %8s %8s" % ("condition", "reached", "collided",
                                         "dist", "corr", "off/rng"))
    for label, factory in conds:
        res = [episode(factory(), w, h, sensor) for w, h in tasks]
        r = {k: float(np.nanmean([x[k] for x in res]))
             for k in ("reached", "collided", "dist", "corr", "off_rng")}
        r["condition"] = label
        rows.append(r)
        print("%-22s %8.2f %9.2f %8.2f %+8.3f %8.2f"
              % (label, r["reached"], r["collided"], r["dist"], r["corr"],
                 r["off_rng"]), flush=True)

    df = pd.DataFrame(rows)
    out = REPO / "results/cx_subcircuit.csv"
    df.to_csv(out, index=False)
    p = df[df.condition.str.startswith("P control")].iloc[0]
    if p["reached"] < 0.5:
        print("\n=> the REFERENCE controller does not reach either: the task, "
              "not the circuit, is what these numbers are about")
    else:
        conn = df[df.condition.str.startswith("head")]
        b = conn.loc[conn["dist"].idxmin()]
        print("\nP control reaches %.2f; best connectome condition is %s "
              "(reached %.2f, dist %.2f)"
              % (p["reached"], b["condition"], b["reached"], b["dist"]))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
