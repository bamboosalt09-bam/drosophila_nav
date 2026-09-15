"""Freeze the core's normalisation constants on the official grid.

Why this exists
---------------
The source model normalises with min/max taken over its entire precomputed
sweep, so its output for one (heading, goal) depends on the rest of the sweep.
We need a pure function, so those limits are computed once here and then
frozen for the whole study (see westeinde2024.py, "The one structural change").

The official grid (notebook cell 5):
    goal_phase = arange(-180, 181, 10)        -> 37 values
    S_vals     = linspace(0, 1, 6)            ->  6 values
    hd_deg     = arange(-180, 181, 1)         -> 361 values
    num_cells  = 1000

Materialising that as the notebook does needs several GB.  We stream over the
goal axis instead: the heavy quantities are recomputed per goal, and the cheap
per-condition scalars are cached, so peak memory stays in the tens of MB.
The arithmetic is identical.

Run:
    python -m core.calibration --out configs/norm_constants.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.westeinde2024 import (CoreParams, NormConstants, StageNorm,
                                WesteindeSteeringCore, elu1, linear_rescale)

OFFICIAL_GRID = dict(goal_step=10, hd_step=1, n_scalars=6)


def official_grid(goal_step: int = 10, hd_step: int = 1, n_scalars: int = 6):
    goals = np.arange(-180, 181, goal_step, dtype=float)
    hds = np.arange(-180, 181, hd_step, dtype=float)
    scalars = np.linspace(0.0, 1.0, n_scalars, endpoint=True)
    return goals, scalars, hds


class _MinMax:
    """Streaming min/max accumulator."""

    def __init__(self):
        self.lo = np.inf
        self.hi = -np.inf

    def update(self, arr) -> None:
        a = np.asarray(arr, dtype=float)
        if a.size:
            self.lo = min(self.lo, float(a.min()))
            self.hi = max(self.hi, float(a.max()))

    def as_pair(self):
        if not np.isfinite(self.lo) or not np.isfinite(self.hi):
            raise RuntimeError("min/max accumulator never saw data")
        return self.lo, self.hi


def calibrate(params: CoreParams | None = None, goal_step: int = 10,
              hd_step: int = 1, n_scalars: int = 6,
              verbose: bool = True) -> NormConstants:
    """Compute every frozen constant, streaming over the grid."""
    p = params if params is not None else CoreParams()
    core = WesteindeSteeringCore(p)  # norm not needed for the raw stages
    goals, scalars, hds = official_grid(goal_step, hd_step, n_scalars)

    n_cond = goals.size * scalars.size * hds.size
    if verbose:
        print("grid: %d goals x %d scalars x %d HD x %d cells  (%d conditions)"
              % (goals.size, scalars.size, hds.size, p.n_units, n_cond))

    # Precompute the HD inputs once: they do not depend on goal or S.
    hd_in_r = np.empty((hds.size, p.n_units))
    hd_in_l = np.empty((hds.size, p.n_units))
    hd_in_2 = np.empty((hds.size, p.n_units))
    for i, hd in enumerate(hds):
        hd_in_r[i], hd_in_l[i], hd_in_2[i] = core.hd_inputs(float(hd))

    # ---- sweep A: pre-activity min/max for the three PFL populations ----
    acc = {k: _MinMax() for k in ("pfl3r", "pfl3l", "pfl2")}
    for g in goals:
        goal = core.goal_input(float(g))[None, :]
        for S in scalars:
            acc["pfl3r"].update(S * (hd_in_r + goal))
            acc["pfl3l"].update(S * (hd_in_l + goal))
            acc["pfl2"].update(S * (hd_in_2 + goal))
    pre = {k: v.as_pair() for k, v in acc.items()}
    if verbose:
        print("sweep A done: PFL pre-activity ranges")
        for k, v in pre.items():
            print("   %-6s [% .6f, % .6f]" % (k, v[0], v[1]))

    # ---- sweep B: post-ELU min/max (needs sweep A) -----------------------
    accb = {k: _MinMax() for k in ("pfl3r", "pfl3l", "pfl2")}

    def _elu_raw(x, lo, hi):
        scaled = linear_rescale(x, lo, hi, -1.0, 1.0)
        return np.where(scaled >= 0.0, scaled, np.exp(scaled) - 1.0)

    for g in goals:
        goal = core.goal_input(float(g))[None, :]
        for S in scalars:
            accb["pfl3r"].update(_elu_raw(S * (hd_in_r + goal), *pre["pfl3r"]))
            accb["pfl3l"].update(_elu_raw(S * (hd_in_l + goal), *pre["pfl3l"]))
            accb["pfl2"].update(_elu_raw(S * (hd_in_2 + goal), *pre["pfl2"]))
    post = {k: v.as_pair() for k, v in accb.items()}
    if verbose:
        print("sweep B done: post-ELU ranges")
        for k, v in post.items():
            print("   %-6s [% .6f, % .6f]" % (k, v[0], v[1]))

    stage_pfl = {k: StageNorm(pre[k][0], pre[k][1], post[k][0], post[k][1])
                 for k in pre}

    # ---- sweep C: cache the per-condition population sums ---------------
    shape = (goals.size, scalars.size, hds.size)
    sum_r = np.empty(shape)
    sum_l = np.empty(shape)
    sum_2 = np.empty(shape)
    for gi, g in enumerate(goals):
        goal = core.goal_input(float(g))[None, :]
        for si, S in enumerate(scalars):
            a_r = elu1(S * (hd_in_r + goal), *stage_pfl["pfl3r"].as_tuple())
            a_l = elu1(S * (hd_in_l + goal), *stage_pfl["pfl3l"].as_tuple())
            a_2 = elu1(S * (hd_in_2 + goal), *stage_pfl["pfl2"].as_tuple())
            if p.pfl3_inhibitory:
                a_r, a_l = -a_r, -a_l
            if p.pfl2_silenced:
                a_2 = np.zeros_like(a_2)
            sum_r[gi, si] = a_r.sum(axis=1)
            sum_l[gi, si] = a_l.sum(axis=1)
            sum_2[gi, si] = a_2.sum(axis=1)
    if verbose:
        print("sweep C done: population sums cached %s" % (shape,))

    # ---- downstream stages are cheap: everything is a small array -------
    dna03r_pre = p.w_pfl3_dna03 * sum_r + p.w_pfl2_dna03 * sum_2
    dna03l_pre = p.w_pfl3_dna03 * sum_l + p.w_pfl2_dna03 * sum_2

    stage_dna03 = {}
    dna03 = {}
    for name, arr in (("dna03r", dna03r_pre), ("dna03l", dna03l_pre)):
        lo, hi = float(arr.min()), float(arr.max())
        raw = _elu_raw(arr, lo, hi)
        st = StageNorm(lo, hi, float(raw.min()), float(raw.max()))
        stage_dna03[name] = st
        dna03[name] = elu1(arr, *st.as_tuple())

    tile = p.dna02_tile_count
    dna02r_pre = tile * p.w_dna03_dna02 * dna03["dna03r"] + p.w_pfl3_dna02 * sum_r
    dna02l_pre = tile * p.w_dna03_dna02 * dna03["dna03l"] + p.w_pfl3_dna02 * sum_l

    stage_dna02 = {}
    dna02 = {}
    for name, arr in (("dna02r", dna02r_pre), ("dna02l", dna02l_pre)):
        lo, hi = float(arr.min()), float(arr.max())
        raw = _elu_raw(arr, lo, hi)
        st = StageNorm(lo, hi, float(raw.min()), float(raw.max()))
        stage_dna02[name] = st
        dna02[name] = elu1(arr, *st.as_tuple())

    steering_raw = dna02["dna02r"] - dna02["dna02l"]
    steering_max = float(steering_raw.max())
    if steering_max <= 0:
        raise RuntimeError("steering_max is not positive (%r); the sign "
                           "convention or the weights are wrong" % steering_max)

    if verbose:
        print("steering_max = %.10f  (raw range [% .6f, % .6f])"
              % (steering_max, steering_raw.min(), steering_raw.max()))

    return NormConstants(
        pfl2=stage_pfl["pfl2"], pfl3r=stage_pfl["pfl3r"],
        pfl3l=stage_pfl["pfl3l"],
        dna03r=stage_dna03["dna03r"], dna03l=stage_dna03["dna03l"],
        dna02r=stage_dna02["dna02r"], dna02l=stage_dna02["dna02l"],
        steering_max=steering_max,
        grid_description=("goal -180:%d:180, S linspace(0,1,%d), "
                          "HD -180:%d:180, n_units %d"
                          % (goal_step, n_scalars, hd_step, p.n_units)),
    )


def save(norm: NormConstants, path: Path, params: CoreParams) -> None:
    payload = {"core_params": params.as_dict(), **norm.as_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(path: Path) -> NormConstants:
    return NormConstants.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="freeze core normalisation")
    ap.add_argument("--goal-step", type=int, default=10)
    ap.add_argument("--hd-step", type=int, default=1)
    ap.add_argument("--n-scalars", type=int, default=6)
    ap.add_argument("--n-units", type=int, default=1000)
    ap.add_argument("--out", default="configs/norm_constants.json")
    args = ap.parse_args(argv)

    params = CoreParams(n_units=args.n_units)
    norm = calibrate(params, args.goal_step, args.hd_step, args.n_scalars)
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    save(norm, out, params)
    print("written ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
