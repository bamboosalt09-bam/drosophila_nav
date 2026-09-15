"""Reduced Drosophila central-complex steering core (Westeinde et al. 2024).

Source
------
Westeinde et al., "Transforming a head direction signal into a goal-oriented
steering command", Nature 2024.  Official implementation:
github.com/wilson-lab/WesteindeWilson_AnalysisCode ->
Westeinde_et_al_model/Westeinde et al Model.ipynb (coded by Lydia Hamburg,
Druckmann and Wilson labs).  A copy is in reference/westeinde_official/.

This module reimplements the notebook's `connectivity_option = 'abstract'`,
`nonlinearity_option = 'ELU1'` model so that it can be called one
(heading, goal) at a time inside a closed loop.  It is a connectome-INFORMED
reduced model, not a connectome simulation.  Never describe its output as
"running the fly connectome".

The one structural change from the source, and why
--------------------------------------------------
The source normalises with `linear_rescale`, which takes min/max over the
WHOLE precomputed array -- all goal phases, all PFL scalars, all head
directions at once.  The source model is therefore not a function of
(heading, goal) alone: its output depends on which other conditions happened
to be in the sweep.

A closed-loop study cannot use that.  Our core must be a pure, frozen function
or the claim "the same core was used in every body condition" is false.  So we
compute those rescale limits ONCE, on the official grid, and freeze them as
explicit parameters (see calibration.py and configs/norm_constants.json).
Inside that grid the two are arithmetically identical; the frozen version
simply stops the answer from moving when the sweep changes.

Verified against the source in tests/test_source_agreement.py.

Conventions
-----------
The public API takes radians, CCW-positive (utils.angles).  The source works
in degrees, so the internal path does too and `evaluate` converts; use
`evaluate_deg` when comparing against the source to avoid a round trip.
Heading error e = wrap(heading - goal).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from utils.angles import rad2deg, wrap


# --------------------------------------------------------------------------
# Source primitives, kept close to the notebook so they can be diffed by eye
# --------------------------------------------------------------------------
def linear_rescale(activity, lo: float, hi: float,
                   new_min: float, new_max: float):
    """Frozen form of the notebook's `linear_rescale`.

    The source calls np.interp(activity, (min(activity), max(activity)),
    (new_min, new_max)).  Here the two anchors are supplied instead of being
    recomputed from whatever happens to be in the array.  np.interp CLAMPS
    outside the anchor range, and that clamping is part of the source
    behaviour, so it is kept.
    """
    if hi <= lo:
        raise ValueError("degenerate rescale range: lo=%r hi=%r" % (lo, hi))
    return np.interp(activity, (lo, hi), (new_min, new_max))


def elu1(activity, pre_lo: float, pre_hi: float,
         post_lo: float, post_hi: float):
    """Frozen form of the notebook's ELU1 nonlinearity.

    Source:
        scaled = linear_rescale(activity, -1.0, 1.0)
        nl     = scaled           where scaled >= 0
                 exp(scaled) - 1  where scaled <  0
        out    = linear_rescale(nl, 0.0, 1.0)

    The leading rescale to [-1, 1] is what pushes low-drive conditions onto
    the exponential branch.  Without it every input stays positive, the ELU is
    the identity, and the bilateral PFL2 drive cancels exactly in the
    left-right readout -- which is the bug this implementation used to have.
    """
    scaled = linear_rescale(activity, pre_lo, pre_hi, -1.0, 1.0)
    # np.exp(x) - 1, not expm1: matches the source's rounding
    nl = np.where(scaled >= 0.0, scaled, np.exp(scaled) - 1.0)
    return linear_rescale(nl, post_lo, post_hi, 0.0, 1.0)


@dataclass(frozen=True)
class StageNorm:
    """The four frozen anchors of one ELU1 stage."""

    pre_lo: float
    pre_hi: float
    post_lo: float
    post_hi: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.pre_lo, self.pre_hi, self.post_lo, self.post_hi)


@dataclass(frozen=True)
class NormConstants:
    """Every frozen normalisation constant of the core.

    Produced by core.calibration.calibrate() on the official grid and then
    never touched again -- in particular never re-derived per body condition.
    """

    pfl2: StageNorm
    pfl3r: StageNorm
    pfl3l: StageNorm
    dna03r: StageNorm
    dna03l: StageNorm
    dna02r: StageNorm
    dna02l: StageNorm
    steering_max: float
    grid_description: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "NormConstants":
        stages = {k: StageNorm(**d[k]) for k in
                  ("pfl2", "pfl3r", "pfl3l", "dna03r", "dna03l",
                   "dna02r", "dna02l")}
        return NormConstants(steering_max=float(d["steering_max"]),
                             grid_description=d.get("grid_description", ""),
                             **stages)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CoreParams:
    """Parameters of the reduced steering core.

    Defaults reproduce the notebook's abstract / ELU1 configuration.
    """

    # discretisation of the neural phase space (source: 1000 "cells")
    n_units: int = 1000

    # phase shifts, degrees, exactly as in the source HD-input expressions
    phase_shift_pfl3_deg: float = 67.5
    phase_shift_pfl2_deg: float = 180.0
    offset_deg: float = 0.0

    # goal amplitude A, and the PFL input scalar S
    goal_amplitude: float = 1.0
    pfl_scalar_S: float = 1.0

    # abstract relative weights
    w_pfl3_dna03: float = 1.0
    w_pfl2_dna03: float = 4.0
    w_pfl3_dna02: float = 1.0
    w_dna03_dna02: float = 12.0

    # The notebook writes np.tile(..., (1000, 1, 1, 1)) in the DNa02 step, so
    # the DNa03 contribution is effectively multiplied by 1000 regardless of
    # num_cells.  Replicated here rather than silently "fixed".
    dna02_tile_count: int = 1000

    pfl3_inhibitory: bool = False   # source default: PFL3 excitatory
    pfl2_silenced: bool = False     # source ablation switch

    label: str = "westeinde2024_reduced_abstract_ELU1"

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CoreState:
    """Snapshot of one core evaluation (for plots and diagnostics)."""

    heading_deg: float
    goal_deg: float
    heading_error_deg: float
    hd_prefs: np.ndarray = field(repr=False)
    act_pfl3r: np.ndarray = field(repr=False)
    act_pfl3l: np.ndarray = field(repr=False)
    act_pfl2: np.ndarray = field(repr=False)
    sum_pfl3r: float = 0.0
    sum_pfl3l: float = 0.0
    sum_pfl2: float = 0.0
    pfl2_bump_amp: float = 0.0
    dna03r: float = 0.0
    dna03l: float = 0.0
    dna02r: float = 0.0
    dna02l: float = 0.0
    steering: float = 0.0          # normalised, source units (peak 1 on grid)


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
class WesteindeSteeringCore:
    """Stateless reduced steering core.

    Memory-less by construction: given (heading, goal) it returns the steering
    command.  All temporal behaviour belongs to the closed-loop simulator, the
    body and the sensors.  That is what makes "the core is frozen across body
    conditions" a structural fact rather than a promise.
    """

    def __init__(self, params: Optional[CoreParams] = None,
                 norm: Optional[NormConstants] = None):
        self.p = params if params is not None else CoreParams()
        self.norm = norm
        self.hd_prefs = self._make_hd_prefs(self.p.n_units)

    # -- source replication ----------------------------------------------
    @staticmethod
    def _make_hd_prefs(n_units: int) -> np.ndarray:
        """Preferred HD of each unit, exactly as the notebook builds it."""
        uncentered = np.linspace(0.0, 360.0, n_units, endpoint=False)
        return np.flip(uncentered - np.mean(uncentered))

    def hd_inputs(self, hd_deg: float):
        """Head-direction input to each population, in degrees-based form."""
        p = self.p
        base = hd_deg - self.hd_prefs - p.offset_deg
        pfl3r = np.cos(np.deg2rad(base + p.phase_shift_pfl3_deg))
        pfl3l = np.cos(np.deg2rad(base - p.phase_shift_pfl3_deg))
        pfl2 = np.cos(np.deg2rad(base + p.phase_shift_pfl2_deg))
        return pfl3r, pfl3l, pfl2

    def goal_input(self, goal_deg: float) -> np.ndarray:
        """Goal-cell output: a bump at the goal phase, same for every HD."""
        p = self.p
        return p.goal_amplitude * np.cos(
            np.deg2rad(-self.hd_prefs + goal_deg - p.offset_deg))

    def pre_activities(self, hd_deg: float, goal_deg: float):
        """S * (HD input + goal input), before any nonlinearity."""
        r_in, l_in, two_in = self.hd_inputs(hd_deg)
        goal = self.goal_input(goal_deg)
        S = self.p.pfl_scalar_S
        return S * (r_in + goal), S * (l_in + goal), S * (two_in + goal)

    # -- public API -------------------------------------------------------
    def _require_norm(self) -> NormConstants:
        if self.norm is None:
            raise RuntimeError(
                "this core has no frozen normalisation constants; run "
                "core.calibration.calibrate() once and pass the result as "
                "norm=..., or load configs/norm_constants.json")
        return self.norm

    def evaluate_deg(self, hd_deg: float, goal_deg: float) -> CoreState:
        """Evaluate the core with angles in DEGREES (source-native path)."""
        p = self.p
        n = self._require_norm()

        pre_r, pre_l, pre_2 = self.pre_activities(hd_deg, goal_deg)

        pfl3r = elu1(pre_r, *n.pfl3r.as_tuple())
        pfl3l = elu1(pre_l, *n.pfl3l.as_tuple())
        pfl2 = elu1(pre_2, *n.pfl2.as_tuple())

        if p.pfl3_inhibitory:
            pfl3r = -pfl3r
            pfl3l = -pfl3l
        if p.pfl2_silenced:
            pfl2 = np.zeros_like(pfl2)

        sum_r = float(np.sum(pfl3r))
        sum_l = float(np.sum(pfl3l))
        sum_2 = float(np.sum(pfl2))

        dna03r_pre = p.w_pfl3_dna03 * sum_r + p.w_pfl2_dna03 * sum_2
        dna03l_pre = p.w_pfl3_dna03 * sum_l + p.w_pfl2_dna03 * sum_2
        dna03r = float(elu1(dna03r_pre, *n.dna03r.as_tuple()))
        dna03l = float(elu1(dna03l_pre, *n.dna03l.as_tuple()))

        # the source tiles the DNa03 term across cells before summing
        tile = p.dna02_tile_count
        dna02r_pre = tile * p.w_dna03_dna02 * dna03r + p.w_pfl3_dna02 * sum_r
        dna02l_pre = tile * p.w_dna03_dna02 * dna03l + p.w_pfl3_dna02 * sum_l
        dna02r = float(elu1(dna02r_pre, *n.dna02r.as_tuple()))
        dna02l = float(elu1(dna02l_pre, *n.dna02l.as_tuple()))

        steering = (dna02r - dna02l) / n.steering_max

        err = float(np.rad2deg(wrap(np.deg2rad(hd_deg - goal_deg))))
        return CoreState(
            heading_deg=float(hd_deg), goal_deg=float(goal_deg),
            heading_error_deg=err, hd_prefs=self.hd_prefs,
            act_pfl3r=pfl3r, act_pfl3l=pfl3l, act_pfl2=pfl2,
            sum_pfl3r=sum_r, sum_pfl3l=sum_l, sum_pfl2=sum_2,
            pfl2_bump_amp=float(np.max(pfl2) - np.min(pfl2)),
            dna03r=dna03r, dna03l=dna03l,
            dna02r=dna02r, dna02l=dna02l,
            steering=float(steering),
        )

    def evaluate(self, heading: float, goal: float) -> CoreState:
        """Evaluate the core with angles in RADIANS (project API)."""
        return self.evaluate_deg(float(rad2deg(heading)), float(rad2deg(goal)))

    def steering(self, heading: float, goal: float) -> float:
        """Normalised steering command for (heading, goal) in radians.

        Unit-less, peak magnitude 1 over the calibration grid.  The source
        multiplies this by k = 200 to get degrees per 0.1 s timestep; doing
        that conversion is the decoder's job, not the core's.
        """
        return self.evaluate(heading, goal).steering

    def sweep_deg(self, hd_degs, goal_deg: float = 0.0):
        """Evaluate over an array of head directions (degrees)."""
        hds = np.atleast_1d(np.asarray(hd_degs, dtype=float))
        keys = ("sum_pfl3r", "sum_pfl3l", "sum_pfl2", "pfl2_bump_amp",
                "dna03r", "dna03l", "dna02r", "dna02l", "steering",
                "heading_error_deg")
        out = {k: np.empty(hds.size) for k in keys}
        for i, hd in enumerate(hds):
            st = self.evaluate_deg(float(hd), goal_deg)
            for k in keys:
                out[k][i] = getattr(st, k)
        out["pfl3_RL_diff"] = out["sum_pfl3r"] - out["sum_pfl3l"]
        return out
