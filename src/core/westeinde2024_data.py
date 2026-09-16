"""The same steering circuit, wired with hemibrain connectome numbers.

This is the source notebook's `connectivity_option = 'data'`, which differs
from the abstract condition in four ways (audited in
reference/check_connectivity_data.py, recorded under
provenance: connectivity_data_audit):

  * 12 cells per PFL population instead of 1000, indexed by protocerebral
    bridge glomerulus
  * per-cell PFL->DN weights (raw hemibrain synapse counts, used directly),
    and DIFFERENT vectors left and right -- the left is the right's mirror,
    i.e. symmetry is assumed rather than measured
  * head-direction preferences predicted per cell from its glomerulus, rather
    than uniformly spaced with a fixed phase shift
  * DNa03 -> DNa02 weight 271 instead of 12

Written as a separate core rather than as options on WesteindeSteeringCore so
that the verified, bit-identical abstract reproduction is not touched.  Both
satisfy the same SteeringCore protocol, which is what that protocol is for.

Two bugs stop the source's data path from running at all.  Both are repaired
here and neither repair carries a free parameter:
  * the source's hd_prefs is a Python list, so `-1*hd_prefs` is repetition by
    -1 (the empty list) rather than negation
  * the source tiles the DNa03 term by a literal 1000 while num_cells is 12;
    broadcasting admits only 1 or num_cells, and those two give bit-identical
    results, so tile = num_cells is forced

R_max is NOT recalibrated for this core.  It is shared with the abstract
condition (user decision, 2026-09-16) so that "body" means the same thing in
both and the weaker data command shows up as behaviour rather than being
normalised away.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from core.westeinde2024 import NormConstants, StageNorm, elu1

N_CELLS = 12

# Predicted HD preferences, per cell, from PB glomerulus (source cell 5).
PFL2_HD_PREFS = np.asarray(
    [0, -45, -90, -90, -135, 180, 180, 135, 90, 90, 45, 0], float)
PFL3R_HD_PREFS = np.asarray(
    [90, 45, 0, 0, -45, -90, -90, -135, 180, 180, 180, 135], float)
PFL3L_HD_PREFS = np.asarray(
    [-135, 180, 180, 180, 135, 90, 90, 45, 0, 0, -45, -90], float)

# Goal-cell preferences, centred on 0.  Identical to the abstract construction
# at 12 cells (verified in the audit).
GOAL_HD_PREFS = np.asarray(
    [165, 135, 105, 75, 45, 15, -15, -45, -75, -105, -135, -165], float)

# Hemibrain synapse counts, used directly as weights (no conversion).
W_PFL2_DNA03R = np.asarray(
    [1., 72., 56., 57., 96., 57., 54., 79., 45., 58., 89., 76.])
W_PFL2_DNA03L = np.flip(W_PFL2_DNA03R)
W_PFL3R_DNA03R = np.asarray(
    [25., 28., 8., 13., 26., 29., 12., 1., 9., 22., 1., 26.])
W_PFL3L_DNA03L = np.flip(W_PFL3R_DNA03R)
W_PFL3R_DNA02R = np.asarray(
    [21., 25., 17., 14., 19., 44., 25., 25., 7., 11., 3., 39.])
W_PFL3L_DNA02L = np.flip(W_PFL3R_DNA02R)
W_DNA03_DNA02 = 271.0


@dataclass
class DataCoreParams:
    """Parameters of the hemibrain-wired core.  The weights are not free."""

    offset_deg: float = 0.0
    goal_amplitude: float = 1.0
    pfl_scalar_S: float = 1.0
    pfl3_inhibitory: bool = False
    pfl2_silenced: bool = False
    dna02_tile_count: int = N_CELLS      # forced; see module docstring
    label: str = "westeinde2024_hemibrain_data_ELU1"

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["n_cells"] = N_CELLS
        d["w_dna03_dna02"] = W_DNA03_DNA02
        d["weights"] = ("hemibrain synapse counts, per cell, "
                        "left = mirror of right")
        return d


def _elu1_band(x) -> StageNorm:
    """The four frozen anchors the source's batch normalisation would give."""
    lo, hi = float(np.min(x)), float(np.max(x))
    scaled = np.interp(x, (lo, hi), (-1.0, 1.0))
    nl = np.where(scaled >= 0.0, scaled, np.exp(scaled) - 1.0)
    return StageNorm(lo, hi, float(nl.min()), float(nl.max()))


class WesteindeDataCore:
    """Hemibrain-wired steering core.  Satisfies core.base.SteeringCore."""

    def __init__(self, params: Optional[DataCoreParams] = None,
                 norm: Optional[NormConstants] = None):
        self.p = params if params is not None else DataCoreParams()
        self.norm = norm

    def hd_inputs(self, hd_deg: float):
        o = self.p.offset_deg
        return (np.cos(np.deg2rad(hd_deg - PFL3R_HD_PREFS - o)),
                np.cos(np.deg2rad(hd_deg - PFL3L_HD_PREFS - o)),
                np.cos(np.deg2rad(hd_deg - PFL2_HD_PREFS - o)))

    def goal_input(self, goal_deg: float) -> np.ndarray:
        p = self.p
        return p.goal_amplitude * np.cos(
            np.deg2rad(-GOAL_HD_PREFS + goal_deg - p.offset_deg))

    def pre_activities(self, hd_deg: float, goal_deg: float):
        r_in, l_in, two_in = self.hd_inputs(hd_deg)
        goal = self.goal_input(goal_deg)
        S = self.p.pfl_scalar_S
        return S * (r_in + goal), S * (l_in + goal), S * (two_in + goal)

    def _raw(self, hd_deg: float, goal_deg: float, n: NormConstants) -> float:
        """DNa02R - DNa02L, before the final steering normalisation."""
        p = self.p
        pre_r, pre_l, pre_2 = self.pre_activities(hd_deg, goal_deg)
        pfl3r = elu1(pre_r, *n.pfl3r.as_tuple())
        pfl3l = elu1(pre_l, *n.pfl3l.as_tuple())
        pfl2 = elu1(pre_2, *n.pfl2.as_tuple())
        if p.pfl3_inhibitory:
            pfl3r, pfl3l = -pfl3r, -pfl3l
        if p.pfl2_silenced:
            pfl2 = np.zeros_like(pfl2)

        dna03r_pre = np.sum(W_PFL3R_DNA03R * pfl3r + W_PFL2_DNA03R * pfl2)
        dna03l_pre = np.sum(W_PFL3L_DNA03L * pfl3l + W_PFL2_DNA03L * pfl2)
        dna03r = float(elu1(dna03r_pre, *n.dna03r.as_tuple()))
        dna03l = float(elu1(dna03l_pre, *n.dna03l.as_tuple()))

        tile = p.dna02_tile_count
        dna02r_pre = (tile * W_DNA03_DNA02 * dna03r
                      + np.sum(W_PFL3R_DNA02R * pfl3r))
        dna02l_pre = (tile * W_DNA03_DNA02 * dna03l
                      + np.sum(W_PFL3L_DNA02L * pfl3l))
        dna02r = float(elu1(dna02r_pre, *n.dna02r.as_tuple()))
        dna02l = float(elu1(dna02l_pre, *n.dna02l.as_tuple()))
        return dna02r - dna02l

    def steering_deg(self, hd_deg: float, goal_deg: float) -> float:
        if self.norm is None:
            raise RuntimeError("no frozen normalisation constants; run "
                               "calibrate() and pass norm=...")
        return float(self._raw(hd_deg, goal_deg, self.norm)
                     / self.norm.steering_max)

    def steering(self, heading: float, goal: float) -> float:
        """SteeringCore protocol: radians in, dimensionless command out."""
        return self.steering_deg(float(np.rad2deg(heading)),
                                 float(np.rad2deg(goal)))

    def as_dict(self) -> Dict:
        return self.p.as_dict()


def official_grid(goal_step: int = 10, hd_step: int = 1, n_scalars: int = 6):
    return (np.arange(-180, 181, goal_step, dtype=float),
            np.linspace(0.0, 1.0, n_scalars),
            np.arange(-180, 181, hd_step, dtype=float))


def calibrate(params: Optional[DataCoreParams] = None, goal_step: int = 10,
              hd_step: int = 1, n_scalars: int = 6,
              verbose: bool = True) -> NormConstants:
    """Freeze this core's batch min/max constants on the official grid.

    Only 12 cells, so the whole grid fits in memory and no streaming is needed
    (unlike the abstract core's 1000-cell calibration).

    These constants are this core's OWN -- they are what the source's batch
    normalisation produces for this connectivity.  The SHARED body scale
    enters downstream, where R_max from the abstract condition is applied.
    """
    p = params if params is not None else DataCoreParams()
    core = WesteindeDataCore(p)
    goals, scalars, hds = official_grid(goal_step, hd_step, n_scalars)

    hd_r = np.stack([core.hd_inputs(float(h))[0] for h in hds])
    hd_l = np.stack([core.hd_inputs(float(h))[1] for h in hds])
    hd_2 = np.stack([core.hd_inputs(float(h))[2] for h in hds])
    goal_in = np.stack([core.goal_input(float(g)) for g in goals])

    # pre-activities over the whole grid, shaped (goal, S, HD, cell)
    S = scalars[None, :, None, None]
    pre_r = S * (hd_r[None, None] + goal_in[:, None, None])
    pre_l = S * (hd_l[None, None] + goal_in[:, None, None])
    pre_2 = S * (hd_2[None, None] + goal_in[:, None, None])

    s_pfl3r, s_pfl3l, s_pfl2 = (_elu1_band(pre_r), _elu1_band(pre_l),
                                _elu1_band(pre_2))
    a_r = elu1(pre_r, *s_pfl3r.as_tuple())
    a_l = elu1(pre_l, *s_pfl3l.as_tuple())
    a_2 = elu1(pre_2, *s_pfl2.as_tuple())
    if p.pfl3_inhibitory:
        a_r, a_l = -a_r, -a_l
    if p.pfl2_silenced:
        a_2 = np.zeros_like(a_2)

    d03r_pre = np.sum(W_PFL3R_DNA03R * a_r + W_PFL2_DNA03R * a_2, axis=-1)
    d03l_pre = np.sum(W_PFL3L_DNA03L * a_l + W_PFL2_DNA03L * a_2, axis=-1)
    s_d03r, s_d03l = _elu1_band(d03r_pre), _elu1_band(d03l_pre)
    d03r = elu1(d03r_pre, *s_d03r.as_tuple())
    d03l = elu1(d03l_pre, *s_d03l.as_tuple())

    tile = p.dna02_tile_count
    d02r_pre = (tile * W_DNA03_DNA02 * d03r
                + np.sum(W_PFL3R_DNA02R * a_r, axis=-1))
    d02l_pre = (tile * W_DNA03_DNA02 * d03l
                + np.sum(W_PFL3L_DNA02L * a_l, axis=-1))
    s_d02r, s_d02l = _elu1_band(d02r_pre), _elu1_band(d02l_pre)
    d02r = elu1(d02r_pre, *s_d02r.as_tuple())
    d02l = elu1(d02l_pre, *s_d02l.as_tuple())

    raw = d02r - d02l
    steering_max = float(raw.max())
    if steering_max <= 0:
        raise RuntimeError("steering_max is not positive (%r); the sign "
                           "convention or the weights are wrong" % steering_max)
    if verbose:
        print("data core: grid %d goals x %d S x %d HD x %d cells"
              % (goals.size, scalars.size, hds.size, N_CELLS))
        print("steering_max = %.10f  (raw range [% .6f, % .6f])"
              % (steering_max, raw.min(), raw.max()))
    return NormConstants(
        pfl2=s_pfl2, pfl3r=s_pfl3r, pfl3l=s_pfl3l,
        dna03r=s_d03r, dna03l=s_d03l, dna02r=s_d02r, dna02l=s_d02l,
        steering_max=steering_max,
        grid_description=("hemibrain data connectivity, goals -180:%d:180, "
                          "hd -180:%d:180, %d scalars, %d cells"
                          % (goal_step, hd_step, n_scalars, N_CELLS)))
