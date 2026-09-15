"""Trial metrics (handoff document section 26).

The success criterion is a SIMULATION ANALYSIS CRITERION, not a biological
threshold (doc section 25): reach |error| <= tol within `timeout_s` and hold it
for at least `hold_s`.  tol is swept (10/15/20 deg) in the sensitivity
analysis, so nothing downstream may hard-code 15 degrees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SuccessCriterion:
    tolerance_rad: float = np.deg2rad(15.0)
    timeout_s: float = 10.0
    hold_s: float = 1.0

    def as_dict(self) -> Dict:
        return {"tolerance_deg": float(np.rad2deg(self.tolerance_rad)),
                "timeout_s": self.timeout_s, "hold_s": self.hold_s,
                "note": "simulation analysis criterion, not a biological threshold"}


@dataclass
class TrialMetrics:
    success: bool
    settling_time_s: Optional[float]
    iae_deg_s: float                  # integral of |error| dt
    max_abs_error_deg: float
    max_overshoot_deg: float
    zero_crossings: int
    tail_zero_crossings: int
    total_rotation_deg: float         # integral of |r| dt
    brain_body_mismatch_deg: float    # integral of |r_brain - r_actual| dt
    final_abs_error_deg: float
    tail_spread_deg: float
    plugin_intervention_fraction: float
    rate_saturation_fraction: float
    accel_saturation_fraction: float
    timeout: bool

    def as_dict(self) -> Dict:
        return asdict(self)


def _first_sustained(mask: np.ndarray, n_hold: int) -> Optional[int]:
    """First index i with mask[i:i+n_hold] all True."""
    if mask.size < n_hold:
        return None
    # rolling all() via cumulative sum of the negation
    bad = np.cumsum(~mask)
    for i in range(mask.size - n_hold + 1):
        if bad[i + n_hold - 1] - (bad[i - 1] if i else 0) == 0:
            return i
    return None


def compute(result, criterion: Optional[SuccessCriterion] = None,
            plugin=None, body=None) -> TrialMetrics:
    """Metrics for one LoopResult."""
    c = criterion if criterion is not None else SuccessCriterion()
    dt = float(result.t[1] - result.t[0])
    err = result.error                       # rad, length n+1
    abs_err = np.abs(err)

    n_hold = max(1, int(round(c.hold_s / dt)))
    within = abs_err <= c.tolerance_rad
    idx = _first_sustained(within, n_hold)
    settling = float(result.t[idx]) if idx is not None else None
    success = bool(settling is not None and settling <= c.timeout_s)

    # overshoot: how far past the goal it went after first crossing zero
    sign = np.sign(err)
    nz = sign[sign != 0]
    crossings = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    first_cross = None
    for i in range(1, err.size):
        if err[i - 1] != 0 and np.sign(err[i]) != np.sign(err[i - 1]):
            first_cross = i
            break
    overshoot = float(np.max(abs_err[first_cross:])) if first_cross is not None else 0.0

    tail = err[-max(2, n_hold):]
    r_actual = result.r[1:]

    # Oscillation has to be judged on the part of the trajectory that matters,
    # and only where the excursion is meaningful: a converged trial flips sign
    # constantly at the 1e-9 level, which is not oscillation.
    tail_start = max(0, err.size - int(round(c.timeout_s / dt)))
    tail_err = err[tail_start:]
    significant = tail_err[np.abs(tail_err) > 0.1 * c.tolerance_rad]
    ts = np.sign(significant)
    ts = ts[ts != 0]
    tail_crossings = int(np.sum(ts[1:] != ts[:-1])) if ts.size > 1 else 0

    return TrialMetrics(
        success=success,
        settling_time_s=settling,
        iae_deg_s=float(np.rad2deg(np.sum(abs_err) * dt)),
        max_abs_error_deg=float(np.rad2deg(np.max(abs_err))),
        max_overshoot_deg=float(np.rad2deg(overshoot)),
        zero_crossings=crossings,
        tail_zero_crossings=tail_crossings,
        total_rotation_deg=float(np.rad2deg(np.sum(np.abs(result.r)) * dt)),
        brain_body_mismatch_deg=float(
            np.rad2deg(np.sum(np.abs(result.r_brain - r_actual)) * dt)),
        final_abs_error_deg=float(np.rad2deg(abs_err[-1])),
        tail_spread_deg=float(np.rad2deg(np.max(tail) - np.min(tail))),
        plugin_intervention_fraction=float(
            getattr(plugin, "intervention_fraction", 0.0)),
        rate_saturation_fraction=float(
            getattr(body, "rate_saturation_fraction", 0.0)),
        accel_saturation_fraction=float(
            getattr(body, "accel_saturation_fraction", 0.0)),
        timeout=not success,
    )
