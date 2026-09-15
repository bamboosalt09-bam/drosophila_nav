"""Condition D: generic P controller baseline (handoff doc section 15).

Drop-in for the fly core -- same `steering(heading, goal)` signature, same
decoder, plugin and body downstream -- so A/B/D differ in exactly one object.
Its job is to answer "is this body condition workable at all by a sane
controller?", separating body infeasibility from fly-circuit failure.

    steering = clip(-K * e, -1, +1),   e = wrap(heading - goal)

K is dimensionless (per radian) and lives in the same normalised output space
as the core, so the frozen decoder gain applies unchanged.  Chosen once on the
CALIBRATION initial conditions with an ideal body, then frozen like kappa
(sections 15, 47).  Run this file to redo that choice.

    PYTHONPATH=src python -m core.p_controller
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.angles import wrap

# calibrated below; frozen thereafter
CALIBRATED_K = 0.35


@dataclass
class PController:
    """Proportional heading controller in the core's output space."""

    K: float = CALIBRATED_K
    name: str = "p_controller"

    def steering(self, heading: float, goal: float) -> float:
        return float(np.clip(-self.K * float(wrap(heading - goal)), -1.0, 1.0))

    def as_dict(self) -> dict:
        return {"name": self.name, "K_per_rad": self.K,
                "law": "steering = clip(-K*wrap(heading-goal), -1, 1)"}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from body.ideal_yaw import IdealYawBody
    from decoder.steering import SteeringDecoder
    from environment.heading_task import CALIBRATION_ERRORS_DEG, trials_from_errors
    from eval import metrics as mx
    from plugins.passthrough import PassthroughPlugin
    from sensors.ideal_heading import IdealHeadingSensor
    from sim.closed_loop import run_closed_loop

    dec, crit = SteeringDecoder(), mx.SuccessCriterion()
    print("K      success   mean IAE   mean settle")
    best = None
    for K in (0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 5.0):
        ok, iae, settle = 0, [], []
        for t in trials_from_errors(CALIBRATION_ERRORS_DEG, duration_s=15.0):
            r = run_closed_loop(PController(K), dec, PassthroughPlugin(),
                                IdealYawBody(), IdealHeadingSensor(), t)
            m = mx.compute(r, crit)
            ok += m.success
            iae.append(m.iae_deg_s)
            settle.append(m.settling_time_s if m.settling_time_s else np.nan)
        n = len(CALIBRATION_ERRORS_DEG)
        print("%-6.2f %2d/%-6d %9.1f %12.2f"
              % (K, ok, n, np.mean(iae), np.nanmean(settle)))
        if ok == n and (best is None or np.mean(iae) < best[1]):
            best = (K, float(np.mean(iae)))

    assert best, "no K succeeded on every calibration heading"
    print("best K = %.2f (IAE %.1f); CALIBRATED_K = %.2f"
          % (best[0], best[1], CALIBRATED_K))
    assert abs(best[0] - CALIBRATED_K) < 1e-9, (
        "calibration moved: update CALIBRATED_K to %.2f" % best[0])
