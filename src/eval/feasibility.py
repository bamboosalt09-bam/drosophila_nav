"""Can the BODY do it at all?  (handoff document section 28)

Before any trial outcome is blamed on the neural controller, we have to know
whether the body could physically have completed the task.  If r_max is so
small that the plant cannot rotate 120 degrees within the timeout, a failure
says nothing about the fly circuit (doc section 27).

With a first-order lag in the plant the analytic bang-bang bound gets messy,
so the document asks for a NUMERICAL reachability bound instead.  That is what
this module computes: drive the plant at maximum effort, in the right
direction, from rest, and integrate how far it can turn.

The bound is deliberately generous to the body -- it ignores the need to stop
at the target, and it lets the plant command its own r_max directly with no
controller in the way.  So "infeasible" here means *provably* impossible, and
a condition that passes this test may still be practically very hard.  That
asymmetry is on purpose: we would rather under-report infeasibility than
excuse a controller failure that was really the controller's fault.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from body.yaw_plant import YawPlant, YawPlantParams


@dataclass
class FeasibilityResult:
    required_rotation_rad: float
    max_rotation_rad: float
    min_time_s: Optional[float]      # None if never reached within the horizon
    feasible: bool
    margin: float                    # max_rotation / required_rotation

    def as_dict(self) -> dict:
        return {"required_rotation_deg": float(np.rad2deg(self.required_rotation_rad)),
                "max_rotation_deg": float(np.rad2deg(self.max_rotation_rad)),
                "min_time_s": self.min_time_s,
                "feasible": bool(self.feasible),
                "margin": float(self.margin)}


def max_rotation(params: YawPlantParams, horizon_s: float,
                 target_rotation_rad: Optional[float] = None) -> FeasibilityResult:
    """Largest angle the plant can sweep in `horizon_s`, starting from rest.

    Commands +r_max throughout, i.e. maximum effort in one direction.
    """
    plant = YawPlant(params)
    plant.reset(psi=0.0, r=0.0)

    dt = params.dt_body
    n = max(1, int(round(horizon_s / dt)))
    r_cmd = params.r_max
    if not np.isfinite(r_cmd):
        # an unconstrained body reaches any angle immediately
        req = float(target_rotation_rad or 0.0)
        return FeasibilityResult(req, np.inf, 0.0, True, np.inf)

    swept = 0.0
    min_time = None
    for k in range(n):
        before = plant.r
        plant.step(r_cmd, dt)
        swept += abs(0.5 * (before + plant.r)) * dt      # trapezoid
        if (target_rotation_rad is not None and min_time is None
                and swept >= target_rotation_rad):
            min_time = (k + 1) * dt

    req = float(target_rotation_rad or 0.0)
    feasible = (min_time is not None) if target_rotation_rad is not None else True
    margin = (swept / req) if req > 0 else np.inf
    return FeasibilityResult(req, float(swept), min_time, bool(feasible), float(margin))


def is_task_feasible(params: YawPlantParams, initial_error_rad: float,
                     timeout_s: float, success_tolerance_rad: float
                     ) -> FeasibilityResult:
    """Feasibility of reducing |error| to the success tolerance in time.

    The plant must sweep at least |e0| - tolerance, taking the short way
    around, before the timeout.
    """
    required = max(0.0, abs(float(initial_error_rad)) - float(success_tolerance_rad))
    if required == 0.0:
        return FeasibilityResult(0.0, np.inf, 0.0, True, np.inf)
    return max_rotation(params, timeout_s, required)
