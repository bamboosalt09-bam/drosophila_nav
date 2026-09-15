"""Plugin C: goal-blind lag compensation.

The handoff document's own condition C (section 14.2) was a feasibility clamp.
Measured, it is bit-identical to plugin B in every cell where B fails, because
alpha_max*tau_r vastly exceeds r_max there -- section 14.2 anticipated exactly
that and said C would have to be redesigned.  This is the redesign.

The failures that survive in the sweep are caused by RESPONSE LAG, so compensate
that instead.  Over one neural cycle the first-order plant does

    r(T) = u + (r0 - u) * exp(-T/tau)

so the command that lands the actual yaw rate on r_brain after one cycle is

    u = (r_brain - r0*a) / (1 - a),    a = exp(-T/tau)

Exact inversion, no tuning knob, then clipped to what the body can hold.

Goal-blind, and measured to be so: with the core output forced to zero this
plugin moves the heading by exactly 0.0 deg (tests/test_plugin_limits.py).
It only ever knows the requested rate, the actual rate, and its own body's
tau and r_max -- all of which section 3.5 explicitly permits.

It does not repeal physics: where the acceleration cap actually binds it helps
only partially (0.22 -> 0.44 at alpha_max = 0.005 R_max/T), because the plant
still cannot deliver the commanded change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LagCompPlugin:
    """Invert one cycle of the body's first-order lag, then clip to r_max."""

    r_max: float = math.inf
    tau_r: float = 0.0
    T_core: float = 0.1
    name: str = "lag_comp"
    _a: float = field(default=0.0, repr=False)
    _n_calls: int = field(default=0, repr=False)
    _n_clipped: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._a = math.exp(-self.T_core / self.tau_r) if self.tau_r > 0 else 0.0

    def command(self, r_brain: float, r_actual: float) -> float:
        self._n_calls += 1
        u = ((r_brain - r_actual * self._a) / (1.0 - self._a)
             if self._a < 1.0 else r_brain)
        out = float(np.clip(u, -self.r_max, self.r_max))
        if out != u:
            self._n_clipped += 1
        return out

    def reset(self) -> None:
        self._n_calls = self._n_clipped = 0

    @property
    def intervention_fraction(self) -> float:
        return self._n_clipped / self._n_calls if self._n_calls else 0.0

    def as_dict(self) -> dict:
        return {"name": self.name, "tau_r": self.tau_r, "T_core": self.T_core,
                "r_max": None if math.isinf(self.r_max) else float(self.r_max),
                "law": "u = (r_brain - r*exp(-T/tau)) / (1 - exp(-T/tau))"}
