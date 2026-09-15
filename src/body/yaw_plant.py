"""Abstract 1-DOF yaw plant with rate, acceleration and lag limits.

This is the body of the main experiment.  It is NOT a model of any particular
drone, robot or animal, and must never be described as one (handoff doc
sections 10 and 57).  It is a deliberately minimal plant with three
independent knobs, so that the study can ask which KIND of body limitation
breaks a fixed neural controller first.

State
-----
    psi : actual yaw angle   [rad]
    r   : actual yaw rate    [rad/s]

Update, per handoff document section 10, sub-stepped at dt_body:

    a_req = (r_cmd - r) / tau_r
    a     = clip(a_req, -alpha_max, +alpha_max)
    r     = clip(r + a*dt_body, -r_max, +r_max)
    psi   = wrap(psi + r*dt_body)

tau_r versus alpha_max (handoff doc section 11)
-----------------------------------------------
They are different things and the write-up must say so, or the two will look
like double counting:

  * tau_r     is a first-order response lag: overall bandwidth.  It governs
              how fast r approaches r_cmd for ANY command size.
  * alpha_max is a hard acceleration ceiling.  It only bites on large command
              steps, where the first-order law would otherwise demand an
              acceleration the actuator cannot produce.

Using both is legitimate; they shape different parts of the response.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from utils.angles import wrap


@dataclass
class YawPlantParams:
    """Body capability.  Defaults are effectively unconstrained.

    In the experiments these are set relative to the neural command scale
    (R99, the 99th percentile of |r_brain| in the ideal baseline) rather than
    from any real vehicle specification -- see handoff doc section 17.
    """

    r_max: float = math.inf         # rad/s, hard rate limit
    alpha_max: float = math.inf     # rad/s^2, hard acceleration limit
    tau_r: float = 0.0              # s, first-order response lag; 0 = none
    dt_body: float = 0.0025         # s, integration sub-step
    label: str = "abstract_yaw_plant"

    def as_dict(self) -> Dict:
        def _f(x):
            return None if math.isinf(x) else float(x)
        return {"r_max": _f(self.r_max), "alpha_max": _f(self.alpha_max),
                "tau_r": float(self.tau_r), "dt_body": float(self.dt_body),
                "label": self.label,
                "note": "abstract plant, not a real vehicle specification"}


@dataclass
class YawPlant:
    """Constrained 1-DOF yaw body."""

    params: YawPlantParams = field(default_factory=YawPlantParams)
    name: str = "yaw_plant"
    _psi: float = field(default=0.0, repr=False)
    _r: float = field(default=0.0, repr=False)

    # saturation bookkeeping, for the metrics in handoff doc section 26
    _n_substeps: int = field(default=0, repr=False)
    _n_rate_sat: int = field(default=0, repr=False)
    _n_accel_sat: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        p = self.params
        if p.dt_body <= 0:
            raise ValueError("dt_body must be positive")
        if p.r_max <= 0 or p.alpha_max <= 0:
            raise ValueError("r_max and alpha_max must be positive")
        if p.tau_r < 0:
            raise ValueError("tau_r must be >= 0")
        # Explicit Euler on a first-order lag is only stable while
        # dt_body < 2*tau_r, and only well-resolved well below that.
        if 0 < p.tau_r < 2.0 * p.dt_body:
            raise ValueError(
                "tau_r (%g s) is too small for dt_body (%g s): explicit "
                "integration would be unstable.  Reduce dt_body to at most "
                "tau_r/5." % (p.tau_r, p.dt_body))

    # -- state ------------------------------------------------------------
    @property
    def psi(self) -> float:
        return self._psi

    @property
    def r(self) -> float:
        return self._r

    def reset(self, psi: float = 0.0, r: float = 0.0) -> None:
        self._psi = float(wrap(psi))
        self._r = float(np.clip(r, -self.params.r_max, self.params.r_max))
        self._n_substeps = self._n_rate_sat = self._n_accel_sat = 0

    # -- dynamics ---------------------------------------------------------
    def step(self, r_cmd: float, duration: float) -> None:
        """Hold r_cmd for `duration` seconds, integrating at dt_body."""
        p = self.params
        r_cmd = float(r_cmd)
        n_sub = max(1, int(round(float(duration) / p.dt_body)))
        dt = float(duration) / n_sub

        for _ in range(n_sub):
            if p.tau_r > 0.0:
                a_req = (r_cmd - self._r) / p.tau_r
            else:
                # no lag: ask for whatever acceleration closes the gap in one
                # sub-step, then let alpha_max decide what is actually possible
                a_req = (r_cmd - self._r) / dt

            a = float(np.clip(a_req, -p.alpha_max, p.alpha_max))
            if a != a_req:
                self._n_accel_sat += 1

            r_new = self._r + a * dt
            r_clipped = float(np.clip(r_new, -p.r_max, p.r_max))
            if r_clipped != r_new:
                self._n_rate_sat += 1

            self._r = r_clipped
            self._psi = float(wrap(self._psi + self._r * dt))
            self._n_substeps += 1

    # -- diagnostics ------------------------------------------------------
    @property
    def rate_saturation_fraction(self) -> float:
        return self._n_rate_sat / self._n_substeps if self._n_substeps else 0.0

    @property
    def accel_saturation_fraction(self) -> float:
        return self._n_accel_sat / self._n_substeps if self._n_substeps else 0.0

    def as_dict(self) -> Dict:
        return {"name": self.name, **self.params.as_dict()}


def from_neural_scale(r99: float, r_max_ratio: float = math.inf,
                      alpha_max_ratio: float = math.inf,
                      tau_ratio: float = 0.0, T_core: float = 0.1,
                      dt_body: float = 0.0025) -> YawPlantParams:
    """Build plant parameters in units of the neural command scale.

    Handoff document section 17: body parameters are dimensionless multiples
    of the brain's own command distribution, not invented vehicle specs.

        r_max     = r_max_ratio     * R99
        alpha_max = alpha_max_ratio * R99 / T_core
        tau_r     = tau_ratio       * T_core

    R99 is the 99th percentile of |r_brain| measured in the ideal baseline.
    """
    return YawPlantParams(
        r_max=(math.inf if math.isinf(r_max_ratio) else r_max_ratio * r99),
        alpha_max=(math.inf if math.isinf(alpha_max_ratio)
                   else alpha_max_ratio * r99 / T_core),
        tau_r=tau_ratio * T_core,
        dt_body=dt_body,
        label=("r_max=%sR99 alpha_max=%sR99/T tau=%sT"
               % (r_max_ratio, alpha_max_ratio, tau_ratio)),
    )
