"""Ideal 1-DOF yaw body: no dynamics at all.

    r(t)   = r_cmd                 (instantaneous, unlimited)
    psi(t) = psi + r_cmd * dt

This is condition A, and it is exactly the body implied by the source model's
own closed loop (`hd[t] = hd[t-1] + k*steering`), where the command IS the
motion.  Keeping it as a body object rather than as a special case in the
simulator means condition A runs through the same scheduler as every other
condition, so the comparison A vs B vs C is not confounded by a different code
path.

The constrained plant (rate limit, acceleration cap, response lag) is Stage 1
and belongs in a separate file; it is deliberately not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils.angles import wrap


@dataclass
class IdealYawBody:
    """Perfect integrator with no actuator limits."""

    name: str = "ideal_yaw"
    _psi: float = field(default=0.0, repr=False)
    _r: float = field(default=0.0, repr=False)

    @property
    def psi(self) -> float:
        return self._psi

    @property
    def r(self) -> float:
        return self._r

    def reset(self, psi: float = 0.0, r: float = 0.0) -> None:
        self._psi = float(wrap(psi))
        self._r = float(r)

    def step(self, r_cmd: float, duration: float) -> None:
        """Follow the command exactly for `duration` seconds."""
        self._r = float(r_cmd)
        self._psi = float(wrap(self._psi + self._r * float(duration)))

    def as_dict(self) -> dict:
        return {"name": self.name, "type": "ideal integrator, no limits"}
