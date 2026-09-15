"""Plugin B: simple rate clip (handoff doc section 14.1).

    r_cmd = clip(r_brain, -r_max, +r_max)

The simplest possible body-side feasibility interface: it refuses to ask the
body for a yaw rate the body cannot reach, and does nothing else.
Acceleration limiting and response lag stay in the body, where they belong.

Goal-blind by construction: `command` receives the brain's requested rate and
the body's actual rate, and the only extra thing this object knows is its own
r_max.  It has no way to learn where the goal is, so the "core removed"
ablation must leave it steering nowhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RateClipPlugin:
    """Clip the requested yaw rate to the body's rate capability."""

    r_max: float = math.inf
    name: str = "rate_clip"
    _n_calls: int = 0
    _n_clipped: int = 0

    def command(self, r_brain: float, r_actual: float) -> float:
        self._n_calls += 1
        out = float(np.clip(r_brain, -self.r_max, self.r_max))
        if out != float(r_brain):
            self._n_clipped += 1
        return out

    def reset(self) -> None:
        self._n_calls = 0
        self._n_clipped = 0

    @property
    def intervention_fraction(self) -> float:
        """Share of cycles where the plugin actually changed the command
        (handoff doc section 26, metric 8)."""
        return self._n_clipped / self._n_calls if self._n_calls else 0.0

    def as_dict(self) -> dict:
        return {"name": self.name,
                "r_max": None if math.isinf(self.r_max) else float(self.r_max)}
