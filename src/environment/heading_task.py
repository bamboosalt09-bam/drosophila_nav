"""The goal-heading recovery / stabilisation task.

The first behavioural task is deliberately NOT target approach (handoff doc
section 9): the core provides a steering output directly, so no forward-speed
generator has to be invented, and the effect of body limits stays confined to
the yaw dimension.

The environment owns the goal.  The core receives it, the plugin never does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from utils.angles import deg2rad, wrap

# Handoff doc section 47: calibration and test initial conditions are kept
# separate so that test angles are not reused for tuning.
CALIBRATION_ERRORS_DEG: Sequence[float] = (-135.0, -90.0, -60.0, -30.0,
                                           30.0, 60.0, 90.0, 135.0)
TEST_ERRORS_DEG: Sequence[float] = (-170.0, -150.0, -120.0, -75.0, -45.0,
                                    45.0, 75.0, 120.0, 150.0, 170.0)
# Section 48: exactly anti-goal is a symmetric unstable equilibrium and is
# reported separately, never folded into the success rate.
STRESS_ERRORS_DEG: Sequence[float] = (180.0, 179.0, -179.0)


@dataclass(frozen=True)
class HeadingTask:
    """One trial specification."""

    goal: float = 0.0                 # rad
    initial_heading: float = np.pi    # rad; source starts anti-goal
    duration_s: float = 100.0
    T_core_s: float = 0.1
    name: str = "goal_heading_recovery"

    @property
    def n_cycles(self) -> int:
        return int(round(self.duration_s / self.T_core_s))

    @property
    def initial_error(self) -> float:
        return float(wrap(self.initial_heading - self.goal))

    def as_dict(self) -> dict:
        return {"name": self.name, "goal_deg": float(np.rad2deg(self.goal)),
                "initial_heading_deg": float(np.rad2deg(self.initial_heading)),
                "initial_error_deg": float(np.rad2deg(self.initial_error)),
                "duration_s": self.duration_s, "T_core_s": self.T_core_s,
                "n_cycles": self.n_cycles}


def trials_from_errors(errors_deg: Sequence[float], goal: float = 0.0,
                       duration_s: float = 100.0,
                       T_core_s: float = 0.1) -> List[HeadingTask]:
    """Build one task per initial heading error."""
    return [HeadingTask(goal=goal,
                        initial_heading=float(wrap(goal + deg2rad(e))),
                        duration_s=duration_s, T_core_s=T_core_s)
            for e in errors_deg]
