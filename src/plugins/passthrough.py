"""Passthrough plugin: the null body-side interface.

Condition A uses this.  It exists so that the closed loop always has the same
shape -- decoder -> plugin -> body -> sensor -- even when no constraint is
being applied, so that conditions A/B/C differ only in which plugin and body
are installed.
"""
from __future__ import annotations


class PassthroughPlugin:
    """r_cmd = r_brain.  Sees no goal, and could not use one if it did."""

    name = "passthrough"

    def command(self, r_brain: float, r_actual: float) -> float:
        return float(r_brain)

    def reset(self) -> None:
        pass
