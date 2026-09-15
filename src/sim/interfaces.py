"""Contracts between the closed-loop modules.

The point of these Protocols is not type checking for its own sake.  Two of the
study's core rules are enforced here, by what each signature is ALLOWED to see:

  * a ConstraintPlugin receives the brain's requested yaw rate and the body's
    actual yaw rate -- and nothing else.  There is no parameter through which
    a goal, a heading error, a target or a path could reach it.  The "C-only /
    core removed" ablation (handoff doc section 16) is only meaningful if the
    plugin physically cannot navigate, and that is a property of this
    signature, not of anyone's good intentions.

  * a SensorModel reads the BODY STATE.  The command never appears in its
    arguments, so "feed the command back as if it were the heading" (doc
    section 3.4) is not expressible.

Angles are radians, CCW-positive; rates are rad/s.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConstraintPlugin(Protocol):
    """Goal-blind body-side interface: shapes a command into something the
    body can actually execute.

    Body capability parameters (r_max, alpha_max, tau_r, ...) belong to the
    plugin instance.  Goal information does not, and cannot be passed here.
    """

    name: str

    def command(self, r_brain: float, r_actual: float) -> float:
        """Return the yaw rate to request of the body, in rad/s."""
        ...

    def reset(self) -> None:
        """Clear any internal state between trials."""
        ...


@runtime_checkable
class BodyModel(Protocol):
    """The plant.  Owns the ACTUAL state; nothing else may write to it."""

    name: str

    @property
    def psi(self) -> float:
        """Actual yaw angle [rad]."""
        ...

    @property
    def r(self) -> float:
        """Actual yaw rate [rad/s]."""
        ...

    def reset(self, psi: float = 0.0, r: float = 0.0) -> None:
        ...

    def step(self, r_cmd: float, duration: float) -> None:
        """Advance the body by `duration` seconds holding r_cmd (zero-order
        hold over one neural cycle).  Sub-stepping is the body's business."""
        ...


@runtime_checkable
class SensorModel(Protocol):
    """Turns body state into the heading the core actually receives."""

    name: str

    def observe(self, body: BodyModel) -> float:
        """Observed heading [rad]."""
        ...

    def reset(self) -> None:
        ...
