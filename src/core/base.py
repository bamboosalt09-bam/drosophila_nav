"""The contract every neural core must satisfy.

Why this file exists
--------------------
The study has two questions, and only the first one is being answered now:

  Q1 (current study)  Does a FIXED fly-derived steering circuit keep working
                      when the body's motion constraints change?
  Q2 (later)          Does the connectome ITSELF compute navigation, when the
                      whole circuit is instantiated and only the sensory input
                      and motor output nodes are designated?

Decision (2026-09-15): run Q1 now; Q2 is a later expansion, and when it comes
its scope is WHOLE-BRAIN FlyWire, in the spirit of Shiu et al. 2024.

For that to be possible without rewriting the study, every core must be
swappable behind ONE narrow interface.

What this interface has actually been shown to do, and what it has not
---------------------------------------------------------------------
SHOWN: a controller with the SAME input/output MEANING -- a heading and a
goal in, a dimensionless steering command out -- can be substituted without
touching the decoder, plugin, body, sensor, sweep, metrics or renderer.
Condition D (the P controller, src/core/p_controller.py) is that
demonstration, and it is the whole of it.

NOT SHOWN, and not to be promised: that a connectome-scale core needs only
to satisfy this signature.  Such a model additionally requires
  * sensory encoding   -- a heading is not an input to a connectome; some set
                          of cells must be driven, with a chosen tuning,
                          gain and baseline
  * temporal alignment -- the core's own dynamics have a timescale that has
                          no reason to equal T_core = 0.1 s, so the loop's
                          fixed-rate scheduling becomes an assumption rather
                          than a given
  * output decoding    -- which cells are read, and how their activity maps
                          onto one steering scalar, is a modelling decision
                          that does not follow from the connectome
None of those three live behind `steering(heading, goal)`.  They are separate
design problems, and each one is a place where a result could be created by
the interface rather than by the circuit.  The interface keeps the DOWNSTREAM
half of the study fixed; it does not make the upstream half free.

The interface deliberately does NOT expose:
  * any internal state (so "the core is frozen across body conditions" is a
    structural fact, not a convention)
  * any body parameter (the core must not know what it is driving)
  * any physical unit (deg/s conversion belongs to the decoder gain kappa)

A future FlyWire core WILL need internal state (membrane potentials, spike
history).  That is allowed -- see StatefulSteeringCore -- but it must then
expose reset() so that every trial starts from an identical condition, which
is what keeps trials independent and paired comparisons valid.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SteeringCore(Protocol):
    """Minimal contract: (heading, goal) -> raw steering drive.

    Angles are radians, CCW-positive, and the heading is the value reported by
    the SENSOR, never the ground-truth body state and never a command.

    The returned value is unit-less.  Its sign convention is fixed by the
    project: positive when e = wrap(heading - goal) > 0.  Turning that into a
    physical yaw rate, including whatever sign the body frame requires, is the
    decoder's job.
    """

    def steering(self, heading: float, goal: float) -> float:
        ...


@runtime_checkable
class StatefulSteeringCore(SteeringCore, Protocol):
    """A core that carries internal dynamics (e.g. a spiking network).

    Not used by the current reduced model, which is deliberately stateless.
    Declared here so that the Q2 expansion has a defined place to land.
    """

    def reset(self, seed: int | None = None) -> None:
        """Return the core to a fixed initial condition before a trial."""
        ...


def assert_is_steering_core(obj: object) -> None:
    """Fail loudly if an object cannot be used as a core."""
    if not isinstance(obj, SteeringCore):
        raise TypeError(
            "object of type %r does not satisfy the SteeringCore contract: "
            "it must provide steering(heading, goal) -> float"
            % type(obj).__name__
        )
