"""Ideal heading sensor: privileged observation of the body's actual yaw.

    theta_obs = psi_actual

This is a PRIVILEGED STATE OBSERVATION, not a model of any fly sensory
pathway, and it must be described that way in any write-up (handoff doc
section 13).  It is the right choice for the first experiments because it
isolates the effect of body constraints: with a perfect sensor, any failure
observed later is attributable to the body or the plugin, never to perception.

Crucially it reads the BODY, so what returns to the core is where the body
actually is -- never the command the core asked for.  Noise, latency, dropout
and partial observation are later extensions and get their own classes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IdealHeadingSensor:
    """Returns the body's true heading, unmodified."""

    name: str = "ideal_heading"

    def observe(self, body) -> float:
        return float(body.psi)

    def reset(self) -> None:
        pass

    def as_dict(self) -> dict:
        return {"name": self.name,
                "type": "privileged state observation (theta_obs = psi_actual)"}
