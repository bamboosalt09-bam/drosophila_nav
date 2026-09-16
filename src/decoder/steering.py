"""Decoder: neural steering drive -> physical yaw rate.

The core returns a unit-less, grid-normalised DNa02R - DNa02L.  The source
paper gives no deg/s conversion constant; what the official notebook gives is

    k  = 200        # factor relating PFL activity to steering
    fs = 10 Hz      # one update every 0.1 s
    hd[t] = wrap(hd[t-1] + k*steering + noise)

so k*steering is a heading INCREMENT in degrees per 0.1 s timestep.  This
module is the single place where that conversion happens, and it is kept out
of the core deliberately: the core must not know what it is driving.

Calibration rule (handoff doc section 46)
-----------------------------------------
kappa is taken from the source, not fitted by us.  It is fixed here once and
frozen for every body condition that follows.  Re-tuning it per body would
destroy exactly the measurement this study is built on.  If it is ever changed,
that is a new experiment and it goes in provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

# Source values, notebook cell 32.
SOURCE_KAPPA_DEG_PER_STEP = 200.0
SOURCE_FS_HZ = 10.0


@dataclass(frozen=True)
class SteeringDecoder:
    """Affine, memory-less decoder.  Frozen after Stage 0."""

    kappa_deg_per_step: float = SOURCE_KAPPA_DEG_PER_STEP
    T_core_s: float = 1.0 / SOURCE_FS_HZ
    name: str = "source_k200"

    # -- source-native form ------------------------------------------------
    def delta_heading_deg(self, steering: float) -> float:
        """Heading increment per core cycle, in degrees (source units)."""
        return self.kappa_deg_per_step * float(steering)

    # -- project form ------------------------------------------------------
    def yaw_rate(self, steering: float) -> float:
        """Requested yaw rate r_brain [rad/s].

        Identical information to delta_heading_deg, expressed as a rate so
        that a body with real dynamics can be driven by it.  For an ideal body
        that simply integrates (psi += r*T_core) the two are equivalent.
        """
        return float(np.deg2rad(self.delta_heading_deg(steering)) / self.T_core_s)

    @property
    def max_yaw_rate(self) -> float:
        """Yaw rate at |steering| = 1, the peak of the calibration grid.

        Useful as the natural scale for body parameters: the handoff document
        (section 17) asks for r_max to be expressed relative to the neural
        command distribution rather than invented from a drone datasheet.
        """
        return self.yaw_rate(1.0)

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["max_yaw_rate_rad_s"] = self.max_yaw_rate
        d["max_yaw_rate_deg_s"] = float(np.rad2deg(self.max_yaw_rate))
        return d


@dataclass(frozen=True)
class DescendingPair:
    """The two-number descending command, as fly simulators actually use it.

    NeuroMechFly 2.0 drives the body with [dL, dR], which modulate the left
    and right leg CPGs: the magnitude sets stepping amplitude and the sign
    sets direction.  Odour taxis, visual taxis and navigation all reduce to
    that 2D action space, and nobody wires motor neurons individually.

    Collapsing the two sides to a single R-L scalar, as this project did up to
    now, throws the forward channel away before anything can look at it -- and
    a visual stimulus may well modulate approach speed rather than turning.
    """

    left: float
    right: float

    @property
    def turn(self) -> float:
        """Positive = turn right."""
        return self.right - self.left

    @property
    def forward(self) -> float:
        """Both sides stepping together."""
        return self.left + self.right

    @property
    def balance(self) -> float:
        """Turn as a fraction of total drive: scale-free, so a network that
        merely gets quieter cannot look like it is steering better."""
        tot = abs(self.left) + abs(self.right)
        return self.turn / tot if tot > 0 else 0.0

    def as_dict(self) -> Dict:
        return {"left": self.left, "right": self.right, "turn": self.turn,
                "forward": self.forward, "balance": self.balance}
