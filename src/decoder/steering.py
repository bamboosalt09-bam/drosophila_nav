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
from typing import Dict, Optional

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


@dataclass(frozen=True)
class MotorReadout:
    """Leg motor activity, read from the connectome rather than invented.

    On FAFB there were no motor neurons, so a steering command had to be
    constructed from 1,303 descending neurons spanning 473 cell types --
    walking, flight, grooming and feeding lumped together, with DNa02, the
    pair the Westeinde model actually reads, just 2 of them.  MaleCNS contains
    the nerve cord, so the readout is simply what the leg motor neurons do.

    The shape matches how fly simulators drive a body: NeuroMechFly takes a
    two-number descending command [dL, dR] into left and right leg CPGs.  Here
    the same two numbers come out of the motor neurons themselves, per leg
    pair, so the turn is the left-right difference and forward drive is the
    sum -- and the per-segment detail is available if it is ever wanted.
    """

    left: Dict[str, float]      # neuromere (T1/T2/T3) -> mean rate
    right: Dict[str, float]

    @property
    def segments(self):
        return sorted(set(self.left) | set(self.right))

    def pair(self, segment: Optional[str] = None) -> DescendingPair:
        """[dL, dR] for one leg pair, or summed over all three."""
        if segment is not None:
            return DescendingPair(self.left.get(segment, 0.0),
                                  self.right.get(segment, 0.0))
        return DescendingPair(sum(self.left.values()),
                              sum(self.right.values()))

    @property
    def turn(self) -> float:
        return self.pair().turn

    @property
    def forward(self) -> float:
        return self.pair().forward

    def as_dict(self) -> Dict:
        d = {"turn": self.turn, "forward": self.forward,
             "balance": self.pair().balance}
        for s in self.segments:
            p = self.pair(s)
            d["turn_%s" % s] = p.turn
            d["forward_%s" % s] = p.forward
        return d


def read_motor(rates, ann, segments=("T1", "T2", "T3")) -> MotorReadout:
    """Mean motor-neuron rate per leg segment and side.

    `rates` is one value per neuron, aligned to `ann`.  Selection is
    anatomical throughout: superclass says which cells are motor neurons,
    somaNeuromere says which leg they drive, somaSide which side.
    """
    import numpy as np

    sc = ann["super_class"].to_numpy(dtype="<U32")
    nm = ann["neuromere"].to_numpy(dtype="<U16")
    side = ann["side"].to_numpy(dtype="<U16")
    motor = np.isin(sc, ("vnc_motor", "cb_motor", "vnc_efferent"))

    out = {"left": {}, "right": {}}
    for s in segments:
        for lr in ("left", "right"):
            m = motor & (nm == s) & (side == lr)
            out[lr][s] = float(np.asarray(rates)[m].mean()) if m.any() else 0.0
    return MotorReadout(left=out["left"], right=out["right"])
