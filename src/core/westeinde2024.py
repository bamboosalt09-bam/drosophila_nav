"""Reduced Drosophila central-complex steering core.

Source model
------------
Westeinde et al., "Transforming a head direction signal into a goal-oriented
steering command", Nature 2024.  Reduced PFL2 / PFL3 -> DNa03 -> DNa02
pathway.  Connectome reference: hemibrain v1.2.1.

This file is a *connectome-informed reduced model*, NOT a raw connectome
simulation.  Never describe its output as "running the fly connectome".

What is SOURCE-DERIVED (handoff document, section 6)
----------------------------------------------------
  S1  phase shift PFL3R = +67.5 deg, PFL3L = -67.5 deg, PFL2 = 180 deg
  S2  goal input enters with amplitude A (default A = 1)
  S3  relative connection weights
        PFL3 -> DNa03 = 1 ,  PFL3 -> DNa02 = 1
        PFL2 -> DNa03 = 4 ,  DNa03 -> DNa02 = 12
  S4  all modelled presynaptic pathways treated as excitatory (cholinergic)
  S5  steering readout = DNa02R - DNa02L
  S6  population discretisation ~1000 units per population

What is an EXPLICIT ASSUMPTION of this implementation
-----------------------------------------------------
  A1  Goal input as a function of neural phase.
      The handoff doc writes "I_goal = A cos(theta_g - theta0 - h)", which
      contains no theta and is therefore CONSTANT across the population; a
      constant input cannot create any left/right asymmetry, so the model
      would output exactly zero steering for every heading error.  We
      implement the standard goal-bump form instead:
          I_goal(theta) = A * cos(theta - theta0 - theta_g)
      i.e. a goal bump at neural phase theta_g, summed with the heading input.
      FLAGGED: must be checked against the official notebook.
  A2  Activation = ELU applied to (gain_S * drive); normalisation mode and the
      placement of the gain are configurable, defaults chosen here.
  A3  Population readout = MEAN over units (not sum), so results do not depend
      on n_units.
  A4  Whether the ELU is also applied at the DNa03 / DNa02 stage is a flag
      (dn_activation).  With a linear DN stage PFL2 cancels exactly in
      DNa02R - DNa02L; only a nonlinear DN stage gives PFL2 a gain-modulating
      role.  Default: True.
  A5  Sign convention.  `steering` returned here is the raw DNa02R - DNa02L.
      Mapping it to a physical yaw rate (including its sign) is the job of the
      decoder, not of this core.

Frame note
----------
Heading h and goal theta_g are radians, CCW-positive (see utils.angles).
Heading error used throughout the project:  e = wrap(h - theta_g).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import numpy as np

from utils.angles import deg2rad, neural_phase_grid, wrap


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CoreParams:
    """Parameters of the reduced steering core.

    Source-derived values keep their documented defaults.  Anything marked
    ASSUMPTION is a simulation choice and must be reported as such.
    """

    # --- source-derived (S1, S2, S6) -------------------------------------
    n_units: int = 1000
    phase_shift_pfl3_deg: float = 67.5
    phase_shift_pfl2_deg: float = 180.0
    goal_amplitude: float = 1.0
    theta0_deg: float = 0.0

    # --- source-derived relative weights (S3) ----------------------------
    w_pfl3_dna03: float = 1.0
    w_pfl3_dna02: float = 1.0
    w_pfl2_dna03: float = 4.0
    w_dna03_dna02: float = 12.0

    # --- ASSUMPTIONS (A2, A4) --------------------------------------------
    gain_S: float = 1.0
    elu_alpha: float = 1.0
    normalize: str = "none"          # none | global_peak | global_rms
    dn_activation: bool = True

    label: str = "westeinde2024_reduced"

    def as_dict(self) -> Dict:
        return asdict(self)

    @property
    def phase_shift_pfl3(self) -> float:
        return float(deg2rad(self.phase_shift_pfl3_deg))

    @property
    def phase_shift_pfl2(self) -> float:
        return float(deg2rad(self.phase_shift_pfl2_deg))

    @property
    def theta0(self) -> float:
        return float(deg2rad(self.theta0_deg))


@dataclass
class CoreState:
    """Full snapshot of one core evaluation (for plotting / diagnostics)."""

    heading: float
    goal: float
    heading_error: float
    theta: np.ndarray = field(repr=False)
    drive_pfl3r: np.ndarray = field(repr=False)
    drive_pfl3l: np.ndarray = field(repr=False)
    drive_pfl2: np.ndarray = field(repr=False)
    act_pfl3r: np.ndarray = field(repr=False)
    act_pfl3l: np.ndarray = field(repr=False)
    act_pfl2: np.ndarray = field(repr=False)
    pfl3r: float = 0.0
    pfl3l: float = 0.0
    pfl2: float = 0.0
    dna03r: float = 0.0
    dna03l: float = 0.0
    dna02r: float = 0.0
    dna02l: float = 0.0
    steering: float = 0.0


def elu(x, alpha: float = 1.0):
    """Exponential linear unit: x for x > 0, else alpha * (exp(x) - 1)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0.0, x, alpha * np.expm1(np.minimum(x, 0.0)))


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
class WesteindeSteeringCore:
    """Stateless (instantaneous rate) reduced steering core.

    The core is deliberately memory-less: given (heading, goal) it returns the
    steering drive.  All temporal behaviour lives in the closed-loop simulator,
    the body and the sensors, never here.  That is what makes it possible to
    freeze the core across every body condition.
    """

    def __init__(self, params: Optional[CoreParams] = None):
        self.p = params if params is not None else CoreParams()
        self.theta = neural_phase_grid(self.p.n_units)

    # -- internal helpers -------------------------------------------------
    def _drives(self, heading: float, goal: float):
        """Summed synaptic drive per population as a function of theta.

        phi = theta - theta0 - heading      (heading-referenced neural phase)
        heading term : cos(phi + shift)
        goal term    : A * cos(theta - theta0 - goal) = A * cos(phi + e)  [A1]
        """
        p = self.p
        phi = self.theta - p.theta0 - float(heading)
        e = float(wrap(float(heading) - float(goal)))

        goal_term = p.goal_amplitude * np.cos(phi + e)

        d_r = np.cos(phi + p.phase_shift_pfl3) + goal_term
        d_l = np.cos(phi - p.phase_shift_pfl3) + goal_term
        d_2 = np.cos(phi + p.phase_shift_pfl2) + goal_term
        return d_r, d_l, d_2, e

    def _normalise(self, d_r, d_l, d_2):
        """Shared (global) normalisation factor -- A2.

        IMPORTANT: normalisation must be COMMON to all populations.  Dividing
        each population by its own peak would erase the R/L amplitude
        difference and the core would output exactly zero steering.
        """
        mode = self.p.normalize
        if mode == "none":
            return d_r, d_l, d_2, 1.0
        stacked = np.concatenate([d_r, d_l, d_2])
        if mode == "global_peak":
            scale = float(np.max(np.abs(stacked)))
        elif mode == "global_rms":
            scale = float(np.sqrt(np.mean(stacked ** 2)))
        else:
            raise ValueError("unknown normalize mode: " + repr(mode))
        if scale <= 1e-12:
            scale = 1.0
        return d_r / scale, d_l / scale, d_2 / scale, scale

    # -- public API -------------------------------------------------------
    def evaluate(self, heading: float, goal: float) -> CoreState:
        """Evaluate the core for one (heading, goal) pair."""
        p = self.p
        d_r, d_l, d_2, e = self._drives(heading, goal)
        n_r, n_l, n_2, _scale = self._normalise(d_r, d_l, d_2)

        a_r = elu(p.gain_S * n_r, p.elu_alpha)
        a_l = elu(p.gain_S * n_l, p.elu_alpha)
        a_2 = elu(p.gain_S * n_2, p.elu_alpha)

        # A3: population readout = mean over the phase grid
        pfl3r = float(np.mean(a_r))
        pfl3l = float(np.mean(a_l))
        pfl2 = float(np.mean(a_2))

        if p.dn_activation:
            def g(x):
                return float(elu(x, p.elu_alpha))
        else:
            def g(x):
                return float(x)

        # PFL2 is bilateral: identical drive to DNa03R and DNa03L (S3, S4)
        dna03r = g(p.w_pfl3_dna03 * pfl3r + p.w_pfl2_dna03 * pfl2)
        dna03l = g(p.w_pfl3_dna03 * pfl3l + p.w_pfl2_dna03 * pfl2)

        dna02r = g(p.w_pfl3_dna02 * pfl3r + p.w_dna03_dna02 * dna03r)
        dna02l = g(p.w_pfl3_dna02 * pfl3l + p.w_dna03_dna02 * dna03l)

        steering = dna02r - dna02l  # S5 (raw, unit-less)

        return CoreState(
            heading=float(heading), goal=float(goal), heading_error=e,
            theta=self.theta,
            drive_pfl3r=d_r, drive_pfl3l=d_l, drive_pfl2=d_2,
            act_pfl3r=a_r, act_pfl3l=a_l, act_pfl2=a_2,
            pfl3r=pfl3r, pfl3l=pfl3l, pfl2=pfl2,
            dna03r=dna03r, dna03l=dna03l,
            dna02r=dna02r, dna02l=dna02l,
            steering=float(steering),
        )

    def steering(self, heading: float, goal: float) -> float:
        """Raw steering drive DNa02R - DNa02L (no physical units)."""
        return self.evaluate(heading, goal).steering

    def sweep(self, heading_errors, goal: float = 0.0):
        """Evaluate over an array of heading errors (radians).

        Returns a dict of 1-D arrays.  With goal = 0, heading = heading_error.
        """
        errs = np.atleast_1d(np.asarray(heading_errors, dtype=float))
        keys = ("pfl3r", "pfl3l", "pfl2", "dna03r", "dna03l",
                "dna02r", "dna02l", "steering")
        out = {k: np.empty(errs.size) for k in keys}
        out["heading_error"] = np.empty(errs.size)
        for i, e in enumerate(errs):
            st = self.evaluate(goal + float(e), goal)
            for k in keys:
                out[k][i] = getattr(st, k)
            out["heading_error"][i] = st.heading_error
        return out
