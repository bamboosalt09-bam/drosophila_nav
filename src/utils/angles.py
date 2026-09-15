"""Angle conventions for the whole project.

CONVENTION (fixed once, never changed silently):
  * All angles INSIDE the code are radians.
  * Degrees appear only at the user interface (configs, plot labels, CLI).
  * psi / heading increases COUNTER-CLOCKWISE (CCW) when viewed from above.
  * yaw rate r > 0  ==  CCW rotation.
  * wrap() maps any angle to the half-open interval [-pi, +pi).

Everything that needs an angle must go through these helpers so that a
degree/radian mistake can only happen in one place.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def deg2rad(x):
    """Degrees -> radians (thin wrapper, kept for explicitness)."""
    return np.deg2rad(x)


def rad2deg(x):
    """Radians -> degrees."""
    return np.rad2deg(x)


def wrap(angle):
    """Wrap angle(s) in radians to [-pi, +pi).

    >>> float(wrap(np.pi))        # +pi wraps to -pi (half-open interval)
    -3.14159...
    """
    return (np.asarray(angle, dtype=float) + np.pi) % TWO_PI - np.pi


def wrap_deg(angle_deg):
    """Wrap angle(s) in degrees to [-180, +180)."""
    return (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def angular_difference(a, b):
    """Shortest signed difference a - b, wrapped to [-pi, +pi).

    Positive result means `a` is CCW of `b`.
    """
    return wrap(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def neural_phase_grid(n_units: int) -> np.ndarray:
    """Uniform phase grid theta over [0, 2pi), used to discretise a population.

    n_units is a *discretisation* parameter of the reduced model (smooth
    approximation of the neural phase space), NOT a biological cell count.
    """
    if n_units < 4:
        raise ValueError("n_units must be >= 4 for a meaningful phase grid")
    return np.linspace(0.0, TWO_PI, n_units, endpoint=False)
