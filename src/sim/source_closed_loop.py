"""Literal replica of the source's closed loop (notebook cell 32).

    hd[0] = 180
    for t in 1..T-1:
        h       = index of the grid HD nearest hd[t-1]
        cmd[t]  = k * steering_grid[h] + noise[t]
        hd[t]   = wrap_to_180(hd[t-1] + cmd[t])

Kept separate from sim/closed_loop.py on purpose.  This loop adds the COMMAND
straight onto the heading: there is no body, no actuator, no actual motion to
feed back.  It is the structure the handoff document (section 3.4) forbids for
the study proper, and reproducing it here is only meaningful as validation --
it is the ideal-body limit that our condition A must agree with.

It also carries an implementation detail of the source worth knowing about:
the heading is SNAPPED to the nearest precomputed grid point on every step, so
the source loop runs on a quantised heading (1 degree on the official grid).
Our own loop evaluates the core at the exact heading instead.  Comparing the
two is how we measure what that quantisation is worth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.angles import wrap_deg


@dataclass
class SourceLoopResult:
    hd_deg: np.ndarray            # heading per timestep
    command_deg: np.ndarray       # k*steering + noise, per timestep
    grid_index: np.ndarray        # which grid point was used
    k: float
    initial_hd_deg: float


def run_source_loop(steering_grid: np.ndarray, hd_grid_deg: np.ndarray,
                    n_steps: int, k: float = 200.0,
                    noise_deg: Optional[np.ndarray] = None,
                    initial_hd_deg: float = 180.0) -> SourceLoopResult:
    """Run the source loop over a precomputed steering lookup table.

    `steering_grid[i]` is the normalised steering command at heading
    `hd_grid_deg[i]`, for one goal phase and one PFL scalar.
    """
    steering_grid = np.asarray(steering_grid, dtype=float)
    hd_grid_deg = np.asarray(hd_grid_deg, dtype=float)
    if steering_grid.shape != hd_grid_deg.shape:
        raise ValueError("steering_grid and hd_grid_deg must have equal shape")

    noise = (np.zeros(n_steps) if noise_deg is None
             else np.asarray(noise_deg, dtype=float))
    if noise.size != n_steps:
        raise ValueError("noise_deg has %d entries, expected %d"
                         % (noise.size, n_steps))

    hd = np.empty(n_steps)
    cmd = np.empty(n_steps)
    idx = np.empty(n_steps, dtype=int)

    hd[0] = initial_hd_deg
    cmd[0] = 0.0
    idx[0] = int(np.argmin(np.abs(wrap_deg(hd_grid_deg - hd[0]))))

    for t in range(1, n_steps):
        current = hd[t - 1]
        h = int(np.argmin(np.abs(wrap_deg(hd_grid_deg - current))))
        idx[t] = h
        cmd[t] = k * steering_grid[h] + noise[t]
        hd[t] = float(wrap_deg(hd[t - 1] + cmd[t]))

    return SourceLoopResult(hd_deg=hd, command_deg=cmd, grid_index=idx,
                            k=float(k), initial_hd_deg=float(initial_hd_deg))


def steering_lookup_table(core, hd_grid_deg, goal_deg: float = 0.0):
    """Evaluate our core on a grid, producing the source's lookup table."""
    return np.array([core.evaluate_deg(float(h), goal_deg).steering
                     for h in np.asarray(hd_grid_deg, dtype=float)])
