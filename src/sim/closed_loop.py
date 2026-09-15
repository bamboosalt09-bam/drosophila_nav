"""The closed-loop scheduler.

Execution order per neural cycle, fixed by handoff document section 12:

    1. read the body's ACTUAL psi
    2. sensor produces the observed heading
    3. core maps (observed heading, goal) -> steering drive
    4. decoder maps steering -> r_brain
    5. command noise, if any, is added to r_brain
    6. plugin maps (r_brain, actual r) -> r_cmd
    7. body integrates r_cmd for one cycle, zero-order hold
    8. the NEXT cycle observes the resulting actual motion

The command is never fed back as if it were the heading.  Step 1 reads the
body and step 7 writes it; there is no path from step 4 or 6 back into step 2.
That is structural, not a convention: the sensor's `observe` takes the body,
and nothing else.

Everything in this module is condition-agnostic.  Conditions A/B/C differ only
in which plugin and which body are passed in; the core and the decoder are the
same objects in all of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from utils.angles import wrap


@dataclass
class LoopResult:
    """Per-cycle log of one trial.  All arrays have length n_cycles + 1 for
    state (including the initial condition) and n_cycles for commands."""

    t: np.ndarray                 # s, state timestamps
    psi: np.ndarray               # rad, ACTUAL heading
    r: np.ndarray                 # rad/s, ACTUAL yaw rate
    heading_obs: np.ndarray       # rad, what the core saw
    steering: np.ndarray          # unit-less core output
    r_brain: np.ndarray           # rad/s, decoder output (noise included)
    r_cmd: np.ndarray             # rad/s, plugin output
    noise: np.ndarray             # rad/s, injected command noise
    goal: float
    meta: Dict = field(default_factory=dict)

    @property
    def error(self) -> np.ndarray:
        """Heading error e = wrap(psi - goal), rad."""
        return wrap(self.psi - self.goal)

    @property
    def error_deg(self) -> np.ndarray:
        return np.rad2deg(self.error)

    @property
    def brain_body_mismatch(self) -> np.ndarray:
        """|r_brain - r_actual| per cycle (handoff doc section 26, metric 7).

        Compared against the yaw rate the body actually had while the command
        was in force, i.e. the state AFTER the step.
        """
        return np.abs(self.r_brain - self.r[1:])

    def as_dataframe(self):
        import pandas as pd
        n = self.steering.size
        return pd.DataFrame({
            "t_s": self.t[:n],
            "psi_deg": np.rad2deg(self.psi[:n]),
            "error_deg": self.error_deg[:n],
            "heading_obs_deg": np.rad2deg(self.heading_obs),
            "steering": self.steering,
            "r_brain_deg_s": np.rad2deg(self.r_brain),
            "r_cmd_deg_s": np.rad2deg(self.r_cmd),
            "r_actual_deg_s": np.rad2deg(self.r[1:n + 1]),
            "noise_deg_s": np.rad2deg(self.noise),
        })


def run_closed_loop(core, decoder, plugin, body, sensor, task,
                    command_noise: Optional[np.ndarray] = None) -> LoopResult:
    """Run one trial.  See the module docstring for the cycle order.

    `command_noise` is in rad/s and is added to r_brain, one value per cycle.
    """
    n = task.n_cycles
    noise = (np.zeros(n) if command_noise is None
             else np.asarray(command_noise, dtype=float))
    if noise.size != n:
        raise ValueError("command_noise has %d entries, expected %d"
                         % (noise.size, n))

    plugin.reset()
    sensor.reset()
    body.reset(psi=task.initial_heading, r=0.0)

    psi = np.empty(n + 1)
    r = np.empty(n + 1)
    heading_obs = np.empty(n)
    steering = np.empty(n)
    r_brain = np.empty(n)
    r_cmd = np.empty(n)

    psi[0] = body.psi
    r[0] = body.r

    for k in range(n):
        # 1-2. the core only ever learns about the body through the sensor
        obs = sensor.observe(body)
        heading_obs[k] = obs

        # 3-4. neural core, then decoder
        s = core.steering(obs, task.goal)
        steering[k] = s
        rb = decoder.yaw_rate(s) + noise[k]
        r_brain[k] = rb

        # 6. goal-blind plugin
        rc = plugin.command(rb, body.r)
        r_cmd[k] = rc

        # 7. the body, and only the body, advances the actual state
        body.step(rc, task.T_core_s)
        psi[k + 1] = body.psi
        r[k + 1] = body.r

    return LoopResult(
        t=np.arange(n + 1) * task.T_core_s,
        psi=psi, r=r, heading_obs=heading_obs, steering=steering,
        r_brain=r_brain, r_cmd=r_cmd, noise=noise, goal=float(task.goal),
        meta={"plugin": getattr(plugin, "name", type(plugin).__name__),
              "body": getattr(body, "name", type(body).__name__),
              "sensor": getattr(sensor, "name", type(sensor).__name__),
              "decoder": getattr(decoder, "name", type(decoder).__name__),
              "task": task.as_dict()},
    )


def resultant_length(angles_rad) -> float:
    """rho = |mean(exp(i*theta))|, the circular concentration of the heading.

    The source reports 1 - circvar(HD) as "consistency of head direction";
    under the current scipy definition that is this quantity.  Computed
    directly here so the number does not depend on a scipy version.
    """
    a = np.asarray(angles_rad, dtype=float)
    return float(np.abs(np.mean(np.exp(1j * a))))
