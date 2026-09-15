"""Command noise, reproducing the source's recipe exactly.

Notebook cell 32:

    random_noise_gaussian = np.random.normal(size=(num_timepoints, 1))
    random_noise_filtered = butter_lowpass_filter(gaussian, fpass, fs)
    random_noise = noise_level * zscore(random_noise_filtered)

with fpass = 2 Hz, fs = 10 Hz, order 5, noise_level = 10, and the result added
to the heading increment in DEGREES PER TIMESTEP.

Two properties of the source recipe worth keeping in mind before interpreting
anything: the filter is `lfilter`, i.e. causal and phase-shifting (not
filtfilt), and the z-score is taken AFTER filtering, so the scaling is set by
the realised sample rather than by the theoretical variance.  Both are
reproduced here rather than "improved", because a reproduction that quietly
fixes the source is no longer a reproduction.

Where the noise enters our architecture
---------------------------------------
The source adds it to the command, in command units, with no body in between.
We inject it on r_brain, immediately after the decoder, i.e. we treat it as
variability in the steering drive itself.  That keeps condition A numerically
identical to the source while leaving the plugin and the body free to react to
it in later conditions -- which is the honest arrangement: a noisy neural
command is something a constrained body has to cope with.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter
from scipy.stats import zscore

SOURCE_NOISE_LEVEL = 10.0     # degrees per timestep
SOURCE_FPASS_HZ = 2.0
SOURCE_FS_HZ = 10.0
SOURCE_ORDER = 5


@dataclass(frozen=True)
class SourceNoiseSpec:
    """Parameters of the source's command-noise generator."""

    noise_level_deg_per_step: float = SOURCE_NOISE_LEVEL
    fpass_hz: float = SOURCE_FPASS_HZ
    fs_hz: float = SOURCE_FS_HZ
    order: int = SOURCE_ORDER
    name: str = "source_lowpass_gaussian"

    def as_dict(self) -> dict:
        return {"noise_level_deg_per_step": self.noise_level_deg_per_step,
                "fpass_hz": self.fpass_hz, "fs_hz": self.fs_hz,
                "order": self.order, "name": self.name,
                "recipe": "gaussian -> butter lfilter -> zscore -> * level"}


def butter_lowpass_filter(data, cutoff: float, fs: float, order: int = 5):
    """Verbatim reproduction of the notebook's helper (causal lfilter)."""
    b, a = butter(order, cutoff, fs=fs, btype="low", analog=False)
    return lfilter(b, a, data)


def source_command_noise_deg(n_steps: int, rng: np.random.Generator,
                             spec: Optional[SourceNoiseSpec] = None):
    """One realisation of the source noise, in degrees per timestep."""
    s = spec if spec is not None else SourceNoiseSpec()
    gaussian = rng.normal(size=(n_steps, 1))
    filtered = butter_lowpass_filter(gaussian, s.fpass_hz, s.fs_hz, s.order)
    return (s.noise_level_deg_per_step * zscore(filtered)).reshape(-1)


def deg_per_step_to_rad_per_s(noise_deg, T_core_s: float):
    """Convert a per-cycle heading increment into the rate the decoder speaks."""
    return np.deg2rad(np.asarray(noise_deg, dtype=float)) / float(T_core_s)


def zero_noise(n_steps: int):
    """No noise.  The default for the body-constraint experiments, which need
    the body effect isolated (handoff doc section 24)."""
    return np.zeros(int(n_steps))
