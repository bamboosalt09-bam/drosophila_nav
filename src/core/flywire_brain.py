"""The whole FlyWire brain as a leaky integrate-and-fire network.

No circuit is designated.  Input goes in at anatomically defined sensory
neurons, output is read at anatomically defined descending neurons, and
everything between is whatever the connectome does.

Neuron parameters are Shiu et al. 2024's (github.com/philshiu/Drosophila_brain_model,
model.py), so they are not ours to tune:

    v_rest = v_reset = -52 mV, v_th = -45 mV
    membrane tau 20 ms, synaptic tau 5 ms
    refractory 2.2 ms, synaptic delay 1.8 ms
    w = sign * synapse_count * 0.275 mV, no synapse-count threshold

`w_syn` is the one free parameter in that model and they say so.  The sign
comes from the predicted transmitter (ACh/DA/OA/5-HT +, GABA/Glu -), which is
an assumption about the connectome rather than a measurement of it.

ponytail: explicit Euler at dt = 0.1 ms and a dense voltage vector.  Fine at
this scale on one core; a batched GPU version is the upgrade if a sweep is
ever needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path("data/flywire")

# Shiu et al. 2024, model.py: default_params
V_REST = -52.0
V_RESET = -52.0
V_TH = -45.0
TAU_MBR_MS = 20.0
TAU_SYN_MS = 5.0
REFRACTORY_MS = 2.2
DELAY_MS = 1.8
W_SYN_MV = 0.275

# ACh / DA / OA / 5-HT excitatory, GABA / Glu inhibitory -- the same rule as
# Shiu et al. 2024.  Sign is a property of the PRESYNAPTIC NEURON (one cell,
# one transmitter), not of the individual edge: the connection table's
# per-edge transmitter averages disagree with the neuron's own call on 20.5%
# of edges and flip the sign on 9.6% of them.
_NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "octopamine": 1.0,
            "serotonin": 1.0, "gaba": -1.0, "glutamate": -1.0}


@dataclass
class BrainState:
    v: np.ndarray
    g: np.ndarray
    refrac_until_ms: np.ndarray
    t_ms: float


def load_connectome(min_synapses: int = 1, w_scale: float = 1.0):
    """Ids, signed weight matrix and the annotation columns anything reads.

    Shared by the spiking and the rate model so both see the same graph.
    Row = source neuron, so one row holds that neuron's out-edges.
    """
    e = np.load(DATA / "edges_783.npz", allow_pickle=True)
    ann = pd.read_csv(DATA / "neuron_annotations_783.tsv", sep="	",
                      low_memory=False)
    ann = ann[["root_id", "super_class", "cell_class", "cell_type",
               "side", "top_nt"]]

    keep = e["syn"] >= min_synapses
    pre, post, syn = e["pre"][keep], e["post"][keep], e["syn"][keep]

    ids = np.union1d(np.union1d(pre, post), ann["root_id"].to_numpy(np.int64))
    idx = pd.Index(ids)
    i_pre, i_post = idx.get_indexer(pre), idx.get_indexer(post)

    # Sign per presynaptic NEURON, through a 139k float32 array rather than a
    # 15M string Series -- the latter allocated hundreds of MB and swapped.
    ann_i = ann.set_index("root_id").reindex(ids)
    sign_by_neuron = (ann_i["top_nt"].map(_NT_SIGN)
                      .fillna(0.0).to_numpy(np.float32))
    sign = sign_by_neuron[i_pre]

    w = (sign * syn * W_SYN_MV * w_scale).astype(np.float32)
    out = sp.csr_matrix((w, (i_pre, i_post)), shape=(len(ids), len(ids)),
                        dtype=np.float32)
    out.sort_indices()
    return ids, out, ann_i, int((sign == 0).sum())


class FlyWireBrain:
    """LIF over the whole connectome.  Indices are positions in `self.ids`."""

    def __init__(self, min_synapses: int = 1, dt_ms: float = 0.1,
                 w_scale: float = 1.0):
        self.ids, self.out, self.ann, self.n_unsigned = load_connectome(
            min_synapses, w_scale)
        self.n = len(self.ids)
        self.n_edges = self.out.nnz
        self.w_scale = float(w_scale)
        self.dt_ms = float(dt_ms)
        self._decay_v = np.float32(np.exp(-dt_ms / TAU_MBR_MS))
        self._decay_g = np.float32(np.exp(-dt_ms / TAU_SYN_MS))
        self._delay_steps = max(1, int(round(DELAY_MS / dt_ms)))
        self.reset()

    # -- anatomical selections, not functional ones ----------------------
    def by_super_class(self, *names: str) -> np.ndarray:
        return np.flatnonzero(self.ann["super_class"].isin(names).to_numpy())

    def by_cell_type(self, *names: str) -> np.ndarray:
        return np.flatnonzero(self.ann["cell_type"].isin(names).to_numpy())

    def side(self) -> np.ndarray:
        # NOTE: Series.to_numpy(str) silently truncates to <U1 when the column
        # holds NaN ("right" -> "r"), which made every side lookup match
        # nothing.  astype(str) first.
        return self.ann["side"].astype(str).to_numpy()

    # -- simulation -------------------------------------------------------
    def reset(self) -> None:
        self.v = np.full(self.n, V_REST, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refrac_until = np.full(self.n, -1.0, dtype=np.float32)
        self.t_ms = 0.0
        self._pending = [np.empty(0, np.int64)
                         for _ in range(self._delay_steps)]
        self._slot = 0

    def step(self, drive_mv: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance dt. `drive_mv` is an external current, in mV per step."""
        dv = np.float32(V_TH)
        spikes = np.flatnonzero((self.v >= dv)
                                & (self.refrac_until <= self.t_ms))

        # synaptic input arrives DELAY_MS after the spike
        arriving = self._pending[self._slot]
        self._pending[self._slot] = spikes
        self._slot = (self._slot + 1) % self._delay_steps

        self.g *= self._decay_g
        if arriving.size:
            contrib = self.out[arriving].sum(axis=0)
            self.g += np.asarray(contrib, dtype=np.float32).ravel()

        # dv/dt = (v_rest - v + g) / tau_mbr
        self.v += (V_REST - self.v + self.g) * np.float32(
            self.dt_ms / TAU_MBR_MS)
        if drive_mv is not None:
            self.v += drive_mv

        if spikes.size:
            self.v[spikes] = V_RESET
            self.g[spikes] = 0.0
            self.refrac_until[spikes] = self.t_ms + REFRACTORY_MS
        self.t_ms += self.dt_ms
        return spikes

    def run(self, duration_ms: float, drive_fn=None, record=None):
        """Run and return spike counts (all neurons) plus optional traces."""
        n_steps = int(round(duration_ms / self.dt_ms))
        counts = np.zeros(self.n, dtype=np.int32)
        traces = ({k: np.zeros((n_steps, len(v)), np.int8)
                   for k, v in record.items()} if record else {})
        for t in range(n_steps):
            drive = drive_fn(self.t_ms) if drive_fn is not None else None
            s = self.step(drive)
            if s.size:
                counts[s] += 1
                for k, v in (record or {}).items():
                    traces[k][t] = np.isin(v, s)
        return counts, traces

    def as_dict(self) -> Dict:
        return {"n_neurons": self.n, "n_edges": int(self.n_edges),
                "dt_ms": self.dt_ms, "w_syn_mV": W_SYN_MV * self.w_scale,
                "params": "Shiu et al. 2024 model.py",
                "sign": "ACh/DA/OA/5-HT +, GABA/Glu -, per presynaptic neuron",
                "n_unsigned_edges": self.n_unsigned}


def demo() -> None:
    """Smoke check: the network builds, runs, and fires at a plausible rate."""
    import time

    t0 = time.perf_counter()
    b = FlyWireBrain()
    print("built %d neurons, %d edges in %.1f s"
          % (b.n, b.n_edges, time.perf_counter() - t0))

    sensory = b.by_super_class("sensory", "sensory_ascending")
    desc = b.by_super_class("descending")
    print("  sensory %d   descending %d" % (len(sensory), len(desc)))
    assert len(sensory) > 10000 and len(desc) > 1000

    # A constant drive to sensory neurons, to see anything move at all.
    # Steady state is V_REST + drive * (tau_mbr / dt) = -52 + drive * 200, so
    # the drive has to exceed 0.035 mV/step to reach the -45 mV threshold.
    drive = np.zeros(b.n, dtype=np.float32)
    drive[sensory] = 0.06

    t0 = time.perf_counter()
    counts, _ = b.run(200.0, drive_fn=lambda t: drive)
    el = time.perf_counter() - t0
    hz = counts / 0.2
    print("  200 ms in %.1f s wall (%.1f s per biological second)"
          % (el, el / 0.2))
    print("  firing: %d of %d neurons spiked, mean %.1f Hz, median active %.1f Hz"
          % ((counts > 0).sum(), b.n, hz.mean(), np.median(hz[counts > 0])
             if (counts > 0).any() else 0.0))
    print("  descending: %d active, mean %.1f Hz"
          % ((counts[desc] > 0).sum(), hz[desc].mean()))
    assert (counts > 0).sum() > 0, "nothing fired at all"
    assert hz.mean() < 500, "runaway excitation"
    print("demo ok")


if __name__ == "__main__":
    demo()
