"""The same connectome as a trainable rate network.

The spiking model in flywire_brain.py answers "does the connectome compute
navigation as published".  It does not.  This one asks the next question:
under identical training, what does optimisation DO to a connectome-wired
network, and does it differ from matched null wirings.

Rate, not spiking, because a hard spike threshold is not differentiable.
This follows Lappalainen et al. 2024 (flyvis), whose dynamics are

    dv/dt = (-v + bias + W @ act(v_source)) / tau

with a ReLU activation.  Their parameters are shared per cell type, which is
what keeps the connectome as DATA: the wiring and the synapse counts stay
fixed and only a few hundred per-type numbers move.  Training free per-synapse
weights would reduce the connectome to a sparsity mask and change what any
result could mean.

Written in PyTorch and device-agnostic from the start, so the same code runs
on this laptop's CPU and on a GPU without a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.flywire_brain import load_connectome

# Shiu et al.'s membrane time constant, as the initial value for every type.
TAU_INIT_MS = 20.0


def activity(v: torch.Tensor) -> torch.Tensor:
    """Firing rate from membrane state: rectified AND saturating.

    A plain ReLU diverges here.  The spiking model is held in check by reset
    and a 2.2 ms refractory period, which caps it at ~455 Hz; a rate model has
    neither, and with raw synapse counts as weights (up to 2,405 synapses on
    one edge) it blows up to NaN at every dt.  Saturation is not a trick to
    make it stable -- a real firing rate IS bounded -- and tanh keeps the
    gradient finite everywhere, unlike a hard clamp.
    """
    return torch.tanh(torch.relu(v))


def _type_labels(ann: pd.DataFrame) -> np.ndarray:
    """One label per neuron: cell_type where known, else super_class.

    Parameters are shared within a label, so this is the axis training moves
    along.  It must be identical across connectome and null conditions or the
    parameter-space distances are not comparable.
    """
    # cell_type is far too fine to share parameters over -- 8,840 distinct
    # values, i.e. 26,520 free numbers, which is no longer a
    # connectome-constrained model.  cell_class is the anatomical grouping one
    # level up: 49 values covering 77% of neurons, and super_class fills the
    # rest.  Together, 57 groups -- the same order as flyvis's 64 cell types.
    #
    # astype(str) on a column with NaN gives an OBJECT array holding floats,
    # so fix the dtype explicitly rather than relying on it.
    cc = ann["cell_class"].astype(str).to_numpy(dtype="<U48")
    sc = ann["super_class"].astype(str).to_numpy(dtype="<U48")
    missing = (cc == "nan") | (cc == "")
    return np.where(missing, np.char.add("sc:", sc), np.char.add("cc:", cc))


class FlyWireRate(nn.Module):
    """Connectome-wired rate network with per-cell-type trainable parameters."""

    def __init__(self, min_synapses: int = 1, w_scale: float = 1.0,
                 device: str | torch.device = "cpu",
                 out_csr=None, ids=None, ann=None):
        super().__init__()
        if out_csr is None:
            ids, out_csr, ann, _ = load_connectome(min_synapses, w_scale)
        self.ids = ids
        self.ann = ann
        self.n = len(ids)
        self.device_ = torch.device(device)

        # W[i, j] = weight from j to i, so a matvec is W @ activity
        w_in = out_csr.T.tocsr()
        self.W = torch.sparse_csr_tensor(
            torch.from_numpy(w_in.indptr.astype(np.int64)),
            torch.from_numpy(w_in.indices.astype(np.int64)),
            torch.from_numpy(w_in.data.astype(np.float32)),
            size=(self.n, self.n)).to(self.device_)
        self.n_edges = int(w_in.nnz)

        labels = _type_labels(ann)
        self.types, inv = np.unique(labels, return_inverse=True)
        self.type_index = torch.from_numpy(inv.astype(np.int64)).to(self.device_)
        n_types = len(self.types)

        # the only things training may move
        self.log_tau = nn.Parameter(torch.full((n_types,),
                                               float(np.log(TAU_INIT_MS))))
        self.bias = nn.Parameter(torch.zeros(n_types))
        self.log_gain = nn.Parameter(torch.zeros(n_types))
        self.to(self.device_)

    # -- per-neuron views of the per-type parameters ----------------------
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()[self.type_index]

    def gain(self) -> torch.Tensor:
        return self.log_gain.exp()[self.type_index]

    def bias_n(self) -> torch.Tensor:
        return self.bias[self.type_index]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- dynamics ---------------------------------------------------------
    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(self.n, batch, device=self.device_)

    def step(self, v: torch.Tensor, drive: torch.Tensor,
             dt_ms: float) -> torch.Tensor:
        """One Euler step.  `v` and `drive` are (n_neurons, batch)."""
        r = activity(v) * self.gain()[:, None]
        inp = torch.sparse.mm(self.W, r)
        dv = (-v + self.bias_n()[:, None] + inp + drive)
        return v + dv * (dt_ms / self.tau()[:, None])

    def run(self, drive: torch.Tensor, duration_ms: float, dt_ms: float,
            record_rows: Optional[torch.Tensor] = None,
            grad_ms: Optional[float] = None):
        """Run with a constant drive; return the final rates and any trace.

        `grad_ms` truncates backpropagation to the last that many ms.  The
        readout is a steady-state rate, so carrying gradients through the whole
        transient costs memory and time for very little signal; running the
        early part under no_grad cuts both roughly in proportion.
        ponytail: truncated BPTT, drop grad_ms if a result ever depends on the
        transient.
        """
        v = self.init_state(drive.shape[1])
        n_steps = int(round(duration_ms / dt_ms))
        n_grad = (n_steps if grad_ms is None
                  else min(n_steps, max(1, int(round(grad_ms / dt_ms)))))
        trace = [] if record_rows is not None else None

        with torch.no_grad():
            for _ in range(n_steps - n_grad):
                v = self.step(v, drive, dt_ms)
                if trace is not None:
                    trace.append(activity(v)[record_rows])
        v = v.detach()
        for _ in range(n_grad):
            v = self.step(v, drive, dt_ms)
            if trace is not None:
                trace.append(activity(v)[record_rows])
        return activity(v), (torch.stack(trace) if trace is not None else None)

    def as_dict(self) -> Dict:
        return {"n_neurons": self.n, "n_edges": self.n_edges,
                "n_types": len(self.types),
                "n_trainable": self.n_trainable(),
                "trains": "per-cell-type tau, bias, gain; weights FIXED",
                "device": str(self.device_)}


def demo() -> None:
    """Check it runs, is differentiable, and how large dt can be."""
    import time

    t0 = time.perf_counter()
    # 0.01, not the spiking model's 2.0: measured, that leaves mean
    # activity at 0.12 with 0.3% of neurons saturated, i.e. responsive.
    # 1.0 saturates 17% before training starts.
    net = FlyWireRate(w_scale=0.01)
    print("built %d neurons, %d edges, %d cell types, %d trainable params "
          "in %.1f s" % (net.n, net.n_edges, len(net.types),
                         net.n_trainable(), time.perf_counter() - t0))
    assert net.n_trainable() < 5000, "too many free parameters to call this "\
        "a connectome-constrained model"

    sensory = np.flatnonzero(net.ann["super_class"]
                             .isin(["sensory", "sensory_ascending"]).to_numpy())
    drive = torch.zeros(net.n, 1)
    drive[sensory] = 1.0

    # dt is a free choice in a rate model: pick the largest that still agrees
    print("  dt (ms)   wall/bio-s   mean rate   vs dt=0.1")
    ref = None
    for dt in (0.1, 0.5, 1.0, 2.0, 5.0):
        t0 = time.perf_counter()
        r, _ = net.run(drive, 200.0, dt)
        el = time.perf_counter() - t0
        m = float(r.mean())
        if ref is None:
            ref = r
        rel = float((r - ref).abs().max() / max(float(ref.abs().max()), 1e-9))
        print("  %7.1f   %10.1f   %9.3f   %8.3f" % (dt, el / 0.2, m, rel))

    # gradients must flow to every parameter group
    r, _ = net.run(drive, 50.0, 1.0)
    loss = r.mean()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        print("  grad %-10s norm %.3e" % (name, float(p.grad.norm())))
    print("demo ok")


if __name__ == "__main__":
    demo()
