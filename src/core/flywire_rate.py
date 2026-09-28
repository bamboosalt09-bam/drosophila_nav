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
# Same numbers on every machine.  The circuit's sparse matvec runs in MKL,
# which picks its kernel by CPU: AVX-512 in the cloud, AVX2 on the Windows
# PC.  The two sum in a different order, the flights part at 1e-9 m by step
# 8 and by metres after 57 s (chaos).  Pinned to AVX2, room 6 v17 on Linux
# is bit-identical to Windows for all 1,200 steps; no slower.  Must be set
# before torch loads MKL.
import os
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")
import torch
import torch.nn as nn

from core.flywire_brain import load_connectome

# Shiu et al.'s membrane time constant, as the initial value for every type.
TAU_INIT_MS = 20.0

# THE DELAY ARM OF THE MOTION DETECTOR.
#
# A Reichardt correlator multiplies a neighbour's signal against a DELAYED
# copy of its own.  In Drosophila the delay is carried by specific medulla
# types: T4 (ON) takes its direct input from Mi1 and Tm3 and its delayed
# input from Mi4 and Mi9, and T5 (OFF) takes direct input from Tm1, Tm2 and
# Tm4 and delayed input from Tm9.  With every cell at the same 20 ms the two
# arms are identical, the product is symmetric, and direction cancels.
#
# These are initial values, not measurements of this model: the slow arm is
# put at 100 ms, inside the 40-150 ms range fitted to the fly, and left as a
# trainable parameter like every other tau.  Any result that depends on the
# exact number has to say that the number was assumed.
TAU_DELAY_MS = 100.0

# FLOOR ON EVERY TIME CONSTANT.  The integrator is explicit Euler at dt = 5 ms,
# which is only stable while dt/tau < 2, i.e. tau > 2.5 ms; below that every
# step overshoots with alternating sign and the state blows up.  Nothing used
# to stop training from walking log_tau there -- lr 0.05 over 250 steps can
# move it by 12.5 in log space -- and the first training run came back with
# all 327 parameters NaN.  10 ms is 2 * dt, a factor of two inside the edge.
TAU_MIN_MS = 10.0
DELAY_LINE_TYPES = ("Mi4", "Mi9", "Tm9")


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


class _SpMM(torch.autograd.Function):
    """y = W @ x for a FIXED sparse W, with an explicit transpose backward.

    ponytail: exists purely because torch's sparse-CSR autograd is ~200x
    slower than the forward it differentiates.  Drop it the day that is fixed
    upstream -- `torch.sparse.mm(W, x)` is the same maths.
    """

    @staticmethod
    def forward(ctx, W, WT, x):
        ctx.WT = WT
        return torch.sparse.mm(W, x)

    @staticmethod
    def backward(ctx, grad_out):
        # W carries no gradient (weights are fixed), so only x gets one
        return None, None, torch.sparse.mm(ctx.WT, grad_out.contiguous())


def _type_labels(ann: pd.DataFrame, by_side: bool = False) -> np.ndarray:
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
    ct = ann["cell_type"].astype(str).to_numpy(dtype="<U48")
    missing = (cc == "nan") | (cc == "")
    lab = np.where(missing, np.char.add("sc:", sc), np.char.add("cc:", cc))

    # THE OPTIC LOBE GETS cell_type, EVERYTHING ELSE KEEPS cell_class.
    #
    # `cell_class` is empty for 45,437 of the 45,532 optic-lobe cells, so
    # they all fell into one `sc:ol_intrinsic` group: T4 and T5 shared a
    # single tau, and so did T4a/b/c/d.  An elementary motion detector is a
    # delay-and-multiply, and a correlator whose two arms have the SAME time
    # constant is symmetric and cancels -- it computes no direction at all.
    # Measured on the lamina-injected subnetwork with the optic lobe fully
    # present (T4 1,410 cells, T5 6,703, LPLC 415, LC 775, giant fibre 24):
    # direction selectivity came out at +-8e-09, four orders below the
    # projection-neuron arm's readout, and the descending common mode
    # tracked 1/tau at -0.13, indistinguishable from having no optic lobe.
    # The wiring was all there; the parameters could not express what it is
    # for, and no amount of training would have fixed that because the
    # groups themselves were too coarse.
    #
    # Every optic-lobe cell has a cell_type and there are only 83 of them --
    # the canonical set, L1-L5, Mi, Tm, T4a-d, T5a-d, LPLC, LC.  With the
    # central brain still on cell_class this is 100 groups and 300 free
    # numbers against 68,045 neurons, so the model stays constrained by the
    # connectome rather than fitted to the task.
    ol = (sc == "ol_intrinsic") & (ct != "nan") & (ct != "")
    lab = np.where(ol, np.char.add("ct:", ct), lab)
    if not by_side:
        return lab
    # Measured in the pilot: left and right descending neurons have the same
    # type composition (648 vs 645 in one label), so side-blind parameters
    # apply an IDENTICAL tau/bias/gain to both sides and cannot touch a
    # left-right asymmetry -- which is exactly what the untrained network's
    # standing offset is.  Splitting by side gives training purchase on it:
    # 118 groups, 354 parameters.  A bilaterally symmetric animal makes this a
    # concession, and it is measurable: if the trained values come out nearly
    # symmetric the asymmetry was in the input, and if they do not, training
    # is compensating for the wiring.
    side = ann["side"].astype(str).to_numpy(dtype="<U8")
    return np.char.add(lab.astype("<U60"), np.char.add("|", side))


class FlyWireRate(nn.Module):
    """Connectome-wired rate network with per-cell-type trainable parameters."""

    def __init__(self, min_synapses: int = 1, w_scale: float = 1.0,
                 device: str | torch.device = "cpu",
                 out_csr=None, ids=None, ann=None, by_side: bool = False):
        super().__init__()
        if out_csr is None:
            ids, out_csr, ann, _ = load_connectome(min_synapses, w_scale)
        self.ids = ids
        self.ann = ann
        self.n = len(ids)
        self.device_ = torch.device(device)

        # W[i, j] = weight from j to i, so a matvec is W @ activity
        w_in = out_csr.T.tocsr()

        def _csr(m):
            # int32 indices, not int64.  A CSR matvec is memory-bound -- 3.3
            # ms for 1.08 M edges is 4 GB/s, which is this machine's
            # bandwidth -- so the only lever is bytes moved.  At int64 each
            # edge costs 4 B of value plus 8 B of index; at int32 it is 4 + 4,
            # a third less traffic.  2.1 G edges would be needed to overflow
            # int32 and the whole connectome has 1.1 M.
            return torch.sparse_csr_tensor(
                torch.from_numpy(m.indptr.astype(np.int32)),
                torch.from_numpy(m.indices.astype(np.int32)),
                torch.from_numpy(m.data.astype(np.float32)),
                size=(self.n, self.n)).to(self.device_)

        # |W| aggregated by SOURCE type: A[i, t] is how much input neuron i
        # receives from type t at unit gain.  Kept for `normalise_gains`.
        self._absA = None
        self._w_in_abs = abs(w_in).tocsr()
        self.W = _csr(w_in)
        # The transpose, kept in CSR, exists only so that backward can be one
        # more CSR matvec.  Torch's own autograd for sparse CSR mm costs
        # 1,840 ms against a 5.5 ms forward -- 200x, for a weight matrix that
        # does not even require grad.  With this it is 5.1 ms and the gradient
        # is bit-identical (verified: max |diff| 0.0).
        self.WT = _csr(w_in.T.tocsr())
        self.n_edges = int(w_in.nnz)

        labels = _type_labels(ann, by_side=by_side)
        self.types, inv = np.unique(labels, return_inverse=True)
        self.type_index = torch.from_numpy(inv.astype(np.int64)).to(self.device_)
        n_types = len(self.types)

        # the only things training may move
        tau0 = np.full(n_types, float(np.log(TAU_INIT_MS)))
        for i, t in enumerate(self.types):
            name = str(t)
            if name.startswith("ct:") and any(
                    name[3:] == d or name[3:].startswith(d + "_")
                    for d in DELAY_LINE_TYPES):
                tau0[i] = float(np.log(TAU_DELAY_MS))
        self.log_tau = nn.Parameter(torch.from_numpy(tau0).float())
        self.bias = nn.Parameter(torch.zeros(n_types))
        self.log_gain = nn.Parameter(torch.zeros(n_types))
        self._frozen = None
        self.to(self.device_)

    # -- per-neuron views of the per-type parameters ----------------------
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=TAU_MIN_MS)[self.type_index]

    def gain(self) -> torch.Tensor:
        return self.log_gain.exp()[self.type_index]

    def bias_n(self) -> torch.Tensor:
        return self.bias[self.type_index]

    def normalise_gains(self, target: float = 1.0, rounds: int = 30,
                        rho_cap: float | None = None) -> dict:
        """Set the per-type gains so every stage carries O(1) signal.

        A GLOBAL spectral radius is the wrong normaliser for a cascade.  It
        is set by the strongest recurrent loop -- here rho 922, so every
        weight is divided by 1,844 -- and a feedforward visual pathway whose
        weights are far weaker than that loop is starved.  Measured on the
        lamina-injected subnetwork, mean |activity| by stage with all gains
        at 1.0:

            lamina   1.8e-03      T4       1.6e-09
            Mi       7.1e-07      T5       3.2e-07
            Tm       1.1e-05      LPLC     4.8e-07
                                  descending 6.7e-08

        That is 3.7e-05 of the injected signal surviving to the descending
        neurons, where a spectral radius of 0.50 over six stages should have
        left 0.5^6 = 1.6e-02.  Four hundred times worse than the radius
        implies, because the radius is not describing this path.

        The model already has the right knob: one gain per cell type, 109 of
        them, all sitting at their initial 1.0.  This solves for gains that
        make each neuron's total input sum to `target`, by a few rounds of
        weighted multiplicative updates.  It is an INITIALISATION, declared
        as one, not a fit to any task -- no behaviour, no trajectory and no
        reference controller is consulted.  Training may move these like any
        other parameter.

        `rho_cap` rescales everything down afterwards if the gain-scaled
        matrix ends up with a larger spectral radius than asked, because
        settling within the 100 ms control cycle is what stopped the readout
        being sampled mid-transient with the wrong sign.
        """
        import scipy.sparse as sp

        n_types = len(self.types)
        if self._absA is None:
            M = sp.csr_matrix(
                (np.ones(self.n), (np.arange(self.n),
                                   self.type_index.cpu().numpy())),
                shape=(self.n, n_types))
            self._absA = np.asarray((self._w_in_abs @ M).todense())
        A = self._absA
        g = np.ones(n_types)
        eps = 1e-30
        for _ in range(rounds):
            inp = A @ g                                   # (n,)
            live = inp > eps
            if not live.any():
                break
            share = (A[live] * g) / inp[live][:, None]    # (n_live, T)
            need = np.log(target / np.maximum(inp[live], eps))
            wsum = share.sum(axis=0)
            step = np.where(wsum > eps, (share * need[:, None]).sum(axis=0)
                            / np.maximum(wsum, eps), 0.0)
            g = g * np.exp(np.clip(step, -2.0, 2.0))
            g = np.clip(g, 1e-6, 1e6)

        # CAP THE LOOP, NOT THE CASCADE.
        #
        # Scaling every gain down until the spectral radius fits is what the
        # global normaliser did, and it is why the visual pathway was
        # starved: measured, 99.1% of the leading eigenvector sits in
        # `cb_intrinsic` and its largest single type is KCg-m, the mushroom
        # body's Kenyon cells.  The whole visual cascade -- lamina, medulla,
        # T4/T5, lobula, projection neurons -- contributes 0.0%.  One dense
        # associative loop was setting the scale for 68,045 neurons, and
        # taming it cost the cascade a factor of 64.
        #
        # "Must settle inside the control cycle" is a constraint on
        # RECURRENT loops.  A feedforward path has no loop and unit gain per
        # stage destabilises nothing.  So the reduction is applied in
        # proportion to each type's share of the leading eigenvector: the
        # types that make the radius take it, the ones that do not are left
        # alone.  Iterated, because shrinking the dominant loop exposes the
        # next one.
        rho = None
        if rho_cap is not None:
            # Power iteration, not ARPACK.  |W| diag(g) is NONNEGATIVE, so
            # Perron-Frobenius applies and the leading pair is exactly what
            # power iteration converges to -- and it is 20 sparse matvecs
            # instead of a general eigensolver.  Measured: 10 s per round
            # with `scipy.sparse.linalg.eigs`, so a 60-round budget ran past
            # a 580 s timeout without finishing; here a round is milliseconds
            # and the whole loop is seconds.
            ti = self.type_index.cpu().numpy()
            A_abs = self._w_in_abs
            x = np.abs(np.random.RandomState(0).randn(self.n))
            x /= np.linalg.norm(x)

            def leading(gv, iters=60):
                y = x.copy()
                r_last = 0.0
                for _ in range(iters):
                    y = A_abs @ (gv * y)
                    nrm = np.linalg.norm(y)
                    if nrm < 1e-300:
                        return 0.0, y
                    y /= nrm
                    r_last = nrm
                return float(r_last), y

            for it in range(200):
                gv = g[ti]
                r_now, vec = leading(gv)
                if rho is None:
                    rho = r_now
                if r_now <= rho_cap or r_now <= 0.0:
                    break
                v = np.abs(vec)
                v = v / max(v.sum(), 1e-300)
                share = np.bincount(ti, weights=v, minlength=n_types)
                share = share / max(share.max(), 1e-300)
                g = g * (rho_cap / r_now) ** share
            self._rho_rounds = it + 1
            self._rho_final = r_now

        with torch.no_grad():
            self.log_gain.copy_(torch.from_numpy(np.log(g)).float())
        self._frozen = None
        inp = A @ g
        live = inp > eps
        return {"gain_min": float(g.min()), "gain_max": float(g.max()),
                "input_median": float(np.median(inp[live])),
                "rho_before_cap": rho, "n_types": n_types}

    def radius(self, iters: int = 60) -> float:
        """Spectral radius of |W| diag(gain), by power iteration.

        |W| diag(gain) is nonnegative, so Perron-Frobenius makes power
        iteration converge to exactly the leading eigenvalue.  60 sparse
        matvecs, under a second, against ~10 s for ARPACK.
        """
        g = self.gain().detach().cpu().numpy()
        y = np.abs(np.random.RandomState(0).randn(self.n))
        y /= np.linalg.norm(y)
        r = 0.0
        for _ in range(iters):
            y = self._w_in_abs @ (g * y)
            r = float(np.linalg.norm(y))
            if r < 1e-300:
                return 0.0
            y /= r
        return r

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- dynamics ---------------------------------------------------------
    def init_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(self.n, batch, device=self.device_)

    # -- inference-time parameter snapshot --------------------------------
    def freeze_params(self, dt_ms: float) -> None:
        """Expand the per-type parameters to per-neuron ONCE.

        `gain()`, `bias_n()` and `tau()` are each an exp over the type table
        followed by a gather to 34,125 rows, and `step` calls all three every
        time.  A control cycle is 20 steps, so that is 60 gathers and 40 exps
        per cycle for numbers that do not move unless training moves them.
        Measured: 79.0 ms -> 66.7 ms for 20 steps, 16%.

        Explicit rather than automatic, because a cache that guessed when
        parameters had changed would be wrong exactly when training was
        running.  Call `unfreeze()` before optimising.
        """
        self._frozen = (self.gain()[:, None].detach().clone(),
                        self.bias_n()[:, None].detach().clone(),
                        (dt_ms / self.tau())[:, None].detach().clone(),
                        float(dt_ms))

    def unfreeze(self) -> None:
        self._frozen = None

    def step(self, v: torch.Tensor, drive: torch.Tensor,
             dt_ms: float) -> torch.Tensor:
        """One Euler step.  `v` and `drive` are (n_neurons, batch)."""
        fz = getattr(self, "_frozen", None)
        if fz is not None and fz[3] == dt_ms:
            g, b, k, _ = fz
            return v + (-v + b + _SpMM.apply(self.W, self.WT,
                                             activity(v) * g) + drive) * k
        r = activity(v) * self.gain()[:, None]
        inp = _SpMM.apply(self.W, self.WT, r)
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
