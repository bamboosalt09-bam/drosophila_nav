"""Train the 327 per-cell-type constants.  The wiring does not move.

WHAT MOVES.  `log_tau`, `bias`, `log_gain`, one of each per cell-type group.
With the optic lobe split by `cell_type` that is 109 groups and 327 numbers
against 68,045 neurons and 1,300,512 synapses, so the model stays constrained
by the connectome rather than fitted to the task.  Every synaptic weight is
frozen; training a free weight per synapse would reduce the connectome to a
sparsity mask and no result would mean anything.

WHY THIS IS NEEDED.  Measured on 2026-09-21, before any training:

    T4 (1,410 cells) and T5 (6,703) shared ONE tau, because their
    `cell_class` is empty and everything with an empty class fell into a
    single `sc:ol_intrinsic` group.  A Reichardt correlator is a
    delay-and-multiply; with both arms at the same time constant the product
    is symmetric and direction cancels.  Direction selectivity came out at
    +-8e-09 and the descending common mode tracked 1/tau at -0.13, which is
    what having no optic lobe at all gives.

    Signal reaching the descending neurons was 3.7e-05 of what was injected,
    because a single global spectral radius -- set by the mushroom body,
    which carries 99.1% of the leading eigenvector -- divided every weight by
    1,844 and starved a cascade that contributes 0.0% to that radius.

Both are parameter problems, not wiring problems, and both are in the 327.

THE LOSS IS A CORRELATION, NOT AN ERROR.  Scale and offset are absorbed
downstream by `gain` and `zero`, so fitting them would spend capacity on
numbers that do not matter.  Negative Pearson correlation against each
target is exactly the quantity the evaluation reports.

WHAT IS NOT IN THE LOSS.  Avoidance.  Both targets are sensory facts about
the world; turning away from an obstacle is a control decision and writing
one into the loss makes the circuit a copy of whoever wrote it.  Flight is
the test, not the objective.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from core.eye_subnet import build_eye_subnet
from core.flywire_rate import activity
from core.vp_subnet import frontal_dn

CLIPS = REPO / "results" / "clips"
N_SUB = 20
DT_MS = 5.0


def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The eps goes INSIDE the square roots.  `a.norm()` has gradient a/|a|,
    # which is 0/0 when the readout is identical across the batch --
    # saturated, or silent -- and one NaN gradient reaches all 327
    # parameters through clip_grad_norm_ and Adam.  The first run ended with
    # every parameter NaN.
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (torch.sqrt((a * a).sum() + 1e-20)
                            * torch.sqrt((b * b).sum() + 1e-20))


class Heads:
    """Reads the two descending channels out of a batch of rate vectors."""

    def __init__(self, info, sub, half_width=30.0):
        self.dl = torch.from_numpy(np.asarray(info["dn_l"])).long()
        self.dr = torch.from_numpy(np.asarray(info["dn_r"])).long()
        fl, fr = frontal_dn(sub, info, half_width)
        self.fl = torch.from_numpy(np.asarray(fl)).long()
        self.fr = torch.from_numpy(np.asarray(fr)).long()

    def steer(self, r):                      # (n, B) -> (B,)
        return r[self.dr].mean(0) - r[self.dl].mean(0)

    def brake(self, r):
        return r[self.fr].mean(0) + r[self.fl].mean(0)


def build_drive(net, inp, lum_prev, lum_cur, scent, mass, k, ramp):
    """Sub-step `k` of the drive, for a whole batch.  (n, B)."""
    B = lum_cur.shape[0]
    val_p = lum_prev * inp.drive_gain
    val_c = lum_cur * inp.drive_gain
    v = val_p * (1.0 - ramp[k]) + val_c * ramp[k]        # (B, n_omma)
    d = torch.zeros(net.n, B)
    d[torch.from_numpy(inp.rows).long()] = v.T
    frac = torch.clamp(scent / (inp.cam.az_span / 2), -1.0, 1.0)
    base = inp.orn_gain * mass
    d[torch.from_numpy(inp.orn_l).long()] = (base * (1 + frac)
                                             * inp.orn_scale_l)[None, :]
    d[torch.from_numpy(inp.orn_r).long()] = (base * (1 - frac)
                                             * inp.orn_scale_r)[None, :]
    return d


def evaluate(net, inp, heads, data, batch=64, limit=None):
    """Held-out correlation for both heads, no gradients."""
    lum, scent, mass, brg, itau = data
    n = lum.shape[0] if limit is None else min(limit, lum.shape[0])
    ramp = np.linspace(1.0 / N_SUB, 1.0, N_SUB)
    S, Bk, TB, TI = [], [], [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            sl = slice(i, min(i + batch, n))
            v = net.init_state(lum[sl].shape[0])
            for f in range(lum.shape[1]):
                lp = torch.from_numpy(lum[sl][:, max(f - 1, 0)])
                lc = torch.from_numpy(lum[sl][:, f])
                for k in range(N_SUB):
                    d = build_drive(net, inp, lp, lc,
                                    torch.from_numpy(scent[sl][:, f]),
                                    torch.from_numpy(mass[sl][:, f]), k, ramp)
                    v = net.step(v, d, DT_MS)
                r = activity(v)
                S.append(heads.steer(r))
                Bk.append(heads.brake(r))
                TB.append(torch.from_numpy(brg[sl][:, f]))
                TI.append(torch.from_numpy(itau[sl][:, f]))
    S, Bk = torch.cat(S), torch.cat(Bk)
    TB, TI = torch.cat(TB), torch.cat(TI)
    return float(corr(S, TB)), float(corr(Bk, TI))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--val", default="val")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.01,
                    help="the parameters are logs; 0.05 was too hot")
    ap.add_argument("--rho", type=float, default=1.0,
                    help="spectral radius to start from")
    ap.add_argument("--grad-sub", type=int, default=40,
                    help="trailing sub-steps that carry gradient; may span "
                         "frames (20 per frame)")
    ap.add_argument("--heads", choices=("both", "steer", "brake"),
                    default="both")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on the braking head")
    args = ap.parse_args(argv)

    net, ann, sub, info = build_eye_subnet()
    from sensors.eye_input import EyeInput
    inp = EyeInput(net, ann, info)
    # Start from the stage-normalised gains, not from 1.0.  At 1.0 only
    # 3.7e-05 of the injected signal reaches the descending neurons and
    # there is no gradient to work with either.
    net.normalise_gains(target=1.0, rho_cap=None)
    # ...then scaled UNIFORMLY to rho = 1.0.  Stage normalisation alone
    # leaves rho at 32, and the rho sweep showed the left/right split of the
    # steering readout is lost from rho 1.5 upward: a wall on either side
    # gives the same sign.  The first run started at 32 anyway.  1.0 is the
    # largest radius that kept the split, and settled to 98.6% in 100 ms.
    # Uniform, so the relative stage gains that let signal cross the
    # cascade are kept -- the share-proportional cap got to 0.52 by driving
    # the central brain to a gain of 3e-11.
    rho0 = net.radius()
    with torch.no_grad():
        net.log_gain.add_(float(np.log(args.rho / rho0)))
    print("stage-normalised gains, rho %.1f -> %.2f"
          % (rho0, net.radius()), flush=True)
    heads = Heads(info, sub)

    def load(name):
        z = np.load(CLIPS / ("%s.npz" % name))
        return (z["lum"], z["scent"], z["mass"], z["bearing"], z["inv_tau"])

    tr, va = load(args.train), load(args.val)
    print("train %d clips, val %d clips, %d frames each"
          % (tr[0].shape[0], va[0].shape[0], tr[0].shape[1]))

    c0 = evaluate(net, inp, heads, va, limit=256)
    print("before training:  steer r %+.3f   brake r %+.3f" % c0, flush=True)

    opt = torch.optim.Adam([net.log_tau, net.bias, net.log_gain], lr=args.lr)
    ramp = np.linspace(1.0 / N_SUB, 1.0, N_SUB)
    rs = np.random.RandomState(0)
    n = tr[0].shape[0]
    t0 = time.perf_counter()
    skipped = 0
    for step in range(args.steps):
        idx = rs.choice(n, args.batch, replace=False)
        lum, scent, mass, brg, itau = (x[idx] for x in tr)
        net._frozen = None
        v = net.init_state(args.batch)
        nf = lum.shape[1]
        # Truncated BPTT over the LAST `grad_sub` sub-steps, which may span
        # frames.  It used to be the last 6 sub-steps of the last frame --
        # 30 ms -- with the state detached at the start of that window.
        # Looming is a change BETWEEN frames, 100 ms apart, carried by delay
        # cells whose time constant is 100 ms; a 30 ms window never showed
        # the gradient the path that computes it, and the braking head came
        # out at +0.029 on held-out rooms, the same as untrained.  40
        # sub-steps is two frames, 200 ms.
        total = nf * N_SUB
        g0 = max(total - args.grad_sub, 0)
        idx_flat = 0
        for f in range(nf):
            lp = torch.from_numpy(lum[:, max(f - 1, 0)])
            lc = torch.from_numpy(lum[:, f])
            sc_f = torch.from_numpy(scent[:, f])
            ms_f = torch.from_numpy(mass[:, f])
            for k in range(N_SUB):
                if idx_flat == g0:
                    v = v.detach()
                if idx_flat < g0:
                    with torch.no_grad():
                        v = net.step(v, build_drive(net, inp, lp, lc, sc_f,
                                                    ms_f, k, ramp), DT_MS)
                else:
                    v = net.step(v, build_drive(net, inp, lp, lc, sc_f,
                                                ms_f, k, ramp), DT_MS)
                idx_flat += 1
        r = activity(v)
        # SIGN-AGNOSTIC.  The controller applies `gain`, whose sign is set
        # by a separate probe, so a correlation of -0.47 is just as usable as
        # +0.47 and driving it toward +1 means fighting the wiring for
        # nothing.  Measured on a 4-step smoke run with `-corr`: held-out
        # steering went -0.470 to +0.006, i.e. the loss destroyed the signal
        # that was already there while its own number improved.  Maximising
        # R^2 asks for the magnitude and leaves the sign alone.
        cs = corr(heads.steer(r), torch.from_numpy(brg[:, nf - 1]))
        cb = corr(heads.brake(r), torch.from_numpy(itau[:, nf - 1]))
        if args.heads == "brake":
            loss = -(cb ** 2)
        elif args.heads == "steer":
            loss = -(cs ** 2)
        else:
            loss = -(cs ** 2) - args.lam * (cb ** 2)
        opt.zero_grad()
        loss.backward()
        params = [net.log_tau, net.bias, net.log_gain]
        # A step with a non-finite loss or gradient is DROPPED, not applied.
        # Adam spreads one NaN to every parameter it touches, and there is
        # no coming back from that.
        if not torch.isfinite(loss) or any(
                not torch.isfinite(q.grad).all() for q in params):
            skipped += 1
            opt.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            print("  step %3d  loss %+.4f   steer %+.3f  brake %+.3f  "
                  "skipped %d  (%.0f s)"
                  % (step, float(loss), float(cs), float(cb), skipped,
                     time.perf_counter() - t0), flush=True)

    net._frozen = None
    c1 = evaluate(net, inp, heads, va, limit=256)
    print("after training:   steer r %+.3f   brake r %+.3f" % c1)
    print("                  was   steer r %+.3f   brake r %+.3f" % c0)
    finite = all(torch.isfinite(q).all() for q in
                 (net.log_tau, net.bias, net.log_gain))
    print("skipped %d of %d steps; parameters finite: %s"
          % (skipped, args.steps, finite))
    if not finite:
        print("NOT saving: parameters are not finite")
        return 1
    out = REPO / "results" / "subnet" / ("eye_trained_%s.npz" % args.heads)
    np.savez(out, log_tau=net.log_tau.detach().numpy(),
             bias=net.bias.detach().numpy(),
             log_gain=net.log_gain.detach().numpy(),
             types=np.array([str(t) for t in net.types]))
    print("wrote %s" % out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
