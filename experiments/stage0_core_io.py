"""Stage 0, part 1: open-loop characterisation of the reduced steering core.

Produces the graphs required by handoff document section 43 items 1-5 and 7,
plus an overlay against the official notebook output (the reproduction gate).
Item 6, the closed-loop heading trajectory, belongs to stage0_ideal_loop.py.

Usage
-----
    python experiments/stage0_core_io.py
    python experiments/stage0_core_io.py --silence-pfl2 --tag pfl2_lesion
    python experiments/stage0_core_io.py --pfl-scalar 0.4 --tag S04

Outputs (results/stage0/<tag>/):
    fig1_steering_vs_error.png      fig5_phase_profiles.png
    fig2_pfl_tuning.png             fig6_source_overlay.png
    fig3_dn_outputs.png             core_io_sweep.csv
    fig4_symmetry_check.png         provenance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.calibration import calibrate, load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore

NORM_JSON = REPO / "configs" / "norm_constants.json"
REF_NPZ = REPO / "reference" / "ref_coarse.npz"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "unavailable"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 0 core I/O characterisation")
    p.add_argument("--n-units", type=int, default=1000)
    p.add_argument("--pfl-scalar", type=float, default=1.0, help="source S")
    p.add_argument("--goal-deg", type=float, default=0.0)
    p.add_argument("--silence-pfl2", action="store_true")
    p.add_argument("--pfl3-inhibitory", action="store_true")
    p.add_argument("--step-deg", type=float, default=0.5)
    p.add_argument("--tag", default="default")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    params = CoreParams(
        n_units=args.n_units,
        pfl_scalar_S=args.pfl_scalar,
        pfl2_silenced=args.silence_pfl2,
        pfl3_inhibitory=args.pfl3_inhibitory,
    )

    if NORM_JSON.exists():
        norm = load(NORM_JSON)
        norm_source = str(NORM_JSON.relative_to(REPO))
    else:
        print("configs/norm_constants.json missing -- calibrating now")
        norm = calibrate(CoreParams(n_units=args.n_units), verbose=True)
        norm_source = "calibrated on the fly (not committed)"

    core = WesteindeSteeringCore(params, norm)
    outdir = REPO / "results" / "stage0" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- sweep -----------------------------------------------------------
    # Sweep the heading ERROR directly on a grid that is symmetric about 0 and
    # includes both +180 and -180.  Do NOT rebuild the axis from the core's
    # wrapped heading_error_deg: wrapping maps +180 to -180, which would put
    # -180 in twice, drop +180, and break the e <-> -e pairing used below.
    err_deg = np.arange(-180.0, 180.0 + args.step_deg, args.step_deg)
    out = core.sweep_deg(args.goal_deg + err_deg, args.goal_deg)
    steer = out["steering"]

    pd.DataFrame({"heading_error_deg": err_deg, **out}).to_csv(
        outdir / "core_io_sweep.csv", index=False)

    # ---- diagnostics -----------------------------------------------------
    i0 = int(np.argmin(np.abs(err_deg)))
    slope0 = float((steer[i0 + 1] - steer[i0 - 1]) /
                   (err_deg[i0 + 1] - err_deg[i0 - 1]))
    trough = int(np.argmin(steer))
    sym_residual = float(np.max(np.abs(steer + steer[::-1])))

    diag = {
        "steering_at_0deg": float(steer[i0]),
        "steering_slope_at_0_per_deg": slope0,
        "peak_abs_steering": float(np.max(np.abs(steer))),
        "peak_abs_steering_at_deg": float(abs(err_deg[trough])),
        "steering_at_180deg": float(steer[-1]),
        "odd_symmetry_max_residual": sym_residual,
        "pfl2_bump_amp_at_0": float(out["pfl2_bump_amp"][i0]),
        "pfl2_bump_amp_at_180": float(out["pfl2_bump_amp"][-1]),
    }

    print("=== Stage 0 core I/O diagnostics (tag: %s) ===" % args.tag)
    for k, v in diag.items():
        print("  %-32s %.6g" % (k, v))

    inner = (err_deg > 0.5) & (err_deg < 179.5)
    checks = [
        ("steering(0) == 0", abs(diag["steering_at_0deg"]) < 1e-9),
        ("odd symmetry", sym_residual < 1e-9),
        # e = +-180 excluded: symmetric unstable equilibrium (doc section 48)
        ("restoring sign for 0 < e < 180", bool(np.all(steer[inner] < 0))),
        ("steering(180) == 0 (unstable eq.)", abs(diag["steering_at_180deg"]) < 1e-9),
        ("PFL2 bump minimal at goal", diag["pfl2_bump_amp_at_0"] < 1e-9),
        ("PFL2 bump maximal anti-goal",
         diag["pfl2_bump_amp_at_180"] > 0.9 * float(out["pfl2_bump_amp"].max())),
    ]
    print("--- qualitative checks ---")
    all_ok = True
    for name, ok in checks:
        all_ok &= bool(ok)
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))

    # ---- figures ---------------------------------------------------------
    def finish(fig, name, title):
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(outdir / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(err_deg, steer, lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.plot(err_deg[trough], steer[trough], "o", color="C3",
            label="peak at %.1f deg" % err_deg[trough])
    ax.set_xlabel("heading error e = wrap(heading - goal)  [deg]")
    ax.set_ylabel("steering command (normalised)")
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    finish(fig, "fig1_steering_vs_error.png",
           "Stage 0 / fig1: steering vs heading error (negative = turn back to goal)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(err_deg, out["sum_pfl3r"], lw=2, label="sum PFL3R")
    axes[0].plot(err_deg, out["sum_pfl3l"], lw=2, label="sum PFL3L")
    axes[0].plot(err_deg, out["sum_pfl2"], lw=2, label="sum PFL2")
    axes[0].set_ylabel("population sum [a.u.]")
    axes[1].plot(err_deg, out["pfl3_RL_diff"], lw=2, color="C3",
                 label="sum PFL3R - sum PFL3L")
    axes[1].plot(err_deg, out["pfl2_bump_amp"] * 50, lw=2, color="C4",
                 label="PFL2 bump amplitude (x50)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("[a.u.]")
    for ax in axes:
        ax.set_xlabel("heading error [deg]")
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    finish(fig, "fig2_pfl_tuning.png", "Stage 0 / fig2: PFL population responses")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(err_deg, out["dna03r"], lw=2, label="DNa03R")
    axes[0].plot(err_deg, out["dna03l"], lw=2, label="DNa03L")
    axes[0].set_title("DNa03 (indirect pathway)")
    axes[1].plot(err_deg, out["dna02r"], lw=2, label="DNa02R")
    axes[1].plot(err_deg, out["dna02l"], lw=2, label="DNa02L")
    axes[1].set_title("DNa02 (output stage)")
    for ax in axes:
        ax.set_xlabel("heading error [deg]")
        ax.set_ylabel("activity [a.u.]")
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    finish(fig, "fig3_dn_outputs.png", "Stage 0 / fig3: descending neuron outputs")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pos = err_deg >= 0
    axes[0].plot(err_deg[pos], steer[pos], lw=2, label="steering(+e)")
    axes[0].plot(err_deg[pos], -steer[::-1][pos], lw=2, ls="--", label="-steering(-e)")
    axes[0].set_xlabel("|heading error| [deg]")
    axes[0].legend(fontsize=8)
    axes[1].plot(err_deg, steer + steer[::-1], lw=1.5, color="C3")
    axes[1].set_xlabel("heading error [deg]")
    axes[1].set_title("residual (max %.2e)" % sym_residual)
    for ax in axes:
        ax.grid(alpha=0.3)
    finish(fig, "fig4_symmetry_check.png", "Stage 0 / fig4: left-right symmetry")

    show = [0.0, 45.0, 90.0, 135.0, 180.0]
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 3.4), sharey=True)
    dist = np.linspace(0, 1, params.n_units)  # source plots "neural space" 0..1
    for ax, e in zip(axes, show):
        st = core.evaluate_deg(args.goal_deg + e, args.goal_deg)
        ax.plot(dist, st.act_pfl3r, lw=1.5, label="PFL3R")
        ax.plot(dist, st.act_pfl3l, lw=1.5, label="PFL3L")
        ax.plot(dist, st.act_pfl2, lw=1.5, label="PFL2")
        ax.set_title("e = %.0f deg" % e, fontsize=9)
        ax.set_xlabel("neural space")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("firing rate [a.u.]")
    axes[0].legend(fontsize=7)
    finish(fig, "fig5_phase_profiles.png",
           "Stage 0 / fig5: population activity across neural space")

    # ---- fig 6: overlay against the official notebook --------------------
    overlay = {"available": False}
    if REF_NPZ.exists() and not args.silence_pfl2 and not args.pfl3_inhibitory:
        ref = np.load(REF_NPZ, allow_pickle=True)
        goals = ref["goal_phase"].astype(float)
        svals = ref["S_vals"].astype(float)
        hds = ref["hd_deg"].astype(float)
        gi = int(np.argmin(np.abs(goals - args.goal_deg)))
        si = int(np.argmin(np.abs(svals - args.pfl_scalar)))
        # the reference was built on a COARSE grid, so recalibrate to match it
        coarse = calibrate(CoreParams(n_units=args.n_units,
                                      pfl_scalar_S=float(svals[si])),
                           goal_step=30, hd_step=4, n_scalars=6, verbose=False)
        ccore = WesteindeSteeringCore(
            CoreParams(n_units=args.n_units, pfl_scalar_S=float(svals[si])), coarse)
        ours = np.array([ccore.evaluate_deg(float(h), float(goals[gi])).steering
                         for h in hds])
        theirs = ref["steering"][gi, si, :]
        resid = float(np.abs(ours - theirs).max())
        overlay = {"available": True, "max_abs_deviation": resid,
                   "goal_deg": float(goals[gi]), "S": float(svals[si]),
                   "grid": "coarse (goal step 30, HD step 4)"}

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
        axes[0].plot(hds, theirs, lw=3, alpha=0.45, label="official notebook")
        axes[0].plot(hds, ours, lw=1.3, ls="--", color="C3", label="this implementation")
        axes[0].set_ylabel("steering (normalised)")
        axes[0].legend(fontsize=8)
        axes[1].plot(hds, ours - theirs, lw=1.5, color="C3")
        axes[1].set_ylabel("ours - official")
        axes[1].set_title("max |deviation| = %.2e" % resid)
        for ax in axes:
            ax.set_xlabel("head direction [deg]")
            ax.set_xticks(np.arange(-180, 181, 90))
            ax.grid(alpha=0.3)
        finish(fig, "fig6_source_overlay.png",
               "Stage 0 / fig6: agreement with Westeinde et al. official code")
        checks.append(("matches official notebook (< 1e-12)", resid < 1e-12))
        all_ok &= resid < 1e-12
        print("  [%s] matches official notebook: max deviation %.3e"
              % ("PASS" if resid < 1e-12 else "FAIL", resid))

    # ---- provenance ------------------------------------------------------
    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/stage0_core_io.py",
        "git_commit": git_commit(),
        "core_source_sha256_16": file_sha256(REPO / "src" / "core" / "westeinde2024.py"),
        "source_paper": {
            "citation": "Westeinde et al., Nature 2024, Transforming a head "
                        "direction signal into a goal-oriented steering command",
            "doi": "10.1038/s41586-024-07039-2",
            "author_correction": "10.1038/s41586-024-08245-8 (genotypes only, "
                                 "model unaffected)",
            "official_code": "github.com/wilson-lab/WesteindeWilson_AnalysisCode",
        },
        "connectome_dataset": "hemibrain v1.2.1 (abstract weights 1:4:12; the "
                              "source's per-cell 'data' variant is not used here)",
        "core_params": params.as_dict(),
        "norm_constants_source": norm_source,
        "norm_constants": norm.as_dict(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "diagnostics": diag,
        "source_overlay": overlay,
        "qualitative_checks": {name: bool(ok) for name, ok in checks},
        "all_checks_passed": bool(all_ok),
    }
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print("--- written to %s ---" % outdir)
    for f in sorted(outdir.iterdir()):
        print("  " + f.name)
    print("STAGE0-PART1: %s" % ("ALL CHECKS PASSED" if all_ok else "CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
