"""Stage 0, part 1: open-loop input-output characterisation of the core.

Produces the graphs required by handoff document section 43, items 1-5 and 7
(item 6, the closed-loop heading trajectory, comes in stage0_ideal_loop.py).

Usage
-----
    python experiments/stage0_core_io.py
    python experiments/stage0_core_io.py --normalize global_rms --tag rmsnorm
    python experiments/stage0_core_io.py --no-dn-activation --tag linear_dn

Outputs (results/stage0/<tag>/):
    fig1_steering_vs_error.png
    fig2_pfl_tuning.png
    fig3_dn_outputs.png
    fig4_symmetry_check.png
    fig5_phase_profiles.png
    core_io_sweep.csv
    provenance.json
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
matplotlib.use("Agg")  # headless: never depend on a GUI backend for results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.westeinde2024 import CoreParams, WesteindeSteeringCore


# --------------------------------------------------------------------------
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
    p.add_argument("--normalize", default="none",
                   choices=["none", "global_peak", "global_rms"])
    p.add_argument("--no-dn-activation", action="store_true",
                   help="linear DNa03/DNa02 stage (diagnostic: PFL2 cancels)")
    p.add_argument("--gain", type=float, default=1.0, help="core gain S")
    p.add_argument("--goal-amplitude", type=float, default=1.0, help="A")
    p.add_argument("--step-deg", type=float, default=0.5)
    p.add_argument("--tag", default="default", help="subfolder under results/stage0")
    return p


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    params = CoreParams(
        n_units=args.n_units,
        normalize=args.normalize,
        dn_activation=not args.no_dn_activation,
        gain_S=args.gain,
        goal_amplitude=args.goal_amplitude,
    )
    core = WesteindeSteeringCore(params)

    outdir = REPO / "results" / "stage0" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- sweep over heading error ---------------------------------------
    err_deg = np.arange(-180.0, 180.0 + args.step_deg, args.step_deg)
    out = core.sweep(np.deg2rad(err_deg))

    df = pd.DataFrame({"heading_error_deg": err_deg, **{k: v for k, v in out.items()
                                                        if k != "heading_error"}})
    df.to_csv(outdir / "core_io_sweep.csv", index=False)

    # ---- scalar diagnostics ---------------------------------------------
    steer = out["steering"]
    i0 = int(np.argmin(np.abs(err_deg)))
    # central-difference slope at e = 0, in steering units per degree
    slope0 = float((steer[i0 + 1] - steer[i0 - 1]) / (err_deg[i0 + 1] - err_deg[i0 - 1]))
    peak_i = int(np.argmax(steer))
    sym_residual = float(np.max(np.abs(steer + steer[::-1])))

    diag = {
        "steering_at_0deg": float(steer[i0]),
        "steering_slope_at_0_per_deg": slope0,
        "steering_slope_at_0_per_rad": slope0 * 180.0 / np.pi,
        "peak_steering": float(steer[peak_i]),
        "peak_steering_at_deg": float(err_deg[peak_i]),
        "steering_at_180deg": float(steer[-1]),
        "odd_symmetry_max_residual": sym_residual,
        "pfl3r_peak_at_deg": float(err_deg[int(np.argmax(out["pfl3r"]))]),
        "pfl3l_peak_at_deg": float(err_deg[int(np.argmax(out["pfl3l"]))]),
        "pfl2_peak_at_deg": float(err_deg[int(np.argmax(out["pfl2"]))]),
        "pfl2_trough_at_deg": float(err_deg[int(np.argmin(out["pfl2"]))]),
    }

    print("=== Stage 0 core I/O diagnostics (tag: %s) ===" % args.tag)
    for k, v in diag.items():
        print("  %-32s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))

    checks = [
        ("steering(0) == 0", abs(diag["steering_at_0deg"]) < 1e-8),
        ("odd symmetry", sym_residual < 1e-8),
        ("restoring sign for e>0", bool(np.all(steer[err_deg > 0.5] > 0))),
        ("PFL3R peak near +67.5", abs(diag["pfl3r_peak_at_deg"] - 67.5) < 3.0),
        ("PFL3L peak near -67.5", abs(diag["pfl3l_peak_at_deg"] + 67.5) < 3.0),
        ("PFL2 trough near 0 (anti-goal)", abs(diag["pfl2_trough_at_deg"]) < 3.0),
        ("PFL2 peak near +-180", abs(abs(diag["pfl2_peak_at_deg"]) - 180.0) < 3.0),
        ("steering(180) == 0 (unstable eq.)", abs(diag["steering_at_180deg"]) < 1e-8),
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

    # fig 1: steering vs heading error
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(err_deg, steer, lw=2, color="C0")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.plot(diag["peak_steering_at_deg"], diag["peak_steering"], "o", color="C3",
            label="peak at %.1f deg" % diag["peak_steering_at_deg"])
    ax.set_xlabel("heading error e = wrap(heading - goal)  [deg]")
    ax.set_ylabel("steering drive  DNa02R - DNa02L  [a.u.]")
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    finish(fig, "fig1_steering_vs_error.png",
           "Stage 0 / fig1: steering output vs heading error")

    # fig 2: PFL tuning curves
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(err_deg, out["pfl3r"], lw=2, label="PFL3R")
    ax.plot(err_deg, out["pfl3l"], lw=2, label="PFL3L")
    ax.plot(err_deg, out["pfl2"], lw=2, label="PFL2")
    for x in (-67.5, 67.5):
        ax.axvline(x, color="gray", ls=":", lw=1)
    ax.axvline(180.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("heading error [deg]")
    ax.set_ylabel("population mean activity [a.u.]")
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    finish(fig, "fig2_pfl_tuning.png",
           "Stage 0 / fig2: PFL3R / PFL3L / PFL2 tuning (dotted: +-67.5, 180 deg)")

    # fig 3: DN outputs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(err_deg, out["dna03r"], lw=2, label="DNa03R")
    axes[0].plot(err_deg, out["dna03l"], lw=2, label="DNa03L")
    axes[0].set_title("DNa03 (indirect pathway, PFL2-dominated)")
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

    # fig 4: symmetry check
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pos = err_deg >= 0
    axes[0].plot(err_deg[pos], steer[pos], lw=2, label="steering(+e)")
    axes[0].plot(err_deg[pos], -steer[::-1][pos], lw=2, ls="--",
                 label="-steering(-e)")
    axes[0].set_xlabel("|heading error| [deg]")
    axes[0].set_ylabel("steering [a.u.]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(err_deg, steer + steer[::-1], lw=1.5, color="C3")
    axes[1].set_xlabel("heading error [deg]")
    axes[1].set_ylabel("steering(e) + steering(-e)")
    axes[1].set_title("residual (max %.2e)" % sym_residual)
    axes[1].grid(alpha=0.3)
    finish(fig, "fig4_symmetry_check.png",
           "Stage 0 / fig4: left-right symmetry of the fixed core")

    # fig 5: phase-resolved population activity at selected errors
    show_errs = [0.0, 45.0, 90.0, 135.0, 180.0]
    fig, axes = plt.subplots(1, len(show_errs), figsize=(3.0 * len(show_errs), 3.4),
                             sharey=True)
    theta_deg = np.rad2deg(core.theta)
    for ax, e in zip(axes, show_errs):
        st = core.evaluate(np.deg2rad(e), 0.0)
        ax.plot(theta_deg, st.act_pfl3r, lw=1.5, label="PFL3R")
        ax.plot(theta_deg, st.act_pfl3l, lw=1.5, label="PFL3L")
        ax.plot(theta_deg, st.act_pfl2, lw=1.5, label="PFL2")
        ax.set_title("e = %.0f deg" % e, fontsize=9)
        ax.set_xlabel("neural phase [deg]")
        ax.set_xticks([0, 180, 360])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("unit activity [a.u.]")
    axes[0].legend(fontsize=7)
    finish(fig, "fig5_phase_profiles.png",
           "Stage 0 / fig5: activity across the neural phase grid")

    # ---- provenance ------------------------------------------------------
    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/stage0_core_io.py",
        "git_commit": git_commit(),
        "core_source_sha256_16": file_sha256(REPO / "src" / "core" / "westeinde2024.py"),
        "source_paper": {
            "citation": "Westeinde et al., Nature 2024, "
                        "Transforming a head direction signal into a "
                        "goal-oriented steering command",
            "doi": "10.1038/s41586-024-07039-2",
            "doi_status": "TO BE VERIFIED against the publisher record",
        },
        "connectome_dataset": "hemibrain v1.2.1 (via published connectivity "
                              "statistics, not re-derived here)",
        "core_params": params.as_dict(),
        "explicit_assumptions": ["A1 goal-bump form", "A2 activation/normalisation",
                                 "A3 mean readout", "A4 DN-stage nonlinearity",
                                 "A5 sign handled by decoder"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "diagnostics": diag,
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
