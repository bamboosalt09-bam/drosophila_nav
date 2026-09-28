# drosophila_nav

Closed-loop simulation study: a **fixed** Drosophila-derived steering circuit
driven into bodies with different motion constraints.

**If you are picking this up cold, read [`docs/HANDOFF.md`](docs/HANDOFF.md) first.**
It says where the work stands, what is settled with the number that settled it,
and what is still open.  Stage 2 (the whole MaleCNS connectome) is the live
work; Stage 1 below is the closed baseline.

Current stage: **Stage 1 complete — body sweep (noise off and on), conditions B/C/D, 3D viewer.**
The core is verified against the authors' own notebook code: PFL population
activity is bit-identical and the steering command agrees to 3.4e-14
(`tests/test_source_agreement.py`, `results/stage0/default/fig6_source_overlay.png`).
The ideal closed loop reproduces the source's Figure 5 trend (heading
consistency rho rises from 0.38 to 0.97 as the PFL scalar goes 0 -> 1), and our
general scheduler agrees with the source's own loop to 0.04 in rho.

Two baseline properties of the ideal loop must not be mistaken for
body-induced failure: with the source gain k = 200 it falls into a period-2
limit cycle near +-100 deg from 26% of initial headings (finding F1), and the
anti-goal equilibrium escapes on floating-point rounding.

**F1 exists only with noise off.**  With the source command noise on, the ideal
body succeeds in 144/144 baseline trials, so `BASELINE_FAILURE` and
`RESCUED_BY_BODY` both drop to zero — the "body limits rescue the core" result
is an artifact of the deterministic protocol and is retracted as a general
claim.  What survives noise is the opposite direction: body constraints still
cause genuine failure, and a lagging body can *improve* navigation by
low-passing the neural command noise (finding R3, optimum near tau = 16 T).
Separately: **no axis in this experiment has a main effect.** Every one
of the three body axes changes its influence depending on where the other
two sit (e.g. the acceleration cap does nothing at r_max <= 0.02 and is the
dominant axis at r_max = 0.25), so no marginal mean over this grid is a
valid summary.  `results/stage1_sweep/<tag>/interaction_split.csv` holds the
conditional tables any such claim must be checked against.
**Stage 1's conclusions are written up in [`docs/stage1_conclusions.md`](docs/stage1_conclusions.md)**, with the
record of how each number was obtained (and what was retracted) in
`provenance/reproduction_adjustments.yaml`.

## Architecture (kept separable from the start, for the later 3D renderer)

```
NeuralCore  ->  Decoder  ->  ConstraintPlugin  ->  BodyModel  ->  actual motion
     ^                                                                  |
     +---------------------  SensorModel  <-----------------------------+
```

* `src/core/`        neural computation only (frozen across all body conditions)
* `src/decoder/`     neural output -> yaw rate (frozen gain k = 200)
* `src/plugins/`     goal-blind command shaping (passthrough; rate clip todo)
* `src/body/`        actual dynamics (ideal yaw; constrained plant todo)
* `src/sensors/`     body state -> neural input (ideal heading)
* `src/environment/` goal, initial conditions
* `src/sim/`         closed-loop scheduler, noise, source-loop replica
* `src/render/`      3D viewer, reads simulation state only              *(todo)*
* `src/eval/`        metrics                                             *(todo)*

Hard rules: the plugin never sees the goal or the heading error; actual body
motion (never the neural command) is fed back through the sensor; the core is
never re-tuned per body.

## Start from a fresh clone (Windows, Python 3.13)

```bash
git clone https://github.com/bamboosalt09-bam/drosophila_nav.git
cd drosophila_nav
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
set PYTHONPATH=src
.venv/Scripts/python.exe -m sim.drone_layer
.venv/Scripts/python.exe -m sensors.vp_input
.venv/Scripts/python.exe experiments/cx_vp_room.py --room 6
```

That is enough for the live experiment: the repo carries the fisheye
extension built for CPython 3.13 / win_amd64
(`src/sensors/_fisheye_cpp/_fisheye.cp313-win_amd64.pyd`) and the
34,125-neuron subnet (`results/subnet/vp_h4_rho0.50_ol_intrinsic.pkl`), so
neither MSVC nor the 2 GB connectome is needed.  Watch a flight with
`experiments/live_view.py` and `--live` (see `CLAUDE.md`).

Where the work stands and how to resume: `docs/HANDOFF.md` (start with
"INPUTS" and "DRONE LAYER" in the 2026-09-28 section), `CLAUDE.md` for a
new Claude session, and the settled research design in
`docs/master_handoff.md`.

## Data (not in git, ~2 GB) — only to rebuild the subnet or query the connectome

* `data/malecns/` — `python scripts/get_malecns.py` downloads it (1.1 GB,
  public FlyEM bucket, CC-BY, no account).
* `data/flywire/` — FlyWire v783 (`proofread_connections_783.feather`,
  `proofread_root_ids_783.npy`, `neuron_annotations_783.tsv`, `edges_783.npz`,
  `column_assignment.csv.gz`, `visual_neuron_types.csv.gz`); only the
  Stage 0/1 FlyWire code uses it.

Other Python versions or platforms: rebuild the extension with
`src/sensors/_fisheye_cpp/build.py build_ext --inplace` (`build_cpp.bat` on
Windows with MSVC).  Ignored and regenerated on demand: training clips
`results/clips/`, most images (the sweep figures `results/sweep_*.png` are
kept).

## Run

```bash
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe experiments/stage0_core_io.py
.venv/Scripts/python.exe experiments/stage0_ideal_loop.py
```

The frozen normalisation constants are committed as `configs/norm_constants.json`.
Regenerate them (16 s) or rebuild the official reference only if needed:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m core.calibration --out configs/norm_constants.json
```

```bash
.venv/Scripts/python.exe reference/run_official.py --goal-step 30 --hd-step 4 --out ref_coarse.npz
```

Outputs go to `results/stage0/default/` (5 figures, `core_io_sweep.csv`,
`provenance.json`).  The script prints qualitative checks and exits non-zero if
any of them fails.

Diagnostic variants:

```bash
.venv/Scripts/python.exe experiments/stage0_core_io.py --silence-pfl2 --tag pfl2_lesion
.venv/Scripts/python.exe experiments/stage0_core_io.py --pfl-scalar 0.4 --tag S04
```

## Conventions

* radians internally, degrees only at the interface
* heading psi CCW-positive, wrapped to [-pi, pi)
* heading error `e = wrap(heading - goal)`
* the core returns the grid-normalised `DNa02R - DNa02L` (peak magnitude 1);
  turning it into a physical yaw rate is the decoder's job.  The source uses
  k = 200 at 10 Hz, i.e. degrees per 0.1 s timestep
* a positive heading error commands a negative steering value: the loop is
  negative feedback, matching the source's `hd += k*steering`

## Provenance

`provenance/` records what is source-derived versus assumed.  Read
`reproduction_adjustments.yaml` before making any claim about reproducing
Westeinde et al. 2024: it records the comparison against the official code,
what the reproduction does and does not license, and what is still open (the
per-cell hemibrain `data` connectivity variant, and wiring k = 200 into the
decoder).

`reference/westeinde_official/` holds an unmodified copy of the authors'
notebook.  It is reference material, not part of this implementation.
