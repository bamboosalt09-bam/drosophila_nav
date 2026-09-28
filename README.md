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

## Setup (Windows, already done in this repo)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Data (not in git, ~2 GB)

`data/` is ignored.  Put the connectome files there before running:

* `data/malecns/` — from `storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/` (CC-BY, no account):
  `body-annotations-male-cns-v1.0-minconf-0.5.feather`,
  `body-neurotransmitters-male-cns-v1.0.feather`,
  `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
* `data/flywire/` — FlyWire v783 (`proofread_connections_783.feather`,
  `proofread_root_ids_783.npy`, `neuron_annotations_783.tsv`, `edges_783.npz`,
  `column_assignment.csv.gz`, `visual_neuron_types.csv.gz`)

Also ignored and rebuilt on demand: the fisheye extension (`build_cpp.bat`,
needs MSVC), subnet pickles in `results/subnet/`, training clips in
`results/clips/`, and result images (`*.png`).

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
