# drosophila_nav

Closed-loop simulation study: a **fixed** Drosophila-derived steering circuit
driven into bodies with different motion constraints.

Current stage: **Stage 0, part 1 — core reproduced and validated open-loop.**
The core is verified against the authors' own notebook code: PFL population
activity is bit-identical and the steering command agrees to 3.4e-14
(`tests/test_source_agreement.py`, `results/stage0/default/fig6_source_overlay.png`).
No body is attached yet.  Per the handoff document, no body-robustness result
may be interpreted before Stage 0 passes.

## Architecture (kept separable from the start, for the later 3D renderer)

```
NeuralCore  ->  Decoder  ->  ConstraintPlugin  ->  BodyModel  ->  actual motion
     ^                                                                  |
     +---------------------  SensorModel  <-----------------------------+
```

* `src/core/`        neural computation only (frozen across all body conditions)
* `src/decoder/`     neural output -> abstract action (gain kappa)      *(todo)*
* `src/plugins/`     goal-blind command shaping                          *(todo)*
* `src/body/`        actual dynamics                                     *(todo)*
* `src/sensors/`     body/environment state -> neural input              *(todo)*
* `src/environment/` goal, perturbations                                 *(todo)*
* `src/sim/`         closed-loop scheduler                               *(todo)*
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

## Run

```bash
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe experiments/stage0_core_io.py
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
