# drosophila_nav

Closed-loop simulation study: a **fixed** Drosophila-derived steering circuit
driven into bodies with different motion constraints.

Current stage: **Stage 0, part 1 — core validated open-loop.**
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

Outputs go to `results/stage0/default/` (5 figures, `core_io_sweep.csv`,
`provenance.json`).  The script prints qualitative checks and exits non-zero if
any of them fails.

Diagnostic variants:

```bash
.venv/Scripts/python.exe experiments/stage0_core_io.py --normalize global_rms --tag rmsnorm
.venv/Scripts/python.exe experiments/stage0_core_io.py --no-dn-activation --tag linear_dn
```

## Conventions

* radians internally, degrees only at the interface
* heading psi CCW-positive, wrapped to [-pi, pi)
* heading error `e = wrap(heading - goal)`
* the core returns the raw `DNa02R - DNa02L`; sign and physical units are the
  decoder's responsibility

## Provenance

`provenance/` records what is source-derived versus assumed.  Read
`reproduction_adjustments.yaml` before making any claim about reproducing
Westeinde et al. 2024 — it lists the open items, including **issue O1**
(PFL2 currently has no effect on the steering output).
