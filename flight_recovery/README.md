# Flight recovery, 2026-10-18

For a fresh clone, first run `python flight_recovery/restore_artifacts.py` to
restore the hash-verified graph/body/readout cache from the repository's GitHub
artifact tag. Source and measured results are tracked in Git. Larger assets are
preserved in [the independent data tag](https://github.com/bamboosalt09-bam/drosophila_nav/tree/flight-artifacts-2026-10-20),
linked from [the release](https://github.com/bamboosalt09-bam/drosophila_nav/releases/tag/flight-recovery-2026-10-20).
See [CONTINUE_HERE.md](../CONTINUE_HERE.md) for the handoff and
[recovery_backups/README.md](../recovery_backups/README.md) for all download options.

This is a fresh, measured engineering reconstruction after the previous nav_v2 source, policies and result packet could not be found in the current workspace or Git. It is not a bitwise restoration of that run or a physiological model of natural flapping flight.

Start with `results/flight_recovery_2026-10-18/results.md`. Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python flight_recovery/run.py --duration 20 --output results/new_flight_trial
```

The cached actual annotated-endpoint MaleCNS operator, binding, annotations, fresh MN readout and exact Flybody mass/inertia cache are in `derived/`. These are sufficient to run the new flight without downloading the 1.1GB raw graph or the Flybody meshes again. The portable runtime archive includes these files and this code. Python dependencies are in `requirements.txt`; the existing environment uses Python 3.13.15.

To rebuild from raw data, run `scripts/get_malecns.py`, `flight_recovery/build_graph.py`, and `flight_recovery/neural.py`. To regenerate body properties, remove only the cache in a separate copy and provide the official source under `reference/flybody_recovered` (commit and XML hash are in `derived/flybody_source.json`). No original input data or historical result is overwritten by this recovery.

Sources: FlyEM MaleCNS v1.0, CC-BY; TuragaLab/flybody source and mass properties (FLYBODY_LICENSE); Three.js 0.158.0, MIT (license header retained in three.min.js).

The original 4,000-window motion table, drone converter and obstacle flight results were not restored. The current checks concern sustained 3D flight of the engineering equivalent plant only.
