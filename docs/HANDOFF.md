# Handoff — resume from here

Written against session compaction.  `provenance/reproduction_adjustments.yaml`
is the full record (29 sections); this is the short version that says where the
work stands and what to do next.

## The question

Can a fly connectome, run as a computable circuit with only its **anatomical**
input and output designated, produce navigation?  Nothing between input and
output may be chosen: no hand-picked "steering circuit", no assigned function.

Long term: is one fly-derived navigation core reusable across different bodies?

## Two stages, and which is live

**Stage 0/1 — the reduced model. DONE, closed, written up.**
Westeinde et al. 2024's 5-population steering circuit, reproduced bit-identically
(PFL activity 0.0 difference, steering 3.4e-14), then driven into 252 bodies.
Conclusions in `docs/stage1_conclusions.md`.  Do not reopen; it is the baseline.

**Stage 2 — the whole connectome. LIVE.**
Everything below is this.

## Dataset: MaleCNS, not FAFB

FAFB (`data/flywire/`) is brain-only.  It has no motor neurons, which is why an
earlier attempt had to invent a steering command out of 1,303 mixed descending
neurons.  **MaleCNS v1.0 (`data/malecns/`) contains the nerve cord** and is the
live dataset.

    166,700 neurons, 6.2M edges at >= 5 synapses, builds in ~4 s
    909 motor neurons, named by muscle ("Ti flexor MN") and leg (T1/T2/T3)
    13,161 vnc_intrinsic -- the CPG, as data rather than hand-written
    assignedOlHex1/2 -- the eye lattice, built into the annotation
    `group` -- explicit left-right pairs, which FAFB lacked

Public bucket, no login:
`storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

FAFB code still exists and still passes its tests.  It is kept as the
reproduction record, not as a live path.

## What runs today

    scene -> 6,199 lamina cells -> 166,700 neurons -> leg motor neurons, 26 s

    src/core/malecns.py          loader, + mirror-average and per-stage balance
    src/core/flywire_rate.py     differentiable rate model, 141 trainable params
    src/core/null_wiring.py      degree-preserving and random-sparse controls
    src/sensors/flywire_eye.py   load_malecns_eye(): hex -> azimuth/elevation
    src/decoder/steering.py      read_motor(): per leg pair, per side
    experiments/flywire_train_pilot.py   training loop (still FAFB-wired)

`pytest tests/ -q` — 110 passing.

## Settled, with the number that settled it

- **Laterality.** Mirror-average where `group` pairs cells (75-88% of motor,
  descending, VNC), then rescale the residue per anatomical stage.  Every stage
  goes to R/L 1.0000 from 0.91-1.26.  Thresholding weak edges does NOT work
  (R/L flat at 1.05-1.10 from threshold 1 to 50).
- **Eye scale.** 5.0 deg per column, checked against the field of view
  (147-162 deg) and independently confirmed by flygym's 157 deg per eye.
- **Weight scale.** w_scale 0.005, not 0.01: at 0.01 the central brain's
  heading variation collapses to 0.9% against 44.8% at 0.005.  Calibrating on
  whole-network mean activity hides this, because the optic lobe dominates it.
- **Sign.** ACh/DA/OA/5-HT excitatory, GABA/Glu inhibitory, per presynaptic
  NEURON not per edge (the per-edge averages flip the sign on 9.6% of edges).
  Same rule Shiu et al. use.
- **Neuron parameters.** Shiu et al. 2024 `model.py`, read from their code.

## Open, and honest about it

- **`turn` does not cross zero.**  It correlates with sin(bar azimuth) at
  -0.67, but offset/range is 0.43.  The wiring is balanced to 1.0000 and
  matching the injected columns between eyes barely moves it (0.48 -> 0.43).
  So it is neither the wiring nor the input count.  Unexplained.
- **No heading bump.**  On FAFB, EPG never formed one.  Not yet re-checked on
  MaleCNS.
- **Training finds degenerate solutions.**  The loss descends only after the
  parameter scales are separated (bias shares the activity scale ~0.01 while
  log_tau/log_gain are exponents; gradient norms differ 13x).  It then reduces
  scale and offset rather than shape, and silences EPG.
- The task may not REQUIRE a bump: an instantaneous stimulus-response map has
  nothing to remember.  A closed loop with a persistent goal would.

## Next: close the loop with flygym

`flygym 2.1.0` + `mujoco 3.9.0` are installed.  It is NeuroMechFly v2, and the
two interfaces already match:

    get_ommatidia_readouts()  ->  our lamina injection     721/eye, 157 deg FOV
    set_actuator_inputs()     <-  our motor readout        48 leg segments

Three mappings remain:

1. **875 hex columns -> 721 ommatidia**, by nearest viewing direction.  Both
   are hex lattices with known directions, so this is a nearest-neighbour join.
2. **909 motor neurons -> joint commands.**  Available, not invented: motor
   types name their muscles and the muscles are antagonists.  `Ti flexor MN`
   (37) against `Ti extensor MN` (12) is the tibia joint; `Tr flexor` /
   `Acc. tr flexor` / `Fe reductor` the femur; sternal and pleural rotators the
   coxa.  349 of 909 name a leg muscle; the rest are abdominal, neck, haltere.
3. **Timescales.**  Our rate model runs at dt 5 ms; flygym's mechanics at
   600 Hz.

## The habit this project runs on

Five times a pooled average has hidden the real behaviour: Stage 1's marginal
means (three separate retractions), pooling 1,303 descending neurons into one
scalar, calibrating w_scale on whole-network activity, averaging over central
cell types, and argmax over one-spike EPG histograms.

**Split every pooled number before believing it.**  Six claims have been
retracted in this project, all of them caught by re-measuring rather than by
reasoning.  Retractions are recorded in provenance, not deleted.

Related: `desktop-fly` uses the same two datasets but keeps 1,045 neurons and
hand-assigns their function (DNa01/02 steering, DNg11 grooming).  That is the
shortcut this project exists to avoid.
