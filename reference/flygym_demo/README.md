# Third-party code — NOT ours, NOT modified

Taken verbatim from [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym),
`src/flygym_demo/complex_terrain/`, retrieved 2026-09-17 from branch `main`.

`flygym` 2.1.0 on PyPI ships the model, the physics and the renderer, but not
the locomotion layer — that lives in the repository's demo package. These files
are copied here unchanged so the closed loop can import them.

## What is in here

| file | what it is |
|---|---|
| `common.py` | `make_locomotion_fly()` — the correctly-parameterised walking fly |
| `preprogrammed.py` | measured single-step leg kinematics, phase → 7 joint angles |
| `cpg_controller.py` | coupled phase-amplitude oscillators; tripod/tetrapod/wave biases |
| `hybrid_controller.py` | CPG + retraction and stumbling corrections + adhesion |
| `turning_controller.py` | `HybridTurningController` — takes `[delta_L, delta_R]` |
| `rule_based_controller.py` | the alternative, rule-based gait (unused so far) |
| `assets/single_steps_untethered.pkl` | the step kinematics data |

## Why this matters for the project

Two facts that changed the architecture:

1. **The descending interface is two numbers.** `HybridTurningController.step`
   takes `descending_signal` of shape `(2,)`. `|delta|` sets each side's CPG
   amplitude; `sign(delta)` runs that side backwards. Nothing else is needed
   from the brain.

2. **The body already closes its own sensory loop.**
   `HybridControllerObservation.from_sim` reads thorax height, all six tarsus5
   heights, and ground-contact forces on tibia/tarsus1/tarsus2, every physics
   step. Retraction and stumbling corrections use them. An earlier note in the
   handoff claiming there was no proprioceptive feedback was wrong.

## Parameters that matter, and that we had wrong

`make_locomotion_fly()` against the naive defaults:

    axis order          YAW_PITCH_ROLL      (not ROLL_PITCH_YAW)
    joint preset        LEGS_ONLY           (keeps the passive tarsal joints)
    joint stiffness     0.05                (default 10.0 — 200x too stiff)
    joint damping       0.06                (default 0.5)
    passive tarsus      stiffness 7.5, damping 1e-2
    actuator kp         45                  (forcerange +-65)
    leg adhesion        gain 40             — without it the fly slides

## Does the project's rule still hold?

The rule: a body-specific plugin must not receive a goal or a heading error and
act as a navigation controller.

The CPG receives neither. It gets a per-side drive amplitude and returns a
gait. Where to go is decided entirely inside the connectome, and the only
designated points remain anatomical — the eye at one end, the descending
neurons at the other. A central pattern generator is body machinery, which is
precisely why the real animal keeps it in the nerve cord and not in the brain.
