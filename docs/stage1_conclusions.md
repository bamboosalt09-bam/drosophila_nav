# Stage 1 conclusions

What a **fixed** Drosophila-derived steering circuit does when the body it
drives is changed, and nothing else is.

Numbers here are reproducible from `results/stage1_sweep/<tag>/`; the record of
how each was obtained, including the retractions, is
`provenance/reproduction_adjustments.yaml`.

## What was held fixed

The neural core (Westeinde et al. 2024 reduced PFL2/PFL3 -> DNa03 -> DNa02),
the decoder gain (kappa = 200), the task, and the success criterion are
byte-identical in every cell of every sweep. The core is verified against the
authors' own notebook: PFL population activity bit-identical, steering command
to 3.4e-14.

The constraint plugin receives `command(r_brain, r_actual)` and **cannot**
receive the goal or the heading error — enforced by the Protocol signature and
measured by ablation (core output forced to zero moves the heading by exactly
0.000e+00 deg).

## What was varied

| axis | meaning | levels (dimensionless) |
|---|---|---|
| `r_max` | rate limit, / R_max | 0.002 … 1.0 |
| `alpha` | acceleration cap, / (R_max/T) | 0.005 … 4.0 |
| `tau` | first-order response lag, / T_core | 1 … 32 |

`R_max = kappa * max|steering| = 2000 deg/s`, the fastest turn the brain can
ever request. 252 bodies x 18 initial headings, x8 noise seeds where noise is on.

## Conclusion 1 — the body's limits are one coupled budget, not three failure modes

**No axis in this experiment has a main effect.** Each one's influence depends
entirely on where the other two sit. Spread of each axis *within* each level of
another (condition B, noise on, `interaction_split.csv`):

```
alpha  within r_max :  0.00  0.00  0.17  0.48  0.40      marginal 0.21
alpha  within tau   :  0.33  0.33  0.32  0.20  0.06  0.02
r_max  within tau   :  0.19  0.19  0.22  0.44  0.68  0.76  marginal 0.41
tau    within r_max :  0.11  0.06  0.25  0.60  0.63      marginal 0.32
```

Read the first row: the acceleration cap changes **nothing at all** when the
rate limit is tight, and is the single dominant axis when it is loose.

The mechanism is direct — whichever limit binds first masks the others. As the
rate limit loosens, the plugin stops clipping and the acceleration cap takes
over instead:

```
r_max / R_max        0.002   0.050   0.250   1.000
cycles clipped       99.6%   53.0%   17.2%    2.2%
acceleration capped   0.0%   15.9%   29.7%   33.6%
```

The constraint does not disappear as the body gets more capable. It moves.

**Therefore**: no sentence of the form "a faster body is worse" or "lag hurts"
is admissible on its own. Every claim about one axis must name the level of the
other two. Marginal means over this grid are not summaries of anything.

## Conclusion 2 — a body constraint can *improve* navigation, by filtering neural noise

Visible only with noise on. At `r_max = 0.05, alpha = 0.005`, success is
**non-monotonic** in the response lag:

```
tau / T_core   1     2     4     8     16    32
success      0.66  0.65  0.75  0.94  1.00  0.75   (noise on)
             1.00  1.00  1.00  1.00  1.00  0.875  (noise off — invisible)
```

Measured mechanism: the injected command noise is flat across tau
(std ~1.5 rad/s), while the fraction reaching the heading falls monotonically.

```
std(r_actual) / std(r_commanded)   0.51  0.51  0.47  0.39  0.31  0.28
```

The body low-passes the neural command noise. Success tracks that attenuation
until the lag's own phase cost takes over, giving an optimum near `tau = 16 T`.
Consistent across all 8 seeds independently.

This is an embodiment result in the opposite direction from the one retracted
below, and unlike that one it **requires** noise rather than being destroyed by
it.

## Conclusion 3 — the lag component is compensable without navigation knowledge; the rest is not

Plugin C inverts one cycle of the body's own lag,
`u = (r_brain - r*exp(-T/tau)) / (1 - exp(-T/tau))`, clipped to `r_max`. It has
no tuning knob and no access to the goal.

It does not merely raise the mean (paired 0.825 -> 0.912 with noise). It
**restructures the grid**:

```
                       condition B            condition C
r_max within tau    0.19 … 0.76 (graded)   0.20 0.20 0.20 0.21 0.20 0.18  (flat)
tau   within r_max  0.11 … 0.63 (graded)   0.11 0.02 0.06 0.06 0.01  (collapsed)
alpha within r_max  0.00 … 0.48            0.00 0.00 0.14 0.70 0.70  (steep)
```

C removes the `r_max x tau` interaction outright and collapses tau's influence
everywhere, leaving **one clean boundary in `alpha x r_max`**.

So of the coupled budget, the lag term can be inverted by a goal-blind plugin
that knows only its own body; the rate and acceleration terms cannot, because
they are hard physical limits rather than a phase relationship. C does not
repeal physics: where the acceleration cap genuinely binds it helps only
partially (0.22 -> 0.44 at alpha = 0.005), and in 20 cells it is *worse*
(max -0.062), all at low alpha, because the inversion commands more
acceleration than the cap can deliver.

## Conclusion 4 — the failures are mostly the body's, not the fly circuit's

Control condition D swaps the fly core for a tuned P controller (K = 0.35
frozen) in the same bodies. Noise-free, 180 paired cells:

```
fly 0.859   vs   P 0.793
59 cells fail for BOTH (body-attributable)   15 fly-only worse   57 P-only worse
```

The fly circuit is the better controller here, and most failures are shared,
i.e. attributable to the body rather than to the circuit.

One failure *is* circuit-specific: the period-2 limit cycle at +-90 deg
(finding F1) belongs to the fly core — the P controller's ideal-body baseline
never fails.

(The two cores' paired samples differ by two headings for exactly that reason.
Recomputed on the 16 common headings the numbers are unchanged, 0.859 vs 0.793,
because P's success at +-90 (0.794) matches its own overall rate.)

## Retracted

**"Body limits rescue the core" is withdrawn.** It said that where the ideal
body failed, a constrained body succeeded — 364 of 504 trials, 72%.

Every one of those 504 trials was an initial heading of +-90 deg, i.e. F1. With
the source command noise on, the ideal body succeeds **144/144** baseline
trials: the limit cycle is an unstable equilibrium that any perturbation
escapes. There is nothing left to rescue, and both `BASELINE_FAILURE` and
`RESCUED_BY_BODY` go to zero.

The claim exists only in the deterministic protocol and must never be repeated
without that qualifier.

Two earlier claims were retracted for the same underlying reason — a marginal
mean over a grid with no main effects. Conclusion 1 is what replaced them.

## What this does not establish

- **Condition D ran noise-free only.** The fly-vs-P comparison has not been
  repeated with noise.
- **Feasibility screening is still noise-free.** `is_task_feasible` uses
  noiseless max-effort reachability, so a marginally feasible trial that noise
  pushed over is counted as a genuine failure. The screen is deliberately
  generous to the body, so this inflates apparent body-attributable failure
  rather than hiding it.
- **The sensor is ideal.** Heading is read from the body without delay or error.
- **One task, one dimension.** Goal-heading recovery in yaw only; no forward
  speed, no obstacles, no target approach.
- **The core is the reduced model**, not the connectome. The source's
  `connectivity_option='data'` variant (real per-cell hemibrain weights) is not
  implemented.
