"""Failure classification (handoff document section 27).

The document is emphatic that failures must not be lumped together, and names
the trap directly: if r_max is so small that the body physically cannot turn
120 degrees within the timeout, that is not a fly-circuit failure.

This study has a SECOND trap of the same kind, found in Stage 0 (finding F1):
with the source's own decoder gain the IDEAL body already falls into a
period-2 limit cycle from about a quarter of initial headings.  So a trial can
fail for three quite different reasons, and only the third is about the body:

    1. the body could not do it            -> BODY_INFEASIBLE
    2. the core+decoder already failed here with a perfect body
                                           -> BASELINE_FAILURE
    3. neither of the above                -> a genuine embodiment failure,
                                              sub-typed by its signature

Order matters: 1 is checked before 2, and 2 before 3.  Never report a class-1
or class-2 trial as evidence about body tolerance.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class Outcome(str, Enum):
    # --- not attributable to the embodiment ------------------------------
    BODY_INFEASIBLE = "body_infeasible"
    BASELINE_FAILURE = "baseline_failure"
    # --- success ----------------------------------------------------------
    RESCUED_BY_BODY = "rescued_by_body"
    SUCCESS = "success"
    SUCCESS_NEAR_INFEASIBLE = "success_near_infeasible"
    SLOW_BUT_STABLE = "slow_but_stable"
    # --- genuine embodiment failures --------------------------------------
    SATURATION_DOMINATED = "saturation_dominated"
    PERSISTENT_OSCILLATION = "persistent_oscillation"
    OVERSHOOT_DOMINANT = "overshoot_dominant"
    CONTROLLER_INSTABILITY = "controller_instability"
    FAILURE_OTHER = "failure_other"


# thresholds are analysis choices, recorded in provenance with every run
NEAR_INFEASIBLE_MARGIN = 1.5      # max reachable rotation < 1.5x required
SATURATION_FRACTION = 0.5         # over half the sub-steps saturated
OSCILLATION_CROSSINGS = 4         # sign changes of the error
OVERSHOOT_RATIO = 0.5             # overshoot > 50% of the initial error


def classify(metrics, feasibility, initial_error_deg: float,
             baseline_success: bool,
             settle_grace_s: Optional[float] = None) -> Outcome:
    """Classify one trial.

    `baseline_success` is the outcome of the SAME initial condition with an
    ideal body, so that a pre-existing core failure is never charged to the
    body.
    """
    if not feasibility.feasible:
        return Outcome.BODY_INFEASIBLE

    if not baseline_success:
        # The ideal body already failed from this initial heading (Stage 0
        # finding F1).  If the CONSTRAINED body nevertheless succeeds, that is
        # a result in its own right and must not be filed as a failure: the
        # body's own limits damped the overshoot that sustained the core's
        # limit cycle.
        return (Outcome.RESCUED_BY_BODY if metrics.success
                else Outcome.BASELINE_FAILURE)

    if metrics.success:
        if feasibility.margin < NEAR_INFEASIBLE_MARGIN:
            return Outcome.SUCCESS_NEAR_INFEASIBLE
        return Outcome.SUCCESS

    # Failed, the body could have done it, and the ideal body did do it.
    if (metrics.settling_time_s is not None
            and (settle_grace_s is None
                 or metrics.settling_time_s <= settle_grace_s)):
        # it did settle, just not inside the timeout
        return Outcome.SLOW_BUT_STABLE

    if max(metrics.rate_saturation_fraction,
           metrics.accel_saturation_fraction) > SATURATION_FRACTION:
        return Outcome.SATURATION_DOMINATED

    if (metrics.tail_zero_crossings >= OSCILLATION_CROSSINGS
            and metrics.tail_spread_deg > 0.5 * abs(initial_error_deg)):
        return Outcome.PERSISTENT_OSCILLATION

    if metrics.max_overshoot_deg > OVERSHOOT_RATIO * abs(initial_error_deg):
        return Outcome.OVERSHOOT_DOMINANT

    if metrics.final_abs_error_deg > abs(initial_error_deg):
        return Outcome.CONTROLLER_INSTABILITY

    return Outcome.FAILURE_OTHER


# Rescues are attributable to the embodiment too, but in the positive
# direction, so they are counted separately from both successes and failures.
RESCUES = {Outcome.RESCUED_BY_BODY}

ATTRIBUTABLE_TO_EMBODIMENT = {
    Outcome.SATURATION_DOMINATED,
    Outcome.PERSISTENT_OSCILLATION,
    Outcome.OVERSHOOT_DOMINANT,
    Outcome.CONTROLLER_INSTABILITY,
    Outcome.FAILURE_OTHER,
    Outcome.SLOW_BUT_STABLE,
}

NOT_ATTRIBUTABLE = {Outcome.BODY_INFEASIBLE, Outcome.BASELINE_FAILURE}

SUCCESSFUL = {Outcome.SUCCESS, Outcome.SUCCESS_NEAR_INFEASIBLE}


def thresholds_as_dict() -> dict:
    return {"near_infeasible_margin": NEAR_INFEASIBLE_MARGIN,
            "saturation_fraction": SATURATION_FRACTION,
            "oscillation_crossings": OSCILLATION_CROSSINGS,
            "overshoot_ratio": OVERSHOOT_RATIO}
