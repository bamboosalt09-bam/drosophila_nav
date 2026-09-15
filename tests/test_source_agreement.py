"""Stage 0 reproduction gate: our core must match the official notebook.

This is the test that decides whether "reduced Westeinde steering core" is an
honest description of src/core/westeinde2024.py.  Everything else in Stage 0
checks that the model is self-consistent; this checks that it is the SOURCE
model.

Ground truth is produced by reference/run_official.py, which exec's the
authors' own notebook cells.  It is compared on a coarsened grid, because the
full official grid needs several GB of RAM.  Coarsening changes the numbers
(the source normalises over the whole sweep), so our core is calibrated on the
SAME coarse grid here -- that equality is the thing being tested.

Regenerate the reference if it is missing:
    python reference/run_official.py --goal-step 30 --hd-step 4 --out ref_coarse.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.calibration import calibrate
from core.westeinde2024 import CoreParams, WesteindeSteeringCore

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "reference" / "ref_coarse.npz"

# grid used to build ref_coarse.npz
GOAL_STEP, HD_STEP, N_SCALARS = 30, 4, 6


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("run reference/run_official.py to generate %s" % REF.name)
    return np.load(REF, allow_pickle=True)


@pytest.fixture(scope="module")
def norm():
    return calibrate(CoreParams(), goal_step=GOAL_STEP, hd_step=HD_STEP,
                     n_scalars=N_SCALARS, verbose=False)


@pytest.fixture(scope="module")
def ours(ref, norm):
    """Evaluate our core over the whole reference grid."""
    goals = ref["goal_phase"].astype(float)
    svals = ref["S_vals"].astype(float)
    hds = ref["hd_deg"].astype(float)
    shape = (goals.size, svals.size, hds.size)
    keys = ("steering", "dna02r", "dna02l", "dna03r", "dna03l",
            "sum_pfl3r", "sum_pfl3l", "sum_pfl2", "pfl2_bump_amp")
    out = {k: np.empty(shape) for k in keys}
    for si, S in enumerate(svals):
        core = WesteindeSteeringCore(CoreParams(pfl_scalar_S=float(S)), norm)
        for gi, g in enumerate(goals):
            for hi, hd in enumerate(hds):
                st = core.evaluate_deg(float(hd), float(g))
                for k in keys:
                    out[k][gi, si, hi] = getattr(st, k)
    return out


def test_hd_prefs_match_the_source(ref):
    core = WesteindeSteeringCore(CoreParams(n_units=int(ref["num_cells"])))
    assert np.array_equal(core.hd_prefs, ref["hd_prefs"])


def test_population_activity_matches_bitwise(ref, norm):
    """Cell-by-cell PFL activity at goal = 0, S = 1."""
    core = WesteindeSteeringCore(
        CoreParams(pfl_scalar_S=float(ref["slice_S"])), norm)
    goal = float(ref["slice_goal_deg"])
    hds = ref["hd_deg"].astype(float)
    for hi, hd in enumerate(hds):
        st = core.evaluate_deg(float(hd), goal)
        assert np.allclose(st.act_pfl3r, ref["pfl3r_slice"][hi], atol=1e-15)
        assert np.allclose(st.act_pfl3l, ref["pfl3l_slice"][hi], atol=1e-15)
        assert np.allclose(st.act_pfl2, ref["pfl2_slice"][hi], atol=1e-15)


@pytest.mark.parametrize("ours_key,ref_key", [
    ("steering", "steering"),
    ("dna02r", "Dna02r"),
    ("dna02l", "Dna02l"),
    ("dna03r", "Dna03r"),
    ("dna03l", "Dna03l"),
    ("pfl2_bump_amp", "pfl2_bump_amp"),
    ("sum_pfl2", "pfl2_sum"),
])
def test_scalar_outputs_match_the_source(ours, ref, ours_key, ref_key):
    a, b = ours[ours_key], ref[ref_key]
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-12, (
        "%s deviates from the source by %.3e" % (ours_key, np.abs(a - b).max()))


def test_pfl3_difference_matches_the_source(ours, ref):
    diff = ours["sum_pfl3r"] - ours["sum_pfl3l"]
    assert np.abs(diff - ref["pfl3_RL_diff"]).max() < 1e-12


def test_steering_is_normalised_to_unit_peak(ours):
    assert np.isclose(np.abs(ours["steering"]).max(), 1.0, atol=1e-9)
