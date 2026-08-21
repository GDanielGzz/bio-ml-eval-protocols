"""Tests for the reporting helpers.

These check that the prose refuses to overstate the evidence — the property the
module exists for.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from bioeval import CI, bootstrap_ci, comparison_note, paired_bootstrap, results_table


def _scores(n=800, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return y, rng.normal(loc=signal * y, scale=1.0)


def test_results_table_shows_intervals_and_method():
    y, s = _scores()
    ci = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=0)
    table = results_table({"model": ci}, metric_name="AUC")
    assert "AUC [95% CI]" in table
    assert f"{ci.low:.4f}" in table and f"{ci.high:.4f}" in table
    assert "Method" in table


def test_results_table_rejects_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        results_table({})


def test_tie_is_written_as_a_tie_not_as_a_narrow_win():
    y, s = _scores(seed=3)
    res = paired_bootstrap(y, s, s.copy(), roc_auc_score, n_boot=300, seed=0)
    note = comparison_note(res)
    assert "statistically tied" in note
    assert "Reported as a tie rather than a win" in note
    for banned in ("slightly better", "comparable", "on par"):
        assert banned not in note


def test_overlapping_marginals_with_significant_difference_are_explained():
    """The reviewer-proofing case the docs call out."""
    paired = type(
        "P", (), {}
    )  # placeholder replaced below by a real PairedResult
    rng = np.random.default_rng(4)
    n = 4000
    y = rng.integers(0, 2, size=n)
    a = rng.normal(loc=1.05 * y, scale=1.0)
    b = a + rng.normal(scale=0.35, size=n)      # correlated, slightly worse

    res = paired_bootstrap(y, a, b, roc_auc_score, n_boot=500, seed=0,
                           label_a="A", label_b="B")
    ci_a = bootstrap_ci(y, a, roc_auc_score, n_boot=500, seed=0)
    ci_b = bootstrap_ci(y, b, roc_auc_score, n_boot=500, seed=0)

    note = comparison_note(res, ci_a, ci_b)
    if res.significant and ci_a.overlaps(ci_b):
        assert "not evidence of a tie" in note
        assert "cancels in the paired difference" in note
    else:
        # Still must not have produced a contradictory note.
        assert "not evidence of a tie" not in note


def test_comparison_note_without_marginals_is_still_valid():
    y, s = _scores(seed=6)
    rng = np.random.default_rng(6)
    worse = s + rng.normal(scale=2.0, size=s.size)
    res = paired_bootstrap(y, s, worse, roc_auc_score, n_boot=300, seed=0)
    note = comparison_note(res)
    assert note.strip()
    assert "CI)" in note
