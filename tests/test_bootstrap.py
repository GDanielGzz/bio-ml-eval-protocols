"""Tests for the bootstrap machinery.

The important ones are the statistical properties: that the interval covers the
truth at roughly the nominal rate, that a row bootstrap on grouped data gives a
too-narrow interval (the error this library exists to prevent), and that the
paired test calls a genuine tie a tie.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from bioeval import CI, bootstrap_ci, grouped_split, paired_bootstrap
from bioeval.datasets import make_clustered_binary


def _simple_scores(n=400, auc_signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = rng.normal(loc=auc_signal * y, scale=1.0)
    return y, s


def test_estimate_matches_the_metric_on_the_full_sample():
    y, s = _simple_scores()
    ci = bootstrap_ci(y, s, roc_auc_score, n_boot=400, seed=0)
    assert ci.estimate == pytest.approx(roc_auc_score(y, s))


def test_interval_brackets_the_estimate():
    y, s = _simple_scores()
    ci = bootstrap_ci(y, s, roc_auc_score, n_boot=500, seed=0)
    assert ci.low < ci.estimate < ci.high
    assert 0.0 <= ci.low and ci.high <= 1.0


def test_more_data_gives_a_narrower_interval():
    y_small, s_small = _simple_scores(n=120, seed=1)
    y_big, s_big = _simple_scores(n=2400, seed=1)
    small = bootstrap_ci(y_small, s_small, roc_auc_score, n_boot=500, seed=0)
    big = bootstrap_ci(y_big, s_big, roc_auc_score, n_boot=500, seed=0)
    assert big.width < small.width


def test_bootstrap_is_deterministic_under_seed():
    y, s = _simple_scores()
    a = bootstrap_ci(y, s, roc_auc_score, n_boot=300, seed=7)
    b = bootstrap_ci(y, s, roc_auc_score, n_boot=300, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_coverage_is_near_nominal():
    """~95% of intervals should contain the population value.

    Approximate by treating a very large sample's AUC as truth, then drawing
    many small samples from the same generator. Tolerance is wide because 60
    replicates is itself noisy; the test is here to catch a broken interval
    (coverage ~0.5 or ~1.0), not to certify second-decimal accuracy.
    """
    rng = np.random.default_rng(123)
    y_pop = rng.integers(0, 2, size=200_000)
    s_pop = rng.normal(loc=1.0 * y_pop, scale=1.0)
    truth = roc_auc_score(y_pop, s_pop)

    hits = 0
    trials = 60
    for t in range(trials):
        y, s = _simple_scores(n=300, seed=1000 + t)
        ci = bootstrap_ci(y, s, roc_auc_score, n_boot=300, seed=t)
        hits += ci.low <= truth <= ci.high
    assert 0.80 <= hits / trials <= 1.0


def test_row_bootstrap_on_grouped_data_is_too_narrow():
    """The error the ``groups=`` argument exists to prevent.

    Ignoring group structure understates uncertainty *when model errors are
    correlated within a group* — which is the real case, because a model that
    gets one member of a sequence family wrong usually gets its homologs wrong
    the same way. The cluster-level offset below reproduces that correlation;
    without it there is nothing for the cluster bootstrap to widen, and the two
    intervals legitimately agree.
    """
    X, y, groups = make_clustered_binary(n_clusters=80, seed=4)
    _, test = grouped_split(groups, test_size=0.4, seed=4)
    g_test = groups[test]

    rng = np.random.default_rng(0)
    cluster_offset = {c: rng.normal(scale=1.1) for c in np.unique(g_test)}
    offsets = np.array([cluster_offset[c] for c in g_test])
    scores = 0.8 * y[test] + offsets + rng.normal(scale=0.35, size=test.size)

    naive = bootstrap_ci(y[test], scores, roc_auc_score, n_boot=800, seed=0)
    correct = bootstrap_ci(
        y[test], scores, roc_auc_score, groups=g_test, n_boot=800, seed=0
    )
    assert naive.width < correct.width, (
        f"row bootstrap width {naive.width:.4f} should be narrower (i.e. "
        f"over-confident) than cluster bootstrap width {correct.width:.4f}"
    )


def test_percentile_and_bca_agree_roughly_on_symmetric_data():
    y, s = _simple_scores(n=1500, seed=2)
    p = bootstrap_ci(y, s, roc_auc_score, n_boot=800, method="percentile", seed=0)
    b = bootstrap_ci(y, s, roc_auc_score, n_boot=800, method="bca", seed=0)
    assert abs(p.low - b.low) < 0.05 and abs(p.high - b.high) < 0.05


def test_bca_fallback_is_reported_not_hidden():
    """A degenerate bootstrap must not silently masquerade as a BCa interval."""
    y = np.array([0] * 40 + [1] * 40)
    s = y.astype(float)                       # perfect separation -> AUC always 1.0
    ci = bootstrap_ci(y, s, roc_auc_score, n_boot=200, method="bca", seed=0)
    assert "fallback" in ci.method


def test_unknown_method_rejected():
    y, s = _simple_scores()
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap_ci(y, s, roc_auc_score, method="jackknife")


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        bootstrap_ci([0, 1, 0], [0.1, 0.2], roc_auc_score)


def test_single_class_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="undefined on the full sample"):
        bootstrap_ci(np.ones(50, dtype=int), np.random.rand(50), roc_auc_score)


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------

def test_identical_models_are_called_a_tie():
    y, s = _simple_scores(n=600, seed=5)
    res = paired_bootstrap(y, s, s.copy(), roc_auc_score, n_boot=400, seed=0)
    assert res.verdict == "tie"
    assert res.difference == pytest.approx(0.0, abs=1e-12)
    assert "statistically tied" in res.sentence()


def test_clearly_better_model_wins():
    rng = np.random.default_rng(9)
    n = 1200
    y = rng.integers(0, 2, size=n)
    good = rng.normal(loc=1.6 * y, scale=1.0)
    poor = rng.normal(loc=0.2 * y, scale=1.0)
    res = paired_bootstrap(y, good, poor, roc_auc_score, n_boot=600, seed=0,
                           label_a="good", label_b="poor")
    assert res.verdict == "a_wins"
    assert res.significant
    assert res.low > 0
    assert "good beats poor" in res.sentence()


def test_direction_is_reported_correctly_when_b_is_better():
    rng = np.random.default_rng(10)
    n = 1200
    y = rng.integers(0, 2, size=n)
    poor = rng.normal(loc=0.2 * y, scale=1.0)
    good = rng.normal(loc=1.6 * y, scale=1.0)
    res = paired_bootstrap(y, poor, good, roc_auc_score, n_boot=600, seed=0,
                           label_a="poor", label_b="good")
    assert res.verdict == "b_wins"
    assert res.high < 0
    assert "good beats poor" in res.sentence()


def test_tiny_difference_is_not_declared_a_win():
    """Guards against the 'slightly better' write-up the docs argue against."""
    rng = np.random.default_rng(11)
    n = 500
    y = rng.integers(0, 2, size=n)
    a = rng.normal(loc=1.0 * y, scale=1.0)
    b = a + rng.normal(scale=0.02, size=n)     # negligible perturbation
    res = paired_bootstrap(y, a, b, roc_auc_score, n_boot=600, seed=0)
    assert res.verdict == "tie"


def test_paired_requires_aligned_predictions():
    y, s = _simple_scores(n=100)
    with pytest.raises(ValueError, match="same records"):
        paired_bootstrap(y, s, s[:50], roc_auc_score)


def test_ci_str_and_overlap_helpers():
    a = CI(estimate=0.9, low=0.85, high=0.95)
    b = CI(estimate=0.87, low=0.80, high=0.92)
    c = CI(estimate=0.50, low=0.45, high=0.55)
    assert str(a) == "0.9000 [0.8500, 0.9500]"
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)
