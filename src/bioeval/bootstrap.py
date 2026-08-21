"""Bootstrap confidence intervals for held-out metrics.

Three rules, each implemented here because each is routinely got wrong:

1. **Resample the unit the split respected.** If the split was cluster-aware,
   the bootstrap must resample *clusters*, not rows. A row bootstrap on grouped
   data produces an interval that is too narrow — the same leakage that inflates
   the point estimate also shrinks the error bar. Pass ``groups=``.
2. **Prefer BCa for bounded metrics.** AUC lives in [0, 1] and its sampling
   distribution skews near the ceiling. A percentile interval overstates the
   upper bound; BCa corrects for bias and skew.
3. **Compare paired, not marginal.** Two overlapping marginal CIs do *not* mean
   the difference is insignificant. Bootstrap the difference on shared test
   records — see :func:`paired_bootstrap`.

Vendor-free: numpy and scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

__all__ = ["CI", "bootstrap_ci", "paired_bootstrap", "PairedResult"]

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class CI:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    alpha: float = 0.05
    method: str = "bca"
    n_boot: int = 0
    n_valid: int = 0

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"

    def overlaps(self, other: "CI") -> bool:
        """Whether two marginal intervals overlap.

        Provided mainly so the docs can point at it and say: **do not use this
        to decide whether two models differ.** Use :func:`paired_bootstrap`.
        """
        return not (self.high < other.low or other.high < self.low)


def _resample_indices(rng, n: int, groups: np.ndarray | None):
    """One bootstrap resample: rows if unGrouped, whole groups if grouped."""
    if groups is None:
        return rng.integers(0, n, size=n)
    labels = np.unique(groups)
    picked = rng.choice(labels, size=labels.size, replace=True)
    # Concatenate member indices of each drawn group (with repetition).
    members = {lab: np.flatnonzero(groups == lab) for lab in labels}
    return np.concatenate([members[lab] for lab in picked])


def _safe_metric(metric: Metric, y: np.ndarray, s: np.ndarray) -> float:
    """Evaluate a metric, returning NaN on degenerate resamples.

    A bootstrap resample can contain only one class, which makes AUC undefined.
    Silently dropping those draws is standard; counting them is honest, so
    :class:`CI` reports ``n_valid``.
    """
    try:
        if np.unique(y).size < 2:
            return float("nan")
        return float(metric(y, s))
    except Exception:
        return float("nan")


def _jackknife(metric, y, s, groups):
    """Leave-one-unit-out estimates, for the BCa acceleration term."""
    if groups is None:
        n = y.size
        out = np.empty(n)
        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False
            out[i] = _safe_metric(metric, y[m], s[m])
    else:
        labels = np.unique(groups)
        out = np.empty(labels.size)
        for k, lab in enumerate(labels):
            m = groups != lab
            out[k] = _safe_metric(metric, y[m], s[m])
    return out[np.isfinite(out)]


def bootstrap_ci(
    y_true: Sequence,
    y_score: Sequence,
    metric: Metric,
    *,
    groups: Sequence | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    seed: int | None = 0,
) -> CI:
    """Confidence interval for ``metric(y_true, y_score)`` on held-out data.

    Parameters
    ----------
    metric:
        Any ``f(y_true, y_score) -> float``: ``roc_auc_score``,
        ``matthews_corrcoef`` on thresholded scores, ``spearmanr``-derived rho,
        anything.
    groups:
        Per-record group label for the *test set*. **Pass this whenever the
        split was group-aware.** Omitting it is the error described in rule 1.
    method:
        ``"bca"`` (default, bias-corrected and accelerated) or ``"percentile"``.

    Notes
    -----
    BCa falls back to the percentile interval when the bootstrap distribution is
    degenerate (all draws identical, or the jackknife yields no variance). The
    returned :attr:`CI.method` records which was actually used, so a fallback is
    visible rather than silent.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"length mismatch: y_true={y.shape[0]}, y_score={s.shape[0]}")
    g = None if groups is None else np.asarray(groups)
    if g is not None and g.shape[0] != y.shape[0]:
        raise ValueError("groups must be the same length as y_true")

    theta_hat = _safe_metric(metric, y, s)
    if not np.isfinite(theta_hat):
        raise ValueError("metric is undefined on the full sample (single class?)")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _resample_indices(rng, y.shape[0], g)
        boots[b] = _safe_metric(metric, y[idx], s[idx])
    valid = boots[np.isfinite(boots)]
    if valid.size < 20:
        raise ValueError(
            f"only {valid.size}/{n_boot} bootstrap draws were valid; "
            "the test set is probably too small or too imbalanced to bootstrap"
        )

    used = method
    if method == "percentile":
        lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    elif method == "bca":
        prop = float(np.mean(valid < theta_hat))
        jack = _jackknife(metric, y, s, g)
        if prop <= 0.0 or prop >= 1.0 or jack.size < 3 or np.ptp(jack) == 0:
            used = "percentile (bca fallback)"
            lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            z0 = stats.norm.ppf(prop)
            jbar = jack.mean()
            d = jbar - jack
            denom = 6.0 * (np.sum(d**2) ** 1.5)
            a = 0.0 if denom == 0 else float(np.sum(d**3) / denom)
            zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
            a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
            a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
            if not (np.isfinite(a1) and np.isfinite(a2)) or a1 >= a2:
                used = "percentile (bca fallback)"
                a1, a2 = alpha / 2, 1 - alpha / 2
            lo, hi = np.percentile(valid, [100 * a1, 100 * a2])
    else:
        raise ValueError(f"unknown method {method!r}; use 'bca' or 'percentile'")

    return CI(
        estimate=theta_hat,
        low=float(lo),
        high=float(hi),
        alpha=alpha,
        method=used,
        n_boot=n_boot,
        n_valid=int(valid.size),
    )


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired head-to-head comparison on shared test records."""

    estimate_a: float
    estimate_b: float
    difference: float
    low: float
    high: float
    alpha: float = 0.05
    n_boot: int = 0
    label_a: str = "A"
    label_b: str = "B"

    @property
    def significant(self) -> bool:
        """True when the interval for (A - B) excludes zero."""
        return not (self.low <= 0.0 <= self.high)

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "tie"
        return "a_wins" if self.difference > 0 else "b_wins"

    def sentence(self) -> str:
        """A publishable one-liner stating the comparison honestly."""
        d = f"{self.difference:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"
        pct = int(round((1 - self.alpha) * 100))
        if self.verdict == "tie":
            return (
                f"{self.label_a} ({self.estimate_a:.4f}) and {self.label_b} "
                f"({self.estimate_b:.4f}) are statistically tied: the paired "
                f"difference is {d} ({pct}% CI), which contains zero."
            )
        winner, loser = (
            (self.label_a, self.label_b) if self.difference > 0
            else (self.label_b, self.label_a)
        )
        return (
            f"{winner} beats {loser}: paired difference "
            f"({self.label_a} - {self.label_b}) = {d} ({pct}% CI), excluding zero."
        )


def paired_bootstrap(
    y_true: Sequence,
    score_a: Sequence,
    score_b: Sequence,
    metric: Metric,
    *,
    groups: Sequence | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
    label_a: str = "A",
    label_b: str = "B",
) -> PairedResult:
    """Bootstrap the *difference* between two models on the same test records.

    This is the function to reach for when asking "is the gap real?". Comparing
    two marginal confidence intervals for overlap is a different, weaker, and
    frequently wrong test: two models can have overlapping marginal intervals
    while their paired difference is clearly nonzero, because the marginal
    intervals include test-set variance that is *shared* between them and
    cancels in the difference.

    Both score vectors must be predictions on the same records in the same
    order. Resampling is over records — or over groups when ``groups`` is given
    — and both models are re-scored on the identical resample.
    """
    y = np.asarray(y_true)
    sa = np.asarray(score_a, dtype=float)
    sb = np.asarray(score_b, dtype=float)
    if not (y.shape[0] == sa.shape[0] == sb.shape[0]):
        raise ValueError(
            "paired comparison needs predictions on the same records: "
            f"y={y.shape[0]}, a={sa.shape[0]}, b={sb.shape[0]}"
        )
    g = None if groups is None else np.asarray(groups)

    ta, tb = _safe_metric(metric, y, sa), _safe_metric(metric, y, sb)
    if not (np.isfinite(ta) and np.isfinite(tb)):
        raise ValueError("metric undefined on the full sample for at least one model")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _resample_indices(rng, y.shape[0], g)
        yb = y[idx]
        da = _safe_metric(metric, yb, sa[idx])
        db = _safe_metric(metric, yb, sb[idx])
        diffs[b] = da - db
    valid = diffs[np.isfinite(diffs)]
    if valid.size < 20:
        raise ValueError(f"only {valid.size}/{n_boot} paired draws were valid")

    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return PairedResult(
        estimate_a=ta,
        estimate_b=tb,
        difference=ta - tb,
        low=float(lo),
        high=float(hi),
        alpha=alpha,
        n_boot=n_boot,
        label_a=label_a,
        label_b=label_b,
    )
