# A bootstrap confidence-interval recipe for held-out metrics

> Part of [`bio-ml-eval-protocols`](README.md). This is the recipe document
> referenced in the README's [What's inside](README.md#whats-inside) table and
> forward-linked twice from [`methodology_traps.md`](methodology_traps.md): once
> from Trap 2 (put intervals on each point of the identity-threshold decay curve
> instead of trusting the across-fold standard deviation) and once from Trap 5
> ("the interval that tells you whether a gap is real"). The honest negative
> results that fall out of this — a method that ties or loses to its baseline —
> are written up using [`negative_results_reporting.md`](negative_results_reporting.md),
> the downstream consumer of everything below.

A single held-out AUC of `0.84` tells you almost nothing on its own. Is the true
value plausibly `0.80`? `0.88`? If a competing model scores `0.86`, is it
actually better, or is the gap inside the noise of a few hundred test records?
Once [`methodology_traps.md`](methodology_traps.md) has bought you a leakage-free
number, the next job is to put an honest interval around it — and the interval has
to inherit the same grouped-split discipline, or it quietly re-introduces the
leakage you just removed.

This document is a small, **vendored** recipe: `numpy` and `scipy.stats` only, no
extra dependency to install or pin. It does three things:

1. a **percentile** bootstrap interval for a single held-out metric (AUC, F1, …);
2. a **BCa** (bias-corrected and accelerated) interval, which is the one to
   reach for when the metric is bounded or the test set is small;
3. a **paired** bootstrap for honest head-to-head comparison — resample the
   *same* test indices for both models and put a CI on the per-resample score
   *difference*, so "is the gap real?" gets a number instead of a vibe.

A short caveat section covers the part most bootstrap tutorials get wrong for
biology: **resample whole clusters, not rows**, so the interval respects the same
grouping that made the point estimate honest in the first place.

Throughout, `metric(y_true, y_score)` is any scoring function with the
scikit-learn signature (e.g. `sklearn.metrics.roc_auc_score`,
`average_precision_score`, or an F1 wrapper), and `cluster_ids[i]` is the
identity-based cluster label of test record `i` — the same array
[`methodology_traps.md`](methodology_traps.md) uses, produced by the repo's thin
`run_mmseqs2(...)` wrapper. The bootstrap never needs the model, only its frozen
predictions on the held-out set, so none of this re-trains anything.

---

## Why bootstrap, and why not the across-fold standard deviation

The tempting shortcut is to report `cross_val_score(...).std()` as the error bar.
It is the wrong quantity for two reasons. First, the fold scores are computed on
disjoint, *small* subsets and are correlated through the shared training data, so
their spread is not a calibrated estimate of the held-out metric's sampling
uncertainty. Second — and this is the biology-specific failure — on clustered
data a shuffled k-fold leaks (Trap 2), so its mean *and* its standard deviation
are both measuring memorised redundancy. A tight `std` on a leaky CV is the most
dangerous error bar there is: it is precise about the wrong number.

The bootstrap instead resamples the **held-out predictions** to estimate how much
the metric would wobble if you had drawn a different test set from the same
population. That is exactly the uncertainty a reviewer cares about, and — done
with cluster resampling — it carries the grouped-split honesty forward into the
interval.

```python
# WRONG: across-fold std as the error bar on a clustered dataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"AUC = {s.mean():.3f} +/- {s.std():.3f}")   # precise about a leaky number
```

```python
# RIGHT: bootstrap the held-out predictions of an already-grouped split
theta_hat, boot = bootstrap_metric(y_test, y_score, roc_auc_score, rng=0)
lo, hi = percentile_ci(boot)
print(f"AUC = {theta_hat:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
```

---

## The vendored helper

Drop this into `examples/bootstrap_ci.py`. It is intentionally ~100 lines of
`numpy` + `scipy.stats`, small enough to read in one sitting and copy into any
project without adding a dependency.

```python
"""Vendor-free bootstrap confidence intervals for held-out metrics.
Dependencies: numpy, scipy.stats. No model re-training — operates on frozen
held-out predictions only."""
import numpy as np
from scipy.stats import norm


def _resample_idx(n, rng):
    """Row resample with replacement (use cluster_bootstrap_idx for grouped data)."""
    return rng.integers(0, n, n)


def cluster_bootstrap_idx(cluster_ids, rng):
    """Resample whole clusters with replacement, return the row indices they cover.
    This is the resample to use whenever the point estimate came from a grouped
    split — it keeps homologs from leaking back in through the bootstrap."""
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    members = {c: np.flatnonzero(cluster_ids == c) for c in uniq}
    sampled = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([members[c] for c in sampled])


def bootstrap_metric(y_true, y_score, metric, n_boot=2000, rng=None,
                     cluster_ids=None):
    """Return (theta_hat, boot_stats). If cluster_ids is given, resample clusters
    rather than rows. Resamples that collapse to a single class are skipped so
    AUC stays defined."""
    rng = np.random.default_rng(rng)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    theta_hat = metric(y_true, y_score)
    boot = []
    while len(boot) < n_boot:
        idx = (cluster_bootstrap_idx(cluster_ids, rng) if cluster_ids is not None
               else _resample_idx(n, rng))
        yt = y_true[idx]
        if yt.min() == yt.max():        # degenerate single-class resample
            continue
        boot.append(metric(yt, y_score[idx]))
    return theta_hat, np.asarray(boot)


def percentile_ci(boot_stats, alpha=0.05):
    """Simple percentile interval. Fine when boot_stats is roughly symmetric."""
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return tuple(np.percentile(boot_stats, [lo, hi]))


def _jackknife(y_true, y_score, metric, cluster_ids=None):
    """Leave-one-out (or leave-one-cluster-out) statistics, for BCa acceleration."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if cluster_ids is None:
        units = np.arange(len(y_true))
        mask = lambda u: np.arange(len(y_true)) != u
    else:
        cluster_ids = np.asarray(cluster_ids)
        units = np.unique(cluster_ids)
        mask = lambda u: cluster_ids != u
    jk = []
    for u in units:
        m = mask(u)
        yt = y_true[m]
        if yt.min() == yt.max():
            continue
        jk.append(metric(yt, y_score[m]))
    return np.asarray(jk)


def bca_ci(theta_hat, boot_stats, jackknife_stats, alpha=0.05):
    """Bias-corrected and accelerated interval (Efron). Corrects for the skew and
    median-bias that a bounded metric like AUC shows near the ceiling."""
    boot_stats = np.asarray(boot_stats)
    prop = np.mean(boot_stats < theta_hat)
    prop = min(max(prop, 1e-6), 1 - 1e-6)        # keep z0 finite
    z0 = norm.ppf(prop)
    jk = np.asarray(jackknife_stats)
    d = jk.mean() - jk
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / denom if denom != 0 else 0.0
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    adjust = lambda z: norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    a1, a2 = adjust(z_lo), adjust(z_hi)
    return tuple(np.percentile(boot_stats, [100 * a1, 100 * a2]))


def paired_bootstrap(y_true, score_a, score_b, metric, n_boot=2000, rng=None,
                     cluster_ids=None):
    """Head-to-head test. Resample the SAME indices for both models each draw and
    take the metric difference; return (obs_diff, (lo, hi), p_two_sided).
    A CI that excludes 0 means the gap is real at the chosen alpha."""
    rng = np.random.default_rng(rng)
    y_true = np.asarray(y_true)
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    n = len(y_true)
    obs = metric(y_true, score_a) - metric(y_true, score_b)
    diffs = []
    while len(diffs) < n_boot:
        idx = (cluster_bootstrap_idx(cluster_ids, rng) if cluster_ids is not None
               else _resample_idx(n, rng))
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        diffs.append(metric(yt, score_a[idx]) - metric(yt, score_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return obs, (lo, hi), min(p, 1.0)
```

---

## 1. Percentile interval for a single metric

The percentile interval is the simplest honest error bar: bootstrap the held-out
predictions `n_boot` times, recompute the metric on each resample, and read off
the 2.5th and 97.5th percentiles of the resulting distribution.

```python
from sklearn.metrics import roc_auc_score

theta_hat, boot = bootstrap_metric(y_test, y_score, roc_auc_score,
                                   n_boot=2000, rng=0)
lo, hi = percentile_ci(boot, alpha=0.05)
print(f"AUC = {theta_hat:.3f}  (95% percentile CI {lo:.3f}-{hi:.3f})")
```

For F1 the only change is the metric, since F1 needs a hard label and a
threshold. Wrap it so the helper sees the scikit-learn `(y_true, y_score)`
signature:

```python
from sklearn.metrics import f1_score

def f1_at(threshold):
    return lambda yt, ys: f1_score(yt, (np.asarray(ys) >= threshold).astype(int))

theta_hat, boot = bootstrap_metric(y_test, y_score, f1_at(0.5), rng=0)
print("F1@0.5  CI:", percentile_ci(boot))
```

The percentile interval is adequate when the bootstrap distribution is roughly
symmetric. For AUC near the ceiling, for small or class-imbalanced test sets, or
for any metric bounded at 0/1, it is biased — which is what BCa fixes.

---

## 2. BCa — the interval to default to for bounded metrics

A percentile interval implicitly assumes the bootstrap distribution is an
unbiased, symmetric image of the sampling distribution. Bounded metrics violate
both assumptions: an AUC of `0.95` cannot have a symmetric error bar because the
metric is capped at `1.0`, so the upper tail is compressed and a naive interval
overstates the upside. **BCa** corrects for this with two terms — a bias
correction `z0` (how far the bootstrap median sits from the point estimate) and
an acceleration `a` (the skew, estimated by jackknife). It is the same
correction `scipy.stats.bootstrap(method="BCa")` applies; the vendored version is
spelled out so it can resample by cluster, which the library helper does not.

```python
theta_hat, boot = bootstrap_metric(y_test, y_score, roc_auc_score,
                                   n_boot=2000, rng=0)
jk = _jackknife(y_test, y_score, roc_auc_score)
lo, hi = bca_ci(theta_hat, boot, jk, alpha=0.05)
print(f"AUC = {theta_hat:.3f}  (95% BCa CI {lo:.3f}-{hi:.3f})")
```

```python
# WRONG: percentile interval on a near-ceiling AUC overstates the upper bound
print(percentile_ci(boot))          # e.g. (0.91, 0.99) — upper tail too generous

# RIGHT: BCa pulls the skewed tail back to an honest place
print(bca_ci(theta_hat, boot, jk))  # e.g. (0.90, 0.97)
```

Rule of thumb: **report BCa by default for AUC, average precision, and F1**, and
fall back to the percentile interval only when you have checked the bootstrap
distribution is symmetric (e.g. for a large, balanced test set where the two
agree to the third decimal anyway). When they disagree, the BCa number is the one
that survives review.

---

## 3. Caveat: resample clusters, not rows

This is the part that makes a bootstrap *biological*. If the point estimate came
from a grouped split (Traps 1–2), a **row**-wise bootstrap silently undoes the
grouping: a resample can place near-duplicate homologs on the same draw and treat
them as independent evidence, so the interval comes out too narrow — the same
leakage, now hiding in the error bar instead of the point estimate. The fix is to
resample whole **clusters** with replacement and take every row they cover, so
the bootstrap inherits the identity discipline.

```python
# WRONG: row bootstrap on grouped data — CI too narrow, leakage via the error bar
theta_hat, boot = bootstrap_metric(y_test, y_score, roc_auc_score, rng=0)
print("row CI:", percentile_ci(boot))            # falsely tight

# RIGHT: cluster bootstrap — the interval respects the same groups as the split
theta_hat, boot = bootstrap_metric(y_test, y_score, roc_auc_score,
                                   cluster_ids=cluster_ids, rng=0)
jk = _jackknife(y_test, y_score, roc_auc_score, cluster_ids=cluster_ids)
print("cluster BCa CI:", bca_ci(theta_hat, boot, jk))   # honest width
```

The cluster bootstrap is usually *wider* than the row bootstrap — that width is
correct. A grouped split with a row-wise error bar is internally inconsistent:
honest about the mean, dishonest about the uncertainty.

---

## 4. The paired bootstrap — "is the gap real?"

This is the interval [`methodology_traps.md`](methodology_traps.md) Trap 5 points
here for. When comparing two models — say a transformer against a
gradient-boosted-tree baseline — comparing two independent CIs and checking
whether they overlap is the wrong test: overlapping CIs can still hide a
significant difference, because the models' errors are *correlated* (they are
scored on the same test records). The correct procedure resamples the **same**
test indices for both models on every draw and builds a CI on the per-resample
**difference**. The pairing cancels the shared test-set variance, so it is both
more correct and more powerful than two separate intervals.

```python
# y_test shared; score_t = transformer probs, score_x = xgboost probs
obs, (lo, hi), p = paired_bootstrap(
    y_test, score_t, score_x, roc_auc_score,
    n_boot=2000, rng=0, cluster_ids=cluster_ids,    # cluster-resample here too
)
print(f"AUC(transformer) - AUC(xgboost) = {obs:+.3f}  "
      f"(95% CI {lo:+.3f}, {hi:+.3f}; p={p:.3f})")
```

The decision rule is a single line:

```python
verdict = "real difference" if (lo > 0 or hi < 0) else "inside the noise (tie)"
print(verdict)
```

If the CI on the difference **excludes 0**, the gap is real at your alpha. If it
**straddles 0**, the honest conclusion is "tied" — and a tie against a simpler,
cheaper baseline is a result worth reporting, not burying. That write-up is what
[`negative_results_reporting.md`](negative_results_reporting.md) is for. Note also
the asymmetry: comparing each model to its own separate CI is the
overlapping-error-bars mistake; only the paired difference answers the question
actually asked.

---

## 5. Putting it together: confidence intervals on the decay curve

Trap 2 argues for reporting AUC across several identity thresholds rather than at
one cutoff. With the recipe above, each point on that decay curve gets a proper
cluster-resampled BCa interval instead of a correlated across-fold `std`:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

rows = []
for thr in (0.90, 0.70, 0.50, 0.40, 0.30):
    cluster_ids = run_mmseqs2(sequences_test, min_seq_id=thr)
    theta, boot = bootstrap_metric(y_test, y_score, roc_auc_score,
                                   cluster_ids=cluster_ids, rng=0)
    jk = _jackknife(y_test, y_score, roc_auc_score, cluster_ids=cluster_ids)
    lo, hi = bca_ci(theta, boot, jk)
    rows.append((thr, theta, lo, hi))
    print(f"identity<= {thr:.2f}:  AUC = {theta:.3f}  (95% BCa {lo:.3f}-{hi:.3f})")
```

Plotted, this is the headline figure of an honest evaluation: the point estimate
falling as the train/test identity gap widens, with an interval on each point
that already respects the clustering. A model that stays well above chance with
non-overlapping intervals down to 30% identity is generalising; one whose
interval reaches chance by 40% was memorising — and you can now say so with a
number.

---

## Worked examples used across this repo

The same three public-framed examples thread through every document here:

- **Cluster-aware splits (AMP classifier).** The identity-threshold decay curve
  of §5, with a cluster-resampled BCa interval on each point, is the honest
  companion to the ~10–15-point random-vs-grouped AUC gap from Traps 1–2.
- **A negative transfer-learning result (CRISPR guide-efficiency).** The §4
  paired bootstrap is exactly how the transformer-vs-XGBoost comparison on
  bacterial guide-RNA data is adjudicated: the CI on the AUC difference straddles
  zero, so the honest verdict is "tied / did not beat the baseline" — the worked
  example carried into [`negative_results_reporting.md`](negative_results_reporting.md).
- **A non-ML methods trap (flux-balance metabolic modelling).** Even where the
  "model" is a genome-scale metabolic reconstruction rather than a classifier,
  the same habit applies — resample the evaluation units, report an interval, and
  check that a reported improvement is not inside the noise of the test set.

---

## Minimum reproducible demo

Per the repository contract, every claim here is backed by a runnable demo. The
accompanying `tests/` entry regenerates the percentile-vs-BCa comparison and the
paired-bootstrap verdict on a small public AMP-style fixture, so the numbers in
§2 and §4 reproduce on a fresh clone:

```bash
pytest tests/test_bootstrap_ci.py && make figures
```

<!-- TODO: Sayaka decide — share one offline fixture (toy FASTA + labels +
     frozen prediction vectors) under examples/ between this demo and
     test_methodology_traps.py, so CI runs without a network pull and the two
     docs do not duplicate a loader. This is the shared-fixture coordination
     point methodology_traps.md (6.2) left open. -->

---

*Citations on this page are limited to the open-access references catalogued in
the project's research note: the over-optimism reviews of
[Ullmann et al. 2021](https://arxiv.org/pdf/2106.02447) and
[Leech et al. 2024](https://arxiv.org/pdf/2407.12220), the
[DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) split-construction
tooling ([addendum](https://www.nature.com/articles/s41467-025-67495-w)), the
[LIT-PCBA leakage audit](https://arxiv.org/html/2507.21404v2), and the
*Briefings in Bioinformatics*
[out-of-distribution learning review](https://academic.oup.com/bib/article/26/3/bbaf294/8176475).
The BCa interval follows Efron's bias-corrected-and-accelerated method as
implemented in `scipy.stats.bootstrap`.*
