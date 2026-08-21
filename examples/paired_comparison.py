#!/usr/bin/env python3
"""Deciding whether a gap is real — and writing it up when it is not.

This is the companion to ``negative_results_reporting.md``. It demonstrates the
two comparisons that get confused:

* **Marginal intervals overlap** — the test most people reach for, and the wrong
  one. Two models evaluated on the same test set share most of their sampling
  noise, so their marginal intervals are wide with variance that cancels in the
  difference.
* **The paired difference interval** — the right one. Resample the test set,
  re-score *both* models on the identical resample, and bootstrap the gap.

Two scenarios are run: a genuine tie, and a real but modest improvement whose
marginal intervals nevertheless overlap. The second is the case where reporting
"overlapping CIs, therefore no difference" would throw away a true result.

Run:

    python examples/paired_comparison.py
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bioeval import (
    audit_split,
    bootstrap_ci,
    comparison_note,
    grouped_split,
    make_clustered_binary,
    paired_bootstrap,
    results_table,
)

SEED = 0
N_BOOT = 1500


def scenario(title: str, model_a, model_b, label_a: str, label_b: str,
             seed: int, n_clusters: int = 300, nonlinearity: float = 0.35):
    print("=" * 74)
    print(title)
    print("=" * 74)

    X, y, groups = make_clustered_binary(
        n_clusters=n_clusters, cluster_size=(4, 14),
        nonlinearity=nonlinearity, seed=seed,
    )
    train, test = grouped_split(groups, test_size=0.30, seed=seed)
    audit_split(train, test, groups=groups).raise_if_leaking()

    a = model_a.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    b = model_b.fit(X[train], y[train]).predict_proba(X[test])[:, 1]

    g_test = groups[test]
    ci_a = bootstrap_ci(y[test], a, roc_auc_score, groups=g_test,
                        n_boot=N_BOOT, seed=SEED)
    ci_b = bootstrap_ci(y[test], b, roc_auc_score, groups=g_test,
                        n_boot=N_BOOT, seed=SEED)
    paired = paired_bootstrap(y[test], a, b, roc_auc_score, groups=g_test,
                              n_boot=N_BOOT, seed=SEED,
                              label_a=label_a, label_b=label_b)

    print(results_table({label_a: ci_a, label_b: ci_b}, metric_name="AUC"))
    print()
    print(f"Marginal intervals overlap? {ci_a.overlaps(ci_b)}")
    print(f"Paired difference excludes zero? {paired.significant}")
    print()
    print(comparison_note(paired, ci_a, ci_b))
    print()
    return paired


def main() -> int:
    scenario(
        "SCENARIO 1 - the same model twice, under two random seeds",
        RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=-1),
        RandomForestClassifier(n_estimators=300, random_state=2, n_jobs=-1),
        "forest (seed 1)",
        "forest (seed 2)",
        seed=1,
    )

    scenario(
        "SCENARIO 2 - a real difference that overlapping intervals would hide",
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        LogisticRegression(max_iter=3000),
        "random forest",
        "logistic regression",
        seed=2,
        n_clusters=900,
        nonlinearity=0.8,
    )

    print("=" * 74)
    print("The point: 'the confidence intervals overlap' is not a test.")
    print("Bootstrap the difference on shared records, and say 'tied' when it is.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
