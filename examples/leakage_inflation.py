#!/usr/bin/env python3
"""Reproduce homology leakage from scratch, in about ten seconds.

This is the central claim of the repository, made runnable:

    A random train/test split on clustered biological data reports a score that
    is not real. The same model, same data, same metric, evaluated on a
    group-aware split, scores substantially lower — and the lower number is the
    honest one.

No data download, no credentials, no licence question. The dataset is generated
by :func:`bioeval.datasets.make_clustered_binary`, which produces exactly the
structure that causes the problem in real sequence corpora: records fall into
clusters, cluster members are near-identical, and the label is a property of the
cluster rather than of the individual record.

Run:

    python examples/leakage_inflation.py

Outputs ``figures/leakage_inflation.png`` and prints the results table.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from bioeval import (
    audit_split,
    bootstrap_ci,
    grouped_split,
    make_clustered_binary,
    results_table,
)

SEED = 0
N_BOOT = 1000
FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"


def random_split(n: int, test_size: float, seed: int):
    """The wrong split, kept honest about what it is."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round((1 - test_size) * n))
    return perm[:cut], perm[cut:]


def fit_and_score(X, y, train, test):
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=1, random_state=SEED, n_jobs=-1
    )
    model.fit(X[train], y[train])
    return model.predict_proba(X[test])[:, 1]


def main() -> int:
    print("Generating clustered data "
          "(300 clusters, near-duplicate members, cluster-level labels)...")
    X, y, groups = make_clustered_binary(
        n_clusters=300, cluster_size=(4, 14), within_cluster_noise=0.25,
        nonlinearity=0.35, seed=SEED,
    )
    print(f"  {X.shape[0]} records, {len(np.unique(groups))} clusters, "
          f"{y.mean():.1%} positive\n")

    # ---------------------------------------------------------------- wrong
    tr_r, te_r = random_split(X.shape[0], test_size=0.30, seed=SEED)
    audit_r = audit_split(tr_r, te_r, groups=groups)
    print("RANDOM SPLIT")
    print(audit_r.summary())
    scores_r = fit_and_score(X, y, tr_r, te_r)
    ci_random = bootstrap_ci(
        y[te_r], scores_r, roc_auc_score, groups=groups[te_r],
        n_boot=N_BOOT, method="bca", seed=SEED,
    )
    print(f"  AUC = {ci_random}\n")

    # ---------------------------------------------------------------- right
    tr_g, te_g = grouped_split(groups, test_size=0.30, seed=SEED)
    audit_g = audit_split(tr_g, te_g, groups=groups)
    print("GROUP-AWARE SPLIT")
    print(audit_g.summary())
    audit_g.raise_if_leaking()
    scores_g = fit_and_score(X, y, tr_g, te_g)
    ci_grouped = bootstrap_ci(
        y[te_g], scores_g, roc_auc_score, groups=groups[te_g],
        n_boot=N_BOOT, method="bca", seed=SEED,
    )
    print(f"  AUC = {ci_grouped}\n")

    inflation = ci_random.estimate - ci_grouped.estimate
    print(results_table(
        {"Random split (leaking)": ci_random,
         "Group-aware split (honest)": ci_grouped},
        metric_name="AUC",
    ))
    print(f"\nInflation from leakage: {inflation:+.4f} AUC "
          f"({inflation * 100:+.1f} points).")
    print("The random-split number is not a better model. It is the same model, "
          "\nrewarded for recognising clusters it had already seen.")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                                    # pragma: no cover
        print("\n(matplotlib not installed - skipping figure)")
        return 0

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    labels = ["Random split\n(homologs leak)", "Group-aware split\n(honest)"]
    ests = [ci_random.estimate, ci_grouped.estimate]
    errs = np.array([
        [ci_random.estimate - ci_random.low, ci_grouped.estimate - ci_grouped.low],
        [ci_random.high - ci_random.estimate, ci_grouped.high - ci_grouped.estimate],
    ])
    colors = ["#c0563a", "#3a6ea5"]
    ax.bar(labels, ests, yerr=errs, capsize=8, color=colors, width=0.55,
           edgecolor="black", linewidth=0.7)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.text(-0.46, 0.512, "chance", fontsize=8, color="grey", ha="left")
    for i, ci in enumerate([ci_random, ci_grouped]):
        ax.text(i, ci.high + 0.02, f"{ci.estimate:.3f}\n[{ci.low:.3f}, {ci.high:.3f}]",
                ha="center", fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUC (95% BCa CI, cluster bootstrap)")
    ax.set_title(f"Same model, same data, same metric\n"
                 f"Leakage inflates AUC by {inflation * 100:.1f} points",
                 fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "leakage_inflation.png"
    fig.savefig(out)
    print(f"\nWrote {out.relative_to(FIG_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
