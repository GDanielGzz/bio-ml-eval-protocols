"""bioeval — the runnable half of bio-ml-eval-protocols.

The markdown documents in this repository state the rules. This package
enforces them, so that "we use cluster-aware splits and bootstrap CIs" is a
dependency rather than a claim.

Typical use::

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from bioeval import grouped_split, audit_split, bootstrap_ci

    train, test = grouped_split(clusters, test_size=0.2, seed=0)
    audit_split(train, test, groups=clusters).raise_if_leaking()

    model = RandomForestClassifier().fit(X[train], y[train])
    scores = model.predict_proba(X[test])[:, 1]

    ci = bootstrap_ci(y[test], scores, roc_auc_score,
                      groups=clusters[test], method="bca")
    print(ci)          # 0.7412 [0.6903, 0.7880]

Every public function is documented with *why* it exists, not just what it
does; the reasoning lives in the accompanying markdown documents.
"""

from .bootstrap import CI, PairedResult, bootstrap_ci, paired_bootstrap
from .datasets import make_clustered_binary
from .leakage import LeakageReport, audit_split, exact_duplicate_pairs, record_hashes
from .report import comparison_note, results_table
from .splits import group_sizes, grouped_kfold, grouped_split

__version__ = "0.2.0"

__all__ = [
    "grouped_split",
    "grouped_kfold",
    "group_sizes",
    "audit_split",
    "LeakageReport",
    "record_hashes",
    "exact_duplicate_pairs",
    "bootstrap_ci",
    "paired_bootstrap",
    "CI",
    "PairedResult",
    "results_table",
    "comparison_note",
    "make_clustered_binary",
    "__version__",
]
