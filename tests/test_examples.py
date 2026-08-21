"""Execute the shipped examples and assert their headline claims.

The README promises that every claim in this repository is runnable. That
promise is only worth something if CI would notice when it stops being true —
so these tests run the example scripts end to end and check the *conclusions*,
not merely that the process exited zero.

They are the slowest tests here (~1 minute). Deselect with ``-m "not slow"``.
"""

import subprocess
import sys
import pathlib

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from bioeval import (
    audit_split,
    bootstrap_ci,
    grouped_split,
    make_clustered_binary,
    paired_bootstrap,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

pytestmark = pytest.mark.slow


def _random_split(n, test_size, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round((1 - test_size) * n))
    return perm[:cut], perm[cut:]


def test_leakage_inflation_is_real_and_substantial():
    """The repository's central empirical claim, asserted directly.

    A random split must score materially higher than a group-aware split on the
    same data with the same model — and the group-aware score must still be well
    above chance, or the demonstration is a caricature rather than a lesson.
    """
    X, y, groups = make_clustered_binary(
        n_clusters=300, cluster_size=(4, 14), within_cluster_noise=0.25,
        nonlinearity=0.35, seed=0,
    )

    def fit_score(train, test):
        m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
        m.fit(X[train], y[train])
        return roc_auc_score(y[test], m.predict_proba(X[test])[:, 1])

    tr_r, te_r = _random_split(X.shape[0], 0.30, 0)
    assert not audit_split(tr_r, te_r, groups=groups).ok      # it must be leaking
    auc_random = fit_score(tr_r, te_r)

    tr_g, te_g = grouped_split(groups, test_size=0.30, seed=0)
    audit_split(tr_g, te_g, groups=groups).raise_if_leaking()
    auc_grouped = fit_score(tr_g, te_g)

    assert auc_random > auc_grouped + 0.05, (
        f"expected material inflation, got random={auc_random:.3f} "
        f"grouped={auc_grouped:.3f}"
    )
    assert auc_grouped > 0.70, (
        f"grouped AUC {auc_grouped:.3f} is too close to chance — the synthetic "
        "task has become unlearnable, which would make the demo a caricature"
    )


def test_identical_architectures_come_out_tied_on_real_fits():
    """Scenario 1 of the paired example: same model, two seeds, must tie."""
    X, y, groups = make_clustered_binary(n_clusters=300, nonlinearity=0.35, seed=1)
    train, test = grouped_split(groups, test_size=0.30, seed=1)
    a = (RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=-1)
         .fit(X[train], y[train]).predict_proba(X[test])[:, 1])
    b = (RandomForestClassifier(n_estimators=300, random_state=2, n_jobs=-1)
         .fit(X[train], y[train]).predict_proba(X[test])[:, 1])
    res = paired_bootstrap(y[test], a, b, roc_auc_score, groups=groups[test],
                           n_boot=800, seed=0)
    assert res.verdict == "tie"


@pytest.mark.parametrize("script", ["leakage_inflation.py", "paired_comparison.py"])
def test_example_script_runs_clean(script, tmp_path):
    """Every shipped example must execute without error on a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stderr[-2000:]}"
    assert proc.stdout.strip()


def test_leakage_example_writes_its_figure():
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "leakage_inflation.py")],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert proc.returncode == 0
    assert (ROOT / "figures" / "leakage_inflation.png").exists()
