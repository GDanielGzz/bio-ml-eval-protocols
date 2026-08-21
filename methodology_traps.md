# Five common evaluation mistakes in biological ML

> Part of [`bio-ml-eval-protocols`](README.md). This is the catalog document
> referenced in the README's [What's inside](README.md#whats-inside) table.
> Companion recipe: [`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) (how to
> put honest confidence intervals on the numbers you keep after avoiding these
> traps).

Most "my model beats the baseline" results in biological ML are inflated by the
same small set of evaluation mistakes. The clearest single example: a binary
antimicrobial-peptide (AMP) classifier looks excellent under a random
train/test split, but the headline AUC drops by roughly **10–15 points** once
the split is made *cluster-aware* at 40% sequence identity. The random-split
number was never real — it measured memorised redundancy in the public corpus,
not generalisation.

That failure mode is not specific to peptides. The same contamination — through
clustered, redundant, or batch-confounded data — recurs across sequence
classifiers, structure predictors, guide-RNA efficiency models, and even
non-ML flux-balance metabolic models. The literature has documented it
thoroughly: the over-optimism reviews of
[Ullmann et al. 2021](https://arxiv.org/pdf/2106.02447) and
[Leech et al. 2024](https://arxiv.org/pdf/2407.12220), the
[DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) split-construction
tooling ([addendum](https://www.nature.com/articles/s41467-025-67495-w)), the
[LIT-PCBA leakage audit](https://arxiv.org/html/2507.21404v2), and the
*Briefings in Bioinformatics* 2025
[out-of-distribution learning review](https://academic.oup.com/bib/article/26/3/bbaf294/8176475)
all converge on the same point. What is still missing is the short, runnable
checklist a working engineer can apply *before* trusting a number. That's this
document.

Each trap below is paired with a **wrong-vs-right** snippet using only `numpy`
and `scikit-learn`. The wrong version is the default that quietly inflates your
score; the right version is the minimal fix.

## The five traps at a glance

| # | Trap | One-line symptom | The fix |
|---|---|---|---|
| 1 | Homology / sequence leakage across the split | Near-identical records sit on both sides of the train/test boundary | Deduplicate, then split on identity-defined groups, never on rows |
| 2 | Random splits on clustered / redundant data | `KFold(shuffle=True)` re-introduces leakage even after you know better | `GroupKFold`; report across several identity thresholds |
| 3 | Batch confounders | The model separates classes on a covariate (length, source, run) instead of biology | Probe the confounder; match distributions or split by batch |
| 4 | Redundancy in public benchmarks | "External" corpus B overlaps training corpus A | Cluster the *union*; drop test records whose cluster appears in train |
| 5 | Co-development of method + benchmark | Selecting hyperparameters on the test set; under-tuned baseline | Touch the test set once; tune the baseline as hard as the method |

A convenience throughout: `cluster_ids[i]` is the cluster label of record `i`
from an identity-based clustering (e.g. MMseqs2 at a chosen `min_seq_id`). The
repo's `examples/` ships a thin `run_mmseqs2(...)` wrapper around the MMseqs2
CLI that returns exactly this array; everything below treats it as given.

---

## Trap 1 — Homology / sequence leakage across the split

**What goes wrong.** Biological records are rarely independent. A sequence in
the test set can be 99% identical to one in the training set — a paralog, an
ortholog, a re-deposit, or a literal duplicate. A row-wise random split treats
these as distinct samples, so the model is effectively tested on data it has
already seen. The reported metric measures memorised redundancy, not
generalisation. This is the mechanism behind the 10–15-point AMP AUC inflation.

```python
# WRONG: row-wise split — homologs leak across the boundary
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
# A near-duplicate of any X_train row may now sit in X_test.
# Nothing in this code can detect that.
```

```python
# RIGHT: split on identity-defined groups, so no cluster spans the boundary
from sklearn.model_selection import GroupShuffleSplit

# cluster_ids[i] = MMseqs2 cluster of record i at, e.g., 40% identity
cluster_ids = run_mmseqs2(sequences, min_seq_id=0.40)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=cluster_ids))

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]
# Whole clusters go to one side or the other — homologs cannot leak.
```

**Cheap detection first.** Before trusting any split, deduplicate and look for
exact or near-exact overlap between the two sides:

```python
import numpy as np

# exact-duplicate check (hash the canonical record representation)
overlap = set(map(hash, X_train.tolist())) & set(map(hash, X_test.tolist()))
assert not overlap, f"{len(overlap)} exact duplicates straddle the split"
```

If even exact duplicates straddle the split, the metric is meaningless before
you get to the subtler homology question. DataSAIL automates the grouped-split
construction for sequences, structures, and molecules; this repo is
complementary — DataSAIL makes the split, the recipes here decide what the
resulting number means.

---

## Trap 2 — Random splits on clustered / redundant data

**What goes wrong.** Knowing about Trap 1 is not enough: the *default*
cross-validation objects re-introduce it. `KFold(shuffle=True)` and
`StratifiedKFold(shuffle=True)` resample rows, so members of the same family
land in different folds. Every fold then leaks, and the cross-validated mean is
optimistic by the same 10–15 points — now wrapped in a tight, falsely
reassuring standard deviation.

```python
# WRONG: stratified k-fold on clustered data — every fold leaks
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())   # high mean, small std, both unreliable
```

```python
# RIGHT: group-aware k-fold — folds respect cluster boundaries
from sklearn.model_selection import GroupKFold, cross_val_score

cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    model, X, y, groups=cluster_ids, cv=cv, scoring="roc_auc"
)
print(scores.mean(), scores.std())   # lower mean — this is the honest number
```

**Report across identity thresholds, not at one.** A single grouped split at
one identity cutoff still hides how fast performance decays as the train/test
gap widens. Re-cluster at several thresholds and report the curve — a model
that holds up at 30% identity is generalising; one that only works at 90% is
memorising:

```python
for thr in (0.90, 0.70, 0.50, 0.40, 0.30):
    cluster_ids = run_mmseqs2(sequences, min_seq_id=thr)
    cv = GroupKFold(n_splits=5)
    s = cross_val_score(model, X, y, groups=cluster_ids, cv=cv, scoring="roc_auc")
    print(f"identity<= {thr:.2f}:  AUC = {s.mean():.3f}")
```

The decay curve is far more informative than any single headline AUC, and it is
exactly the kind of figure the *Briefings in Bioinformatics* OOD review argues
for. Put confidence intervals on each point with
[`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) rather than relying on the
across-fold standard deviation, which is itself correlated across folds.

---

## Trap 3 — Batch confounders

**What goes wrong.** When positives and negatives come from different sources,
runs, or sampling procedures, the model can separate the classes on the
*batch* signal rather than on biology. A classic shape: AMP positives are
curated short peptides from one database, while negatives are sampled from a
general protein database — so sequence length, amino-acid composition, or a
source tag alone predicts the label. The split is leakage-free, every duplicate
is gone, and the AUC is still a lie because the discriminative feature is an
artifact. This is the "shortcut learning" the over-optimism literature warns
about.

```python
# WRONG: negatives sampled with a different length distribution than positives
X = np.vstack([pos_features, neg_features])
y = np.r_[np.ones(len(pos_features)), np.zeros(len(neg_features))]
model.fit(X_train, y_train)
# 0.97 AUC — but length alone may carry most of it.
```

```python
# RIGHT: probe each suspected confounder BEFORE trusting the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def confounder_auc(covariate, y):
    """How much of the label can a single covariate explain on its own?"""
    c = np.asarray(covariate).reshape(-1, 1)
    p = LogisticRegression(max_iter=1000).fit(c, y).predict_proba(c)[:, 1]
    return roc_auc_score(y, p)

for name, cov in {"length": seq_len, "source": source_id, "run": batch_id}.items():
    auc = confounder_auc(cov, y)
    flag = "  <-- leaks the label" if auc > 0.65 else ""
    print(f"{name:8s} alone: AUC = {auc:.3f}{flag}")
```

If a trivial covariate scores well above chance on its own, fix the sampling
before reporting anything: match the negative distribution to the positives
(e.g. length-matched negative sampling), regress the confounder out, or — when
the batch *is* the unit of analysis — split by batch with `GroupKFold(groups=batch_id)`
so no run appears on both sides. The honest test is an external set drawn the
*same way for both classes*.

---

## Trap 4 — Redundancy in public benchmarks

**What goes wrong.** "We trained on corpus A and validated on independent
corpus B" is only as strong as the independence claim. Public biological
databases share deposits and are frequently re-derived from the same primary
sources, so B routinely overlaps A. The external validation is then
contaminated in exactly the way Trap 1 describes, just across corpus
boundaries. The LIT-PCBA audit documents how severe this cross-dataset overlap
can be in practice.

```python
# WRONG: assume two public corpora are independent
train = load_public_dataset("corpusA")
test  = load_public_dataset("corpusB")        # assumed independent
model.fit(train.X, train.y)
print(model.score(test.X, test.y))            # inflated by shared records
```

```python
# RIGHT: cluster the UNION, then drop test records that share a train cluster
seqs = np.concatenate([train.seqs, test.seqs])
union_clusters = run_mmseqs2(seqs, min_seq_id=0.40)

n_train = len(train.seqs)
train_clusters = set(union_clusters[:n_train].tolist())
test_clusters  = union_clusters[n_train:]

keep = np.array([c not in train_clusters for c in test_clusters])
test_X, test_y = test.X[keep], test.y[keep]
print(f"dropped {(~keep).sum()} / {len(keep)} test records as cross-corpus leakage")
model.fit(train.X, train.y)
print(roc_auc_score(test_y, model.predict_proba(test_X)[:, 1]))  # the real number
```

Always report how many external records survived the dedup. A held-out set that
loses 60% of its rows to cross-corpus overlap was never the independent test it
was advertised as, and the reviewer deserves to know the surviving `n`.

---

## Trap 5 — Co-development of method + benchmark (over-optimism)

**What goes wrong.** When the same workflow designs the method and evaluates it
on the same benchmark, the test set leaks through the *researcher*. Selecting
hyperparameters by test-set score, iterating the architecture against the test
metric, or running the proposed method through ten tuning rounds while the
baseline gets one — all of these report a number fitted to the test set.
Ullmann et al. (2021) frame this as over-optimism in benchmark studies; Leech
et al. (2024) catalog it among questionable ML practices. The under-tuned
baseline is the most common and least visible version.

```python
# WRONG: pick hyperparameters by the test score
best = None
for params in grid:
    m = fit(X_train, y_train, params)
    s = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])  # selecting on test
    if best is None or s > best[1]:
        best = (params, s)
print("reported AUC:", best[1])   # this is a fitted-to-test number
```

```python
# RIGHT: tune on an inner split; touch the test set exactly once
from sklearn.model_selection import GroupKFold, GridSearchCV

inner = GroupKFold(n_splits=4)
search = GridSearchCV(model, grid, cv=inner, scoring="roc_auc", refit=True)
search.fit(X_dev, y_dev, groups=cluster_ids_dev)   # test set never seen here

final_auc = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])  # once
print("honest AUC:", final_auc)
```

Two disciplines make this robust. First, **tune the baseline as hard as the
proposed method** — give a gradient-boosted-tree baseline the same grid budget
the deep model gets; a fair fight is the whole point. Second, **decide the
evaluation before you see the result**: lock the held-out set, pre-commit the
metric and the split, and report a confidence interval rather than a point
estimate. When the honest result is "the new method did not beat the baseline,"
that is a publishable finding, not a failure — see
[`negative_results_reporting.md`](negative_results_reporting.md) for how to ship
it, and [`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) for the interval that
tells you whether a gap is real.

---

## Worked examples used across this repo

The same three public-framed examples thread through every document here, to
show the discipline is model-agnostic:

- **Cluster-aware splits (AMP classifier).** MMseqs2 clustering at 40% identity
  is the worked example for Traps 1 and 2: the honest, cluster-aware AUC is the
  headline; the inflated random-split AUC is the cautionary contrast (~10–15
  points apart).
- **A negative transfer-learning result (CRISPR guide-efficiency).** A
  transformer that did not beat an XGBoost baseline on bacterial guide-RNA data
  — the worked example for Trap 5's "tune the baseline as hard as the method"
  discipline and for [`negative_results_reporting.md`](negative_results_reporting.md).
- **A non-ML methods trap (flux-balance metabolic modelling).** A genome-scale
  metabolic-model debugging trail, included to show the same evaluation
  discipline applies whether the model is a neural network or a flux-balance
  model — the "verify the result isn't an artifact of how you set up the test"
  habit transfers directly.

## Minimum reproducible demo

Per the repository contract, this document's claims are backed by a runnable
demo. The accompanying `tests/` entry regenerates the random-vs-grouped AUC gap
on a small public AMP-style fixture, so the 10–15-point inflation reproduces on
a fresh clone:

```bash
pytest tests/test_methodology_traps.py && make figures
```

<!-- TODO: Sayaka decide — fixture source for the demo. Smallest honest option
     is a bundled toy FASTA + labels under examples/ so the test runs offline in
     CI without a network pull. Wire this when bootstrap_ci_recipe.md (6.3) lands
     its shared fixtures, to avoid duplicating the loader. -->

---

*Citations on this page are limited to the open-access references catalogued in
the project's research note: DataSAIL ([Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-58606-8)
and [addendum](https://www.nature.com/articles/s41467-025-67495-w)),
[Ullmann et al. 2021](https://arxiv.org/pdf/2106.02447),
[Leech et al. 2024](https://arxiv.org/pdf/2407.12220), the
[LIT-PCBA leakage audit](https://arxiv.org/html/2507.21404v2), and the
*Briefings in Bioinformatics*
[OOD-learning review](https://academic.oup.com/bib/article/26/3/bbaf294/8176475).*
