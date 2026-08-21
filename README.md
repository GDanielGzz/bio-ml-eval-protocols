# bio-ml-eval-protocols

**An opinionated, runnable standard for evaluating biological machine-learning
models without lying to yourself.**

[![CI](https://github.com/GDanielGzz/bio-ml-eval-protocols/actions/workflows/ci.yml/badge.svg)](https://github.com/GDanielGzz/bio-ml-eval-protocols/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Most "my model beats the baseline" results in biological ML are inflated by the
same handful of evaluation mistakes. This repository is two things: three
documents explaining those mistakes, and a small Python package — `bioeval` —
that makes them hard to commit.

The distinction matters. A document saying *you should use cluster-aware splits*
is a claim. A function that refuses to hand you a leaking split is a dependency.

---

## The problem, reproduced from scratch in ten seconds

```bash
pip install -e ".[examples]"
python examples/leakage_inflation.py
```

![Leakage inflates AUC by 11.8 points](figures/leakage_inflation.png)

| Model | AUC [95% CI] | Interval width |
|---|---|---|
| Random split (leaking) | 0.9817 [0.9664, 0.9910] | 0.0246 |
| Group-aware split (honest) | 0.8640 [0.7795, 0.9178] | 0.1384 |

**Same data. Same model. Same metric. An 11.8-point difference**, which is
squarely inside the 10–15 points reported when real antimicrobial-peptide
benchmarks are re-split cluster-aware. The random-split number is not a better
model; it is the same model being rewarded for recognising clusters it had
already seen.

Note the second column too. The leaking split does not merely overstate the
score — it understates the uncertainty by a factor of five, because near-duplicate
homologs are counted as independent evidence twice over: once in the point
estimate, once in the error bar.

No data download, no credentials, no licence question: the dataset is generated
by `bioeval.datasets.make_clustered_binary`, which reproduces the structure that
causes the problem — records in clusters, cluster members near-identical, labels
that are a property of the cluster.


### The same effect, on a real corpus

The demonstration above is synthetic so that it runs anywhere. The effect is not.

[**amp-classifier**](https://github.com/GDanielGzz/amp-classifier) applies this
standard to DRAMP 3.0: MMseqs2 clustering at 40% identity, 1,969 held-out
sequences across non-overlapping clusters, bootstrap CIs on every reported
metric. A frozen ESM2-650M head reaches **AUC 0.919 [0.906-0.931]** against an
XGBoost baseline over 428 engineered features at **0.864 [0.845-0.880]** -
non-overlapping intervals on both AUC and MCC.

That repository is where these protocols came from. `ml/scripts/make_clusters.py`,
`make_splits.py` and `ml/eval_common.py` are the problem-specific ancestors of
`bioeval.splits`, `bioeval.leakage` and `bioeval.bootstrap`: the same
discipline, written once for one corpus and then generalised.

---
## Quickstart

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bioeval import grouped_split, audit_split, bootstrap_ci, paired_bootstrap

# 1. Split so that no identity cluster spans the boundary.
train, test = grouped_split(clusters, test_size=0.2, seed=0)

# 2. Prove it. Fails loudly in CI if anything leaks.
audit_split(train, test, groups=clusters, records=sequences).raise_if_leaking()

# 3. Report an interval, not a point — resampling clusters, not rows.
model = RandomForestClassifier().fit(X[train], y[train])
scores = model.predict_proba(X[test])[:, 1]
ci = bootstrap_ci(y[test], scores, roc_auc_score,
                  groups=clusters[test], method="bca")
print(ci)                      # 0.8640 [0.7795, 0.9178]

# 4. Compare paired, and accept a tie when it is one.
result = paired_bootstrap(y[test], scores, baseline_scores, roc_auc_score,
                          groups=clusters[test],
                          label_a="transformer", label_b="xgboost")
print(result.sentence())
# "transformer (0.8640) and xgboost (0.8592) are statistically tied:
#  the paired difference is +0.0048 [-0.0210, +0.0301] (95% CI),
#  which contains zero."
```

### What's in the package

| Module | Purpose |
|---|---|
| `bioeval.splits` | `grouped_split`, `grouped_kfold`, `group_sizes` — splits where no group crosses the boundary |
| `bioeval.leakage` | `audit_split`, `record_hashes`, `exact_duplicate_pairs` — detect leakage in a split you did *not* construct |
| `bioeval.bootstrap` | `bootstrap_ci` (percentile + BCa, cluster-aware), `paired_bootstrap` — intervals and honest head-to-head tests |
| `bioeval.report` | `results_table`, `comparison_note` — prose that refuses to overstate the interval |
| `bioeval.datasets` | `make_clustered_binary` — synthetic homologous data so every claim here is reproducible offline |

**`bioeval` does not cluster your data.** Clustering is a domain decision and
belongs to a tool that understands your records — MMseqs2, CD-HIT, or
[DataSAIL](https://www.nature.com/articles/s41467-025-58606-8). This package
takes the group labels those produce and guarantees everything downstream
respects them. DataSAIL makes the split; this decides what the numbers mean.

---

## The documents

| Document | What it covers |
|---|---|
| [`methodology_traps.md`](methodology_traps.md) | Five evaluation mistakes — homology leakage, random splits on clustered data, batch confounders, redundancy in public benchmarks, and co-development of method and benchmark. Wrong-vs-right code for each. |
| [`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) | A vendor-free `numpy` + `scipy` bootstrap pipeline, including the paired variant and why the across-fold standard deviation is not an error bar. |
| [`negative_results_reporting.md`](negative_results_reporting.md) | A pre-registration checklist and fill-in template for shipping a result that lost to the baseline, with a worked example where a transformer failed to beat gradient-boosted trees. |

---

## Repository contract

Every claim in the documents is executable, and CI fails if one stops being true:

```bash
make test       # 66 tests, including the example scripts end to end
make fast       # unit tests only, skipping the ~1 min example runs
make figures    # regenerate every figure the documents cite
```

The test suite asserts *properties*, not shapes — that no group ever spans a
split across many seeds, that bootstrap coverage is near nominal, that a row
bootstrap on grouped data really is over-confident, and that two identical
models are called a tie. Tests that only checked return types would pass on an
implementation that leaks.

Two findings from building it, both left in because they are the point:

- The `grouped_split` seed originally did almost nothing — sorting clusters
  largest-first made the result nearly independent of it, which would have made
  multi-seed robustness checks meaningless. Shuffled-order packing costs about a
  percentage point of accuracy against the target fraction and was the right
  trade.
- The demonstration was first tuned to a fully nonlinear task, where the honest
  AUC came out at 0.52 — chance. A 46-point gap looks impressive and proves
  nothing, because an unlearnable task is not evidence about leakage. The
  `nonlinearity` parameter exists so the task stays genuinely learnable and the
  remaining gap is attributable.

---

## Why this repository exists

The tooling conversation has matured fast. Split-construction tools such as
DataSAIL ([Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-58606-8))
make leakage-free splits straightforward, and the over-optimism literature
([Ullmann et al. 2021](https://arxiv.org/pdf/2106.02447);
[Leech et al. 2024](https://arxiv.org/pdf/2407.12220)) has documented the problem
thoroughly, including a 2025 *Briefings in Bioinformatics* review on
out-of-distribution learning
([bbaf294](https://academic.oup.com/bib/article/26/3/bbaf294/8176475)).

What is still missing is the short, opinionated thing a working bio-ML engineer
would tape to their monitor: given a leakage-free split, how do you decide what
the numbers mean, and how do you write up a result that lost to the baseline
without burying it?

## Worked examples

All examples use public framing only — no unpublished strain data appears here.

- **Cluster-aware splits** — MMseqs2 clustering at 40% identity, from an
  antimicrobial-peptide classifier where the honest number is the headline and
  the inflated random-split number is the cautionary contrast.
- **A negative transfer-learning result** — a transformer that did not beat an
  XGBoost baseline on bacterial guide-RNA data, reframed as a teaching example
  for reporting a clean loss.
- **A non-ML trap** — a genome-scale metabolic-model debugging trail, included
  to show the same discipline applies whether the model is a neural net or a
  flux-balance model.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
