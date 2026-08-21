# Reporting a result that did not beat the baseline

> Part of [`bio-ml-eval-protocols`](README.md). This is the reporting document
> referenced in the README's [What's inside](README.md#whats-inside) table, and
> it is the downstream consumer of the other two. [`methodology_traps.md`](methodology_traps.md)
> buys you a leakage-free split and a fairly-tuned baseline (Trap 5);
> [`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) §4 gives you the paired
> interval that decides whether a gap is real. This document is what you do
> *after* that interval comes back straddling zero: how to write up, and ship, a
> result where the new method did not beat the baseline.

Careful evaluation has an uncomfortable endpoint. You constructed a cluster-aware
split, tuned the baseline as hard as the method, ran the paired bootstrap on the
shared test set — and the confidence interval on the metric difference straddles
zero. The transformer did not beat the gradient-boosted trees. The honest
conclusion is "tied," or sometimes "the simpler model is reliably better."

Most of that work never gets written up. It goes into the file drawer, and the
field is worse for it. This document argues that a pre-committed, leakage-free
negative result is a publishable finding, gives a pre-registration checklist that
makes the negative *trustworthy*, restates the paired-bootstrap verdict as the
statistical backbone, and ships a fill-in template so the write-up is a
half-day's work rather than a research project.

---

## 1. Why a negative result is publishable — and why the file drawer is expensive

The published record over-represents wins. Methods that beat their baselines get
written up; methods that tie or lose get quietly shelved. This selection — the
*file-drawer effect* — is not a minor hygiene problem. It is a direct cause of the
over-optimism the literature keeps re-discovering: when only the winning
configurations survive to publication, the field's sense of what works is
systematically inflated. [Ullmann et al. (2021)](https://arxiv.org/pdf/2106.02447)
document this over-optimism in benchmark studies; [Leech et al.
(2024)](https://arxiv.org/pdf/2407.12220) catalog the questionable practices —
selective reporting, under-tuned baselines, post-hoc metric choice — that convert
a would-be tie into a reported win. The honest mirror of that catalog is simply to
report the tie.

A clean negative result carries real information:

- **It saves the next person three months.** A documented "transformer did not
  beat XGBoost on this data" tells the field not to re-spend the compute. The
  absence of that record is why dead ends get reproduced.
- **It constrains the hypothesis space.** "Deep representation learning does not
  add signal here, given this much data" is a finding about the *problem*, not
  just the model — exactly the out-of-distribution-generalisation question the
  *Briefings in Bioinformatics*
  [2025 review](https://academic.oup.com/bib/article/26/3/bbaf294/8176475)
  foregrounds.
- **The cost asymmetry makes it actionable.** A transformer that merely *ties* a
  gradient-boosted-tree baseline has lost on every axis that is not the metric:
  data hunger, training cost, inference latency, interpretability. "Tied" is, in
  practice, a recommendation to ship the simpler model.

What separates a publishable negative from "we couldn't make it work" is entirely
in the method, not the outcome. A negative result is trustworthy when:

1. the evaluation was **pre-committed** and leakage-free, so the null reflects the
   biology and not an under-powered or contaminated test;
2. the **baseline was tuned as hard as the method**, so it is a fair fight and not
   a straw man the method failed to clear;
3. the **effect size and its interval are reported**, so "tied" is a quantified
   claim with a known precision, not an asserted one;
4. the result is **informative** — the reader learns something about the problem
   from the null.

Sections 2–4 are the machinery for each of those four points.

---

## 2. Pre-registration checklist — lock it before you see the test result

The single discipline that converts a negative from "we gave up" into "we ran a
fair, pre-committed test and it tied" is the operational form of
[`methodology_traps.md`](methodology_traps.md) Trap 5: **decide the evaluation
before you see the test number.** Once the held-out metric is in front of you,
every subsequent choice — which metric to headline, which threshold, whether to
run the baseline through one more tuning round — is contaminated by knowing the
answer. Freeze the decisions while you are still blind to the outcome.

Lock the following, in writing and in the repo, before the held-out set is
touched even once:

- [ ] **The held-out split.** Constructed cluster-aware (Traps 1–2), frozen, and
      hashed so you can later prove it did not move.
- [ ] **The primary metric.** Exactly one, pre-named (AUC, average precision, or
      F1 at a stated threshold). Secondary metrics are fine but flagged as
      secondary.
- [ ] **The baseline(s) and their tuning budget.** Name them, and commit the
      *same* grid size / compute budget the proposed method gets. An under-tuned
      baseline is the most common way a tie is mis-reported as a win.
- [ ] **The comparison procedure.** The paired bootstrap of §3 on the shared test
      set, with `alpha` and `n_boot` fixed in advance.
- [ ] **The decision rule.** Write down — before the run — what CI outcome you
      will call "real improvement," "tie," and "clean loss" (§3). Pre-committing
      the rule is what stops the result from choosing the rule.
- [ ] **The stopping rule.** The test set is touched once, to produce the final
      number. Everything else happens on a development split.

This needs no external registry to be effective: a committed file with a
timestamp in the repository's history is enough provenance for a methods note.
(OSF or AsPredicted pre-registration is available if a venue wants it, but the
git history is the cheap default.) Hash the frozen split so the claim "the split
never moved" is checkable rather than promised:

```python
import hashlib, json
import numpy as np

def freeze_split(test_idx, path="frozen_test_idx.json"):
    """Hash the held-out indices and record them. Commit the output BEFORE
    scoring on the test set, so the split is provably immutable afterwards."""
    test_idx = np.sort(np.asarray(test_idx))
    digest = hashlib.sha256(test_idx.tobytes()).hexdigest()
    json.dump({"sha256": digest, "n": int(test_idx.size)},
              open(path, "w"), indent=2)
    return digest

# later, before reporting, prove it did not change:
def assert_split_unchanged(test_idx, path="frozen_test_idx.json"):
    recorded = json.load(open(path))["sha256"]
    now = hashlib.sha256(np.sort(np.asarray(test_idx)).tobytes()).hexdigest()
    assert recorded == now, "the held-out split moved after pre-registration"
```

The checklist is deliberately the same six items whether the result turns out to
be a win, a tie, or a loss. That symmetry is the point: you cannot have
pre-committed to it if it only appears in the papers that won.

---

## 3. The statistical backbone — tie versus clean loss

The verdict comes from the paired bootstrap of
[`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) §4. Resample the **same** test
indices for both models on each draw (cluster-resampled, to inherit the grouped
split), build a confidence interval on the per-resample metric *difference*
`metric(method) − metric(baseline)`, and read the verdict off where that interval
sits relative to zero. There are three outcomes, and a negative result is two of
them — they are different claims, and the template (§4) forces you to say which:

| Paired-CI on `method − baseline` | Verdict | What you report |
|---|---|---|
| Excludes 0, positive side (`lo > 0`) | **Real improvement** | The method beats the baseline at your alpha |
| **Straddles 0** | **Honest tie** | The method did not beat the baseline; the two are indistinguishable on this test set |
| Excludes 0, negative side (`hi < 0`) | **Clean loss** | The simpler baseline is reliably better |

```python
from examples.bootstrap_ci import paired_bootstrap
from sklearn.metrics import roc_auc_score

# score_method = transformer probs, score_base = xgboost probs, shared y_test
obs, (lo, hi), p = paired_bootstrap(
    y_test, score_method, score_base, roc_auc_score,
    n_boot=2000, rng=0, cluster_ids=cluster_ids,   # cluster-resample, per §3 caveat
)

if lo > 0:
    verdict = "real improvement (CI excludes 0, method side)"
elif hi < 0:
    verdict = "clean loss to baseline (CI excludes 0, baseline side)"
else:
    verdict = "honest tie (CI straddles 0)"
print(f"AUC(method) - AUC(baseline) = {obs:+.3f}  "
      f"(95% CI {lo:+.3f}, {hi:+.3f}; p={p:.3f})  ->  {verdict}")
```

**Do not confuse "tied" with "under-powered."** A CI that straddles zero on a tiny
test set may simply lack the resolution to separate the models, which is a
different — and weaker — statement than "they perform the same." Guard against it
by reporting the test size in both units that matter (records *and* clusters,
since the clusters are the effective sample size after grouping) and by stating
the interval width. A tie with a tight interval is informative ("indistinguishable
to within ±0.01 AUC"); a tie with an interval spanning half the metric range is an
admission the experiment could not answer the question, and should say so. This is
the honest counterpart, on the null side, of not over-claiming a win.

---

## 4. The fill-in reporting template

Drop this block into the manuscript or methods note and fill every field. Each
field maps to one of the four trustworthiness conditions from §1; leaving one
blank is the signal that the result is not yet shippable.

```text
NEGATIVE RESULT REPORT
----------------------
Claim:            <one sentence: "Method M did not beat baseline B at task T.">
Verdict:          <tie | clean loss>   (from the §3 decision rule)

Split:            <how the held-out set was built; identity threshold; n records;
                   n clusters; sha256 of frozen_test_idx.json>
Pre-registration: <where the locked decisions live: repo path + commit, or OSF id>

Baseline parity:  <the baseline and its tuning budget, stated as equal to the
                   method's: same grid size / compute / feature budget. One line
                   on why the comparison is fair.>

Primary metric:   <metric name (+ threshold if F1)>
  Method:         <point estimate>  (95% CI <lo>-<hi>)
  Baseline:       <point estimate>  (95% CI <lo>-<hi>)
  Paired diff:    <obs>  (95% CI <lo>, <hi>; p=<p>)   <- this is the verdict line

What would have   <the pre-committed effect size / CI that WOULD have flipped the
changed it:        conclusion: e.g. "a paired-CI lower bound > +0.02 AUC would have
                   been a real win." States what the experiment could detect.>

Data / code:      <link to the runnable demo that regenerates the verdict line>
```

A worked, filled-in version using the repo's CRISPR guide-efficiency example
(public framing — see §5):

```text
NEGATIVE RESULT REPORT
----------------------
Claim:            A transformer guide-efficiency model did not beat a
                  gradient-boosted-tree (XGBoost) baseline on bacterial guide-RNA
                  activity prediction.
Verdict:          tie

Split:            mmseqs2 cluster-aware at 40% identity; whole clusters held out;
                  n = <records> across <clusters> clusters; split hashed and
                  committed before scoring.
Pre-registration: docs/preregistration.md @ <commit>; metric, baseline budget,
                  and decision rule locked before the test set was scored.

Baseline parity:  XGBoost given the same hyperparameter-search budget (grid size
                  and CV folds) as the transformer's; both selected on an inner
                  cluster-aware development split, test set touched once.

Primary metric:   ROC-AUC
  Method:         <transformer AUC>  (95% BCa CI <lo>-<hi>)
  Baseline:       <xgboost AUC>      (95% BCa CI <lo>-<hi>)
  Paired diff:    <obs ~ near 0>     (95% CI straddles 0; p > alpha)

What would have   A paired-CI lower bound above +0.02 AUC in the transformer's
changed it:        favour would have been reported as a real improvement; the
                   observed interval includes 0, so the honest verdict is a tie.

Data / code:      tests/test_negative_results_reporting.py reproduces the paired
                  verdict on the shared fixture.
```

The "what would have changed it" field is the one most negative-result write-ups
omit, and the one reviewers most want. It demonstrates that the experiment had the
*power* to detect a meaningful effect and did not, rather than leaving open the
suspicion that the null is an artifact of an under-resolved test.

---

## 5. Worked example — transformer versus gradient-boosted baseline

The repo's running negative-result example is a transformer guide-RNA efficiency
model evaluated against a gradient-boosted-tree (XGBoost) baseline on **bacterial
guide-RNA activity data**, in public framing throughout. The chain across the
three documents is the whole point of the repo:

1. [`methodology_traps.md`](methodology_traps.md) Trap 5 sets up the fair fight —
   the baseline gets the *same* tuning budget as the transformer, and the test set
   is touched once on a cluster-aware split.
2. [`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) §4 adjudicates it — the
   paired bootstrap on the AUC difference returns an interval that **straddles
   zero**.
3. This document reports it — the §3 decision rule reads that straddle as an
   **honest tie**, and the §4 template turns it into a paragraph: the transformer
   did not beat the cheaper baseline, the comparison was fair and pre-committed,
   and on the cost asymmetry the recommendation is the simpler model.

That tie is the seed of a standalone paper. The full manuscript that *uses* this
template — corpora, transfer-learning setup, the cross-kingdom framing, and the
"honestly tied on one corpus, cleanly lost on another" result — is drafted
separately as `cross_kingdom_negative_result_draft.md` in the CRISPR Designer
project (repo idea #7). This document is the reusable scaffold; that paper is the
first instance of it.

The same discipline is model-agnostic. The repo's non-ML example — a genome-scale
metabolic-model debugging trail — lands the same way: a reported improvement that,
once the evaluation units are resampled and an interval put on the difference,
turns out to sit inside the noise. "Did not beat the baseline" is a verdict you
can reach with a neural network or a flux-balance model; the reporting template
does not care which.

---

## 6. Venue notes

A pre-committed, reproducible negative result fits the current methods-commentary
appetite well — the editorial conversation has moved from "leakage exists" to
"here is the discipline," and a working-practitioner write-up with a `pytest`-able
companion is a hard package to reject. Candidate venues, from the project research
note:

- ***Briefings in Bioinformatics* — Methods.** The natural primary target: it ran
  the 2025 out-of-distribution-learning review
  ([bbaf294](https://academic.oup.com/bib/article/26/3/bbaf294/8176475)), so the
  board has documented appetite for exactly this axis. A short methods piece
  (≤ ~2,500 words + one figure) citing the repo's release DOI is the intended
  shape.
- ***PLOS Computational Biology* — Perspective.** A good home for the
  field-practice argument rather than a single result, and the natural citation
  neighbour to the continuous-benchmarking ecosystem the field is converging on.
- ***Biology Methods and Protocols* (Oxford) — "AI in Biology and
  Bioinformatics" CFP.** A fit for the template-and-checklist framing; verify the
  call deadline before committing.
- ***Bioinformatics* — Application Note.** Possible, but the App Note track
  expects a pip-installable package rather than docs-plus-demos, which is roughly
  one to two extra weeks of packaging. Worth it only if the repo is being shipped
  as an installable tool anyway.

Whichever venue, the structure should mirror what the reproducibility ecosystem
(MLRC, and ISMB's code-or-reject proceedings policy) accepts, so the artifact
slots in cleanly: the claim is the paper, and the green CI run is the evidence.

---

## Worked examples used across this repo

The same three public-framed examples thread through every document here:

- **Cluster-aware splits (AMP classifier).** mmseqs2 clustering at 40% identity —
  the headline honest AUC versus the inflated random-split number from Traps 1–2.
  The discipline that makes a *win* trustworthy is the same one that makes a
  negative trustworthy.
- **A negative transfer-learning result (CRISPR guide-efficiency).** The
  transformer that did not beat an XGBoost baseline on bacterial guide-RNA data —
  the running example of this document, adjudicated by the §4 paired bootstrap and
  carried forward to the standalone idea-#7 paper.
- **A non-ML methods trap (flux-balance metabolic modelling).** A genome-scale
  metabolic-model debugging trail, included to show "did not beat the baseline" is
  a model-agnostic verdict.

## Minimum reproducible demo

Per the repository contract, the verdict this document is built around is backed
by a runnable demo. The accompanying `tests/` entry replays the paired-bootstrap
comparison on the shared public AMP-style fixture and asserts the honest-tie
verdict (the CI on the metric difference straddles zero), so the §4 worked example
reproduces on a fresh clone:

```bash
pytest tests/test_negative_results_reporting.py && make figures
```


---

*Citations on this page are limited to the open-access references catalogued in
the project's research note, anchored on the two over-optimism / file-drawer
references: [Ullmann et al. 2021](https://arxiv.org/pdf/2106.02447) and
[Leech et al. 2024](https://arxiv.org/pdf/2407.12220), with the
*Briefings in Bioinformatics*
[out-of-distribution learning review](https://academic.oup.com/bib/article/26/3/bbaf294/8176475)
as the venue-appetite reference. No citations are invented; the statistical
method is the paired bootstrap of [`bootstrap_ci_recipe.md`](bootstrap_ci_recipe.md) §4.*
