"""Turning numbers into claims you can defend.

``negative_results_reporting.md`` argues that the hard part of an honest
evaluation is not computing the interval, it is writing the sentence. This
module writes the sentence, and refuses to write a stronger one than the
interval supports.
"""

from __future__ import annotations

from .bootstrap import CI, PairedResult

__all__ = ["results_table", "comparison_note"]


def results_table(rows: dict[str, CI], metric_name: str = "metric") -> str:
    """Render ``{model_name: CI}`` as a markdown table with intervals, not points.

    The column header names the interval explicitly. A table of bare point
    estimates is the most common way a paper implies a precision it did not
    measure.
    """
    if not rows:
        raise ValueError("no rows to render")
    alpha = next(iter(rows.values())).alpha
    pct = int(round((1 - alpha) * 100))
    out = [
        f"| Model | {metric_name} [{pct}% CI] | Interval width | Method |",
        "|---|---|---|---|",
    ]
    for name, ci in rows.items():
        out.append(
            f"| {name} | {ci.estimate:.4f} [{ci.low:.4f}, {ci.high:.4f}] "
            f"| {ci.width:.4f} | {ci.method} |"
        )
    return "\n".join(out)


def comparison_note(
    paired: PairedResult,
    ci_a: CI | None = None,
    ci_b: CI | None = None,
) -> str:
    """Write the head-to-head paragraph, including the tie case.

    When the paired difference contains zero the note says *tied* — not
    "comparable", not "on par with", and never "slightly better". If marginal
    intervals are supplied and they overlap while the paired difference does
    not contain zero, the note flags that too, because the overlap will
    otherwise be used against the result by a reviewer applying the wrong test.
    """
    lines = [paired.sentence()]

    if ci_a is not None and ci_b is not None:
        overlap = ci_a.overlaps(ci_b)
        if overlap and paired.significant:
            lines.append(
                f"Note that the marginal intervals for {paired.label_a} and "
                f"{paired.label_b} overlap. That is not evidence of a tie: the "
                "marginal intervals carry test-set variance shared by both models, "
                "which cancels in the paired difference. The paired interval is "
                "the correct test here."
            )
        elif not overlap and not paired.significant:
            lines.append(
                "The marginal intervals do not overlap, but the paired difference "
                "does contain zero. The paired test is the one to report."
            )

    if paired.verdict == "tie":
        lines.append(
            "Reported as a tie rather than a win. The honest reading is that this "
            "experiment did not distinguish the two models at this sample size — "
            "which is a result, not a failure to obtain one."
        )
    return "\n\n".join(lines)
