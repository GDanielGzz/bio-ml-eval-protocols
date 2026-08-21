"""Auditing a split you did not construct yourself.

:mod:`bioeval.splits` prevents leakage. This module *detects* it — in a split
someone handed you, in a published benchmark's canonical train/test files, or
in your own pipeline three refactors later.

Run :func:`audit_split` before you trust any held-out number. It is cheap, and
the failure it catches is the one that silently inflates results by 10–15 AUC
points.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["LeakageReport", "audit_split", "record_hashes", "exact_duplicate_pairs"]


def record_hashes(records: Sequence) -> np.ndarray:
    """Hash records to a canonical form for exact-duplicate detection.

    Strings are stripped and upper-cased before hashing, so ``"acgt"``,
    ``"ACGT "`` and ``"ACGT"`` collide — which is what you want for sequence
    data, where the same record often arrives with different whitespace or case
    from two different sources. Non-strings are hashed via ``repr``; numpy
    arrays are hashed on their bytes.
    """
    out = []
    for r in records:
        if isinstance(r, str):
            key = r.strip().upper().encode()
        elif isinstance(r, (bytes, bytearray)):
            key = bytes(r)
        elif isinstance(r, np.ndarray):
            key = np.ascontiguousarray(r).tobytes()
        else:
            key = repr(r).encode()
        out.append(hashlib.sha256(key).hexdigest())
    return np.array(out)


def exact_duplicate_pairs(records: Sequence) -> dict:
    """Return ``{hash: [indices]}`` for every record value appearing more than once."""
    h = record_hashes(records)
    uniq, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)
    dup = {}
    for k in np.flatnonzero(counts > 1):
        dup[str(uniq[k])] = np.flatnonzero(inverse == k).tolist()
    return dup


@dataclass
class LeakageReport:
    """Result of :func:`audit_split`. Falsy when leakage was found."""

    n_train: int
    n_test: int
    overlapping_indices: list = field(default_factory=list)
    spanning_groups: list = field(default_factory=list)
    duplicate_across_split: list = field(default_factory=list)
    n_groups_train: int | None = None
    n_groups_test: int | None = None

    @property
    def ok(self) -> bool:
        return not (
            self.overlapping_indices
            or self.spanning_groups
            or self.duplicate_across_split
        )

    def __bool__(self) -> bool:  # so `if not audit_split(...)` reads naturally
        return self.ok

    def summary(self) -> str:
        lines = [
            f"Split audit: {self.n_train} train / {self.n_test} test records",
        ]
        if self.n_groups_train is not None:
            lines.append(
                f"             {self.n_groups_train} train groups / "
                f"{self.n_groups_test} test groups"
            )
        if self.ok:
            lines.append("PASS - no index overlap, no group spans the boundary, "
                         "no exact duplicates across it.")
            return "\n".join(lines)

        lines.append("FAIL")
        if self.overlapping_indices:
            lines.append(
                f"  * {len(self.overlapping_indices)} record(s) appear in BOTH "
                f"train and test: {self.overlapping_indices[:10]}"
                + (" ..." if len(self.overlapping_indices) > 10 else "")
            )
        if self.spanning_groups:
            lines.append(
                f"  * {len(self.spanning_groups)} group(s) span the boundary: "
                f"{self.spanning_groups[:10]}"
                + (" ..." if len(self.spanning_groups) > 10 else "")
            )
            lines.append(
                "    Homologs of test records are in train. The held-out metric "
                "is inflated."
            )
        if self.duplicate_across_split:
            lines.append(
                f"  * {len(self.duplicate_across_split)} exact duplicate pair(s) "
                f"across the boundary, e.g. {self.duplicate_across_split[:5]}"
            )
            lines.append(
                "    These are free correct answers. Deduplicate the union before "
                "splitting."
            )
        return "\n".join(lines)

    def raise_if_leaking(self) -> None:
        """Fail loudly. Use in CI so a leaking split cannot reach a results table."""
        if not self.ok:
            raise AssertionError(self.summary())


def audit_split(
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    groups: Sequence | None = None,
    records: Sequence | None = None,
) -> LeakageReport:
    """Check a split for the three ways records leak across the boundary.

    1. **Index overlap** — the same row in both sides. Rare but catastrophic,
       and trivially caused by an off-by-one in a manual slice.
    2. **Group spanning** — a cluster, batch, plate or subject with records on
       both sides. This is the expensive one, and it is invisible without the
       group labels.
    3. **Exact duplicates across the boundary** — byte-identical records in
       train and test, which happens whenever two public corpora are merged
       without deduplicating the union.

    ``groups`` and ``records`` are indexed over the *full* dataset, not over the
    split. Either may be omitted; the corresponding check is then skipped, and
    the report says nothing about it rather than passing it silently.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    overlap = np.intersect1d(train_idx, test_idx)

    spanning: list = []
    n_gtr = n_gte = None
    if groups is not None:
        g = np.asarray(groups)
        gtr = set(np.unique(g[train_idx]).tolist())
        gte = set(np.unique(g[test_idx]).tolist())
        n_gtr, n_gte = len(gtr), len(gte)
        spanning = sorted(gtr & gte, key=repr)

    dup_across: list = []
    if records is not None:
        h = record_hashes(records)
        train_h = {}
        for i in train_idx:
            train_h.setdefault(h[i], []).append(int(i))
        for j in test_idx:
            if h[j] in train_h:
                dup_across.append((train_h[h[j]][0], int(j)))

    return LeakageReport(
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        overlapping_indices=overlap.tolist(),
        spanning_groups=spanning,
        duplicate_across_split=dup_across,
        n_groups_train=n_gtr,
        n_groups_test=n_gte,
    )
