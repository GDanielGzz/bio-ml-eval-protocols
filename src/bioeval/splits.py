"""Group-aware splitting.

The single rule this module enforces: **no group may span the train/test
boundary.** For biological sequence data the group is normally an identity
cluster (e.g. MMseqs2 at 40% identity); for assay data it may be a plate, a
batch, or a subject. The failure this prevents is the one described in
``methodology_traps.md`` — near-duplicates of test records sitting in train,
so the reported metric measures memorised redundancy rather than
generalisation.

Nothing here computes clusters. Clustering is a domain decision and belongs to
a tool that understands your data (MMseqs2, CD-HIT, DataSAIL). This module takes
the resulting group labels and guarantees the split respects them.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

__all__ = ["grouped_split", "grouped_kfold", "group_sizes"]


def _as_group_array(groups: Sequence) -> np.ndarray:
    arr = np.asarray(groups)
    if arr.ndim != 1:
        raise ValueError(f"groups must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("groups is empty")
    return arr


def group_sizes(groups: Sequence) -> dict:
    """Return ``{group_label: n_records}``, largest first.

    Useful before splitting: if one group holds 40% of the data, no group-aware
    split can give you a 20% test set without that group dominating one side.
    Knowing this up front is better than discovering it in the variance.
    """
    arr = _as_group_array(groups)
    labels, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    return {labels[i].item() if hasattr(labels[i], "item") else labels[i]: int(counts[i])
            for i in order}


def grouped_split(
    groups: Sequence,
    test_size: float = 0.2,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so that whole groups land on one side or the other.

    Groups are shuffled, then greedily assigned to the test set (largest first,
    which packs closer to the requested fraction than random order) until adding
    another group would overshoot ``test_size``.

    Parameters
    ----------
    groups:
        Per-record group label. Records sharing a label are never separated.
    test_size:
        Target *fraction of records* — not of groups — in the test set. The
        realised fraction will differ when group sizes are uneven; that is a
        property of the data, not a bug, and :func:`group_sizes` will tell you
        why before you split.
    seed:
        Seed for the group shuffle. ``None`` for nondeterministic.

    Returns
    -------
    ``(train_idx, test_idx)`` as integer arrays into the original record order.

    Raises
    ------
    ValueError
        If the split would leave either side empty — usually a sign there are
        too few groups for the requested fraction.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    arr = _as_group_array(groups)
    n = arr.size

    labels, counts = np.unique(arr, return_counts=True)
    if labels.size < 2:
        raise ValueError(
            "cannot make a group-aware split with fewer than 2 groups; "
            "every record shares a group label"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(labels.size)

    # Fill in shuffled order, taking any group that still fits under the target.
    # Sorting largest-first would pack marginally tighter, but it makes the
    # result almost independent of `seed` whenever group sizes are distinct —
    # which silently defeats multi-seed robustness checks. Seed sensitivity is
    # worth more here than a percentage point of packing accuracy.
    target = test_size * n
    test_labels: list = []
    running = 0
    for i in order:
        if running == 0 or running + counts[i] <= target:
            test_labels.append(labels[i])
            running += counts[i]

    test_mask = np.isin(arr, test_labels)
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"split left one side empty (train={train_idx.size}, test={test_idx.size}); "
            f"with {labels.size} groups, test_size={test_size} is not achievable"
        )
    return train_idx, test_idx


def grouped_kfold(
    groups: Sequence,
    n_splits: int = 5,
    seed: int | None = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` for k folds that respect group boundaries.

    Groups are visited in shuffled order and each is assigned to the currently
    smallest fold — a bin-packing heuristic that keeps folds close to equal size
    even when group sizes are skewed, while remaining genuinely sensitive to
    ``seed`` (see the note in :func:`grouped_split` on why that matters).
    """
    arr = _as_group_array(groups)
    labels, counts = np.unique(arr, return_counts=True)
    if labels.size < n_splits:
        raise ValueError(
            f"cannot make {n_splits} group-aware folds from {labels.size} groups"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(labels.size)

    fold_of_label: dict = {}
    fold_load = np.zeros(n_splits, dtype=np.int64)
    for i in order:
        f = int(np.argmin(fold_load))
        fold_of_label[labels[i].item() if hasattr(labels[i], "item") else labels[i]] = f
        fold_load[f] += counts[i]

    record_fold = np.array(
        [fold_of_label[g.item() if hasattr(g, "item") else g] for g in arr]
    )
    for f in range(n_splits):
        test_idx = np.flatnonzero(record_fold == f)
        train_idx = np.flatnonzero(record_fold != f)
        if test_idx.size == 0:
            continue
        yield train_idx, test_idx
