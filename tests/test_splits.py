"""Tests for group-aware splitting.

These assert the *property* the module exists to guarantee — no group spans the
boundary — rather than checking that a function returns something of the right
shape. A test that only checks shapes would pass on an implementation that
leaks.
"""

import numpy as np
import pytest

from bioeval import group_sizes, grouped_kfold, grouped_split
from bioeval.datasets import make_clustered_binary


@pytest.mark.parametrize("seed", range(8))
def test_no_group_spans_the_split(seed):
    _, _, groups = make_clustered_binary(n_clusters=60, seed=seed)
    train, test = grouped_split(groups, test_size=0.25, seed=seed)
    assert set(groups[train]).isdisjoint(set(groups[test]))


@pytest.mark.parametrize("seed", range(8))
def test_split_is_a_partition(seed):
    _, _, groups = make_clustered_binary(n_clusters=60, seed=seed)
    train, test = grouped_split(groups, test_size=0.25, seed=seed)
    assert np.intersect1d(train, test).size == 0
    assert np.union1d(train, test).size == groups.size


def test_split_is_deterministic_under_seed():
    _, _, groups = make_clustered_binary(n_clusters=50, seed=3)
    a = grouped_split(groups, test_size=0.2, seed=11)
    b = grouped_split(groups, test_size=0.2, seed=11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_split_changes_with_seed():
    _, _, groups = make_clustered_binary(n_clusters=50, seed=3)
    a = grouped_split(groups, test_size=0.2, seed=1)
    b = grouped_split(groups, test_size=0.2, seed=2)
    assert not np.array_equal(a[1], b[1])


def test_test_fraction_is_approximately_respected():
    _, _, groups = make_clustered_binary(n_clusters=200, cluster_size=(3, 9), seed=5)
    _, test = grouped_split(groups, test_size=0.25, seed=5)
    frac = test.size / groups.size
    # Whole groups must move together, so exactness is impossible by construction.
    assert 0.18 < frac < 0.30


def test_single_group_is_rejected_rather_than_silently_leaking():
    groups = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="fewer than 2 groups"):
        grouped_split(groups, test_size=0.2)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_size_rejected(bad):
    _, _, groups = make_clustered_binary(n_clusters=20, seed=0)
    with pytest.raises(ValueError, match="test_size"):
        grouped_split(groups, test_size=bad)


@pytest.mark.parametrize("n_splits", [3, 5, 8])
def test_kfold_folds_never_share_a_group(n_splits):
    _, _, groups = make_clustered_binary(n_clusters=90, seed=1)
    folds = list(grouped_kfold(groups, n_splits=n_splits, seed=1))
    assert len(folds) == n_splits
    for train, test in folds:
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_kfold_test_sets_tile_the_dataset_exactly_once():
    _, _, groups = make_clustered_binary(n_clusters=90, seed=2)
    tests = [t for _, t in grouped_kfold(groups, n_splits=5, seed=2)]
    stacked = np.concatenate(tests)
    assert stacked.size == groups.size
    assert np.array_equal(np.sort(stacked), np.arange(groups.size))


def test_kfold_rejects_too_few_groups():
    groups = np.repeat(np.arange(3), 10)
    with pytest.raises(ValueError, match="cannot make 5 group-aware folds"):
        list(grouped_kfold(groups, n_splits=5))


def test_group_sizes_sorted_descending_and_totals():
    groups = np.array([0, 0, 0, 1, 1, 2])
    sizes = group_sizes(groups)
    assert list(sizes.values()) == [3, 2, 1]
    assert sum(sizes.values()) == groups.size
