"""Tests for split auditing.

The audit must fire on leakage it should catch, and stay quiet on a clean
split. A leakage detector with false negatives is worse than none, because it
converts an open question into a false reassurance — so most of these tests
construct leakage deliberately and assert it is found.
"""

import numpy as np
import pytest

from bioeval import audit_split, exact_duplicate_pairs, grouped_split, record_hashes
from bioeval.datasets import make_clustered_binary


def test_clean_grouped_split_passes():
    _, _, groups = make_clustered_binary(n_clusters=60, seed=0)
    train, test = grouped_split(groups, test_size=0.2, seed=0)
    report = audit_split(train, test, groups=groups)
    assert report.ok
    assert bool(report) is True
    assert "PASS" in report.summary()


def test_random_split_on_clustered_data_is_caught():
    _, _, groups = make_clustered_binary(n_clusters=60, seed=0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(groups.size)
    cut = int(0.8 * groups.size)
    report = audit_split(perm[:cut], perm[cut:], groups=groups)
    assert not report.ok
    assert report.spanning_groups          # this is the whole point
    assert "FAIL" in report.summary()
    assert "inflated" in report.summary()


def test_index_overlap_is_caught():
    groups = np.repeat(np.arange(10), 5)
    train = np.arange(0, 40)
    test = np.arange(35, 50)          # 35..39 appear on both sides
    report = audit_split(train, test, groups=groups)
    assert not report.ok
    assert set(report.overlapping_indices) == set(range(35, 40))


def test_exact_duplicates_across_boundary_are_caught():
    records = ["AAAA", "CCCC", "GGGG", "AAAA"]   # 0 and 3 identical
    groups = np.array([0, 1, 2, 3])              # different groups: only dedup finds it
    report = audit_split([0, 1], [2, 3], groups=groups, records=records)
    assert not report.ok
    assert report.duplicate_across_split == [(0, 3)]
    assert not report.spanning_groups            # groups alone would have passed it


def test_raise_if_leaking_raises_only_when_leaking():
    groups = np.repeat(np.arange(10), 5)
    clean_train, clean_test = grouped_split(groups, test_size=0.2, seed=0)
    audit_split(clean_train, clean_test, groups=groups).raise_if_leaking()

    with pytest.raises(AssertionError, match="FAIL"):
        audit_split([0, 1, 2], [2, 3, 4], groups=groups).raise_if_leaking()


def test_hashes_normalise_case_and_whitespace():
    h = record_hashes(["acgt", "  ACGT ", "ACGA"])
    assert h[0] == h[1] != h[2]


def test_hashes_handle_arrays_and_non_strings():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 4.0])
    h = record_hashes([a, b, c])
    assert h[0] == h[1] != h[2]


def test_exact_duplicate_pairs_groups_all_occurrences():
    dup = exact_duplicate_pairs(["A", "B", "A", "C", "A"])
    assert len(dup) == 1
    assert list(dup.values())[0] == [0, 2, 4]


def test_omitting_groups_does_not_fabricate_a_pass():
    """Without group labels the audit cannot see homology leakage.

    It must therefore report nothing about groups rather than implying the split
    is clean on that axis.
    """
    groups = np.repeat(np.arange(10), 5)
    rng = np.random.default_rng(0)
    perm = rng.permutation(50)
    report = audit_split(perm[:40], perm[40:])      # groups deliberately omitted
    assert report.ok                                 # nothing it can check has failed
    assert report.n_groups_train is None             # and it says so
    leaky = audit_split(perm[:40], perm[40:], groups=groups)
    assert not leaky.ok                              # the same split, seen properly
