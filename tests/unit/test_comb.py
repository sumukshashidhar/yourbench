import random
from math import comb
from itertools import combinations

import pytest

from yourbench.utils.dataset_engine import _unrank_comb, _floyd_sample_indices, _sample_exact_combinations


def test_comb_basic_cases():
    assert _unrank_comb(5, 3, 0) == [0, 1, 2]
    assert _unrank_comb(5, 3, 1) == [0, 1, 3]
    assert _unrank_comb(5, 3, 9) == [2, 3, 4]  # Last combination


def test_comb_k_equals_n():
    assert _unrank_comb(4, 4, 0) == [0, 1, 2, 3]


def test_comb_k_zero():
    assert _unrank_comb(5, 0, 0) == []


def test_comb_invalid_k_n():
    with pytest.raises(ValueError):
        _unrank_comb(4, 5, 0)


def test_comb_invalid_rank_negative():
    with pytest.raises(ValueError):
        _unrank_comb(5, 2, -1)


def test_comb_invalid_rank_too_large():
    with pytest.raises(ValueError):
        _unrank_comb(5, 2, comb(5, 2))  # rank == C(5,2) is out of bounds


def test_comb_against_colex_order():
    for n in range(1, 15):
        for k in range(0, n + 1):
            # Generate all combinations and sort by colex order (right-to-left)
            expected_combs = sorted(combinations(range(n), k), key=lambda x: x[::-1])
            for rank, expected in enumerate(expected_combs):
                actual = _unrank_comb(n, k, rank)
                assert actual == list(expected), f"Mismatch at n={n}, k={k}, rank={rank}"


def test_floyd_basic_properties():
    result = _floyd_sample_indices(10, 5)
    assert len(result) == 5
    assert all(0 <= x < 10 for x in result)


def test_floyd_full_sample():
    result = _floyd_sample_indices(10, 10)
    assert len(result) == 10
    assert result == set(range(10))


def test_floyd_zero_sample():
    result = _floyd_sample_indices(10, 0)
    assert result == set()


def test_floyd_invalid_sample_size():
    with pytest.raises(ValueError):
        _floyd_sample_indices(5, 6)


def test_floyd_deterministic_with_seed():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    result1 = _floyd_sample_indices(100, 10, rng=rng1)
    result2 = _floyd_sample_indices(100, 10, rng=rng2)
    assert result1 == result2
    assert len(result1) == 10


def test_sample_basic_output():
    objects = ["a", "b", "c", "d"]
    result = _sample_exact_combinations(objects, k=2, N=3)
    assert len(result) == 3
    for combination in result:
        assert len(comb) == 2
        assert all(obj in objects for obj in comb)
        assert len(set(comb)) == 2  # Unique elements within combination


def test_sample_no_duplicates():
    objects = list(range(6))
    N = 10
    samples = _sample_exact_combinations(objects, k=3, N=N)
    assert len(samples) == N
    assert len({tuple(sorted(s)) for s in samples}) == N  # All unique combinations


def test_sample_all_possible():
    objects = ["a", "b", "c", "d"]
    total = comb(len(objects), 2)
    samples = _sample_exact_combinations(objects, k=2, N=total)
    expected = set(combinations(objects, 2))
    assert {tuple(sorted(s)) for s in samples} == expected


def test_sample_zero():
    result = _sample_exact_combinations(["x", "y"], k=0, N=1)
    assert result == [[]]


def test_sample_invalid_request():
    with pytest.raises(ValueError):
        _sample_exact_combinations([1, 2, 3], k=2, N=4)  # C(3,2)=3 < 4


def test_sample_deterministic_with_seed():
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    result1 = _sample_exact_combinations(list(range(20)), k=4, N=5, rng=rng1)
    result2 = _sample_exact_combinations(list(range(20)), k=4, N=5, rng=rng2)
    assert result1 == result2
