"""
Data Predicates

Defines several assertions about data that are useful for tests,
e.g. checks for monotonicity, whether the data approximately follows a Poisson distribution, etc.

All functions expect a 1D numpy array as first parameter.

TODO:
- implement faster numpy-based solution for tests of increasing/decreasing
"""

import numpy as np
from scipy.stats import chisquare, poisson
from collections import Counter


def strictly_increasing(data):
    return all(a < b for a, b in zip(data, data[1:]))


def strictly_decreasing(data):
    return all(a > b for a, b in zip(data, data[1:]))


def monotonically_increasing(data):
    return all(a <= b for a, b in zip(data, data[1:]))


def monotonically_decreasing(data):
    return all(a >= b for a, b in zip(data, data[1:]))


def all_positive(data):
    return np.all(data > 0)


def all_negative(data):
    return np.all(data < 0)


def all_nonnegative(data):
    return np.all(data >= 0)


def all_nonpositive(data):
    return np.all(data <= 0)


def approx_poisson(data, rate=None, significance=0.05, verbose=False):
    """
    Test whether data appears to follow Poisson distribution, using Chi-sq goodness of fit.
    Does not do particularly well comparing poisson data of rate r_1 vs. poisson distribution of rate r_2.
    Args:
        data: 1D array where index i corresponds the number of events observed in interval i.
        rate: rate (lambda) of the Poisson distribution against which to compare. If None, rate is estimated from the data.
        significance: for p > significance, fail to reject that the data is not Poisson-distributed.
        verbose: if True, prints estimated rate, and results (chi-sq, p-value) of the goodness-of-fit test.
    """

    if rate is None:
        rate = np.mean(data)

    counts = Counter(list(data))
    counts = [counts[i] if i in counts.keys() else 0 for i in range(max(data) + 1)]

    res = chisquare(
        np.array(counts) / sum(counts),
        poisson(rate).pmf(range(len(counts)))
        / sum(poisson(rate).pmf(range(len(counts)))),
    )

    if verbose:
        print(f"Estimated rate (lambda): {rate}")
        print(f"Chi-sq: {res[0]}")
        print(f"p: {res[1]}")

    return res[1] > significance


def test_data_predicates():
    assert strictly_increasing(np.array([1, 2, 3])) and not strictly_increasing(
        np.array([1, 1, 2])
    )
    assert strictly_decreasing(np.array([3, 2, 1])) and not strictly_decreasing(
        np.array([3, 3, 2])
    )
    assert monotonically_increasing(
        np.array([1, 1, 2])
    ) and not monotonically_increasing(np.array([1, 0, 1]))
    assert monotonically_decreasing(
        np.array([2, 2, 1])
    ) and not monotonically_decreasing(np.array([1, 2, 1]))
    assert all_positive(np.array([1, 2, 3])) and not all_positive(np.array([1, 1, 0]))
    assert all_negative(np.array([-1, -2, -3])) and not all_negative(
        np.array([-1, -1, 0])
    )
    assert all_nonnegative(np.array([0, 1, 2])) and not all_nonnegative(
        np.array([-1, 0, 1])
    )
    assert all_nonpositive(np.array([0, -1, -2])) and not all_nonpositive(
        np.array([-1, 0, 1])
    )

    poisson_data = np.random.poisson(lam=2, size=1000)
    geom_data = np.random.geometric(p=0.1, size=1000)
    assert approx_poisson(poisson_data) and not approx_poisson(geom_data)

    print("Passed all tests.")


if __name__ == "__main__":
    test_data_predicates()
