import numpy as np
import matplotlib.pyplot as plt


def qqplot(actual, expected, quantile_precision=0.01, include_line=True):
    """
    Do a QQ-plot of expected and actual distributions.
    The plot should be approximately linear if the distributions have the same shape.

    Args:
        actual: 1D numpy array representing draws from actual distribution
        expected: 1D numpy array representing draws from expected distribution
        quantile_precision: use smaller values for smoother plots, or if points appear to be missing
        include_line: whether to include a straight line for reference

    Returns: matplotlib Plot object
    """

    # theoretical line
    if include_line:
        plt.plot([min(expected), max(expected)], [min(actual), max(actual)], 'r--')

    # quantiles
    # TODO: could improve by sampling more where quantiles change more rapidly, instead of with constant increment
    quantiles = np.arange(0, 1, quantile_precision)
    actual_quantiles = np.quantile(actual, quantiles)
    expected_quantiles = np.quantile(expected, quantiles)

    return plt.scatter(expected_quantiles, actual_quantiles)