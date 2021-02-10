import numpy as np
import matplotlib.pyplot as plt


def qqplot(actual, expected, quantile_precision=0.01):

    # theoretical line
    plt.plot([min(expected), max(expected)], [min(actual), max(actual)], 'r--')

    # quantiles
    quantiles = np.arange(0, 1, quantile_precision)
    actual_quantiles = np.quantile(actual, quantiles)
    expected_quantiles = np.quantile(expected, quantiles)

    return plt.scatter(expected_quantiles, actual_quantiles)