import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon

DATA = "data/lysis_initiation/time_for_lysis.csv"


def sample_histogram(lower, upper, p, n=None, rng=None):
    if not n:
        n = 1

    if not rng:
        rng = np.random.default_rng()

    bins_chosen = rng.choice(len(p), size=n, replace=True, p=p)

    result = []
    for bin in bins_chosen:
        l_val, u_val = lower[bin], upper[bin]
        result.append(rng.uniform(l_val, u_val))

    return result


def main():
    data = pd.read_csv(DATA, skipinitialspace=True)
    center_time = data[["T_lower(sec)", "T_upper(sec)"]].mean(axis=1)
    width = data["T_upper(sec)"] - data["T_lower(sec)"]

    fig, ax = plt.subplots()

    # Plot data
    # NOTE: Heights are divided by bin width, *so that the plot integrates to 1*.
    # This is done for ease of comparison with the fitting and sampling methods below,
    # for which a pmf or pdf is the most reasonable presentation.
    ax.bar(
        x=center_time,
        height=data["P(T)"] / width,
        width=width,
        color="b",
        alpha=0.5,
        label="Data",
    )

    # Test exponential fit
    mean = 192.8
    x_range = np.arange(0, 500, 10)
    ax.plot(x_range, expon.pdf(x_range, scale=mean), "r--", label="Exponential fit")

    # Test exponential sampling
    rng = np.random.default_rng(324)
    exp_sample = rng.exponential(mean, size=10000)
    bins = list(data["T_lower(sec)"]) + [list(data["T_upper(sec)"])[-1]]
    ax.hist(
        exp_sample,
        bins,
        density=True,
        color="r",
        alpha=0.5,
        label="Exponential sampled data",
    )

    # Test histogram sampling
    N = 10000
    sampled = sample_histogram(
        data["T_lower(sec)"], data["T_upper(sec)"], data["P(T)"], N, rng=rng
    )
    hist, _ = np.histogram(sampled, bins, density=True)
    ax.bar(
        x=center_time,
        height=hist,
        width=width,
        color="c",
        alpha=0.5,
        label="Sampled_data",
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig("out/lysis_time_fit.png")


if __name__ == "__main__":
    main()
