import os
from collections import Counter
from itertools import accumulate, chain, takewhile, tee

import matplotlib.pyplot as plt
import numpy as np


def length_distributions(column):
    strand_lengths = []
    gap_lengths = []

    in_strand = bool(column[0])
    current_length = 0
    for x in column:
        if x == int(in_strand):
            current_length += 1
        else:
            (strand_lengths if in_strand else gap_lengths).append(current_length)
            in_strand = not in_strand
            current_length = 1

    return strand_lengths, gap_lengths


def poisson_sampler(rng, rate):
    def sampler():
        while True:
            yield rng.poisson(rate)

    return sampler


def sample_column(rows, murein, strand_sampler, rng):
    result = np.zeros(rows)

    strand_length, total_length = tee(strand_sampler())
    total_length = accumulate(total_length)

    # Sample strand lengths
    strands = [
        s
        for s, _ in takewhile(lambda s: s[1] < murein, zip(strand_length, total_length))
        if s > 0
    ]
    remaining = murein - sum(strands)
    if remaining:
        strands.append(remaining)

    total_gap = rows - sum(strands)
    p_s = total_gap / len(strands)

    result_i = 0
    for i in range(rows):
        if len(strands) == 0:
            break

        if rng.uniform() <= p_s:
            # insert strand
            strand = strands.pop()
            result[result_i : (result_i + strand)] = 1
            result_i += strand
        else:
            result_i += 1

    return result


def plot_length_distributions(strand_lengths, gap_lengths):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for c, data in enumerate([strand_lengths, gap_lengths]):
        flat_data = list(chain(*data))

        # Create histogram of each run in the data
        histograms = np.zeros((len(data), max(flat_data) + 1))
        for run, run_data in enumerate(data):
            hist = Counter(run_data)
            for k, v in hist.items():
                histograms[run, k] = v

        # Normalize as probability distributions
        distributions = histograms / histograms.sum(axis=1)[:, np.newaxis]

        # Calculate means, error bars
        y = distributions.mean(axis=0)
        err = distributions.std(axis=0)
        bincenters = np.arange(y.size)
        axs[c].bar(bincenters, y, yerr=err)
        axs[c].set_xticks(bincenters)
        axs[c].set_xticklabels(axs[c].get_xticks(), rotation=45)
        axs[c].set_title(f"{('Strand', 'Gap')[c]} Length Distribution")

    return fig, axs


def plot_locational(columns):
    columns = np.array(columns)
    runs, positions = columns.shape

    strands = columns.sum(axis=0)
    gaps = runs - strands

    fig, ax = plt.subplots()
    x = np.arange(positions)
    ax.bar(x, strands, width=1, label="Strand")
    ax.bar(x, gaps, width=1, bottom=strands, label="Gap")
    ax.set_title(f"Count of Strand/Gap By Position, over {runs} runs")
    ax.legend()

    return fig, ax


def test_column_sampler(outdir="out/murein_sampling"):
    rng = np.random.default_rng(0)

    columns = []
    strand_lengths = []
    gap_lengths = []

    N = 100
    for i in range(N):
        col = sample_column(300, 200, poisson_sampler(rng, 0.5), rng)
        s, g = length_distributions(col)
        columns.append(col)
        strand_lengths.append(s)
        gap_lengths.append(g)

    # Diagnostic plots ====================================================
    os.makedirs(outdir, exist_ok=True)

    fig, _ = plot_length_distributions(strand_lengths, gap_lengths)
    fig.set_size_inches((10, 6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "length_distributions.png"))

    fig, _ = plot_locational(columns)
    fig.set_size_inches((8, 6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "locational.png"))


def main():
    test_column_sampler()


if __name__ == "__main__":
    main()
