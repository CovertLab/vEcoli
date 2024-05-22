import os

import matplotlib.pyplot as plt
import numpy as np


def geom_sampler(rng, p):
    def sampler():
        while True:
            yield rng.geometric(p)

    return sampler


def fit_strand_term_p(df, upper_mean):
    # Take only the wild-type strain
    clean = df[df["Strain"] == "MC4100"].drop(columns=["Strain"]).set_index("Length")

    # Normalize to sum to 1
    clean["Percent"] /= clean["Percent"].sum()

    # Get mean of strands <= 30 in length
    lower_data = clean[clean.index != ">30"]
    lower_mean = (
        np.array([int(x) for x in lower_data.index]) * lower_data["Percent"]
    ).sum()

    # Get overall mean
    p_upper = clean.loc[">30"]
    mean = (1 - p_upper) * lower_mean + (p_upper * upper_mean)

    # return p for a geometric distribution with corresponding mean
    return 1 / mean.values[0]


def sample_column(rows, murein_monomers, strand_sampler, rng, shift=True):
    result = np.zeros(rows, dtype=int)

    # If no murein available, return full column hole
    if murein_monomers == 0:
        return result

    # Don't try to assign more murein than can fit in the column
    murein_monomers = int(min(murein_monomers, rows))

    # Sample strand lengths such that
    # as much as possible of the available murein is used,
    # while having at least one gap per strand
    strands = []
    for s in strand_sampler():
        # Stop if adding this strand (and its associated minimum gap)
        # would exceed murein or column length constraint
        if s + sum(strands) > murein_monomers:
            break
        if s + 1 + sum(strands) + len(strands) >= rows:
            break
        strands.append(s)

    # add remaining strand if there is space
    remaining_strand = min(
        murein_monomers - sum(strands), rows - sum(strands) - len(strands) - 1
    )
    assert remaining_strand >= 0
    if remaining_strand:
        strands.append(remaining_strand)

    # Get probability for initiating a strand
    total_gap = rows - sum(strands)
    strand_starts = list(rng.choice(total_gap, size=len(strands), replace=False))
    strand_starts.sort(reverse=True)

    # Assemble the resulting column from strands, start positions, # rows
    result_i = 0
    next_start = strand_starts.pop()
    # iterate over all gap, inserting when next strand start reached
    for gap_i in range(total_gap + 1):
        if next_start == gap_i:
            strand = strands.pop()
            result[result_i : (result_i + strand)] = 1
            result_i += strand
            next_start = strand_starts.pop() if len(strands) > 0 else -1
        result_i += 1

    if shift:
        result = np.roll(result, rng.integers(len(result)))

    return result


def sample_lattice(murein_monomers, rows, cols, strand_sampler, rng):
    result = np.zeros((rows, cols), dtype=int)

    # Get murein in each column
    murein_per_column = rng.multinomial(murein_monomers, np.repeat([1 / cols], cols))

    for c, murein in enumerate(murein_per_column):
        result[:, c] = sample_column(rows, murein, strand_sampler, rng)

    return result


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
    p = 0.058

    columns = []

    rows = 3050
    cols = 599
    murein = 450000
    for i in range(cols):
        col = sample_column(rows, int(murein // cols), geom_sampler(rng, p), rng)
        columns.append(col)

    # Diagnostic plots ====================================================
    os.makedirs(outdir, exist_ok=True)

    fig, _ = plot_locational(columns)
    fig.set_size_inches((8, 6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "locational.png"))


def main():
    test_column_sampler()


if __name__ == "__main__":
    main()
