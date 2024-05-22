import matplotlib.pyplot as plt
import numpy as np


def rotate(px, py, theta):
    return np.array(
        [
            px * np.cos(theta) - py * np.sin(theta),
            px * np.sin(theta) + py * np.cos(theta),
        ]
    )


def main():
    HIGHLIGHT_BLUE = "#1a66ffff"
    STROKE_GRAY = "#404040ff"
    BG_GRAY = "#ccccccff"

    gen = 2271
    sub = 2417
    antigen = 63
    antisub = 43

    fig, ax = plt.subplots()

    outer_rad = 1.8
    ring_width = 0.3
    margin = 0.1

    ax.pie(
        [sub, gen],
        radius=outer_rad,
        colors=[HIGHLIGHT_BLUE, BG_GRAY],
        wedgeprops=dict(width=ring_width, edgecolor=STROKE_GRAY, linewidth=0.75),
        startangle=90,
        frame=False,
    )
    ax.pie(
        [antisub, antigen],
        radius=outer_rad - ring_width - margin,
        colors=[HIGHLIGHT_BLUE, BG_GRAY],
        wedgeprops=dict(width=ring_width, edgecolor=STROKE_GRAY, linewidth=0.75),
        startangle=90,
        frame=False,
    )

    ax.text(
        -outer_rad - 2 * margin,
        0,
        "All genes",
        fontdict={"fontsize": 10, "family": "Arial"},
        horizontalalignment="right",
        verticalalignment="center",
    )
    ax.text(
        -outer_rad + 2 * ring_width + 3 * margin,
        0,
        "Antibiotic\nResponse genes",
        fontdict={"fontsize": 10, "family": "Arial"},
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.hlines(0, -outer_rad, -outer_rad - 1.5 * margin, color=STROKE_GRAY, linewidth=1)
    ax.hlines(
        0,
        -outer_rad + 2 * ring_width + margin,
        -outer_rad + 2 * ring_width + 2.5 * margin,
        color=STROKE_GRAY,
        linewidth=1,
    )

    ax.text(
        *rotate(outer_rad + margin, 0, 3 * np.pi / 4),
        f"{sub} genes",
        fontdict={"fontsize": 8, "family": "Arial"},
        horizontalalignment="right",
    )
    ax.text(
        *rotate(outer_rad + 1.5 * margin, 0, -np.pi / 4),
        f"{gen} genes",
        fontdict={"fontsize": 8, "family": "Arial"},
        horizontalalignment="left",
    )
    ax.text(
        *rotate(outer_rad - 2 * ring_width - 2.5 * margin, 0, 0.8 * np.pi),
        f"{antisub} genes",
        fontdict={"fontsize": 8, "family": "Arial"},
        horizontalalignment="left",
    )
    ax.text(
        *rotate(outer_rad - 2 * ring_width - 2 * margin, 0, -0.25 * np.pi),
        f"{antigen} genes",
        fontdict={"fontsize": 8, "family": "Arial"},
        horizontalalignment="right",
    )

    ax.set_xlim(-outer_rad - 3 * margin, outer_rad)

    fig.tight_layout()
    fig.set_size_inches(3.480, 2.334)
    fig.savefig("out/figure_1/fig1b.svg")


if __name__ == "__main__":
    main()
