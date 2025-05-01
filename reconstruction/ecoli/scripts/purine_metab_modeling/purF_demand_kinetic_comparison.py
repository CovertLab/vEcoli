import matplotlib.pyplot as plt
import os

from model_purine_metabolism import OUTPUT_DIR

PURF_DEMAND = {
    "minimal": 0.378,
    "plus aas only": 0.756,
    "acetate": 0.225,
    "succinate": 0.321,
}

PURF_MONOMER_CONC = {
    "minimal": 1.06e-3,
    "plus aas only": 4.43e-4,
    "acetate": 9.53e-4,
    "succinate": 1.08e-3,
}

PPGPP_CONC = {
    "minimal": 49.2e-6,
    "plus aas only": 20.6e-6,
    "acetate": 120.6e-6,
    "succinate": 92.9e-6,
}
# PPGPP_CONC = {
#     "minimal": 15e-6,
#     "plus aas only": 6e-6,
#     "acetate": 26e-6,
#     "succinate": 16e-6
# }
# 16.4 * 10^8 aas/per cell at 40min, then ppGpp conc at 4.3pmol / 10^17 aas,
# so that's 4.3 * 16.4 * 10^8 / 10^17 pmol / cell = 7.052e-9 pmol/cell * 1 cell/1fL = 7.052 e-9 pmol/fL = 7.052 uM

GLN_CONC = {"minimal": 3.81e-3}
GLN_REL_CONC = {
    "minimal": 1.0,
    "plus aas only": 1.498,
    "acetate": 1.086,
    "succinate": 1.724,
}
for x in GLN_REL_CONC:
    GLN_CONC[x] = GLN_CONC["minimal"] * GLN_REL_CONC[x]

PRPP_CONC = {
    "minimal": 2.58e-4,
    "plus aas only": 2.58e-4,
    "acetate": 2.58e-4,
    "succinate": 2.58e-4,
}
R5P_REL_CONC = {
    "minimal": 1.0,
    "plus aas only": 1.342,
    "acetate": 0.29,
    "succinate": 0.382,
}
for x in R5P_REL_CONC:
    PRPP_CONC[x] = PRPP_CONC["minimal"] * R5P_REL_CONC[x]

AMP_CONC = {"minimal": 2.10e-4}
AMP_REL_CONC = {
    "minimal": 1.0,
    "plus aas only": 0.768,
    "acetate": 0.883,
    "succinate": 0.795,
}

for x in AMP_REL_CONC:
    AMP_CONC[x] = AMP_CONC["minimal"] * AMP_REL_CONC[x]

GMP_CONC = {"minimal": 1.92e-5}
GMP_REL_CONC = {
    "minimal": 1.0,
    "acetate": 0.84,
    "succinate": 0.743,
    "plus aas only": 1.559,
}

for x in GMP_REL_CONC:
    GMP_CONC[x] = GMP_CONC["minimal"] * GMP_REL_CONC[x]

# Kinetic parameters
KM_GLN = 2.1e-3
KM_PRPP = 60e-6
KI_AMP = 1000e-6
KI_GMP = 220e-6
KI_GA = 64e-6
KI_PPGPP = 50e-6
HA = 2.0
HG = 4.5
HP = 2.5


def plot():
    def kinetics(amp, gmp, ppgpp, prpp, gln):
        gln_factor = gln / (gln + KM_GLN)
        comp_factor = prpp / (
            prpp + KM_PRPP * (1 + (amp / KI_AMP) ** HA + (ppgpp / KI_PPGPP))
        )
        noncomp_factor = (prpp / KM_PRPP) ** HP / (
            (prpp / KM_PRPP) ** HP
            + (gmp / KI_GMP) ** HG
            + (amp / KI_AMP) ** HA * (gmp / KI_GA) ** HG
        )

        return gln_factor * comp_factor * noncomp_factor

    vs = {}
    vs_max = {}
    vs_const_prpp = {}
    for media in PURF_DEMAND:
        amp = AMP_CONC[media]
        gmp = GMP_CONC[media]
        ppgpp = PPGPP_CONC[media]
        prpp = PRPP_CONC[media]
        gln = GLN_CONC[media]

        purF = PURF_MONOMER_CONC[media]

        vs[media] = kinetics(amp, gmp, ppgpp, prpp, gln) * purF
        vs_const_prpp[media] = (
            kinetics(amp, gmp, ppgpp, PRPP_CONC["minimal"], gln) * purF
        )
        vs_max[media] = purF

    rel_demand = {
        media: PURF_DEMAND[media] / PURF_DEMAND["minimal"] for media in PURF_DEMAND
    }
    rel_vs = {media: vs[media] / vs["minimal"] for media in PURF_DEMAND}
    rel_vs_const_prpp = {
        media: vs_const_prpp[media] / vs["minimal"] for media in PURF_DEMAND
    }
    rel_vs_max = {media: vs_max[media] / vs["minimal"] for media in PURF_DEMAND}

    fig, axs = plt.subplots(2, figsize=(8, 12))
    media_colors = {
        "minimal": "black",
        "plus aas only": "blue",
        "acetate": "green",
        "succinate": "red",
    }
    for media in rel_demand:
        axs[0].scatter(
            rel_demand[media], rel_vs[media], color=media_colors[media], label=media
        )
        axs[0].scatter(
            rel_demand[media],
            rel_vs_max[media],
            color=media_colors[media],
            marker="x",
            label=f"{media} (max)",
        )
        axs[0].scatter(
            rel_demand[media],
            rel_vs_const_prpp[media],
            color=media_colors[media],
            marker="*",
            label=f"{media} (minimal media PRPP)",
        )
    axs[0].set_title("PurF demand vs. PurF kinetic rate (relative)", size=18)
    axs[0].set_xlabel("Relative demand", size=18)
    axs[0].set_ylabel("Relative PurF kinetic rate", size=18)
    axs[0].set_xlim(0, 3)
    axs[0].set_ylim(0, 3)
    axs[0].plot([0, 3], [0, 3], color="black", linestyle="--")
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "PurF kinetic vs. demand plot.png"))
    plt.close("all")


if __name__ == "__main__":
    plot()
