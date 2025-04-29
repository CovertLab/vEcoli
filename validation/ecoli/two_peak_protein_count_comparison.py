import os
import csv

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")

SCHMIDT_FILE = os.path.join(
    os.path.join(BASE_DIR, "flat"), "schmidt2015_javier_table.tsv"
)
SCHMIDT_CONDITIONS_FILE = os.path.join(
    os.path.join(BASE_DIR, "flat"), "schmidt2015_table_conditions.tsv"
)

SIM_CONDITIONS = ["minimal", "rich", "acetate", "succinate"]
SIM_DTS = {"minimal": 48, "rich": 23, "acetate": 80, "succinate": 55}
SIM_SIZES = {"minimal": 1.06, "rich": 2.54, "acetate": 0.35, "succinate": 0.72}
SIM_PURF_COUNTS = {"minimal": 968, "rich": 939, "acetate": 268, "succinate": 661}
SIM_PURF_CONCS = {c: SIM_PURF_COUNTS[c] / SIM_SIZES[c] for c in SIM_CONDITIONS}
SIM_PURF_CONCS_LIST = [SIM_PURF_CONCS[c] for c in SIM_CONDITIONS]
SIM_PURF_CONCS_LIST_NORM = [
    SIM_PURF_CONCS[c] / SIM_PURF_CONCS["minimal"] for c in SIM_CONDITIONS
]
SIM_DTS_LIST = [SIM_DTS[c] for c in SIM_CONDITIONS]

GENES = ["purF", "purC", "purD", "purH", "purM", "purN", "purT", "purE", "purK", "purL"]
CONDITIONS = [
    "Glucose",
    "Glycerol + AA",
    "Acetate",
    "Fumarate",
    "Glucosamine",
    "Glycerol",
    "Pyruvate",
    "Xylose",
    "Mannose",
    "Galactose",
    "Succinate",
    "Fructose",
]
# Note: all from the BW25113 strain
# Note: also from Table S6, coefficient of variation is in 5-20% range


def make_plot():
    protein_counts = {gene: {} for gene in GENES}
    condition_sizes = {}
    condition_dts = {}
    with open(SCHMIDT_FILE, "r") as file:
        csv_reader = csv.reader(file, delimiter="\t")
        csv_header = next(csv_reader)
        condition_to_idx = {condition: i for i, condition in enumerate(csv_header)}
        for line in csv_reader:
            if line[condition_to_idx["GeneName"]] in GENES:
                for condition in CONDITIONS:
                    protein_counts[line[1]][condition] = int(
                        line[condition_to_idx[condition]]
                    )

    with open(SCHMIDT_CONDITIONS_FILE, "r") as file:
        csv_reader = csv.reader(file, delimiter="\t")
        csv_header = next(csv_reader)
        name_to_idx = {name: i for i, name in enumerate(csv_header)}
        for line in csv_reader:
            if line[name_to_idx["Growth condition"]] in CONDITIONS:
                condition_sizes[line[name_to_idx["Growth condition"]]] = float(
                    line[name_to_idx["Single cell volume [fl]"]]
                )
                condition_dts[line[name_to_idx["Growth condition"]]] = (
                    float(line[name_to_idx["Doubling time (h)"]]) * 60
                )

    protein_concs = {}
    for gene in GENES:
        protein_concs[gene] = {}
        for condition in CONDITIONS:
            protein_concs[gene][condition] = (
                protein_counts[gene][condition] / condition_sizes[condition]
            )

    protein_concs_list = {
        gene: [protein_concs[gene][condition] for condition in CONDITIONS]
        for gene in GENES
    }
    protein_concs_list_norm = {
        gene: [
            protein_concs[gene][condition] / protein_concs[gene]["Glucose"]
            for condition in CONDITIONS
        ]
        for gene in GENES
    }
    condition_dts_list = [condition_dts[condition] for condition in CONDITIONS]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(
        condition_dts_list,
        protein_concs_list["purF"],
        label="PurF validation",
        color="blue",
    )
    axs[0].scatter(SIM_DTS_LIST, SIM_PURF_CONCS_LIST, label="PurF sim", color="red")
    axs[0].set_xlabel("Doubling time (min)")
    axs[0].set_ylabel("PurF concentration (counts/fL)")
    axs[0].set_xlim(0, 200)
    axs[0].set_ylim(0, 1200)
    for i, c in enumerate(CONDITIONS):
        axs[0].annotate(
            c, (condition_dts_list[i], protein_concs_list["purF"][i]), color="blue"
        )
    for i, c in enumerate(SIM_CONDITIONS):
        axs[0].annotate(c, (SIM_DTS_LIST[i], SIM_PURF_CONCS_LIST[i]), color="red")

    axs[1].scatter(
        condition_dts_list,
        protein_concs_list_norm["purF"],
        label="PurF validation",
        color="blue",
    )
    axs[1].scatter(
        SIM_DTS_LIST, SIM_PURF_CONCS_LIST_NORM, label="PurF sim", color="red"
    )
    axs[1].set_xlabel("Doubling time (min)")
    axs[1].set_ylabel("PurF concentration normalized to Glucose or Minimal")
    axs[1].set_xlim(0, 200)
    axs[1].set_ylim(0, 1.2)
    for i, c in enumerate(CONDITIONS):
        axs[1].annotate(
            c, (condition_dts_list[i], protein_concs_list_norm["purF"][i]), color="blue"
        )
    for i, c in enumerate(SIM_CONDITIONS):
        axs[1].annotate(c, (SIM_DTS_LIST[i], SIM_PURF_CONCS_LIST_NORM[i]), color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "protein_conc_validation.png"))
    plt.close("all")

    # 1. Get directory of validation, and read it into tables for relevant proteins in relevant conditions
    # 2. Get the cell sizes of each condition, to convert counts to concentrations
    # 3. Get the growth rates of each condition, to make a plot of concentration versus growth rate
    # 4. Get the concentrations of proteins from EcoCyc modeling tab, plot them against growth rate
    # 5. Normalize to the concentration at ~45 minute growth rate, and plot the two sets of concentrations together.
    # 6. Determine: if the curve that the EcoCyc modeling tab follows (about constant at slower than ~45 minute growth rate, but drops at higher),
    # is followed by the validation data (so they should be on similar lines), in particular if the concentration at higher growth rates
    # also drops for the validation data, then that's a good match and supports the two-peak assumption.
    # However, if they don't match (i.e. slower than ~45 minute growth rate there's signifcant changes in concentration, or it doesn't drop or
    # even rises at higher growth rates), then that does not support the two-peak assumption.


if __name__ == "__main__":
    make_plot()
