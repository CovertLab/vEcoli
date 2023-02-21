import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import variation

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR

GLUCOSE_DATA = "data/glc_10000/2022-12-08_00-35-28_562633+0000.pkl"
TET_DATA = "data/tet_0/2023-01-05_01-00-44_215314+0000.pkl"
AMP_DATA = "data/amp_0/2 mg_L/2022-12-08_17-03-56_357734+0000.pkl"
PERIPLASMIC_VOLUME_FRACTION = 0.2


def calculate_glucose_metrics(data, out="out/figure_2/glc_metrics.txt"):
    metrics = {}
    max_time = data.Time.max()

    # get birth, end time for each agent (excluding those present at end
    # of simulation)
    agent_lifetimes = data.groupby("Agent ID").agg(
        birth_time=("Time", "min"), end_time=("Time", "max")
    )

    # get lifetime for each agent, which generation it was in
    agent_lifetimes["gen_time"] = agent_lifetimes.end_time - agent_lifetimes.birth_time
    agent_lifetimes["generation"] = agent_lifetimes.index.map(len)

    # get mean, std. dev generation time
    non_final_agent_lifetimes = agent_lifetimes[
        data.groupby("Agent ID").Time.max() != max_time
    ]
    metrics[
        "Mean doubling time"
    ] = f"{non_final_agent_lifetimes.gen_time.mean() / 60} min"
    metrics[
        "Std dev doubling time"
    ] = f"{non_final_agent_lifetimes.gen_time.std() / 60} min"

    # get standard deviation in start time for each generation
    for g, stddev, cv in zip(
        range(agent_lifetimes.generation.min(), agent_lifetimes.generation.max()),
        agent_lifetimes.groupby("generation").birth_time.std(),
        agent_lifetimes.groupby("generation").birth_time.agg(variation),
    ):
        metrics[
            f"Standard deviation in start time of generation {g}"
        ] = f"{stddev / 60} min"
        metrics[f"CV in start time of generation {g}"] = cv

    # Get average proportion of cell lifetime in which ompF, tolC, ampC, or marR mRNAs are ever zero
    def prop_zero(series):
        return (series.values == 0).mean()

    def all_zero(series):
        return all(series.values == 0)

    prop_lifetime_zero = (
        data.groupby("Agent ID").agg(
            ompF=("ompF mRNA", prop_zero),
            tolC=("tolC mRNA", prop_zero),
            ampC=("ampC mRNA", prop_zero),
            marR=("marR mRNA", prop_zero),
        )
    ).mean()

    prop_all_zero = (
        data.groupby("Agent ID").agg(
            ompF=("ompF mRNA", all_zero),
            tolC=("tolC mRNA", all_zero),
            ampC=("ampC mRNA", all_zero),
            marR=("marR mRNA", all_zero),
        )
    ).mean()

    for mrna in ["ompF", "tolC", "ampC", "marR"]:
        metrics[
            f"Average proportion of cell lifetime in which {mrna} mRNA is zero"
        ] = prop_lifetime_zero[mrna]

        metrics[f"Proportion of cells in which {mrna} was always zero"] = prop_all_zero[
            mrna
        ]

    # End-point expression
    monomers = ["OmpF monomer", "TolC monomer", "AmpC monomer", "MarR monomer"]
    endpoint_expression = variation(data[data.Time == max_time][monomers])

    for i, monomer in enumerate(monomers):
        metrics[f"CV for endpoint count of {monomer}"] = endpoint_expression[i]

    # Output results
    with open(out, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric} = {value}\n")


def calculate_tet_metrics(data, out="out/figure_3/tet_metrics.txt"):
    metrics = {}

    min_time = data.Time.min()

    # Mean, std dev in OmpF, AcrAB-TolC, active ribosome, micF-ompF duplex concentration
    # (immediately when tet added)
    data_t0 = data[data.Time == min_time]
    for molecule, periplasmic in {
        "OmpF complex": True,
        "AcrAB-TolC": True,
        "Active ribosomes": False,
        "micF-ompF duplex": False,
    }.items():
        molecule_t0 = (
            (
                data_t0[molecule]
                / (
                    (
                        PERIPLASMIC_VOLUME_FRACTION
                        if periplasmic
                        else 1 - PERIPLASMIC_VOLUME_FRACTION
                    )
                    * data_t0["Volume"]
                )
            )
            * COUNTS_PER_FL_TO_NANOMOLAR
            / 10**3
        )
        metrics[
            f"Mean {molecule} {'periplasmic' if periplasmic else 'cytoplasmic'} concentration at instant tet added (μM)"
        ] = molecule_t0.mean()
        metrics[
            f"Std. dev. in {molecule} {'periplasmic' if periplasmic else 'cytoplasmic'} concentration at instant tet added (μM)"
        ] = molecule_t0.std()

    # Time to reach equilbrium in periplasmic, cytoplasmic tet conc
    epsilon = 1e-7
    DT = 2

    # (Throwing away first few seconds to account for time to actually start importing tetracycline)
    discard = 5
    metrics[
        f"Periplasmic tet equilibration time (abs(d[Tet]) <= {epsilon * 10**6:.1e} nM) (seconds)"
    ] = (
        (
            np.abs(
                data[data["Time"] <= min_time + 2 * 60]
                .groupby("Time")["Periplasmic tetracycline"]
                .mean()
                .diff()
            )
            < 1e-7
        )
        .values[discard:]
        .argmax()
        + discard
    ) * DT

    metrics[
        f"Cytoplasmic tet equilibration time (abs(d[Tet]) <= {epsilon * 10**6:.1e} nM) (seconds)"
    ] = (
        (
            np.abs(
                data[data["Time"] <= min_time + 2 * 60]
                .groupby("Time")["Cytoplasmic tetracycline"]
                .mean()
                .diff()
            )
            < 1e-7
        )
        .values[discard:]
        .argmax()
        + discard
    ) * DT

    # Mean, std. dev. ribosome concentration
    # 1.5 (2? may line up with a division event, have outlier(s)) minutes after
    # tet added
    data_t1_5 = data[data.Time == data.Time.min() + 1.5 * 60]

    ribosomes_t1_5 = (
        (
            data_t1_5["Active ribosomes"]
            / ((1 - PERIPLASMIC_VOLUME_FRACTION) * data_t1_5["Volume"])
        )
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 10**3
    )
    metrics[
        "Mean ribosome concentration 1.5 mins after tet added (μM)"
    ] = ribosomes_t1_5.mean()
    metrics[
        "Std. dev. in ribosome concentration 1.5 mins after tet added (μM)"
    ] = ribosomes_t1_5.std()

    # Mean, std. dev. in micF-ompF duplex concentration
    # 10 minutes after tet added
    data_t10 = data[data.Time == data.Time.min() + 10 * 60]

    micF_ompF_t10 = (
        (
            data_t10["micF-ompF duplex"]
            / ((1 - PERIPLASMIC_VOLUME_FRACTION) * data_t10["Volume"])
        )
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 10**3
    )
    metrics[
        "Mean micF-ompF duplex concentration 10 mins after tet added (μM)"
    ] = micF_ompF_t10.mean()
    metrics[
        "Std. dev. in micF-ompF duplex concentration 10 mins after tet added (μM)"
    ] = micF_ompF_t10.std()

    # "Half-life" of ompF monomer, mRNA
    data["OmpF monomer conc"] = (
        data["OmpF monomer"] / (PERIPLASMIC_VOLUME_FRACTION * data.Volume)
    ) * COUNTS_PER_FL_TO_NANOMOLAR
    data["ompF mRNA conc"] = (
        data["ompF mRNA"] / (PERIPLASMIC_VOLUME_FRACTION * data.Volume)
    ) * COUNTS_PER_FL_TO_NANOMOLAR

    ompf_monomer_trace = data.groupby("Time")["OmpF monomer conc"].mean()
    ompf_mRNA_trace = data.groupby("Time")["ompF mRNA conc"].mean()

    metrics["OmpF monomer half-life (min)"] = (
        np.argmin(np.abs(ompf_monomer_trace - ompf_monomer_trace[min_time] / 2))
        * DT
        / 60
    )

    metrics["ompF mRNA half-life (min)"] = (
        np.argmin(np.abs(ompf_mRNA_trace - ompf_mRNA_trace[min_time] / 2)) * DT / 60
    )

    # Output results
    with open(out, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric} = {value}\n")


def calculate_amp_metrics(data, out="out/figure_4/amp_metrics.txt"):
    pass


def main():
    print("Calculating glucose metrics...")
    with open(GLUCOSE_DATA, "rb") as f:
        glc_data = pickle.load(f)
    calculate_glucose_metrics(glc_data)
    del glc_data

    print("Calculating tetracycline metrics...")
    with open(TET_DATA, "rb") as f:
        tet_data = pickle.load(f)
    calculate_tet_metrics(tet_data)
    del tet_data

    print("Calculating ampicillin metrics...")
    with open(AMP_DATA, "rb") as f:
        amp_data = pickle.load(f)
    calculate_amp_metrics(amp_data)
    del amp_data


if __name__ == "__main__":
    main()
