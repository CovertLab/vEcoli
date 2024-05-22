import ast
import os

import numpy as np
import pandas as pd
from scipy.stats import variation

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR

GLUCOSE_DATA = "data/colony_data/sim_dfs/2022-12-08_00-35-28_562633+0000.csv"
TET_DATA = "data/colony_data/sim_dfs/2023-01-05_01-00-44_215314+0000.csv"
AMP_DATA = "data/colony_data/sim_dfs/2022-12-08_17-03-56_357734+0000.csv"
PERIPLASMIC_VOLUME_FRACTION = 0.2


def calculate_glucose_metrics(
    data, out="out/analysis/paper_figures/metrics/glc_metrics.txt"
):
    metrics = {}
    max_time = data.Time.max()

    # get birth, end time for each agent
    agent_lifetimes = data.groupby("Agent ID").agg(
        birth_time=("Time", "min"), end_time=("Time", "max")
    )

    # get lifetime for each agent, which generation it was in
    agent_lifetimes["gen_time"] = agent_lifetimes.end_time - agent_lifetimes.birth_time
    agent_lifetimes["generation"] = agent_lifetimes.index.map(len)

    # get mean, std. dev generation time
    # exclude cells present at end of sim and those that died
    non_final_agent_lifetimes = agent_lifetimes[
        [
            (agent_id + "0" in agent_lifetimes.index)
            for agent_id in agent_lifetimes.index
        ]
    ]
    metrics["Mean doubling time"] = (
        f"{non_final_agent_lifetimes.gen_time.mean() / 60} min"
    )
    metrics["Std dev doubling time"] = (
        f"{non_final_agent_lifetimes.gen_time.std() / 60} min"
    )

    # get standard deviation in start time for each generation
    for g, stddev, cv in zip(
        range(agent_lifetimes.generation.min(), agent_lifetimes.generation.max()),
        agent_lifetimes.groupby("generation").birth_time.std(),
        agent_lifetimes.groupby("generation").birth_time.agg(variation),
    ):
        metrics[f"Standard deviation in start time of generation {g}"] = (
            f"{stddev / 60} min"
        )
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
        metrics[f"Average proportion of cell lifetime in which {mrna} mRNA is zero"] = (
            prop_lifetime_zero[mrna]
        )

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


def calculate_tet_metrics(
    data, out="out/analysis/paper_figures/metrics/tet_metrics.txt"
):
    metrics = {}

    min_time = data.Time.min()

    # Mean, std dev, CV in OmpF, AcrAB-TolC, active ribosome, micF-ompF duplex
    # concentrations when tet first added
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
        metrics[
            f"CV in {molecule} {'periplasmic' if periplasmic else 'cytoplasmic'} concentration at instant tet added (μM)"
        ] = molecule_t0.std() / molecule_t0.mean()

    # Mean, std. dev., CV in outer membrane permeability when tet. first added
    om_perm = data.loc[data.Time == min_time, "Outer tet. permeability (cm/s)"]
    metrics["Mean OM perm. when tet. first added (nm/s)"] = om_perm.mean() * 1e7
    metrics["Std. dev. OM perm. when tet. first added (nm/s)"] = om_perm.std() * 1e7
    metrics["CV OM perm. when tet. first added"] = om_perm.std() / om_perm.mean()

    # Mean, std. dev., CV in outer membrane permeability 4 hours post-tet.
    om_perm = data.loc[
        data.Time == min_time + 3600 * 4, "Outer tet. permeability (cm/s)"
    ]
    metrics["Mean OM perm. 4 hrs after tet. added (nm/s)"] = om_perm.mean() * 1e7
    metrics["Std. dev. OM perm. 4 hrs after tet. added (nm/s)"] = om_perm.std() * 1e7
    metrics["CV OM perm. 4 hrs after tet. added"] = om_perm.std() / om_perm.mean()

    # Mean, std. dev., CV in periplasmic tetracycline conc. 8 sec after tet. added
    peri_tet = data.loc[data.Time == min_time + 8, "Periplasmic tetracycline"]
    metrics[
        "Mean periplasmic tetracycline concentration 8 sec. after tet. added (μM)"
    ] = peri_tet.mean()
    metrics[
        "Std. dev. periplasmic tetracycline concentration 8 sec. after tet. added (μM)"
    ] = peri_tet.std()
    metrics["CV periplasmic tetracycline concentration 8 sec. after tet. added"] = (
        peri_tet.std() / peri_tet.mean()
    )

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

    # Mean, std. dev. ribosome concentration 2 minutes after tet added
    data_t2 = data[data.Time == data.Time.min() + 2 * 60]

    ribosomes_t2 = (
        (
            data_t2["Active ribosomes"]
            / ((1 - PERIPLASMIC_VOLUME_FRACTION) * data_t2["Volume"])
        )
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 10**3
    )
    metrics["Mean ribosome concentration 2 mins after tet added (μM)"] = (
        ribosomes_t2.mean()
    )
    metrics["Std. dev. in ribosome concentration 2 mins after tet added (μM)"] = (
        ribosomes_t2.std()
    )

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
    metrics["Mean micF-ompF duplex concentration 10 mins after tet added (μM)"] = (
        micF_ompF_t10.mean()
    )
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


def calculate_amp_metrics(
    data, out="out/analysis/paper_figures/metrics/amp_metrics.txt"
):
    metrics = {}

    min_time = data.Time.min()
    data_t60 = data[data.Time == min_time + 60]

    # Mean, std. dev., CV in AmpC concentration 1 min. post-amp.
    ampc_data = (
        data_t60["AmpC monomer"]
        / (data_t60["Volume"] * PERIPLASMIC_VOLUME_FRACTION)
        * COUNTS_PER_FL_TO_NANOMOLAR
        / 1000
    )
    metrics["Mean AmpC when amp. first added (μM)"] = ampc_data.mean()
    metrics["Std. dev. AmpC when amp. first added (μM)"] = ampc_data.std()
    metrics["CV AmpC when amp. first added"] = ampc_data.std() / ampc_data.mean()

    # Mean, std. dev., CV in ampicillin 1 min. post-amp.
    amp_conc = data_t60["Periplasmic ampicillin"]
    metrics["Mean amp. 1 min after amp. added (μM)"] = amp_conc.mean() * 1000
    metrics["Std. dev. amp. 1 min after amp. added (μM)"] = amp_conc.std() * 1000
    metrics["CV amp. 1 min after amp. added"] = amp_conc.std() / amp_conc.mean()

    # Mean, std. dev., CV in PBP1A active frac. 1 min. post-amp.
    pbp1a = data_t60["Active fraction PBP1a"]
    metrics["Mean PBP1A active frac. 1 min after amp. added"] = pbp1a.mean()
    metrics["Std. dev. PBP1A active frac. 1 min after amp. added"] = pbp1a.std()
    metrics["CV PBP1A active frac. 1 min after amp. added"] = pbp1a.std() / pbp1a.mean()

    # Mean, std. dev., CV in PBP1B active frac. 1 min. post-amp.
    pbp1b = data_t60["Active fraction PBP1b"]
    metrics["Mean PBP1B active frac. 1 min after amp. added"] = pbp1b.mean()
    metrics["Std. dev. PBP1B active frac. 1 min after amp. added"] = pbp1b.std()
    metrics["CV PBP1B active frac. 1 min after amp. added"] = pbp1b.std() / pbp1b.mean()

    # Mean, std. dev., CV in ampicillin 15 min. post-amp.
    amp_conc = data.loc[data.Time == min_time + 60 * 15, "Periplasmic ampicillin"]
    metrics["Mean amp. 15 min after amp. added (μM)"] = amp_conc.mean() * 1000
    metrics["Std. dev. amp. 15 min after amp. added (μM)"] = amp_conc.std() * 1000
    metrics["CV amp. 15 min after amp. added"] = amp_conc.std() / amp_conc.mean()

    # Output results
    with open(out, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric} = {value}\n")


def main():
    os.makedirs("out/analysis/paper_figures/metrics/", exist_ok=True)
    print("Calculating glucose metrics...")
    glc_data = pd.read_csv(
        GLUCOSE_DATA, dtype={"Agent ID": str, "Seed": str}, index_col=0
    )
    glc_data["Boundary"] = glc_data["Boundary"].apply(ast.literal_eval)
    calculate_glucose_metrics(glc_data)
    del glc_data

    print("Calculating tetracycline metrics...")
    tet_data = pd.read_csv(TET_DATA, dtype={"Agent ID": str, "Seed": str}, index_col=0)
    tet_data["Boundary"] = tet_data["Boundary"].apply(ast.literal_eval)
    calculate_tet_metrics(tet_data)
    del tet_data

    print("Calculating ampicillin metrics...")
    amp_data = pd.read_csv(AMP_DATA, dtype={"Agent ID": str, "Seed": str}, index_col=0)
    amp_data["Boundary"] = amp_data["Boundary"].apply(ast.literal_eval)
    calculate_amp_metrics(amp_data)
    del amp_data


if __name__ == "__main__":
    main()
