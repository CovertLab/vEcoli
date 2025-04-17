import os
import csv

import matplotlib.pyplot as plt

this_file = os.path.dirname(os.path.realpath(__file__))
Li_2014_file = os.path.join(this_file, "Li_2014_protein_counts.tsv")
wcm_monomer_counts_dir = os.path.join(this_file, "wcm_full_monomers_for_mica")

OUTPUT_DIR = os.path.join(this_file, "plots")


def get_protein_counts_Li():
    gene_counts = []
    gene_names = []
    with open(Li_2014_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for line in reader:
            gene_names.append(line[0])
            minimal_count = line[2]
            if minimal_count[0] == "[":
                minimal_count = minimal_count[1:-1]
            gene_counts.append(int(minimal_count))

    # total_counts = np.sum(gene_counts)


def get_protein_count_distr_WCM(plot_name):
    protein_name = "AIRS-MONOMER"

    name_1 = "wcm_full_monomers_var_0_seed_"
    name_2 = "_gen_"
    name_3 = ".csv"

    avg_counts = []
    for seed in range(64):
        if seed == 61:
            continue
        for gen in [2, 3]:
            name = name_1 + str(seed) + name_2 + str(gen) + name_3
            file = os.path.join(wcm_monomer_counts_dir, name)

            counts = 0
            num_timesteps = 0
            with open(file, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                protein_idx = header.index(protein_name)
                for line in reader:
                    counts += float(line[protein_idx])
                    num_timesteps += 1

            avg_counts.append(counts / num_timesteps)

    fig, axs = plt.subplots(2, figsize=(5, 10))
    axs[0].hist(avg_counts)
    axs[0].set_xlabel("Average counts across cell cycle of " + protein_name)
    axs[0].set_ylabel("Number of simulations")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, plot_name))
    plt.close("all")


if __name__ == "__main__":
    get_protein_count_distr_WCM("purM_histogram")
