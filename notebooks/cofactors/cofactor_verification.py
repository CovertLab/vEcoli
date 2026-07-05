# %%
import os
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import warnings

mpl.rcParams["figure.dpi"] = 300

sns.set(style="darkgrid", palette="Set2", context="notebook")
warnings.filterwarnings(action="ignore")

# %%
path_complex_ids = "/Users/adrianjuarez/Documents/Covert_lab/repos/vEcoli/notebooks/cofactors/complex_ids.txt"
path_protein_ids = "/Users/adrianjuarez/Documents/Covert_lab/repos/vEcoli/notebooks/cofactors/protein_ids.txt"

base_dir = "/Users/adrianjuarez/Documents/Covert_lab/projects/Metals"
sim_sets = ["sims_set_1", "sims_set_2", "sims_set_3"]


# %%
complex_ids_truth = pd.read_csv(path_complex_ids)
protein_ids_truth = pd.read_csv(path_protein_ids)

# %%
print(complex_ids_truth["Complex_id"].iloc[0])
print(type(complex_ids_truth["Complex_id"].iloc[0]))
complex_names = complex_ids_truth["Complex_id"].tolist()
protein_names = protein_ids_truth["protein_id"].tolist()


# %%
def filter_counts(df, names):
    existing = [c for c in names if c in df.columns]
    missing = [c for c in names if c not in df.columns]
    out = df[["seed", "generation"] + existing].copy()
    for c in missing:
        out[c] = 0
    return out, missing


# %%
for s in sim_sets:
    set_dir = os.path.join(base_dir, s)
    for media in ["rich", "minimal"]:
        df = pd.read_csv(os.path.join(set_dir, f"complex_counts_{media}.tsv"), sep="\t")
        filtered_complexes, missing_complexes = filter_counts(df, complex_names)
        filtered_proteins, missing_proteins = filter_counts(df, protein_names)
        filtered_complexes.to_csv(os.path.join(set_dir, f"complexes_{media}.csv"))
        filtered_proteins.to_csv(os.path.join(set_dir, f"proteins_{media}.csv"))
        print(
            f"{s}/{media}: complexes missing {len(missing_complexes)} {missing_complexes}, "
            f"proteins missing {len(missing_proteins)} {missing_proteins}"
        )
