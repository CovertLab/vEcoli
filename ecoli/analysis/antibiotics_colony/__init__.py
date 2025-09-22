import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import N_A
from wholecell.utils.filepath import ROOT_PATH

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"

try:
    DE_GENES = pd.read_csv(os.path.join(ROOT_PATH, "data/marA_binding/gene_fc.csv"))
except FileNotFoundError:
    raise FileNotFoundError(
        "Please run data/marA_binding/get_TU_ID.py first to generate the gene_fc.csv file."
    )
SPLIT_TIME = 11550
MAX_TIME = 26000
COUNTS_PER_FL_TO_NANOMOLAR = 1 / (1e-15) / N_A * (1e9)

# Average generation time (average of all cells except those
# in final generation of simulation IDs: '2022-12-08_00-33-56_581605+0000',
# '2022-12-08_00-35-12_754291+0000', '2022-12-08_00-35-28_562633+0000')
AVG_GEN_TIME = 3137.0669216061187

# Mapping: Condition -> Seed -> Experiment ID
EXPERIMENT_ID_MAPPING = {
    "Glucose": {
        0: "2022-12-08_00-33-56_581605+0000",
        100: "2022-12-08_00-35-12_754291+0000",
        10000: "2022-12-08_00-35-28_562633+0000",
    },
    "Tetracycline (1.5 mg/L)": {
        0: "2023-01-05_01-00-44_215314+0000",
        100: "2023-01-05_01-00-52_399146+0000",
        10000: "2023-01-05_01-01-02_513996+0000",
    },
    "Tetracycline (4 mg/L)": {0: "2023-01-05_01-01-09_970526+0000"},
    "Tetracycline (2 mg/L)": {0: "2023-01-05_01-01-16_783140+0000"},
    "Tetracycline (1 mg/L)": {
        0: "2023-01-05_01-01-23_555787+0000",
    },
    "Tetracycline (0.5 mg/L)": {
        0: "2023-01-05_01-01-32_083738+0000",
    },
    "Ampicillin (2 mg/L)": {
        0: "2022-12-08_17-03-56_357734+0000",
        100: "2022-12-08_17-04-20_544970+0000",
        10000: "2022-12-08_17-04-52_137662+0000",
    },
    "Ampicillin (4 mg/L)": {
        0: "2022-12-08_17-08-04_777218+0000",
    },
    "Ampicillin (1.5 mg/L)": {
        0: "2022-12-08_17-07-14_437731+0000",
    },
    "Ampicillin (1 mg/L)": {
        0: "2022-12-08_17-06-35_367185+0000",
    },
    "Ampicillin (0.5 mg/L)": {
        0: "2022-12-08_19-13-14_431590+0000",
    },
}

PATHS_TO_LOAD = {
    "Dry mass": ("listeners", "mass", "dry_mass"),
    "Protein mass": ("listeners", "mass", "proteinMass"),
    "Dry mass fold change": ("listeners", "mass", "dryMassFoldChange"),
    "Protein mass fold change": ("listeners", "mass", "proteinMassFoldChange"),
    "RNA mass fold change": ("listeners", "mass", "rnaMassFoldChange"),
    "Small molecule fold change": ("listeners", "mass", "smallMoleculeFoldChange"),
    "Cell mass": ("listeners", "mass", "cell_mass"),
    "Water mass": ("listeners", "mass", "water_mass"),
    "RNA mass": ("listeners", "mass", "rnaMass"),
    "rRNA mass": ("listeners", "mass", "rRnaMass"),
    "tRNA mass": ("listeners", "mass", "tRnaMass"),
    "mRNA mass": ("listeners", "mass", "mRnaMass"),
    "DNA mass": ("listeners", "mass", "dnaMass"),
    "Small molecule mass": ("listeners", "mass", "smallMoleculeMass"),
    "Projection mass": ("listeners", "mass", "projection_mass"),
    "Cytosol mass": ("listeners", "mass", "cytosol_mass"),
    "Extracellular mass": ("listeners", "mass", "extracellular_mass"),
    "Flagellum mass": ("listeners", "mass", "flagellum_mass"),
    "Membrane mass": ("listeners", "mass", "membrane_mass"),
    "Outer membrane mass": ("listeners", "mass", "outer_membrane_mass"),
    "Periplasm mass": ("listeners", "mass", "periplasm_mass"),
    "Pilus mass": ("listeners", "mass", "pilus_mass"),
    "Inner membrane mass": ("listeners", "mass", "inner_membrane_mass"),
    "Growth rate": ("listeners", "mass", "growth"),
    "AcrAB-TolC": ("bulk", "TRANS-CPLX-201[m]"),
    "Periplasmic tetracycline": ("periplasm", "concentrations", "tetracycline"),
    "Cytoplasmic tetracycline": ("cytoplasm", "concentrations", "tetracycline"),
    "Periplasmic ampicillin": ("periplasm", "concentrations", "ampicillin"),
    "Active MarR": ("bulk", "CPLX0-7710[c]"),
    "Inactive MarR": ("bulk", "marR-tet[c]"),
    "micF-ompF duplex": ("bulk", "micF-ompF[c]"),
    "micF RNA": (
        "bulk",
        "MICF-RNA[c]",
    ),
    "30S subunit": ("bulk", "CPLX0-3953[c]"),
    "Inactive 30S subunit": ("bulk", "CPLX0-3953-tetracycline[c]"),
    "Active ribosomes": ("listeners", "aggregated", "active_ribosome_len"),
    "Active RNAP": ("listeners", "aggregated", "active_RNAP_len"),
    "Outer tet. permeability (cm/s)": (
        "kinetic_parameters",
        "outer_tetracycline_permeability",
    ),
    "Murein tetramer": ("bulk", "CPD-12261[p]"),
    "PBP1a complex": ("bulk", "CPLX0-7717[p]"),
    "PBP1a mRNA": ("mrna", "EG10748_RNA"),
    "PBP1b alpha complex": ("bulk", "CPLX0-3951[i]"),
    "PBP1b mRNA": ("mrna", "EG10605_RNA"),
    "PBP1b gamma complex": ("bulk", "CPLX0-8300[c]"),
    "Wall cracked": ("wall_state", "cracked"),
    "AmpC monomer": ("monomer", "EG10040-MONOMER"),
    "ampC mRNA": ("mrna", "EG10040_RNA"),
    "Extension factor": ("wall_state", "extension_factor"),
    "Wall columns": ("wall_state", "lattice_cols"),
    "Unincorporated murein": ("murein_state", "unincorporated_murein"),
    "Incorporated murein": ("murein_state", "incorporated_murein"),
    "Shadow murein": ("murein_state", "shadow_murein"),
    "Max hole size": ("listeners", "hole_size_distribution"),
    "Porosity": ("listeners", "porosity"),
    "Active fraction PBP1a": ("pbp_state", "active_fraction_PBP1A"),
    "Active fraction PBP1b": ("pbp_state", "active_fraction_PBP1B"),
    "Boundary": ("boundary",),
    "Volume": ("listeners", "mass", "volume"),
    "Total mRNA": ("total_mrna",),
    "OmpF complex": ("bulk", "CPLX0-7534[o]"),
    "Exchanges": ("environment", "exchange"),
}
for gene_data in DE_GENES[["Gene name", "id", "monomer_ids"]].values:
    if gene_data[0] != "MicF":
        PATHS_TO_LOAD[f"{gene_data[0]} mRNA"] = ("mrna", gene_data[1])
        PATHS_TO_LOAD[f"{gene_data[0]} synth prob"] = ("rna_synth_prob", gene_data[1])
    gene_data[2] = eval(gene_data[2])
    if len(gene_data[2]) > 0:
        monomer_name = gene_data[0][0].upper() + gene_data[0][1:]
        PATHS_TO_LOAD[f"{monomer_name} monomer"] = ("monomer", gene_data[2][0])
# Housekeeping gene GAPDH for normalization between samples
PATHS_TO_LOAD["GAPDH mRNA"] = ("mrna", "EG10367_RNA")
PATHS_TO_LOAD["GAPDH synth prob"] = ("rna_synth_prob", "EG10367_RNA")
PATHS_TO_LOAD["GAPDH monomer"] = ("monomer", "GAPDH-A-MONOMER")
# RNAP monomers and mRNAs
PATHS_TO_LOAD["rpoA mRNA"] = ("mrna", "EG10893_RNA")
PATHS_TO_LOAD["rpoB mRNA"] = ("mrna", "EG10894_RNA")
PATHS_TO_LOAD["rpoC mRNA"] = ("mrna", "EG10895_RNA")
PATHS_TO_LOAD["RpoA monomer"] = ("monomer", "EG10893-MONOMER")
PATHS_TO_LOAD["RpoB monomer"] = ("monomer", "RPOB-MONOMER")
PATHS_TO_LOAD["RpoC monomer"] = ("monomer", "RPOC-MONOMER")


def restrict_data(data: pd.DataFrame):
    """If there is more than one condition in data, keep up
    to SPLIT_TIME from the first condition and between SPLIT_TIME
    and MAX_TIME from the the rest."""
    conditions = data.loc[:, "Condition"].unique()
    if len(conditions) > 1:
        data = data.set_index(["Condition"])
        condition_1_mask = data.loc[conditions[0]]["Time"] <= SPLIT_TIME
        filtered_data = [data.loc[conditions[0]].loc[condition_1_mask, :]]
        for exp_condition in conditions[1:]:
            condition_mask = (data.loc[exp_condition]["Time"] >= SPLIT_TIME) & (
                data.loc[exp_condition]["Time"] <= MAX_TIME
            )
            filtered_data.append(data.loc[exp_condition].loc[condition_mask, :])
        data = pd.concat(filtered_data)
        data = data.reset_index()
    else:
        data = data.loc[data.loc[:, "Time"] <= MAX_TIME, :]
    return data
