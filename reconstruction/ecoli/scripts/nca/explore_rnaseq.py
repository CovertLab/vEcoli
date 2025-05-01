import csv
import os
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from rnaseq_utils import *
from get_media_composition import MediaComps
from sklearn import mixture

DATA_DIR = os.path.join(BASE_DIR, "data")

# TODO: plot normal and other on same

# Sequencing related
WCM_FILE = os.path.join(DATA_DIR, "wcm_fold_changes.tsv")
COMPENDIUM_DIR = os.path.join(DATA_DIR, "compendium")
SAMPLES_FILE = os.path.join(COMPENDIUM_DIR, "samples.tsv")
GENE_NAMES_FILE = os.path.join(COMPENDIUM_DIR, "gene_names.tsv")
GENE_SYNONYMS_FILE = os.path.join(COMPENDIUM_DIR, "gene_synonyms.tsv")
SEQ_DIR = os.path.join(COMPENDIUM_DIR, "seq")
SEQ_FILES = [
    "EcoMAC.tsv",
    # 'RNAseqEcoMACFormat.tsv',
    # 'GSE29076.tsv',
    # 'GSE72525.tsv',
    # 'GSE55662.tsv',
    # 'GSE50529.tsv',
    # 'GSE55365aerobic.tsv',
    # 'GSE55365anaerobic.tsv',
]
PRECISE2_SEQ_DIR = os.path.join(DATA_DIR, "PRECISE2_data")
PRECISE2_SEQ_FILE = os.path.join(PRECISE2_SEQ_DIR, "PRECISE2_counts.csv")


# Output plots
SAMPLE_REG_DIR = os.path.join(OUTPUT_DIR, "detect_sample_regs")

# Constants for EM-GMM
MAX_ITER = 1000
CUTOFF_THRESHOLD = 1

# Replacing in curating sample_data
# Questioning the LB/M9 becoming LB. Also then the thiamine and glucose supplements shouldn't be there?
REPlACE_MEDIA = {
    "M10": "M9",
    "M11": "M9",
    "M12": "M9",
    "M13": "M9",
    "M14": "M9",
    "M15": "M9",
    "M9+MOPS": "MOPS",
    "M9 MOPS": "MOPS",
    "M9 ": "M9",
    "BHIB": "BHI",
    "DM": "Davis MM",
    "LB or DMEM": "DMEM",
    "LB/M9": "LB",
    "MOPS + M9": "MOPS",
}

REPLACE_STRAIN = {
    "DH5α": "DH5alpha",
    "BW25114": "BW25113",
    "BW25115": "BW25113",
    "BW25116": "BW25113",
    "EHEC": "EDL933",
    "O157H7": "O157:H7",
}


REPLACE_GSM_STRAIN = {
    "GSM540105": "rpoA27",
    "GSM540096": "P2",
    "GSM540101": "rpoD3",
    "GSM540100": "rpoD3",
    "GSM540106": "rpoA27",
    "GSM540098": "P2",
    "GSM540107": "rpoA27",
    "GSM540104": "rpoA14",
    "GSM540097": "P2",
    "GSM540103": "rpoA14",
    "GSM540099": "rpoD3",
    "GSM540102": "rpoA14",
}
# TODO: 10g/Lglu is glucose?
# TODO: 0mM iptg for WT, could replace with just treatment is ''
REPLACE_TREATMENT = {
    "100uMIPTG": "0.1mMiptg",
    "10tetracycline": "10ugtetracycline",
    "128tetracycline": "128ugtetracycline",
    "1mMIPTG": "1mMiptg",
    "2.7g/L glucose ": "2.7g/L glucose",
    "8g/Lglu ": "8g/Lglu",
    "not foundgluconate ": "not foundgluconate",
    "not foundmannitol ": "not foundmannitol",
    "5g/Lglu ": "5g/Lglu",
    "temperature biofilm DMF": "temperature DMF biofilm",
    "control": "",
    "negative control": "",
    "250ngnorf": "250 ng/mlnorfloxacin",
    "251 ng/ml 100uMnor chelator": "250 ng/ml 100uMnor chelator",
    "252 ng/ml 100uMnor chelator": "250 ng/ml 100uMnor chelator",
    "253 ng/ml 100uMnor chelator": "250 ng/ml 100uMnor chelator",
}

REPLACE_GENE = {"wt": "", "WT": "", "WTnoIPTG": ""}
# TODO: check 'gly' is indeed glyceorl for teh T. allen one
GENE_TO_TREATMENT = {"WTIPTG": "5mMiptg"}

# TODO: maybe merge exponential with mid-exponential (would have to change more,
# but actually changing mid-exponential to exponential, and maybe more, check)
REPLACE_TIME = {
    "log phase": "exponential",
    "mid log phase": "mid-exponential",
    "stat": "stationary",
    "mid-log phase": "mid-exponential",
    "WT": "mid-exponential",
    "mid-exponential phase": "mid-exponential",
}

# TODO: could try taking this out, to see if our data could predict it?
TIME_TO_TREATMENT = {"-300": "glu"}

MEDIA_COMPOSITIONS = {}


def load_data_exclude_rewiring(
    no_pathogen=True,
    no_evolved=True,
    no_rewiring=True,
    no_gene_perturb=True,
    no_nongrowing=True,
):
    # Load sequencing data related genes
    # b_numbers = []
    symbols = []
    with open(GENE_NAMES_FILE) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            # b_numbers.append(line[1])
            symbols.append(line[2])

    geos = []
    authors = []
    strains = []
    media = []
    treatments = []
    gene_perturb = []
    is_env = []
    is_genetic = []
    is_WT = []
    gr = []
    components = []

    included_sample = []
    with open(SAMPLES_FILE) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        geo_idx = header.index("GEO accession")
        cel_idx = header.index("CEL file name")
        author_idx = header.index("Author")
        strain_idx = header.index("Strain")
        medium_idx = header.index("Medium")
        treatment_idxs = (
            header.index("Treatment (value)"),
            header.index("Treatment (conditions)"),
            header.index("Time (min)"),
        )
        gene_perturb_idxs = (
            header.index("Gene perturbated"),
            header.index("Type of perturbation"),
        )
        perturb_flags = (
            header.index("Environmental perturbations"),
            header.index("Genetic perturbations"),
            header.index("Arrays in WT condition"),
        )
        gr_idx = header.index("Growth rate (1/h)")

        for i, line in enumerate(reader):
            if no_rewiring:
                if line[author_idx] == "M. Isalan":
                    continue

            if no_pathogen:
                disease_strains = [
                    "EDL933",
                    "86-24",
                    "VS94",
                    "CFT073",
                    "UTI89",
                    "EHEC",
                    "KMD",
                    "B41",
                    "APEC",
                ]
                if "O157" in line[strain_idx]:
                    continue
                if line[strain_idx] in disease_strains:
                    continue

            if no_evolved:
                if line[strain_idx] == "strain evolved":
                    continue
                if line[author_idx] == "S.S. Fong":
                    if line[treatment_idxs[2]] != "WT":
                        continue
                if line[author_idx] == "V. Portnoy":
                    continue
                if line[author_idx] == "D. Lee":
                    # Paper says evolved stress
                    continue

            if no_gene_perturb:
                gene_perturb_annotated_right = ["GSE21995", "GSE22829"]
                if line[perturb_flags[1]] == "1":
                    continue
                if line[strain_idx] == "P4XB2":
                    continue
                if line[author_idx] == "G. Vemuri":
                    if line[cel_idx] not in ["GSM510322", "GSM510323", "GSM510324"]:
                        continue
                if line[author_idx] == "D. Vinella":
                    # Mislabeled two genotype perturbations as not a genotype perturbation
                    continue
                if line[author_idx] == "J. Faith":
                    if line[cel_idx] in ["GSM157595", "GSM157596"]:
                        continue  # lucU overexpression, don't rly need these two bc lots of J. Faith normal LB samples already
                if line[author_idx] == "S. Durand":
                    if line[cel_idx] in ["GSM490262", "GSM490264", "GSM490266"]:
                        continue
                if line[author_idx] == "J. Wang":
                    if line[cel_idx] in ["GSM862032", "GSM862033", "GSM862034"]:
                        continue
                if line[author_idx] == "M. Caldara":
                    # Since the strain used has methionine biosynthesis gene deletion (even though methionine is added to medium)
                    # TODO: whether to still keep this?
                    continue
                if line[author_idx] == "C. Peano":
                    continue

            if no_nongrowing:
                if line[author_idx] == "F. Haddadin":
                    if line[treatment_idxs[2]] == "240":
                        continue
                    if line[gene_perturb_idxs[0]] == "IPTG":
                        continue
                if line[author_idx] == "M. Kohanski":
                    if line[treatment_idxs[1]] in [
                        "ampicillin",
                        "kanamycin",
                        "norfloxacin",
                        "spectinomycin",
                    ]:
                        continue
                if line[author_idx] == "D. Dwyer":
                    if line[treatment_idxs[1]] in [
                        "norfloxacin",
                        "nor chelator",
                        "UV",
                        "MMC",
                        "ciprofloxacin",
                        "chelator",
                    ]:
                        continue  # For chelator, also shows not rly growing bc it has ccdB toxin upregulation
                if "biofilm" in line[treatment_idxs[1]]:
                    continue
                if "stationary" in line[treatment_idxs[2]]:
                    continue
                if line[author_idx] == "T. Dong":
                    if line[treatment_idxs[2]] == "stat":
                        continue
                if line[treatment_idxs[1]] == "nitrofurazone":
                    continue
                if line[treatment_idxs[1]] == "HU":
                    # TODO: maybe keep this? might be interesting to be able to see dna damage responses, etc.?
                    continue
                if line[treatment_idxs[1]] == "amp":
                    continue
                if line[treatment_idxs[1]] == "AI2":
                    continue
                if line[author_idx] == "J. Faith":
                    if line[treatment_idxs[1]] == "nor":
                        continue
                    if "nor" in line[treatment_idxs[1]]:
                        continue
                if line[author_idx] == "J. Winkler":
                    if line[treatment_idxs[1]] == "tetracycline":
                        continue
                if line[author_idx] == "H. Lee":
                    # TODO: can't find publication, plus most are antibiotics, so just skip
                    continue
                    # if line[cel_idx] not in ["PR00099", "PR00100"]
                    # if line[treatment_idxs[1]] in ["amp", "gent", "norf", "spec", "tetracycline"]:
                if line[author_idx] == "T. Allen":
                    if line[treatment_idxs[1]] in ["ciprofloxacin", "glu, heat shock"]:
                        continue
                    if line[treatment_idxs[1]] == "glu":
                        if line[treatment_idxs[2]] in ["330", "480", "720"]:
                            continue
                if line[author_idx] == "M. Laubacher":
                    if "mecillinam" in line[treatment_idxs[1]]:
                        continue
                    if "cefsulodin" in line[treatment_idxs[1]]:
                        continue
                if line[author_idx] == "A. Ito":
                    if line[treatment_idxs[2]] == "stationary":
                        continue
                    if "rpoS" in line[gene_perturb_idxs[0]]:
                        continue  # Not annotated as gene-perturbation
                if line[author_idx] == "J. Oberto":
                    if line[treatment_idxs[1]] in ["stationary", "transition"]:
                        continue
                if line[author_idx] == "Y. Asakura":
                    # TODO: try keeping the 0 time-point ones? YA027 parent strain?
                    continue
                if line[author_idx] == "H. Alper":
                    if line[treatment_idxs[0]] in ["20g/L 5g/L 1mM", "40g/L 5g/L 1mM"]:
                        continue
                if line[author_idx] == "J. Soini":
                    # TODO: whether to keep this? the paper says high-cell densities
                    continue
                if line[author_idx] == "S. Shabala":
                    # Paper says grown to early stationary phase
                    continue
                if line[author_idx] == "L. Nobre":
                    if line[treatment_idxs[1]] == "CORM-2":
                        # TODO: whether to keep? paper says it's 50% of MIC, cells still growing
                        continue
                if line[author_idx] == "M. Lin":
                    continue
                    # TODO: change if we end up using the ethanol ones. Since we're not using
                    # the perturbations of M. Lin, only thing left is JM109 during LB, which probably
                    # is not that useful since we have a lot of LB already, and JM109 has slight genetic
                    # modifications from K-12.
                    # if "ethanol" in line[treatment_idxs[1]]:
                    #     # TODO: whether to keep ethanol here? also note, some of the samples
                    #     # are under stationary condition, should remove those
                    #     continue
                    # if "glyphosate" in line[treatment_idxs[1]]:
                    #     continue
                    # if line[geo_idx] == "GSE17465":
                    #     continue # IrrE upregulation, or 1M NaCl shock which would probably kill bacteria/inhibit growth
                if line[author_idx] == "C. Yang":
                    # TODO: might want to keep this bc it includes the interesting phosphate starvation?
                    # It's at early stationary phase though
                    continue
                if line[author_idx] == "T. Wood":
                    if line[treatment_idxs[1]] in ["nalidixic acid", "azlocillin"]:
                        # TODO: maybe keep this cuz cells r still growing despite these antibiotics?
                        continue
                # TODO: whether to keep H2O2 oxidative stress in Q. Ma paper?
                if line[author_idx] == "T. Chen":
                    # TODO: this is ethanol. Not sure if keep or change. IrrE deletion produces
                    # big genome-wide effects
                    continue
                if line[author_idx] == "Y. HU":
                    # Used erythromycin, which stops growth
                    continue
                if line[medium_idx] == "BHI Agar":
                    continue  # Paper said grown after overnight, might be stationary phase?? also shows high deviations.
                # if line[author_idx] == "B. Mensa":
                #    if line[treatment_idxs[1]] in ["polymyxin B", "PMX10070"]:
                # TODO: maybe still remove B. Mensa this? could help look at stress response,
                # but see what it looks like in actual graphs
                # genes. Causes only a slight
                # attenuation in growth.
                # Maybe keep all antimicrobials which don't kill or stop
                # growth completely?
                # continue
                if line[geo_idx] == "GSE34275":
                    # Paper says is actually staionary or long-term stationary phase growth
                    continue
                if line[author_idx] == "I. Keren":
                    # TODO: pretty arbitrary, idk whether to keep? bc might be stationary phase after 5 hrs, idk.
                    if int(line[treatment_idxs[2]]) > 300:
                        continue
                if line[author_idx] == "K. Zhu":
                    # CHIR-090 is at 10x MIC, so kills the cells.
                    # Adding the extra three DMSO samples probably doesn't do much, so we'll take those out as well.
                    # TODO: whether to keep the DMSO? it seems to cause some slight differences in e.g. purC, but
                    # that's probably just due to condition effects bc CHIR-090 does the same thing.
                    continue

            if "magnetic field" in line[treatment_idxs[1]]:
                continue
            if line[gene_perturb_idxs[0]] == "ala2 supressor":
                continue
            if line[gene_perturb_idxs[0]] == "recombinant pPROEx-CAT":
                continue
            if line[author_idx] == "C. Cardinale":
                if "bicyclomycin" in line[treatment_idxs[1]]:
                    continue
            if line[author_idx] == "Y. Yang":
                if "PenG" in line[treatment_idxs[1]]:
                    continue
            if "microgravity" in line[treatment_idxs[1]]:
                continue
            if line[medium_idx] == "K medium":
                if line[cel_idx] in [
                    "GSM183473",
                    "GSM183490",
                    "GSM183493",
                    "GSM183496",
                    "GSM183499",
                ]:
                    continue

            # TODO: take glucose out of LB, but check where u should add it back in! Check each LB media to find what the added
            # carbon source is, if there is one!
            # K. Aggrawal what to do? (ok bc most r genotype so skip)
            # TODO: J. Faith EMG2, LB has glucose concentration annotation
            # what to do about J. Winkler tetracycline? not sure so keep out for now
            # TODO: whether to keep antibiotic stuff like D. Dwyer?
            # TODO: get all the 0 time and replace with nothing treatment, e.g. J. faith, etc.
            # TODO: maybe add K. Anderson ampicillin if the cells are still growign then?
            # TODO: K medium doesn't have K+. Maybe could use K+ then as a component? check genes
            # TODO: Na+ not in M63 medium apparently, so maybe could use that as a component? check genes

            included_sample.append(i)
            geos.append(line[geo_idx])
            if line[author_idx] == "":
                authors.append("Not found")
            else:
                authors.append(line[author_idx])

            if line[cel_idx] in REPLACE_GSM_STRAIN:
                strains.append(REPLACE_GSM_STRAIN[line[cel_idx]])
            elif line[strain_idx] in REPLACE_STRAIN:
                strains.append(REPLACE_STRAIN[line[strain_idx]])
            else:
                strains.append(line[strain_idx])

            if line[medium_idx] in REPlACE_MEDIA:
                media.append(REPlACE_MEDIA[line[medium_idx]])
            elif line[author_idx] == "L. Nobre":
                media.append("minimal medium salts")
            else:
                media.append(line[medium_idx])

            if line[author_idx] == "G. Kannan":
                if line[treatment_idxs[2]] == "0":
                    treatment = ""
                    component = ""
            elif line[author_idx] == "K. Aggarwal":
                if line[treatment_idxs[0]] == "0mM":
                    treatment = ""
                    component = ""
            elif line[author_idx] == "S. Harcum":
                if line[gene_perturb_idxs[1]] == "IPTG":
                    treatment = "IPTG"
                    component = "IPTG"
            elif line[author_idx] == "H. Alper":
                if line[treatment_idxs[0]] == "0g/L 5g/L 1mM":
                    treatment = "5g/L 1mM" + "D-glucose thiamine"
                    component = "D-glucose thiamine"
            elif line[author_idx] == "Y. Li":
                if line[treatment_idxs[1]] == "glucose saline":
                    treatment = "4g/L" + "glucose"
                    component = "glucose"
            elif line[author_idx] == "L. Nobre":
                if line[cel_idx] == "GSM351280":
                    treatment = "anaerobic"
                    component = "anaerobic"
                elif line[cel_idx] == "GSM351282":
                    treatment = "anaerobic"
                    component = "anaerobic"
            elif line[author_idx] == "G. Vemuri":
                treatment = "1g/L" + "glucose"
                component = "glucose"
            elif line[author_idx] == "L. Waters":
                treatment = "10uMHighMnCl2"
                component = "HighMn2+"
                # TODO: make sure this is treated as highmn2+ in the final compositions,
                # not just mn2+?
            elif line[author_idx] == "J. Wang":
                treatment = "furfural"
                component = "furfural"
            elif line[geo_idx] == "GSE37026":
                if line[cel_idx] in [
                    "GSM909096",
                    "GSM909098",
                    "GSM909100",
                    "GSM909102",
                ]:
                    treatment = "colicin M"
                    component = "colicin M"
            elif line[medium_idx] == "K medium":
                if line[cel_idx] in ["GSM175728", "GSM183488", "GSM183494"]:
                    treatment = "glucose 30C methionine"
                elif line[cel_idx] in [
                    "GSM175737",
                    "GSM183470",
                    "GSM183489",
                    "GSM183495",
                ]:
                    treatment = "glucose 30C methionine highOsm"
                elif line[cel_idx] in ["GSM183471", "GSM183491", "GSM183497"]:
                    treatment = "glucose 43C methionine"
                elif line[cel_idx] in ["GSM183472", "GSM183492", "GSM183498"]:
                    treatment = "glucose 43C methionine highOsm"
                component = treatment
            else:
                treatment = line[treatment_idxs[0]] + line[treatment_idxs[1]]
                component = line[treatment_idxs[1]]

            treatment = REPLACE_TREATMENT.get(treatment, treatment)
            # if treatment in REPLACE_TREATMENT:
            #    treatments.append(REPLACE_TREATMENT[treatment])
            # else:
            #    treatments.append(treatment)
            time = line[treatment_idxs[2]]
            time = REPLACE_TIME.get(time, time)
            treatment = TIME_TO_TREATMENT.get(time, treatment)
            # if time in TIME_TO_TREATMENT:
            #    treatment = TIME_TO_TREATMENT[time]
            # if line[treatment_idxs[2]] in REPLACE_TIME:
            #    time.append(REPLACE_TIME[line[treatment_idxs[2]]])
            # else:
            #    time.append(line[treatment_idxs[2]])
            gene = line[gene_perturb_idxs[0]] + line[gene_perturb_idxs[1]]
            gene = REPLACE_GENE.get(gene, gene)
            gene_perturb.append(gene)
            # if gene in REPLACE_GENE:
            #    gene_perturb.append(REPLACE_GENE[gene])
            # else:
            #    gene_perturb.append(gene)
            if gene in GENE_TO_TREATMENT:
                treatment = GENE_TO_TREATMENT[gene]
                gene_perturb[-1] = ""

            treatment_all = treatment + time
            treatments.append(treatment_all)
            is_env.append(int(line[perturb_flags[0]]))
            is_genetic.append(int(line[perturb_flags[1]]))
            is_WT.append(int(line[perturb_flags[2]]))
            gr.append(float(line[gr_idx]))

            components.append(component)

    seq_data = []
    for filename in SEQ_FILES:
        path = os.path.join(SEQ_DIR, filename)
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            seq_data.append(list(reader))

    data = np.hstack(seq_data).astype(np.float64)
    data = data[:, included_sample]

    ### Make vectorized samples
    dont_split = [
        "response control - water added",
        "response control - H2O2",
        "propylene glycol",
        "polymyxin B",
        "nalidixic acid",
        "negative control",
        "magnetic field - sinusoidal continuous",
        "magnetic field - sinusoidal intermittent",
        "magnetic field - powerline intermittent",
        "casamino acids",
        "heat shock",
        "clinostat - microgravity",
        "biofilm with R1drd19 plasmid",
        "acid shock",
        "pH 8.7",
        "KCl acid shift",
        "colicin M",
    ]

    convert_comp = {
        "norf": "norfloxacin",
        "iptg": "IPTG",
        "arab": "arabinose",
        "pH 8.7": "pH8.7",
        "response control - H2O2": "H2O2",
        "nor": "norfloxacin",
        "D-glucose": "glucose",
        "+O2": None,  # TODO: check this, should we default everything to aerobic
        "aerobic": None,  # TODO: check this
        "spec": "spectinomycin",
        "EtOH": "ethanol",  # TODO: check this
        "control": None,  # TODO: check this,
        "amp": "ampicillin",
        "negative control": None,  # TODO: check this,
        "acid shock": "pH2",  # for 10 min? TODO: check this
        "pH": "pH2",  # for 10 min?
        "gent": "gentamicin",
        "-O2": "anaerobic",
        "glu": "glucose",
        "ASN": "acidified-sodium-nitrate",
        "AI-3": "autoinducer-3",
        "saline": "NaCl",
        "KCl acid shift": "pH5.5",
        "gly": "glycerol",
        "AI2": "autoinducer-2",
        "temperature": "low-temp",
        "response control - water added": None,
        "H202": "H2O2",
        "MMC": "mitomycin-C",
        "Cm": "chloramphenicol",  # TODO: check whether they used Kanamycin in the exp too? https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655446/
        "polymyxin B": "polymyxin-B",
        "glu,": "glucose",
        "chelator": "iron-chelator",
        "HU": "hydroxyurea",
        "pH7": None,
    }
    all_split_components = []
    for sample in components:
        split_components = []
        name = sample
        for x in dont_split:
            if x in name:
                split_components.append(x)
                name = name.replace(x, " ")

        split_components.extend(name.split())
        all_split_components.append(split_components)

    all_split_components_converted = []
    for x in all_split_components:
        all_split_components_converted.append([convert_comp.get(y, y) for y in x if y])

    # Convert to compositions
    media_comps = MediaComps()
    all_components_converted = media_comps.convert_to_composition(
        all_split_components_converted
    )

    # Get unique sets
    unique_components = set()
    for x in all_components_converted:
        unique_components.update(x)

    # Get media
    for i, x in enumerate(media):
        if authors[i] == "M. Laubacher" and x == "M9C":
            media[i] = "LB"
    unique_media = set(media)

    # M9C for x yang is M9 with casamino acids and glucose: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394867/
    # M9C for m laubacher: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2258881/

    # Media in ecocyc: BHI on metacyc, LB, M9, LB + glycerol (check),
    # MOPS, M63, LB + HOMOPIPES (check), LB low salt (check), EZ defined rich medium,
    # MES-LB (check), M9C (check),
    # Not in ecocyc: BHI agar?, MOPS + M9 or M9 + MOPS + tannic acid (?),
    # Terrific Broth, DMEM, K medium, modified complex (read paper),
    # glucose-limited minimal media in chemostat (check), Davis MM

    media_compositions = media_comps.media_comp
    all_media_compounds = media_comps.all_compounds

    media_components = [set(media_compositions[x].keys()) for x in media]
    all_components_converted_set = [set(x) for x in all_components_converted]

    for i, x in enumerate(media_components):
        all_components_converted_set[i].update(x)

    all_components_together = [list(x) for x in all_components_converted_set]

    # Add markers for relevant combinations of components
    for x in all_components_together:
        if "nitrate" in x:
            if "anaerobic" in x:
                x.append("nitrate-anaerobic")
                unique_components.add("nitrate-anaerobic")
            else:
                x.append("nitrate-aerobic")
                unique_components.add("nitrate-aerobic")
        if "anaerobic" in x:
            if "nitrate" not in x:
                x.append("anaerobic-no nitrate")
                unique_components.add("anaerobic-no nitrate")
        if "fumarate" in x:
            if "anaerobic" in x:
                x.append("fumarate-anaerobic")
                unique_components.add("fumarate-anaerobic")
            else:
                # NOTE: not present currently
                x.append("fumarate-aerobic")
                unique_components.add("fumarate-aerobic")
        if "succinate" in x:
            if "anaerobic" in x:
                # NOTE: not present currently
                x.append("succinate-anaerobic")
                unique_components.add("succinate-anaerobic")
            else:
                x.append("succinate-aerobic")
                unique_components.add("succinate-aerobic")
        if "acetate" in x:
            if "anaerobic" in x:
                # NOTE: not present currently
                x.append("acetate-anaerobic")
                unique_components.add("acetate-anaerobic")
            else:
                x.append("acetate-aerobic")
                unique_components.add("acetate-aerobic")
        if "glycerol" in x:
            if "anaerobic" in x:
                x.append("glycerol-anaerobic")
                unique_components.add("glycerol-anaerobic")
            # NOTE: not present currently
        if "anaerobic" in x:
            if "threonine" in x:
                # NOTE: only just doing threonine, bc threonine is present with many other aas like serine
                # that would be relevant to tdc operon
                x.append("anaerobic-threonine")
                unique_components.add("anaerobic-threonine")
            if "molybdate" in x:
                x.append("molybdate-anaerobic")
                unique_components.add("molybdate-anaerobic")
        if "fatty acids" in x:
            if "anaerobic" not in x:
                x.append("fatty acids-aerobic")

    # Get all compounds
    total_compounds = set(all_media_compounds)
    total_compounds.update(unique_components)
    total_compounds.remove(None)
    total_compounds = np.array(sorted(list(total_compounds)))

    # TODO: phosphate with high-affinity and low-affinity. So probably don't have phosphate as an
    # all or none component?? idk
    # TODO: look into more of these high-affinity, low-affinity regulation.
    # TODO: continue fixing to have components as sets not longer strings
    # TODO: generally clean up the media to be right
    # TODO: add sulfate to LB, other small metabolites too that are present in M9, etc.
    # TODO: LB with no glucose too for alternative carbon sources!
    # TODO: ethanol? bicarbonate?
    # TODO: do smth about the carbon source issue?
    # carbon_sources = ['gluconate', 'sucrose', 'succinate', 'glucose', 'fumarate', 'mannitol', 'lactate',
    #                  'propylene glycol', 'acetate', 'glycerol', 'citrate', 'arabinose', 'glycerophosphate', 'inositol',
    #                  'proline']

    # no_carbon_source = []
    # for i, x in enumerate(all_components_together):
    #   has_c = False
    #   for y in carbon_sources:
    #       if y in x:
    #           has_c = True
    #           break
    #   if not has_c:
    #       no_carbon_source.append(i)

    # minimal_no_carbon = [i for i in no_carbon_source if media_converted[i] == "M9" or media_converted[i] == "MOPS"]
    # chemical_minimal_no_carbon = [chemicals[i] for i in minimal_no_carbon]
    # TODO: check these, also maybe redo the operations before with new chemicals maybe
    # TODO: should check if the pH 5.3 or 5.7, treating that as a pH5 acidic, whether that messes up the acidic statistics
    # for i in minimal_no_carbon:
    #    all_components_together[i].append('glucose')

    # TODO: divide compounds into aas, carbon source, N source, antibiotic, etc.

    # Vectorize the treatments
    vectorized_treatments = np.zeros(
        (len(all_components_together), len(total_compounds))
    )
    for i, treat in enumerate(all_components_together):
        vectorized_treatments[i, :] = np.isin(total_compounds, treat)

    # Combine molecules that are always co-present
    reduced_vectorized_treatments, inverse = np.unique(
        vectorized_treatments, axis=1, return_inverse=True
    )
    # Where 1 appears in inverse, is where the 1-index pattern of reduced_vectorized_treatments' components appears in total_compounds
    # So, for each number in inverse, combine the places where it is
    reduced_total_components = []
    for i in range(np.shape(reduced_vectorized_treatments)[1]):
        overlapping_components = total_compounds[(inverse == i)]
        reduced_total_components.append(set(overlapping_components))
    reduced_total_components = np.array(reduced_total_components)

    ###

    # TODO: could maybe add the Shabala sucrose and NaCl (Sucrose, Shabala is at v high molarity so osmotic stress, should check if Y. Yang is at normal levels of sucrose?
    #     # in which case should add a component that marks osmotic stress?
    #     # Also early stationary phase, but maybe could use??)
    # TODO: fix the LB, which ones have glucose, etc.

    return (
        symbols,
        {
            "geos": np.array(geos),
            "authors": np.array(authors),
            "strains": np.array(strains),
            "media": np.array(media),
            "treatments": np.array(treatments),
            "gene_perturb": np.array(gene_perturb),
            "is_env": np.array(is_env),
            "is_genetic": np.array(is_genetic),
            "is_WT": np.array(is_WT),
            "gr": np.array(gr),
        },
        data,
        reduced_vectorized_treatments,
        reduced_total_components,
    )


def load_PRECISE2_data():
    seq_data = []
    symbols = []
    with open(PRECISE2_SEQ_FILE, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            seq_data.append([int(x) for x in row[1:]])
            symbols.append(row[0])

    return symbols, np.array(seq_data)


# TODO: other functions still need this, so could uncomment when needed
# def test_treatments(included_sample):
#     chemicals = []
#     times = []
#     media = []
#     authors = []
#
#     with open(SAMPLES_FILE) as f:
#         reader = csv.reader(f, delimiter='\t')
#         header = next(reader)
#         author_idx = header.index("Author")
#         media_idx = header.index("Medium")
#         treatment_idx = header.index("Treatment (conditions)")
#         time_idx = header.index("Time (min)")
#         strain_idx = header.index("Strain")
#         for i, line in enumerate(reader):
#             if i in included_sample:
#                 chemicals.append(line[treatment_idx])
#                 times.append(line[time_idx])
#                 media.append(line[media_idx])
#                 authors.append(line[author_idx])
#
#     times_converted = [REPLACE_TIME.get(x, x) for x in times]
#     # TODO: get concentration, so if concentration is 0, should delete it.
#
#     dont_split = ["response control - water added", "response control - H2O2",
#                   "propylene glycol", "polymyxin B", "nalidixic acid",
#                   "negative control", "magnetic field - sinusoidal continuous",
#                   "magnetic field - sinusoidal intermittent", "magnetic field - powerline intermittent",
#                   "casamino acids", "heat shock", 'clinostat - microgravity', 'biofilm with R1drd19 plasmid',
#                  'acid shock', "pH 8.7", "KCl acid shift"]
#
#     convert_comp = {"norf": "norfloxacin",
#                     "iptg": "IPTG",
#                     "arab": "arabinose",
#                     "pH 8.7": "pH8.7",
#                     "response control - H2O2": "H2O2",
#                     "nor": "norfloxacin",
#                     "D-glucose": "glucose",
#                     "+O2": None, # TODO: check this, should we default everything to aerobic
#                     "aerobic": None, # TODO: check this
#                     "spec": "spectinomycin",
#                     "EtOH": "ethanol", # TODO: check this
#                     "control": None, # TODO: check this,
#                     "amp": "ampicillin",
#                     "negative control": None, # TODO: check this,
#                     "acid shock": "pH2", # for 10 min? TODO: check this
#                     "pH": "pH2", # for 10 min?
#                     "gent": "gentamicin",
#                     "-O2": "anaerobic",
#                     "glu": "glucose",
#                     "ASN": "acidified-sodium-nitrate",
#                     "AI-3": "autoinducer-3",
#                     "saline": "NaCl",
#                     "KCl acid shift": "pH5.5",
#                     "gly": "glycerol",
#                     "AI2": "autoinducer-2",
#                     "temperature": "low-temp",
#                     "response control - water added": None,
#                     "H202": "H2O2",
#                     "MMC": "mitomycin-C",
#                     "Cm": "chloramphenicol", # TODO: check whether they used Kanamycin in the exp too? https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655446/
#                     "polymyxin B": "polymyxin-B",
#                     "glu,": "glucose",
#                     "chelator": "iron-chelator",
#                     "HU": "hydroxyurea",
#                     }
#
#     # TODO: ask about chemostats
#     # TODO: ask about hydroxyindole vs indole
#     # https://asap.genetics.wisc.edu/asap/experiment_data.php: the T. Allen stuff?
#     # https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15319 UTI ASN stuff, pH 5
#     # https://journals.asm.org/doi/10.1128/aac.00052-07?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed Y. Li glucose saline
#
#     # So I want to get a list of every chemical or treatment that they add.
#     # Then, given a certain treatment, I want to split it into these treatments
#     # One method is to say, if a certain string is present, then remove it. Another is to say,
#     # certain ones don't split, and then split the rest. Let's try this second?
#
#     all_components = []
#     for treatment in chemicals:
#         components = []
#         name = treatment
#         for x in dont_split:
#             if x in name:
#                 components.append(x)
#                 name = name.replace(x, ' ')
#
#         components.extend(name.split())
#         all_components.append(components)
#
#     all_components_converted = []
#     for x in all_components:
#         all_components_converted.append([convert_comp.get(y, y) for y in x if y])
#
#     # Convert to compositions
#     media_comps = MediaComps()
#     all_components_converted = media_comps.convert_to_composition(all_components_converted)
#
#     # treatment_phase = []
#     # phase_treatments = ['biofilm with R1drd19 plasmid', 'planktonic', 'biofilm']
#     # for i, treatment in enumerate(all_components_converted):
#     #     type = ''
#     #     for phase in phase_treatments:
#     #         if phase in treatment:
#     #             all_components_converted[i] = [x for x in all_components_converted[i] if x != phase]
#     #             type = phase
#     #
#     #
#     #     treatment_phase.append(type)
#
#
#
#     #time_phase = []
#     phase_times = ['mid-exponential', 'transition', 'stationary', 'formation', 'maturation',
#                    'exponential', 'attachment', 'Adaptive evolution']
#     for i, time in enumerate(times_converted):
#         #type = ''
#         for phase in phase_times:
#             if time == phase:
#                 times_converted[i] = ''
#                 all_components_converted[i].append(phase)
#                 #type = phase
#
#         #time_phase.append(type)
#
#     #combined_phase = [x + y for x, y in zip(treatment_phase, time_phase)]
#
#     # # Get unique sets
#     unique_components = set()
#     for x in all_components_converted:
#         unique_components.update(x)
#     #
#     # unique_times = set(times_converted)
#
#     # TODO: fix none problem
#
#     # biofilm with R1drd19 plasmid, planktonic, biofilm
#     # mid-exponential phase, transition, stationary, formation, maturation,
#     # exponential, mid-exponential, attachment
#     # Adaptive evolution: strain?
#
#     # Get media
#     media_converted = [REPlACE_MEDIA.get(x, x) for x in media]
#     for i, x in enumerate(media_converted):
#         if authors[i] == "M. Laubacher" and x == "M9C":
#             media_converted[i] = "LB"
#     unique_media = set(media_converted)
#
#     # M9C for x yang is M9 with casamino acids and glucose: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394867/
#     # M9C for m laubacher: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2258881/
#
#     # Media in ecocyc: BHI on metacyc, LB, M9, LB + glycerol (check),
#     # MOPS, M63, LB + HOMOPIPES (check), LB low salt (check), EZ defined rich medium,
#     # MES-LB (check), M9C (check),
#     # Not in ecocyc: BHI agar?, MOPS + M9 or M9 + MOPS + tannic acid (?),
#     # Terrific Broth, DMEM, K medium, modified complex (read paper),
#     # glucose-limited minimal media in chemostat (check), Davis MM
#
#     media_compositions = media_comps.media_comp
#     all_media_compounds = media_comps.all_compounds
#
#     media_components = [set(media_compositions[x].keys()) for x in media_converted]
#     all_components_converted_set = [set(x) for x in all_components_converted]
#
#     for i, x in enumerate(media_components):
#         all_components_converted_set[i].update(x)
#
#     #def optional_replacements(components):
#     #    replacements = {
#     #        "casamino acids": "amino acids"
#     #    }#
#        #
#        # for x in components:
#
#
#     all_components_together = [list(x) for x in all_components_converted_set]
#
#     total_compounds = set(all_media_compounds)
#     total_compounds.update(unique_components)
#     total_compounds.remove(None)
#     total_compounds = np.array(list(total_compounds))
#
#     # TODO: LB with no glucose too for alternative carbon sources!
#     # TODO: ethanol? bicarbonate?
#     carbon_sources = ['gluconate', 'sucrose', 'succinate', 'glucose', 'fumarate', 'mannitol', 'lactate',
#                       'propylene glycol', 'acetate', 'glycerol', 'citrate', 'arabinose', 'glycerophosphate', 'inositol',
#                       'proline']
#
#     no_carbon_source = []
#     for i, x in enumerate(all_components_together):
#         has_c = False
#         for y in carbon_sources:
#             if y in x:
#                 has_c = True
#                 break
#         if not has_c:
#             no_carbon_source.append(i)
#
#     minimal_no_carbon = [i for i in no_carbon_source if media_converted[i] == "M9" or media_converted[i] == "MOPS"]
#     chemical_minimal_no_carbon = [chemicals[i] for i in minimal_no_carbon]
#     # TODO: check these, also maybe redo the operations before with new chemicals maybe
#     for i in minimal_no_carbon:
#         all_components_together[i].append('glucose')
#
#     # TODO: divide compounds into aas, carbon source, N source, antibiotic, etc.
#
#     ## Analysis of the compounds present and their frequency
#     vectorized_treatments = np.zeros((len(all_components_together), len(total_compounds)))
#     for i, treat in enumerate(all_components_together):
#         vectorized_treatments[i, :] = np.isin(total_compounds, treat)
#
#     # num_of_treatments = np.sum(vectorized_treatments, axis=1)
#     # treatment_freqs = np.sum(vectorized_treatments, axis=0, dtype=int)
#     # treatment_freq_dict = {x: y for x, y in zip(total_compounds, treatment_freqs)}
#     # # fig, ax = plt.subplots(1)
#     # # ax.hist(treatment_freqs, bins=PlotUtils().n_bins(treatment_freqs))
#     # # plt.savefig(os.path.join(OUTPUT_DIR, "treatment_freqs_hist"))
#     # treatment_freq_compounds = {
#     #     "1": [k for k, v in treatment_freq_dict.items() if v == 1],
#     #     "2": [k for k, v in treatment_freq_dict.items() if v == 2],
#     #     "3": [k for k, v in treatment_freq_dict.items() if v == 3],
#     #     "4": [k for k, v in treatment_freq_dict.items() if v == 4],
#     #     "5-10": [k for k, v in treatment_freq_dict.items() if (5 <= v and v <= 10)],
#     #     "11-20": [k for k, v in treatment_freq_dict.items() if (10 < v and v <= 20)],
#     #     "21-30": [k for k, v in treatment_freq_dict.items() if (21 <= v and v <= 30)],
#     #     "31-40": [k for k, v in treatment_freq_dict.items() if (31 <= v and v <= 40)],
#     #     "41-50": [k for k, v in treatment_freq_dict.items() if (41 <= v and v <= 50)],
#     #     "51-100": [k for k, v in treatment_freq_dict.items() if (51 <= v and v <= 100)],
#     #     "101-500": [k for k, v in treatment_freq_dict.items() if (101 <= v and v <= 300)],
#     #     "501-1000": [k for k, v in treatment_freq_dict.items() if (501 <= v and v <= 1000)],
#     #     "1001-1750": [k for k, v in treatment_freq_dict.items() if (1001 <= v and v <= 1750)],
#     #     "1751-2189": [k for k, v in treatment_freq_dict.items() if (1751 <= v and v <= 2189)]
#     # }
#
#     #tuple_x = [tuple(set(all_components_together[i])) for i in minimal_no_carbon]
#     #no_glucose_minimal = minimal.intersection(no_glucose)
#     #all_components_weird = [all_components_together[i] for i in no_glucose_minimal]
#
#     return total_compounds, vectorized_treatments
#     #compare_components = CompareComponents(total_compounds, vectorized_treatments)
#     #compare_components.compare(np.random.default_rng().choice(vectorized_treatments, size=100, replace=False, axis=0))
#     # TODO: pH, alike compounds like benzoate, indole, peptone/tryptone/amino acids/casamino acids, BHI/yeast extract/etc.,
#     # TODO: figure out when molecules are used as inducers
#     # TODO: statistical stuff
#
#     # TODO: we probably need both a subset statistic, also a total freq statistic, also a statistic about whether
#     # a given property overall is associated with smth? a t-test? 1. enriched fraction, 2. overall fraction, 3. t-test?
#
#
#     # TODO: test this on a certain class I candidate genes
#
#     # We wish to assess enrichment in certain condition property. A certain
#     # property that is present in the subset at a higher probability than would be
#     # expected (given the total frequency). This depends on the size of the subset, of course.
#     # So, we thus compare the fraction of the subset that has a certain property,
#     # to the overall fraction that has that property.
#     # The power of our statistics falls off for low-occurrence properties. So we
#     # also introduce a complementary criteria for these.
#     # So generally, the property is presence or absence of some compound, and combinations of this.
#     # We run into issues for things like


def load_gene_synonyms():
    # synonyms = []
    # with open(GENE_SYNONYMS_FILE, 'r') as f:
    #    lines = f.readlines()
    #    for line in lines:
    #        synonyms.append(line.split()[1:])

    # return synonyms
    b_numbers = []
    symbols = []
    with open(GENE_NAMES_FILE) as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            b_numbers.append(line[1])
            symbols.append(line[2])

    return b_numbers, symbols


def make_table(samples, sample_data):
    def extract_info(all, specific):
        unique = np.unique(all)
        transform = np.array([np.where(unique == x)[0][0] for x in specific], dtype=int)
        return unique, transform

    unique_authors, author_idx = extract_info(
        sample_data["authors"], samples["authors"]
    )
    unique_media, media_idx = extract_info(sample_data["media"], samples["media"])
    unique_strains, strains_idx = extract_info(
        sample_data["strains"], samples["strains"]
    )
    unique_treatments, treatments_idx = extract_info(
        sample_data["treatments"], samples["treatments"]
    )
    gr = samples["gr"]
    unique_gene, gene_idx = extract_info(
        sample_data["gene_perturb"], samples["gene_perturb"]
    )

    total_data = {
        "authors": author_idx,
        "media": media_idx,
        "strains": strains_idx,
        "treatments": treatments_idx,
        "gr": gr,
        "gene": gene_idx,
    }
    total_info = {
        "authors": unique_authors,
        "media": unique_media,
        "strains": unique_strains,
        "treatments": unique_treatments,
        "gene": unique_gene,
    }

    import ipdb

    ipdb.set_trace()
    return total_data, total_info


def test_bootstrap(sample_data):
    total_data, total_info = make_table(sample_data, sample_data)

    fig, axs = plt.subplots(10, figsize=(5, 50))
    for i in range(10):
        scores, _ = RnaseqUtils().group_bootstrap(
            score=100, group_A_size=100, group_B_size=10, all_samples=total_data
        )
        n_bins = int(
            np.ceil(
                (np.max(scores) - np.min(scores))
                / (2 * stats.iqr(scores) / len(scores) ** (1 / 3))
            )
        )
        axs[i].hist(scores, bins=n_bins)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_group_bootstrap"))
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def test_classify(sample_data):
    total_data, total_info = make_table(sample_data, sample_data)

    fig, axs = plt.subplots(10, figsize=(5, 50))
    for i in range(10):
        scores, _ = ClassifyScore().bootstrap(
            1, 100, 100, total_data, features=["media", "authors"]
        )
        n_bins = int(
            np.ceil(
                (np.max(scores) - np.min(scores))
                / (2 * stats.iqr(scores) / len(scores) ** (1 / 3))
            )
        )
        axs[i].hist(scores, bins=n_bins)
        axs[i].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_classify_bootstrap"))
    plt.close("all")

    import ipdb

    ipdb.set_trace()


# I have many samples, each sample has many gene expression data. Want to parameterize
# some genes being controlled by transcription factors. So first look into the sample data.
def classify_samples(sample_data, method="media"):
    (
        geos,
        media,
        treatment_values,
        treatment_conditions,
        treatment_time,
        gene_perturb_id,
        gene_perturb_dir,
        is_env,
        is_genetic,
        is_WT,
        gr,
    ) = sample_data
    condition = [
        x
        for x in zip(
            media,
            treatment_values,
            treatment_conditions,
            treatment_time,
            gene_perturb_id,
            gene_perturb_dir,
        )
    ]
    condition_with_gr = [
        x
        for x in zip(
            media,
            treatment_values,
            treatment_conditions,
            treatment_time,
            gene_perturb_id,
            gene_perturb_dir,
            gr,
        )
    ]
    condition_no_time = [
        x
        for x in zip(
            media,
            treatment_values,
            treatment_conditions,
            gene_perturb_id,
            gene_perturb_dir,
        )
    ]
    media_types = list(set(media))
    all_conditions = list(set(condition))
    all_conditions_with_gr = list(set(condition_with_gr))
    all_conditions_no_time = list(set(condition_no_time))

    sample_to_media = np.array([media_types.index(medium) for medium in media])
    sample_to_condition = np.array([all_conditions.index(x) for x in condition])
    sample_to_condition_gr = np.array(
        [all_conditions_with_gr.index(x) for x in condition_with_gr]
    )
    sample_to_condition_no_time = np.array(
        [all_conditions_no_time.index(x) for x in condition_no_time]
    )

    if method == "media":
        return sample_to_media, np.array(media_types)
    elif method == "condition":
        return sample_to_condition, np.array(all_conditions)
    elif method == "condition_with_gr":
        return sample_to_condition_gr, np.array(all_conditions_with_gr)
    elif method == "condition_no_time":
        return sample_to_condition_no_time, np.array(all_conditions_no_time)


def average_exp(expression, sample_data, method="media"):
    if method is None:
        return expression, np.array([data for data in sample_data]).T

    classification, types = classify_samples(sample_data, method=method)
    averaged_exp = np.zeros((np.shape(expression)[0], len(np.unique(classification))))
    for i in np.unique(classification):
        averaged_exp[:, i] = np.mean(expression[:, (classification == i)], axis=1)
    return averaged_exp, types


def get_gene_exp(symbols, seq_data, gene_idxs=None, gene_names=None):
    if gene_names is not None:
        gene_idxs = [symbols.index(name) for name in gene_names]
    elif gene_idxs is not None:
        gene_names = [symbols[idx] for idx in gene_idxs]

    expression = np.array(seq_data[gene_idxs, :])
    return gene_names, expression


def get_one_gene_exp(symbols, seq_data, gene_idx=None, gene_name=None):
    if gene_name is not None:
        gene_idx = symbols.index(gene_name)
    elif gene_idx is not None:
        gene_name = symbols[gene_idx]

    expression = np.array(seq_data[gene_idx, :])
    return gene_name, expression


# If we assumed good representation of all conditions that might cause a regulated TF to turn on.
# Then we'd see that if a TF is regulated and indeed has two states that differ
# much in expression, then it'll have significant amount in both situations.
# And so if we see a histogram that doesn't have significant amounts of samples in
# different enough expression-then it's not regulated?
# And so maybe we could use normal-fitting statistics here.

# On the other hand, we might not have good representation of conditions.
# Then if a TF is regulated and indeed has two states that differ much in expression,
# it might not have significant amounts in one of the situations.
# And so, if we see a histogram that has some amount of samples in a different
# enough expression (by which we mean, distinguishable from a distribution in the
# "other" case), it might still be regulated.
# So then, we want to see whether there's any cause for believing this.

# If it has a low standard deviation, that means genes are tightly clustered.
# If it fits a Gaussian, that means doesn't have a big tail, so it could
# possibly be not regulated.


def compare_stimulon(idxs1, idxs2, sample_data):
    sample_data_1 = {}
    sample_data_2 = {}

    for k, v in sample_data.items():
        sample_data_1[k] = v[idxs1]
        sample_data_2[k] = v[idxs2]

    stimulon_1, info = make_table(sample_data_1, sample_data)
    stimulon_2, _ = make_table(sample_data_2, sample_data)
    sample_table, _ = make_table(sample_data, sample_data)

    # So we've gotten the stimulons (the table of features) from the idxs, and ran the
    # group similarity score and bootstrap p-value on it.
    score, p_value = RnaseqUtils().run_group_test(stimulon_1, stimulon_2, sample_table)

    return score, p_value


def compare_stimulon_classify(idxs1, idxs2, sample_data, features_tuple):
    sample_data_1 = {}
    sample_data_2 = {}

    for k, v in sample_data.items():
        sample_data_1[k] = v[idxs1]
        sample_data_2[k] = v[idxs2]

    stimulon_1, info = make_table(sample_data_1, sample_data)
    stimulon_2, _ = make_table(sample_data_2, sample_data)
    sample_table, _ = make_table(sample_data, sample_data)

    results = {}
    classifiers = {}
    for features in features_tuple:
        score, p_value, classify = ClassifyScore().run_test(
            stimulon_1, stimulon_2, sample_table, features
        )
        results[tuple(features)] = (score, p_value)

        classify_info = {}
        for k, v in classify.items():
            features_info = tuple([info[f][k[i]] for i, f in enumerate(features)])
            classify_info[features_info] = v

        classifiers[tuple(features)] = classify_info

    return results, classifiers


def compare_tail_stimulon_classify(
    gene_names,
    all_genes,
    seq_data,
    sample_data,
    features_tuple=(["authors"], ["strains"], ["media"], ["treatments"], ["gene"]),
):
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)

    all_results = []
    all_classifiers = []
    sizes = []
    for i, exp in enumerate(expression):
        mean = np.mean(exp)
        std = np.std(exp)
        big_idxs = np.where(exp > mean + 1.3 * std)[0]
        small_idxs = np.where(exp < mean - 1.3 * std)[0]
        results, classifiers = compare_stimulon_classify(
            big_idxs, small_idxs, sample_data, features_tuple
        )
        all_results.append(results)
        all_classifiers.append(classifiers)
        sizes.append((len(big_idxs), len(small_idxs)))

    import ipdb

    ipdb.set_trace()
    return all_results, sizes


# def compare_tail_stimulons(gene_names, all_genes, seq_data, sample_data):
#     # TODO: can't use averaging here because then the indexes of exp wouldn't match
#     # the sample idxs; maybe could do something if we're averaging replicates?
#
#     _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)
#
#     scores = []
#     ps = []
#     sizes = []
#     for i, exp in enumerate(expression):
#         mean = np.mean(exp)
#         std = np.std(exp)
#         big_idxs = np.where(exp > mean + 1.3 * std)[0]
#         small_idxs = np.where(exp < mean - 1.3 * std)[0]
#         score, p = compare_stimulon(big_idxs, small_idxs, sample_data)
#         scores.append(score)
#         ps.append(p)
#         sizes.append((len(big_idxs), len(small_idxs)))
#
#     import ipdb; ipdb.set_trace()
#     return scores, ps, sizes


def compare_tail_stimulon_enrichment(
    gene_name, cutoff, all_genes, seq_data, sample_data
):
    # total_compounds, vectorized_treatments = test_treatments()
    # compare_components = CompareComponents(total_compounds, vectorized_treatments)

    _, expression = get_gene_exp(all_genes, seq_data, gene_names=[gene_name])

    media = sample_data["media"]
    treatments = sample_data["treatments"]
    authors = sample_data["authors"]
    strains = sample_data["strains"]
    # enriched = []
    # depleted = []
    fractions = []
    cdfs = []
    sizes = []
    for i, exp in enumerate(expression):
        # mean = np.mean(exp)
        # std = np.std(exp)
        big_idxs = np.where(exp > cutoff)[0]
        small_idxs = np.where(exp <= cutoff)[0]

        big_media = np.array(media)[big_idxs]
        small_media = np.array(media)[small_idxs]
        big_treatments = np.array(treatments)[big_idxs]
        small_treatments = np.array(treatments)[small_idxs]
        big_authors = np.array(authors)[big_idxs]
        small_authors = np.array(authors)[small_idxs]
        small_strains = np.array(strains)[small_idxs]
        big_strains = np.array(strains)[big_idxs]

        # big_enriched, big_depleted, big_fractions = compare_components.compare(vectorized_treatments[big_idxs])
        # small_enriched, small_depleted, small_fractions = compare_components.compare(vectorized_treatments[small_idxs])
        # enriched.append((big_enriched, small_enriched))
        # depleted.append((big_depleted, small_depleted))

        # big_cdfs, big_fractions = compare_components.compare(vectorized_treatments[big_idxs])
        # small_cdfs, small_fractions = compare_components.compare(vectorized_treatments[small_idxs])
        # fractions.append((big_fractions, small_fractions))
        # cdfs.append((big_cdfs, small_cdfs))
        # sizes.append((len(big_idxs), len(small_idxs)))

        import ipdb

        ipdb.set_trace()


def gene_exp_hist(
    gene_names,
    all_genes,
    seq_data,
    sample_data,
    title,
    avg_method=None,
    has_gaussian=None,
):
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)
    avg_expression, types = average_exp(expression, sample_data, method=avg_method)

    n_genes = len(gene_names)
    # is_WT = np.array(sample_data['is_WT'])
    # WT_exp = expression[:, np.where(is_WT)[0]]
    mean = np.mean(avg_expression, axis=1)
    std = np.std(avg_expression, axis=1)

    n_bins = PlotUtils().n_bins
    fig, axs = plt.subplots(n_genes, figsize=(5, 5 * n_genes))
    for i, (exp, gene) in enumerate(zip(expression, gene_names)):
        # axs[i % 50, int(i/50)].hist(exp, bins=n_bins(exp))
        axs[i].hist(exp, bins=n_bins(exp))
        axs[i].set_xlim(0, 15)
        axs[i].set_title(str(gene), size=14)
        axs[i].set_xlabel("Log2(gene expression fraction), arbitrary units", size=14)
        axs[i].set_ylabel("Counts of microarray experiments", size=14)
        # axs[i % 50, int(i/50)].set_title(str(gene))
        # axs[i % 50, int(i/50)].set_xlim(0, 15)
        xs = np.linspace(0, 15, 100)
        if has_gaussian:
            if has_gaussian[i]:
                axs[i].plot(
                    xs, stats.norm.pdf(xs, mean[i], std[i]) * len(exp) * 15 / 100
                )

        # axs[i, 1].hist(np.squeeze(WT_exp[i, :]), bins=10)
        # axs[i, 1].set_title("WT"+str(gene))

        # axs[i, 1].set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, title))
    # if avg_method:
    #    plt.savefig(os.path.join(OUTPUT_DIR, title+avg_method))
    # else:
    #    plt.savefig(os.path.join(OUTPUT_DIR, title+"_no_avg"))
    plt.close("all")


def hist_from_exp(expression, gene_names, title):
    """
    Creates a histogram given the expression and gene names.
    """
    n_genes = len(expression)
    n_bins = PlotUtils().n_bins
    fig, axs = plt.subplots(n_genes, figsize=(5, 5 * n_genes))
    if n_genes > 1:
        for i, (exp, gene) in enumerate(zip(expression, gene_names)):
            axs[i].hist(exp, bins=n_bins(exp))
            axs[i].set_xlim(0, 15)
            axs[i].set_title(str(gene))
            axs[i].set_xlabel("Log2(gene expression fraction), arbitrary units")
            axs[i].set_ylabel("Counts of microarray experiments")
            xs = np.linspace(0, 15, 100)
            axs[i].plot(
                xs, stats.norm.pdf(xs, np.mean(exp), np.std(exp)) * len(exp) * 15 / 100
            )
    else:
        exp = expression[0]
        gene = gene_names[0]
        axs.hist(exp, bins=n_bins(exp))
        axs.set_xlim(0, 15)
        axs.set_title(str(gene))
        axs.set_xlabel("Log2(gene expression fraction), arbitrary units")
        axs.set_ylabel("Counts of microarray experiments")
        xs = np.linspace(0, 15, 100)
        axs.plot(
            xs, stats.norm.pdf(xs, np.mean(exp), np.std(exp)) * len(exp) * 15 / 100
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, title))
    plt.close("all")


def PRECISE2_hist_from_exp(expression, gene_names, title):
    """
    Creates a histogram given the expression and gene names.
    """
    n_genes = len(expression)
    n_bins = PlotUtils().n_bins
    fig, axs = plt.subplots(n_genes, figsize=(5, 5 * n_genes))
    if n_genes > 1:
        for i, (exp, gene) in enumerate(zip(expression, gene_names)):
            axs[i].hist(np.log10(exp + 1), bins=n_bins(np.log10(exp + 1)))
            axs[i].set_title(str(gene))
            axs[i].set_xlabel("Log10(gene expression counts), arbitrary units")
            axs[i].set_ylabel("Counts of microarray experiments")
    else:
        exp = expression[0]
        gene = gene_names[0]
        axs.hist(np.log10(exp), bins=n_bins(exp))
        axs.set_title(str(gene))
        axs.set_xlabel("Log2(gene expression fraction), arbitrary units")
        axs.set_ylabel("Counts of microarray experiments")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, title))
    plt.close("all")


def run_gmm(expression, clusters=2):
    """
    Runs the EM algorithm for Gaussian mixture model for the expression data of one gene
    """

    split_exp = []
    for i in range(clusters):
        split_exp.append([])
    for x in expression:
        idx = np.random.randint(0, clusters)
        split_exp[idx].append(x)
    phis_old = [len(exp) / len(expression) for exp in split_exp]
    mus_old = [np.mean(exp) for exp in split_exp]
    sigmas_old = [np.sqrt(np.var(exp)) for exp in split_exp]
    expression = np.array(expression)

    def gaussian(mu, sigma, x):
        return np.exp(-1 / 2 * ((mu - x) / sigma) ** 2) / (np.sqrt(2 * math.pi) * sigma)

    def log_likelihood(mus, sigmas, phis, w_array):
        ll = 0
        for i, x in enumerate(expression):
            temp_ll = 0
            for j, case in enumerate(zip(mus, sigmas, phis)):
                temp_ll += w_array[i, j] * np.log(
                    gaussian(case[0], case[1], x) * case[2]
                )
            ll += temp_ll

        return ll

    for i in range(MAX_ITER):
        # Expectation
        probs = [
            [gaussian(mu, sigma, x) for mu, sigma in zip(mus_old, sigmas_old)]
            for x in expression
        ]
        w_array = np.array([np.array(ws) / np.sum(ws) for ws in probs])
        # TODO: divide by zero errors?
        # Maximization
        phis_new = np.sum(w_array, axis=0) / np.sum(w_array)
        mus_new = np.divide(w_array.T @ expression, np.sum(w_array, axis=0))
        deviation = np.array([expression - mu for mu in mus_old]).T ** 2
        sigmas_new = np.sqrt(
            np.divide(
                np.array([(w_array.T @ deviation)[i, i] for i in range(len(mus_old))]),
                np.sum(w_array, axis=0),
            )
        )

        # Update
        ll_old = log_likelihood(mus_old, sigmas_old, phis_old, w_array)
        ll_new = log_likelihood(mus_new, sigmas_new, phis_new, w_array)
        if np.abs(ll_old - ll_new) < CUTOFF_THRESHOLD:
            return mus_new, sigmas_new, phis_new, True, ll_new
        else:
            phis_old = phis_new
            mus_old = mus_new
            sigmas_old = sigmas_new

    return mus_new, sigmas_new, phis_new, False, ll_new


def run_gmm_sklearn(expression, clusters=2):
    gmm = mixture.GaussianMixture(n_components=clusters)
    mus = []
    sigmas = []
    phis = []
    scores = []
    for x in expression:
        gmm.fit(x)
        params = gmm.get_params()
        score = gmm.score(x)
        scores.append(score)
        import ipdb

        ipdb.set_trace()
    gmm.fit(
        expression,
    )
    # TODO: fix this


def fit_gaussian(expression, scaled=True):
    def gaussian(x, mu, sigma):
        return np.exp(-1 / 2 * ((mu - x) / sigma) ** 2) / (np.sqrt(2 * math.pi) * sigma)

    def log_likelihood(exp, mu, sigma):
        ll = 0
        for x in exp:
            ll += np.log(gaussian(x, mu, sigma))
        return ll

    mu = np.mean(expression)
    sigma = np.std(expression)
    if scaled:
        scaled_expression = (expression - mu) / sigma
        ll = log_likelihood(scaled_expression, 0, 1)
    else:
        ll = log_likelihood(expression, mu, sigma)

    return mu, sigma, ll


def rank_by_std_ll(seq_data, all_genes, gene_names, plot_file, write_file):
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)

    stds = []
    lls = []
    for i, exp in enumerate(expression):
        _, sigma, ll = fit_gaussian(exp)
        stds.append(sigma)
        lls.append(ll)

    stds = np.array(stds)
    lls = np.array(lls)

    if plot_file:
        fig, ax = plt.subplots(1)
        ax.scatter(stds, lls, s=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, plot_file))
        plt.close("all")

    rank_stds = np.argsort(stds)
    rank_lls = np.argsort(lls)[::-1]

    output_file = os.path.join(OUTPUT_DIR, write_file)
    with open(output_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = [
            "genes_rank_std",
            "std_rank_std",
            "ll_rank_std",
            "genes_rank_ll",
            "std_rank_ll",
            "ll_rank_ll",
        ]
        writer.writerow(header)
        for i in range(len(stds)):
            writer.writerow(
                [
                    np.array(gene_names)[rank_stds][i],
                    stds[rank_stds][i],
                    lls[rank_stds][i],
                    np.array(gene_names)[rank_lls][i],
                    stds[rank_lls][i],
                    lls[rank_lls][i],
                ]
            )

    import ipdb

    ipdb.set_trace()


# Given the wild-type expression, when is smth a good candidate for under a certain set of conditions, it's far away?
# Maybe if it fits two gaussians well, where one is wild-type and other is the other?

# So thing we want to do: a plot for, given all the samples,
# we want to plot out the correlation between sample and the
# expression.
# Case 0: most samples around a value, gaussian. Then maybe shud be a straight line?
# Case 1: most samples around x, one sample lower. Then shud be straight line at the around-x, and a jump at that one sample
# Case 2: most samples around a value, then a bridge going out. Then shud be a decreasing line to straight line.
# Case 3: samples clustered around two values. Then shud be two straight lines connected by a jump.

# Case A: samples clustered around two values, with clear differences in sample. then you'd get a nice decrease
# Case B: samples clustered around two values, no clear difference in sample. Then you'd get no nice line.
# Make a plot where x-axis is samples, y-axis is exp-level. If u just did this, it'd kinda be a messy bar plot.
# Order the samples given some rules, so that the high-exp samples are near the front.
# And then maybe plot differences until next highest sample??
# Choice 1: order so that, you have to move media as a block, but within media you can order in any way.
# Then, within media you'd always see a nice decreasing line. Between media, the jump depends on the lowest vs highest within the two media.
# So maybe we want to look at average within the media. So plot out average with grey line for standard deviation?
# So then we know lowest to highest media, and their standard deviations, and how quickly they drop.
# The thing we want to look at is between diff conditions, which ones have what levels of expression, which ones deviate the most?


def samples_plot(all_genes, gene_names, seq_data, sample_data, avg_method=None):
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)
    avg_exp, samples = average_exp(expression, sample_data, method=avg_method)

    fig, axs = plt.subplots(len(avg_exp), figsize=(5, 5 * len(avg_exp)))
    output_file = os.path.join(OUTPUT_DIR, "samples_plot_samples_test" + avg_method)
    if avg_method is None:
        header = [
            "GEO id",
            "media",
            "treatment value",
            "treatment condition",
            "treatment time",
            "gene perturb id",
            "gene perturb dir",
            "environmental",
            "genetic",
            "WT",
            "growth rate",
        ]
    elif avg_method == "media":
        header = ["media"]
    with open(output_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        writer.writerow(header)

    for i, exp in enumerate(avg_exp):
        sort = np.argsort(exp)
        exp = exp[sort]
        samples_sort = np.array(samples)[sort]

        axs[i].plot(list(range(len(exp))), exp)
        axs[i].set_xticks(list(range(len(exp))), [str(i) for i in range(len(exp))])
        axs[i].set_title(gene_names[i])

        with open(output_file, "a") as f:
            writer = csv.writer(
                f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
            )
            writer.writerow([gene_names[i]])
            for row in samples_sort:
                writer.writerow([row])
            writer.writerow(["\n"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "samples_plot_test" + avg_method))
    plt.close("all")

    # output_file = os.path.join(OUTPUT_DIR, "samples_plot_samples_test"+avg_method)
    # with open(output_file, 'w') as f:
    #    writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

    #    if avg_method is None:
    #        header = ["GEO id", "media", "treatment value", "treatment condition", "treatment time",
    #                  "gene perturb id", "gene perturb dir", "environmental", "genetic", "WT", "growth rate"]
    #    elif avg_method == "media":
    #        header = ["media"]
    #    writer.writerow(header)
    #    for row in samples:
    #        writer.writerow([row])

    import ipdb

    ipdb.set_trace()


def make_gene_rankings(seq_data, all_genes, gene_names, output_name):
    ll_two_cluster = []
    ll_one_cluster = []
    difference = []
    for gene in gene_names:
        _, expression = get_one_gene_exp(all_genes, seq_data, gene_name=gene)
        mu_1, sigma_1, ll_one = fit_gaussian(expression, scaled=False)
        mu_2, sigma_2, phi_2, _, ll_two = run_gmm(expression, clusters=2)
        difference.append(ll_two - ll_one)
        ll_two_cluster.append(ll_two)
        ll_one_cluster.append(ll_one)

    # two_cluster_rank = np.array(gene_names)[np.argsort(np.array(ll_two_cluster))[::-1]]
    # one_cluster_rank = np.array(gene_names)[np.argsort(np.array(ll_one_cluster))[::-1]]
    sort = np.argsort(np.array(difference))[::-1]
    ranked = np.array(gene_names)[sort]
    ll_two_cluster = np.array(ll_two_cluster)[sort]
    ll_one_cluster = np.array(ll_one_cluster)[sort]
    difference = np.array(difference)[sort]

    output_file = os.path.join(OUTPUT_DIR, output_name)
    with open(output_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["ranked genes", "ll difference", "ll two-cluster", "ll one-cluster"]
        writer.writerow(header)
        for i in range(len(ranked)):
            writer.writerow(
                [ranked[i], difference[i], ll_two_cluster[i], ll_one_cluster[i]]
            )

    import ipdb

    ipdb.set_trace()
    return ranked, difference, ll_two_cluster, ll_one_cluster


def rank_by_outliers(seq_data, all_genes, gene_names):
    # TODO: work on this
    # So we want something that, given a dataset, tells us if there's
    # some of it deviating a lot from the major part of the dataset.
    # Maybe we can say that like, give a cutoff and fit a gaussian,
    # and then get max threshold based on this? and what cutoff to use,
    #
    # Or just keep with this outlier thing. This one detects minor outliers,
    # but misses big chunks.
    # The likelihood ratio detects big chunks, but false positives on
    # a non-gaussian looking chunk, and also misses minor outliers?
    # So how to find a thing that finds minor outliers? maybe the outlier thing.
    # the worry is how do we find it, and how do we know it's not an artifact because
    # of high numbers?
    # And also, another thing we want to do is find the samples. If something
    # is really deviating consistently in a certain environment, then we'll count
    # it as real.
    outlier_z = []
    for gene in gene_names:
        _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene)
        mu, sigma, _ = fit_gaussian(expression)
        max_outlier = np.max(np.abs(expression - mu) / sigma)
        outlier_z.append(max_outlier)
        import ipdb

        ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    ranked = np.array(gene_names)[np.argsort(np.array(outlier_z))[::-1]]
    outlier_z = outlier_z[np.argsort(np.array(outlier_z))[::-1]]
    return ranked, outlier_z


def plot_rRNAs(symbols, seq_data, sample_data):
    rrns = [
        gene
        for gene in symbols
        if (("rrs" in gene) | ("rrl" in gene)) | ("rrf" in gene)
    ]
    gene_exp_hist(
        rrns, symbols, seq_data, sample_data, title="rrn_exp_hist", avg_method=None
    )
    import ipdb

    ipdb.set_trace()


# We want to get genes that have low spread in their expression. Ways to look:
# 1. lowest range, 2. std (95% confidence interval), 3. interquartile range, 4. range of 95% of data points
def rank_by_spread(
    gene_names,
    symbols,
    seq_data,
    sample_data,
    output_name,
    output_plot_name,
    avg_method=None,
):
    gene_names = np.array(gene_names)

    _, expression = get_gene_exp(symbols, seq_data, gene_names=gene_names)
    expression, _ = average_exp(expression, sample_data, method=avg_method)

    maxs = np.max(expression, axis=1)
    mins = np.min(expression, axis=1)
    ranges = maxs - mins

    stds = np.std(expression, axis=1)
    ci_95 = 1.96 * stds

    iqrs = stats.iqr(expression, axis=1)

    top_95 = maxs - np.quantile(expression, 0.05, axis=1)
    bottom_95 = np.quantile(expression, 0.95, axis=1) - mins
    middle_95 = np.quantile(expression, 0.975, axis=1) - np.quantile(
        expression, 0.025, axis=1
    )
    best_95 = np.minimum(np.minimum(top_95, bottom_95), middle_95)

    fig, axs = plt.subplots(6, figsize=(5, 30))
    axs[0].scatter(ranges, ci_95, s=0.5)
    axs[0].set_title("Ranges vs ci_95")
    axs[1].scatter(ranges, iqrs, s=0.5)
    axs[1].set_title("Ranges vs IQRs")
    axs[2].scatter(ranges, best_95, s=0.5)
    axs[2].set_title("Ranges vs best 95")
    axs[3].scatter(ci_95, iqrs, s=0.5)
    axs[3].set_title("ci_95 vs IQRs")
    axs[4].scatter(ci_95, best_95, s=0.5)
    axs[4].set_title("CI_95 vs best 95")
    axs[5].scatter(iqrs, best_95, s=0.5)
    axs[5].set_title("IQRs vs best 95")

    output_plot_file = os.path.join(OUTPUT_DIR, output_plot_name + ".png")
    plt.tight_layout()
    plt.savefig(output_plot_file)
    plt.close("all")

    sort_range = np.argsort(ranges)
    sort_ci_95 = np.argsort(ci_95)
    sort_iqr = np.argsort(iqrs)
    sort_best_95 = np.argsort(best_95)

    gene_rank_range = gene_names[sort_range]
    gene_rank_ci_95 = gene_names[sort_ci_95]
    gene_rank_iqr = gene_names[sort_iqr]
    gene_rank_best_95 = gene_names[sort_best_95]
    ranges = ranges[sort_range]
    ci_95 = ci_95[sort_ci_95]
    iqrs = iqrs[sort_iqr]
    best_95 = best_95[sort_best_95]

    output_file = os.path.join(OUTPUT_DIR, output_name + ".txt")
    with open(output_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = [
            "rank_range",
            "range",
            "rank_ci_95",
            "ci_95",
            "rank_iqr",
            "iqr",
            "rank_best_95",
            "best_95",
        ]
        writer.writerow(header)
        for tuple in zip(
            gene_rank_range,
            ranges,
            gene_rank_ci_95,
            ci_95,
            gene_rank_iqr,
            iqrs,
            gene_rank_best_95,
            best_95,
        ):
            writer.writerow([x for x in tuple])

    # TODO: look at data, graph, think about what it means for certain genes graphically, look at rankings and get generally high-ranking, etc.?
    #
    import ipdb

    ipdb.set_trace()


def plot_mean_std(gene_names, symbols, seq_data, output_name):
    gene_names = np.array(gene_names)
    _, expression = get_gene_exp(symbols, seq_data, gene_names=gene_names)
    means = np.mean(expression, axis=1)
    stds = np.std(expression, axis=1)

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.scatter(means, stds, s=0.5)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.set_title("Standard deviation vs. mean of gene expression")

    output_plot_file = os.path.join(OUTPUT_DIR, output_name + ".png")
    plt.tight_layout()
    plt.savefig(output_plot_file)
    plt.close("all")
    import ipdb

    ipdb.set_trace()


# So the deal is, we have an approx normal profile with std sigma, and want to know
# whether it probably has outliers somewhere. we can see, for instance, how many it has outside
# 2 stds (which is supposed to be 5% of 2000 or around 100 data points).


def get_gene_spread_data(gene_names, spread_rank_file):
    # Get the gene's data, i.e. rankings and values
    # Want to get some sort of criteria for if a gene is truly seemingly "unregulated"?
    data = {gene: {} for gene in gene_names}
    with open(spread_rank_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, row in enumerate(reader):
            for gene in gene_names:
                if gene in row[0]:
                    data[gene]["range"] = (i, row[1])
                if gene in row[2]:
                    data[gene]["ci_95"] = (i, row[3])
                if gene in row[4]:
                    data[gene]["iqr"] = (i, row[5])
                if gene in row[6]:
                    data[gene]["best_95"] = (i, row[7])

    return data


def plot_gene_spread_data(spread_rank_file, plot_name):
    ranges = []
    ci_95 = []
    iqrs = []
    best_95 = []
    with open(spread_rank_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for row in reader:
            ranges.append(row[1])
            ci_95.append(row[3])
            iqrs.append(row[5])
            best_95.append(row[7])

    ranges = ranges[1:]
    ci_95 = ci_95[1:]
    iqrs = iqrs[1:]
    best_95 = best_95[1:]

    output_file = os.path.join(OUTPUT_DIR, plot_name + ".png")
    fig, axs = plt.subplots(4, figsize=(5, 30))
    axs[0].hist(ranges, bins=30)
    axs[0].set_title("Ranges")
    axs[1].hist(ci_95, bins=30)
    axs[1].set_title("95% confidence intervals")
    axs[2].hist(iqrs, bins=30)
    axs[2].set_title("IQRs")
    axs[3].hist(best_95, bins=30)
    axs[3].set_title("Range of 95% of data")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def calc_with_moments(seq_data, all_genes, gene_names, plot_file, write_file):
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)
    plot_path = os.path.join(OUTPUT_DIR, plot_file)
    write_path = os.path.join(OUTPUT_DIR, write_file)

    Rankings().rank_by_moments(expression, np.array(gene_names), plot_path, write_path)


def std_and_prob_plot(seq_data, all_genes, gene_names, plot_file, write_file):
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=gene_names)
    plot_path = os.path.join(OUTPUT_DIR, plot_file)
    write_path = os.path.join(OUTPUT_DIR, write_file)
    Rankings().rank_by_std_prob_plot(
        expression, np.array(gene_names), plot_path, write_path
    )


def get_candidate_genes(all_genes, input_file, write_file):
    # Get the gene's data, i.e. rankings and values
    # Want to get some sort of criteria for if a gene is truly seemingly "unregulated"?
    data = {gene: {} for gene in all_genes}
    with open(input_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, row in enumerate(reader):
            for gene in all_genes:
                if gene in row[0]:
                    data[gene]["std"] = row[1]
                    data[gene]["prob_r"] = row[2]

    low_std = set([g for g in all_genes if data[g]["std"] < 0.7])
    high_r = set([g for g in all_genes if data[g]["prob_r"] > 0.995])
    candidates = low_std.intersection(high_r)

    with open(write_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["gene", "std", "prob_plot_r"]
        writer.writerow(header)
        for gene in candidates:
            writer.writerow([gene, data[gene]["std"], data[gene]["prob_r"]])
    import ipdb

    ipdb.set_trace()
    return data


# Idea 1: so get an idea of how the max z-scores are looking. If a lot of them are e.g. < 2.5, and
# another chunk has > 2.7, then maybe we can take some cutoff as 2 or smth?
# Forward from that, we'd look at, e.g. the number of points with z-scores > 2.5 or smth. A tradeoff between
# number of z-scores, and value of the z-scores. But then, if u e.g. have a sample with a few, and they're high
# z-score, versus a sample with many and high z-score, would you want to weigh the many with high z-score?
# but that creates a difference between large and small samples, and you're looking for small outliers.
# but indeed, having one sample at z-score > 2.5 is more likely than having 6 there. hmm.
# Maybe smth like, if there's more than some cutoff number of points above some cutoff z-score? so it
# incorporates this statistical element but is not too sensitive to big or small samples (assuming a small
# true-positive sample is probably going to pass the cutoff? most sensitive would just be 1 point above
# some cutoff z-score. Can bootstrap to see false-positive rate).


def plot_max_z_score_for_candidates(all_genes, seq_data, candidates_file):
    # if you do it for all the genes, that'd be like saying, what's the max z-score for these, but I rly care about
    # applying a cutoff for these candidates, so less important? It'd still be good to see though.
    _, expression = get_gene_exp(all_genes, seq_data, gene_names=all_genes)
    stds = np.std(expression, axis=1)
    means = np.mean(expression, axis=1)
    z_of_max = (np.max(expression, axis=1) - means) / stds
    z_of_min = (np.min(expression, axis=1) - means) / stds
    largest_z = np.maximum(z_of_max, np.abs(z_of_min))

    candidate_genes = []
    prob_r_candidates = []
    with open(os.path.join(OUTPUT_DIR, candidates_file), "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader, None)
        for line in reader:
            candidate_genes.append(line[0])
            prob_r_candidates.append(line[2])
    candidate_genes = np.array(candidate_genes)

    is_candidate = np.isin(all_genes, candidate_genes)
    candidate_largest_z = largest_z[is_candidate]

    low_largest_z = largest_z < 4.5
    final_candidate = np.logical_and(is_candidate, low_largest_z)

    with open(os.path.join(OUTPUT_DIR, "candidates_low_z.txt"), "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["gene", "mean", "std", "largest_z"]
        writer.writerow(header)
        for i in np.where(final_candidate)[0]:
            writer.writerow([all_genes[i], means[i], stds[i], largest_z[i]])

    fig, axs = plt.subplots(5, figsize=(5, 10))
    axs[0].hist(candidate_largest_z, bins=PlotUtils().n_bins(candidate_largest_z))
    axs[0].set_title("Largest z-score of candidates")
    axs[1].hist(largest_z, bins=PlotUtils().n_bins(largest_z))
    axs[1].set_title("Largest z-score of all genes")
    axs[2].scatter(stds[is_candidate], candidate_largest_z, s=0.5)
    axs[2].set_title("Largest z vs std of candidates")
    axs[3].scatter(stds, largest_z, s=0.5)
    axs[3].set_title("Largest z vs std of all genes")
    axs[4].scatter(candidate_largest_z, prob_r_candidates, s=0.5)
    axs[4].set_title("Prob plot r vs. largest z of candidates")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "largest_z_candidates"))
    plt.close("all")

    # So low largest-z, based on a few samples, seems to include many transcriptional regulators.
    # TODO: plot the number of points above a certain std cutoff to see if that might be better
    # So, how many points does it have about 3 std? expect to be around 6.
    # Well each point has a 0.3% probability of being above 3 std. So for 2000 points, that's sampling
    # a binomial.
    import ipdb

    ipdb.set_trace()


def bootstrap_max_normal(n, iter):
    maxs = []
    for i in range(iter):
        sample = np.random.normal(0, 1, n)
        max = np.max(np.abs(sample))
        maxs.append(max)

    fig, ax = plt.subplots(1)
    ax.hist(maxs, bins=PlotUtils().n_bins(maxs))
    ax.set_title("Maximum of normal")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "max_z_normal_bootstrap"))
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def get_genes_from_file(file):
    genes = []
    with open(os.path.join(OUTPUT_DIR, file), "r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        next(csv_reader)
        for line in csv_reader:
            genes.append(line[0])

    return genes


def get_llr_genes():
    genes = []
    with open(os.path.join(OUTPUT_DIR, "gmm_llr_rankings_no_rewiring"), "r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        next(csv_reader)
        for line in csv_reader:
            genes.append(line[0])

    return genes


# One hypothesis: 4000 genes in e coli, 1000 have 1 gaussian, 2000 have 2 gaussians


def plot_gene_profiles(symbols, gene_names, seq_data, sample_data, ecocyc):
    fig, axs = plt.subplots(len(gene_names), figsize=(5, 5 * len(gene_names)))
    _, exp_all = get_gene_exp(symbols, seq_data, gene_names=gene_names)
    for i, gene in enumerate(gene_names):
        exp = exp_all[i, :]
        axs[i].hist(exp, bins=PlotUtils().n_bins(exp))
        axs[i].set_xlim(0, 15)
        xs = np.linspace(0, 15, 100)
        mean = np.mean(exp)
        std = np.std(exp)
        regulators = None
        if gene in ecocyc.gene_to_regulators:
            regulators = ecocyc.gene_to_regulators[gene]

        axs[i].plot(xs, stats.norm.pdf(xs, mean, std) * len(exp) * 15 / 100)
        axs[i].set_title(
            str(gene) + " mean: " + str(round(mean, 2)) + " std: " + str(round(std, 2))
        )
        if regulators:
            label = ""
            for reg in regulators:
                label += str(reg) + ", "
            axs[i].set_xlabel(label)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gene_profile_candidates"))
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def plot_gmm_llr(symbols, seq_data, output_name):
    genes_ranked = get_llr_genes()
    fig, axs = plt.subplots(nrows=50, ncols=2, figsize=(5 * 2, 5 * 50))
    high_genes = genes_ranked[:50]
    low_genes = genes_ranked[-50:]
    _, exp_high = get_gene_exp(symbols, seq_data, gene_names=high_genes)
    _, exp_low = get_gene_exp(symbols, seq_data, gene_names=low_genes)

    plotutils = PlotUtils()
    for i, (gene, exp) in enumerate(zip(high_genes, exp_high)):
        axs[i, 0].hist(exp, bins=plotutils.n_bins(exp))
        axs[i, 0].set_xlim(0, 15)
        xs = np.linspace(0, 15, 100)
        mean = np.mean(exp)
        std = np.std(exp)
        axs[i, 0].plot(xs, stats.norm.pdf(xs, mean, std) * len(exp) * 15 / 100)
        axs[i, 0].set_title(str(gene))

    for i, (gene, exp) in enumerate(zip(low_genes, exp_low)):
        axs[i, 1].hist(exp, bins=plotutils.n_bins(exp))
        axs[i, 1].set_xlim(0, 15)
        xs = np.linspace(0, 15, 100)
        mean = np.mean(exp)
        std = np.std(exp)
        axs[i, 1].plot(xs, stats.norm.pdf(xs, mean, std) * len(exp) * 15 / 100)
        axs[i, 1].set_title(str(gene))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_name))
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def plot_exp(seq_data, idxs):
    fig, axs = plt.subplots(len(idxs), figsize=(5, 5 * len(idxs)))
    for i, idx in enumerate(idxs):
        exp = seq_data[:, idx]
        axs[i].hist(exp, bins=PlotUtils().n_bins(exp))
        axs[i].set_xlim(0, 20)
        axs[i].set_title(str(idx))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_level_expression"))
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def plot_histograms(symbols, seq_data, gene_names, title):
    _, expression = get_gene_exp(symbols, seq_data, gene_names=gene_names)
    heatmap = MakeHeatmaps(OUTPUT_DIR)
    heatmap.heatmap(expression, title)


def get_candidate_two_peaks():
    file = os.path.join(OUTPUT_DIR, "peak_finder")
    file = os.path.join(file, "all_peaks_nopathogen.txt")
    plot_genes = []
    import json

    with open(file, "r") as f:
        for line in f.readlines()[1:]:
            gene, num_peaks, _, _, standard_devs = line.split("\t")
            if int(num_peaks) == 2:
                standard_devs = json.loads(standard_devs[1:-2])
                if float(standard_devs[0]) < 0.04:
                    if float(standard_devs[1]) < 0.04:
                        plot_genes.append(gene[1:-1])

    return plot_genes


def get_samples_in_range(exp, samples, range):
    in_range = np.logical_and((exp > range[0]), (exp < range[1]))
    return {k: samples[k][in_range] for k in samples.keys()}, in_range


def get_candidate_one_peak(expression, gene_names, write_file, method="twofold"):
    num_samples = np.shape(expression)[1]
    candidate_genes = []
    for i, exp in enumerate(expression):
        mean = np.mean(exp)

        if method == "twofold":
            within_twofold = np.abs(exp - mean) < 1.0
            frac_twofold = np.sum(within_twofold) / num_samples
            if frac_twofold > 0.95:
                candidate_genes.append(gene_names[i])
        elif method == "standard_dev":
            standard_dev = np.std(exp)
            if standard_dev < 0.6:
                within_two = np.abs(exp - mean) < 2 * standard_dev
                frac_within_two = np.sum(within_two) / num_samples
                if frac_within_two > 0.95:
                    candidate_genes.append(gene_names[i])

    with open(write_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["gene"]
        writer.writerow(header)
        for gene in candidate_genes:
            writer.writerow([gene])


def one_peak_num_reg_histogram(all_genes, one_peak_file, plot_file, exclude_sigma=True):
    gene_names = []
    with open(one_peak_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, line in enumerate(reader):
            if i > 0:
                gene_names.append(line[0])
    ecocyc = EcocycReg(BASE_DIR, exclude_sigma=exclude_sigma)

    regulation = ecocyc.find_gene_regulation(all_genes)

    def _get_num_reg(genes):
        num_reg = []
        for gene in genes:
            reg_info = regulation[gene]
            if reg_info == "N/A":
                num_reg.append(-1)
            else:
                num_reg.append(len(reg_info))
        # TODO: account for special regs like lrp, etc.
        return np.array(num_reg)

    one_peak_num_regs = _get_num_reg(gene_names)
    all_num_regs = _get_num_reg(all_genes)
    max_num_reg = max(np.max(one_peak_num_regs), np.max(all_num_regs))
    one_peak_hist, bins = np.histogram(
        one_peak_num_regs, bins=np.arange(-1, max_num_reg + 1, 1)
    )
    one_peak_hist = one_peak_hist / np.sum(one_peak_hist)
    all_hist, _ = np.histogram(all_num_regs, bins=np.arange(-1, max_num_reg + 1, 1))
    all_hist = all_hist / np.sum(all_hist)

    fig, axs = plt.subplots(2, figsize=(20, 10))
    axs[0].bar(
        np.arange(0, 3 * (max_num_reg + 1), step=3),
        one_peak_hist,
        width=1,
        align="edge",
        color="r",
        tick_label=bins[:-1],
        label="One-peak",
    )
    axs[0].bar(
        np.arange(1, 3 * (max_num_reg + 1) + 1, step=3),
        all_hist,
        width=1,
        align="edge",
        color="b",
        label="All",
    )
    axs[0].legend()
    axs[0].set_title("Histogram of fraction of genes with n regulators")
    axs[0].set_xlabel("Number of regulators")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close("all")


def genes_reg_by_x(regulator, all_genes, one_peak_file, write_file):
    gene_names = []
    with open(one_peak_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, line in enumerate(reader):
            if i > 0:
                gene_names.append(line[0])
    ecocyc = EcocycReg(BASE_DIR)

    # NOTE: this all_genes must be ALL ecomac gene names
    regulation = ecocyc.find_gene_regulation(all_genes)

    target_genes = []
    for gene in gene_names:
        reg_info = regulation[gene]
        if reg_info != "N/A":
            reg_gene_names = [x[0] for x in reg_info]
            if regulator in reg_gene_names:
                target_genes.append(gene)

    with open(write_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["gene", "notes"]
        writer.writerow(header)
        for gene in target_genes:
            writer.writerow([gene, ""])

    import ipdb

    ipdb.set_trace()


def crp_investigation(all_genes):
    ecocyc = EcocycReg(BASE_DIR)

    # NOTE: this all_genes must be ALL ecomac gene names
    regulation = ecocyc.find_gene_regulation(all_genes)

    reg_by_crp_genes = []
    for gene in all_genes:
        reg_info = regulation[gene]
        reg_gene_names = [x[0] for x in reg_info]
        if "crp" in reg_gene_names:
            reg_by_crp_genes.append(gene)

    def _get_regulators(genes):
        regulators = np.array(
            ecocyc.ecocyc_to_ecomac_names(ecocyc.regulators, all_genes)
        )
        is_regulator = np.isin(np.array(genes), regulators)

        return is_regulator

    is_reg = _get_regulators(reg_by_crp_genes)

    import ipdb

    ipdb.set_trace()


# NOTE: don't use this one anymore
def one_peak_reg_info(
    all_genes, one_peak_file, write_file, plot_file, exclude_sigma=True
):
    gene_names = []
    with open(one_peak_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, line in enumerate(reader):
            if i > 0:
                gene_names.append(line[0])
    ecocyc = EcocycReg(BASE_DIR, exclude_sigma=exclude_sigma)

    regulation = ecocyc.find_gene_regulation(all_genes)

    special_regs = ["rpoD", "hns", "lrp", "ihfA", "ihfB", "fis", "rpoE", "rpoH", "rpoS"]

    def _categorize(genes):
        category = []
        reg_gene_names = []
        for gene in genes:
            reg_info = regulation[gene]
            if reg_info == "N/A":
                category.append(-1)
            elif len(reg_info) == 0:
                category.append(0)
            else:
                reg_names = [x[0] for x in reg_info]
                reg_gene_names.extend(reg_names)

                if len(reg_info) == 1:
                    if (reg_info[0][0] == gene) & (reg_info[0][1] == "-"):
                        category.append(1)
                        continue

                reg_is_special = [x in special_regs for x in reg_names]
                if np.sum(reg_is_special) == len(reg_is_special):
                    category.append(3)
                    continue
                category.append(2)
        return category, np.unique(reg_gene_names, return_counts=True)

    def _get_regulators(genes):
        regulators = np.array(
            ecocyc.ecocyc_to_ecomac_names(ecocyc.regulators, all_genes)
        )
        is_regulator = np.isin(np.array(genes), regulators)

        return is_regulator

    is_regulator = _get_regulators(gene_names)
    all_is_regulator = _get_regulators(all_genes)
    frac_regulator = np.sum(is_regulator) / len(is_regulator)
    all_frac_regulator = np.sum(all_is_regulator) / len(all_is_regulator)

    category, (reg_gene_names, reg_gene_counts) = _categorize(gene_names)
    all_category, (all_reg_gene_names, all_reg_gene_counts) = _categorize(all_genes)

    all_category_stats = np.array(np.unique(all_category, return_counts=True)[1])
    all_num_regulated_genes = np.sum(all_category_stats[2:])
    all_category_stats = all_category_stats / np.sum(all_category_stats)

    category_stats = np.unique(category, return_counts=True)[1]
    num_regulated_genes = np.sum(category_stats[2:])
    category_stats = category_stats / np.sum(category_stats)

    reg_gene_sort = np.argsort(reg_gene_counts)
    reg_gene_counts = reg_gene_counts[reg_gene_sort]
    reg_gene_names = reg_gene_names[reg_gene_sort]
    reg_gene_frac = reg_gene_counts / num_regulated_genes
    reg_gene_other_frac = reg_gene_counts / np.sum(reg_gene_counts)
    reg_gene_name_to_frac = {
        gene: frac for gene, frac in zip(reg_gene_names, reg_gene_frac)
    }
    reg_gene_name_to_other_frac = {
        gene: frac for gene, frac in zip(reg_gene_names, reg_gene_other_frac)
    }

    all_reg_gene_sort = np.argsort(all_reg_gene_counts)
    all_reg_gene_names = all_reg_gene_names[all_reg_gene_sort]
    all_reg_gene_counts = all_reg_gene_counts[all_reg_gene_sort]
    all_reg_gene_frac = all_reg_gene_counts / all_num_regulated_genes
    all_reg_gene_other_frac = all_reg_gene_counts / np.sum(all_reg_gene_counts)

    fig, axs = plt.subplots(2, figsize=(100, 10))
    axs[0].bar(
        np.arange(0, 3 * len(all_reg_gene_names), step=3),
        all_reg_gene_frac,
        width=1,
        align="edge",
        color="r",
        tick_label=all_reg_gene_names,
        label="All",
    )
    axs[0].bar(
        np.arange(1, 3 * len(all_reg_gene_names) + 1, step=3),
        [reg_gene_name_to_frac.get(gene, 0) for gene in all_reg_gene_names],
        width=1,
        align="edge",
        color="b",
        label="One-peak",
    )
    axs[0].set_title("Fraction of regulated genes that are regulated by each regulator")
    axs[0].legend()

    axs[1].bar(
        np.arange(0, 3 * len(all_reg_gene_names), step=3),
        all_reg_gene_other_frac,
        width=1,
        align="edge",
        color="r",
        tick_label=all_reg_gene_names,
        label="All",
    )
    axs[1].bar(
        np.arange(1, 3 * len(all_reg_gene_names) + 1, step=3),
        [reg_gene_name_to_other_frac.get(gene, 0) for gene in all_reg_gene_names],
        width=1,
        align="edge",
        color="b",
        label="One-peak",
    )
    axs[1].set_title("Fraction of regulatory interactions that are each regulator")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close("all")

    with open(write_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["gene", "reg_info", "category"]
        writer.writerow(header)
        for gene, cat in zip(gene_names, category):
            writer.writerow([gene, regulation[gene], cat])

    import ipdb

    ipdb.set_trace()


def comp_null_standard_dev_vs_reg(
    all_genes, gene_file, expression, plot_file, exclude_sigma=True
):
    gene_names = []
    with open(gene_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, line in enumerate(reader):
            if i > 0:
                gene_names.append(line[0])
    ecocyc = EcocycReg(BASE_DIR, exclude_sigma=exclude_sigma)

    regulation = ecocyc.find_gene_regulation(all_genes)

    special_regs = ["rpoD", "hns", "lrp", "ihfA", "ihfB", "fis", "rpoE", "rpoH", "rpoS"]

    def _find_num_regs(genes):
        num_regs = []
        for gene in genes:
            reg_info = regulation[gene]
            if reg_info == "N/A":
                num_regs.append(-1)
            elif len(reg_info) == 0:
                num_regs.append(0)
            else:
                reg_names = [x[0] for x in reg_info if x not in special_regs]
                num_regs.append(len(reg_names))

        return num_regs

    standard_devs = np.std(expression, axis=1)
    is_gene = np.isin(all_genes, gene_names)
    standard_devs_genes = standard_devs[is_gene]
    fig, axs = plt.subplots(2, figsize=(20, 20))
    num_regs = _find_num_regs(gene_names)
    axs[0].scatter(standard_devs_genes, num_regs, s=0.5)
    axs[0].set_xlabel("Standard deviation of each gene")
    axs[0].set_ylabel("Number of regulators of each gene (excluding special regs)")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close("all")
    import ipdb

    ipdb.set_trace()


def write_one_peak_genes(all_genes, gene_file, expression, write_file):
    gene_names = []
    with open(gene_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        for i, line in enumerate(reader):
            if i > 0:
                gene_names.append(line[0])

    standard_devs = np.std(expression, axis=1)
    is_gene = np.isin(all_genes, gene_names)
    standard_devs_genes = standard_devs[is_gene]
    one_peak_genes = np.array(gene_names)[standard_devs_genes < 0.6]

    with open(write_file, "w") as f:
        writer = csv.writer(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        header = ["gene"]
        writer.writerow(header)
        for gene in one_peak_genes:
            writer.writerow([gene])


def plot_overlapping_one_peak_genes(one_peak_file, plot_file):
    symbols, sample_data, seq_data, vectorized_samples, total_components = (
        load_data_exclude_rewiring()
    )
    one_peak_genes = []
    with open(one_peak_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader)
        for line in reader:
            one_peak_genes.append(line[0])

    fig, axs = plt.subplots(2, figsize=(20, 40))
    symbols = np.array(symbols)
    for gene in one_peak_genes:
        idx = np.where(symbols == gene)[0][0]
        exp = seq_data[idx]
        exp = exp - np.mean(exp)
        axs[0].hist(exp, bins=80, range=(-8, 8), alpha=0.1)
    axs[0].set_title(
        "Overlapping histograms of " + str(len(one_peak_genes)) + " one-peak genes"
    )
    axs[0].set_xlabel("log2(expression value) minus mean of expression")
    axs[0].set_ylabel("Frequency (out of ~800)")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close("all")


def two_peak_reg_analyze(
    all_genes, two_peak_file, plot_file, plot_file_reg, exclude_sigma=True
):
    excluded_components = [
        "selenate",
        "formate",
        "tannin-extract",
        "propylene glycol",
        "MOPS",
        "clinostat - microgravity",
        "cefsulodin",
        "mecillinam",
        "spectinomycin",
        "kanamycin",
        "tetracycline",
        "CHIR-090",
        "PMX10070",
        "lactate",
        "polymyxin-B",
        "DMSO",
        "HOMOPIPES",
    ]  # HOMOPIPES?

    genes_to_components = {}
    component_to_genes = {}
    with open(two_peak_file, "r") as f:
        reader = csv.reader(
            f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader)
        for line in reader:
            if line[1] in excluded_components:
                continue
            if line[0] not in genes_to_components:
                genes_to_components[line[0]] = []
            genes_to_components[line[0]].append(line[1])

            if line[1] not in component_to_genes:
                component_to_genes[line[1]] = []
            component_to_genes[line[1]].append(line[0])

    gene_names = list(genes_to_components.keys())
    ecocyc = EcocycReg(BASE_DIR, exclude_sigma=exclude_sigma)
    regulation = ecocyc.find_gene_regulation(all_genes)

    def _get_num_reg(genes):
        num_reg = []
        for gene in genes:
            reg_info = regulation[gene]
            if reg_info == "N/A":
                num_reg.append(-1)
            else:
                num_reg.append(len(reg_info))
        # TODO: account for special regs like lrp, etc.
        return np.array(num_reg)

    two_peak_num_regs = _get_num_reg(gene_names)
    all_num_regs = _get_num_reg(all_genes)
    max_num_reg = max(np.max(two_peak_num_regs), np.max(all_num_regs))
    two_peak_hist, bins = np.histogram(
        two_peak_num_regs, bins=np.arange(-1, max_num_reg + 1, 1)
    )
    two_peak_hist = two_peak_hist / np.sum(two_peak_hist)
    all_hist, _ = np.histogram(all_num_regs, bins=np.arange(-1, max_num_reg + 1, 1))
    all_hist = all_hist / np.sum(all_hist)

    fig, axs = plt.subplots(2, figsize=(20, 10))
    axs[0].bar(
        np.arange(0, 3 * (max_num_reg + 1), step=3),
        two_peak_hist,
        width=1,
        align="edge",
        color="r",
        tick_label=bins[:-1],
        label="Two-peak",
    )
    axs[0].bar(
        np.arange(1, 3 * (max_num_reg + 1) + 1, step=3),
        all_hist,
        width=1,
        align="edge",
        color="b",
        label="All",
    )
    axs[0].legend()
    axs[0].set_title("Histogram of fraction of genes with n regulators")
    axs[0].set_xlabel("Number of regulators")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close("all")

    one_reg_genes = np.array(gene_names)[two_peak_num_regs == 1]

    def _get_reg_gene_counts(genes):
        reg_gene_names = []
        for gene in genes:
            reg_info = regulation[gene]
            if reg_info == "N/A":
                continue
            if len(reg_info) > 0:
                reg_names = [x[0] for x in reg_info]
                reg_gene_names.extend(reg_names)

        return np.unique(reg_gene_names, return_counts=True)

    def _get_regulators(genes):
        regulators = np.array(
            ecocyc.ecocyc_to_ecomac_names(ecocyc.regulators, all_genes)
        )
        is_regulator = np.isin(np.array(genes), regulators)

        return is_regulator

    is_regulator = _get_regulators(one_reg_genes)
    all_is_regulator = _get_regulators(all_genes)
    frac_regulator = np.sum(is_regulator) / len(is_regulator)
    all_frac_regulator = np.sum(all_is_regulator) / len(all_is_regulator)

    (reg_gene_names, reg_gene_counts) = _get_reg_gene_counts(gene_names)
    (one_reg_gene_names, one_reg_gene_counts) = _get_reg_gene_counts(one_reg_genes)
    (all_reg_gene_names, all_reg_gene_counts) = _get_reg_gene_counts(all_genes)

    reg_gene_sort = np.argsort(reg_gene_counts)
    reg_gene_counts = reg_gene_counts[reg_gene_sort]
    reg_gene_names = reg_gene_names[reg_gene_sort]
    reg_gene_name_to_count = {
        gene: count for gene, count in zip(reg_gene_names, reg_gene_counts)
    }

    one_reg_gene_sort = np.argsort(one_reg_gene_counts)
    one_reg_gene_counts = one_reg_gene_counts[one_reg_gene_sort]
    one_reg_gene_names = one_reg_gene_names[one_reg_gene_sort]
    one_reg_gene_name_to_count = {
        gene: count for gene, count in zip(one_reg_gene_names, one_reg_gene_counts)
    }

    all_reg_gene_sort = np.argsort(all_reg_gene_counts)
    all_reg_gene_names = all_reg_gene_names[all_reg_gene_sort]
    all_reg_gene_counts = all_reg_gene_counts[all_reg_gene_sort]

    fig, axs = plt.subplots(2, figsize=(100, 10))
    axs[0].bar(
        np.arange(0, 3 * len(all_reg_gene_names), step=3),
        all_reg_gene_counts,
        width=1,
        align="edge",
        color="r",
        tick_label=all_reg_gene_names,
        label="All",
    )
    axs[0].bar(
        np.arange(1, 3 * len(all_reg_gene_names) + 1, step=3),
        [reg_gene_name_to_count.get(gene, 0) for gene in all_reg_gene_names],
        width=1,
        align="edge",
        color="b",
        label="Two-peak",
    )
    axs[0].set_title("Number of genes regulated by each regulator for two-peak genes")
    axs[0].legend()

    axs[1].bar(
        np.arange(0, 3 * len(all_reg_gene_names), step=3),
        [reg_gene_name_to_count.get(gene, 0) for gene in all_reg_gene_names],
        width=1,
        align="edge",
        color="r",
        label="Two-peak all",
    )
    axs[1].bar(
        np.arange(1, 3 * len(all_reg_gene_names) + 1, step=3),
        [one_reg_gene_name_to_count.get(gene, 0) for gene in all_reg_gene_names],
        tick_label=all_reg_gene_names,
        width=1,
        align="edge",
        color="b",
        label="Two-peak one reg",
    )
    axs[1].set_title(
        "Number of genes regulated by each regulator for two-peak one-reg genes"
    )
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(plot_file_reg)
    plt.close("all")
    # For two-peak genes: some of them are annotated unregulated (opportunities for discovery), interesting what the component is.
    # some of them are annotated with having 1 regulator (that's our hypothesis), where either the component matches the regulator or not
    # some of them are annotated with having >1 that are either mostly firm (opportunities for discovery),
    # some of them are annotated with having >1 where only 1 is firm. Where the component matches the regulator or not.
    # To validate that two-peak genes are mainly single regulator: we should show that there's a big enrichment
    # in having 1 regulator for which the component that regulates it is what we expect it to be?
    # If a regulon's member is two-peak, we might expect other members of that regulon to be two-peak (tho not necessarily).

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    # neg_autoreg, pos_autoreg, both_autoreg = ecocyc.find_autoregulated_genes()
    # num_regulators, regulators = ecocyc.classify_regulation(neg_autoreg)
    #
    #
    # #import ipdb; ipdb.set_trace()
    # #symbols, sample_data, seq_data = load_data()
    # colinetdata = coliNetData()
    # simple_operon, simple_tf, one_to_one = colinetdata.get_simple_reg()
    # symbols, seq_data = load_PRECISE2_data()
    #
    # dpiAB_regulon = ['b0620', 'b0619']
    # purRC = ['b1658', 'b2476']
    # birA_regulon = ['b0775', 'b0777', 'b0778', 'b0776', 'b3973']
    # rutA_regulon = ['b1012', 'b1011', 'b1010', 'b1009', 'b1008', 'b1007', 'b1006']
    # trpR_regulon = ['b1260', 'b1261', 'b1262', 'b1263', 'b1264', 'b4393']
    # _, exp = get_gene_exp(symbols, seq_data, gene_names=trpR_regulon)
    # PRECISE2_hist_from_exp(exp, ['trpA', 'trpB', 'trpC', 'trpD', 'trpE', 'trpR'], 'trpR_operon_PRECISE2')
    # plot_overlapping_one_peak_genes(
    #     os.path.join(SAMPLE_REG_DIR, "one_peak_genes_v5_3"),
    # os.path.join(SAMPLE_REG_DIR, "all_one_peak_genes_overlap_plot.png"))

    ##### Honors proposal figures #####
    ## Figure 1
    # symbols, sample_data, seq_data, vectorized_samples, total_components = load_data_exclude_rewiring()
    # gene_exp_hist(["lolB", "purC", "hdeB"], symbols, seq_data, sample_data, "Honors_thesis_three_genes",
    #               has_gaussian=[1, 0, 0])

    ## Figure 2
    # symbols, sample_data, seq_data, vectorized_samples, total_components = load_data_exclude_rewiring()
    # plot_components = PlotComponents(total_components, vectorized_samples)
    # plot_components.plot_freq_components_grid(os.path.join(SAMPLE_REG_DIR, "sample_components_grid"))

    # ecocyc = EcocycReg(BASE_DIR)
    # ecocyc.init_transcr_reg_data(os.path.join(SAMPLE_REG_DIR, "ecocyc_reg_tfs"))
    # ecocyc.one_peak_transcr_reg_info(symbols, os.path.join(SAMPLE_REG_DIR,
    #                                                        "one_peak_genes_v5"),
    #                                  os.path.join(SAMPLE_REG_DIR, "ecocyc_reg_tfs_annotated"),
    #                                               "",
    #                                  os.path.join(SAMPLE_REG_DIR, "one_peak_genes_num_reg_v5"))

    ## Figure 3
    # def plot_gene_components(genes, comp1, comp2, title):
    #     symbols, sample_data, seq_data, vectorized_samples, total_components = load_data_exclude_rewiring()
    #     _, exp = get_gene_exp(symbols, seq_data, gene_names=genes)
    #     plot_components = PlotComponents(total_components, vectorized_samples)
    #     plot_components.plot_exp_control_components(exp, title, genes, comp1, comp2)
    #
    # def plot_gene_all_components(gene, title):
    #     symbols, sample_data, seq_data, vectorized_samples, total_components = load_data_exclude_rewiring()
    #     _, exp = get_gene_exp(symbols, seq_data, gene_names=[gene])
    #     plot_components = PlotComponents(total_components, vectorized_samples)
    #     plot_components.plot_all_components(exp[0], title)
    #
    # plot_gene_components(['purC', 'purF'],
    # {'adenine'}, {'methionine'}, "Honors thesis purC components")

    import ipdb

    ipdb.set_trace()
    # gene_names = []
    # with open(os.path.join(SAMPLE_REG_DIR, "one_peak_genes_v5_3"), 'r') as f:
    #     reader = csv.reader(f, delimiter='\t')
    #     next(reader)
    #     for line in reader:
    #         gene_names.append(line[0])
    #
    # with open(os.path.join(SAMPLE_REG_DIR, "one_peak_genes_v5_3_gene_only"), 'w') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     for gene in gene_names:
    #         writer.writerow([gene])
    # import ipdb; ipdb.set_trace()

    # symbols, sample_data, seq_data, vectorized_samples, total_components = load_data_exclude_rewiring()
    # compare_tail_stimulon_enrichment("gmk", 11.5, symbols, seq_data, sample_data)
    # plot_gene_all_components('adk', "adk_all_components")
    import ipdb

    ipdb.set_trace()
    # plot_gene_all_components('lldP', 'lldP_all_comp')
    # plot_gene_components(['yaiE', 'yfiH', 'deoD', 'add', 'xapA', 'gsk', 'gpt', 'hpt', 'rihC', 'apt'],
    #                     {'adenine'},
    #                     {'pantothenate', 'cytosine', 'uracil', 'guanine'},
    #                     "adenine_salvage_genes")

    # import ipdb; ipdb.set_trace()

    candidate_genes = [
        "FruR",
        "acrR",
        "adiA_adiY",
        "alpA",
        "arcA",
        "arsR",
        "atoC",
        "birA_murA",
        "cadC",
        "cbl",
        "cspA",
        "cytR",
        "fadR",
        "flhDC",
        "gatR_1",
        "gntR",
        "kdpDE",
        "leuO",
        "malT",
        "mhpR",
        "mlc",
        "modE",
        "nadR",
        "narL",
        "nlpD_rpoS",
        "rbsR",
        "rob",
        "rpoH",
        "rpoN",
        "rtcR",
        "treR",
        "xapR",
        "yjbK",
        "zntR",
    ]

    # load_data_exclude_rewiring()

    # ArsEFG changeed to ArsRBC
    # EmrR changed to mprA
    one_to_one_genes = [
        "acrA",
        "acrB",
        "adiA",
        "arsR",
        "arsB",
        "arsC",
        "cadB",
        "cadA",
        "mprA",
        "emrA",
        "emrB",
        "gatY",
        "gatZ",
        "gatA",
        "gatB",
        "gatC",
        "gatD",
        "gatR_2",
        "hipB",
        "hipA",
        "idnD",
        "idnO",
        "idnT",
        "idnR",
        "kdpA",
        "kdpB",
        "kdpC",
        "leuL",
        "leuA",
        "leuB",
        "leuC",
        "leuD",
        "mhpA",
        "mhpB",
        "mhpC",
        "mhpD",
        "mhpF",
        "mhpE",
        "moaA",
        "moaB",
        "moaC",
        "moaD",
        "moaE",
        "modA",
        "modB",
        "modC",
        "mtlA",
        "mtlD",
        "mtlR",
        "rbsD",
        "rbsA",
        "rbsC",
        "rbsB",
        "rbsK",
        "rtcA",
        "rtcB",
        "slp",
        "treB",
        "treC",
        "xapA",
        "xapB",
        "zntA",
        "znuA",
        "znuB",
        "znuC",
    ]

    # components, samples = test_treatments()
    curated_one_to_one_genes = [
        "xapR",
        "xapA",
        "xapB",
        "zntR",
        "zntA",
        "zur",
        "znuC",
        "znuB",
        "zinT",
        "znuA",
        "pliG",
        "ykgM",  #'ykgO',
        "moaA",
        "modA",
        "modE",
        "leuA",
        "leuO",
        "kdpD",
        "kdpE",  #'kdpF',
        "kdpA",
        "kdpB",
        "kdpC",
        "idnR",
        "idnD",
        "idnO",
        "hipA",
        "hipB",
        "cadC",
        "lysP",
        "cadB",
        "adiY",
        "adiA",
    ]

    # NOTE: ykgO, kdpF, not in dataset. lysP is synonym for cadR.

    # not_in_symbols = []
    # for gene in one_to_one_genes:
    #     if gene not in symbols:
    #         not_in_symbols.append(gene)

    # converted_names = ecocyc.ecocyc_to_ecomac_names(neg_autoreg, symbols)
    # converted_names = np.array(converted_names)
    # mask = (converted_names != "")
    # converted_names = converted_names[mask]
    # num_regulators = num_regulators[mask]

    # converted_names = [x for x in converted_names if x != ""]
    # updated_gene_names, duplicated_genes, missing_genes = ecocyc.curate_gene_names(symbols)
    # rank, difference, ll_two_cluster, ll_one_cluster = make_gene_rankings(seq_data, symbols, symbols, "gmm_llr_rankings_no_rewiring")

    # gene_exp_hist(["yejH", "araE", "hdeB"], symbols, seq_data, sample_data, "honors_proposal_genes")
    # ["yejH", "araE", "hdeB", "treR", "treB", "dnaA"]
    # low STD genes: yejH, yfgD
    # low STD and low z: rutR, paaX, ydfH
    # Two-peak genes: araE, ygeF
    # treR, treB, dnaA, hdeB
    # arabinose_gene_names = ['araB', 'araA', 'araD', 'araE',
    #     'ytfQ', 'ytfR', 'ytfT', 'yjfF', 'araF', 'araG', 'araH_1', 'araH_2', 'ygeA', 'ydeN', 'ydeM',
    #     'xylA', 'xylB']
    # mannitol_gene_names = ['mtlA', 'mtlD', 'mtlR', 'srlA', 'srlB', 'srlE']
    rutR_regulon = [
        "rutR",
        "rutA",
        "rutB",
        "rutC",
        "rutD",
        "rutE",
        "rutF",
        "rutG",
        "carA",
        "carB",
        "fepB",
        "gmr",
        "nemR",
        "nemA",
        "gloA",
        "gadX",
        "gadW",
    ]
    # # TODO: add more genes to rutR_regulon
    thiamine_riboswitch = [
        "thiM",
        "thiD",
        "thiC",
        "thiE",
        "thiF",
        "thiG",
        "thiH",
    ]  # EcoMAC is missing thiS
    sgrR_regulon = ["sgrR", "tbpA", "thiP", "thiQ", "setA"]  # small RNA sroA,
    # and divergently small RNAs sgrR, sgrT with setA
    orphan_thiamine_synth_genes = ["thiL", "thiI", "thiK"]
    cytR_regulon = [
        "tsx",
        "ycdZ",
        "cdd",
        "nupC",
        "nupG",
        "ppiA",
        "rpoH",
        "udp",
        "cytR",
        "deoC",
        "deoA",
        "deoB",
        "deoD",
    ]
    pyridoxal_salvage = ["ydbC", "pdxY", "pdxH", "pdxK"]
    lipoate_salvage = ["lplA", "ytjB"]
    nadR_regulon = ["nadR", "nadA", "pnuC", "pncB", "nadB"]
    modE_regulon = [
        "modE",
        "modA",
        "modB",
        "modC",
        "moaA",
        "moaB",
        "moaC",
        "moaD",
        "moaE",
        "dmsA",
        "dmsB",
        "dmsC",
        "narX",
        "narL",
        "oppA",
        "oppB",
        "oppC",
        "oppD",
        "oppF",
        "napF",
        "napD",
        "napA",
        "napG",
        "napH",
        "napB",
        "napC",
        "ccmA",
        "ccmB",
        "ccmC",
        "ccmD",
        "ccmE",
        "ccmF",
        "ccmG",
        "ccmH",
        "hycA",
        "hycB",
        "hycC",
        "hycD",
        "hycE",
        "hycF",
        "hycG",
        "hycH",
        "hycI",
        "deoC",
        "deoA",
        "deoB",
        "deoD",
    ]
    zur_regulon = ["zur", "ykgM", "pliG", "znuC", "znuB", "znuA", "zinT"]

    arg_regulon = ["argD", "argE", "argF", "argI", "argR"]

    purR_regulon = [
        "purR",
        "speA",
        "speB",
        "purL",
        "glnB",
        "purC",
        "cvpA",
        "purF",
        "ubiX",
        "purT",
        "purR",
        "prs",
        "hflD",
        "purB",
        "pyrD",
        "purE",
        "purK",
        "codB",
        "codA",
        "carA",
        "carB",
        "pyrC",
        "purM",
        "purN",
        "glyA",
        "guaB",
        "guaA",
        "purA",
        "purH",
        "purD",
    ]
    cysB_regulon = [
        "cysB",
        "yciW",
        "ydjN",
        "fliY",
        "cysK",
        "tauA",
        "tauB",
        "tauC",
        "tauD",
        "ybdN",
        "ssuE",
        "ssuA",
        "ssuD",
        "ssuC",
        "ssuB",
        "cysP",
        "cysU",
        "cysW",
        "cysA",
        "cysM",
        "yfbR",
        "cbl",
        "yoaC",
        "ydeH",
        "hslJ",
        "ariR",
    ]

    purR_regulon_filtered = [
        "purL",
        "purC",
        "cvpA",
        "purF",
        "purT",
        "purE",
        "purK",
        "codB",
        "codA",
        "purM",
        "purN",
        "purH",
        "purR",
    ]

    birA_regulon = ["birA", "bioA", "bioB", "bioC", "bioF", "bioD"]

    cueR_regulon = ["cueR", "cueO", "copA"]
    rbs_regulon = ["rbsA", "rbsB", "rbsC", "rbsD", "rbsK", "rbsR"]

    # ['ilvY', 'ilvC']
    # gene_names = get_candidate_two_peaks()
    # gene_names = ['ribD']

    # samples, in_range = get_samples_in_range(exp[0], sample_data, (0, 6.5))

    # # NOTE: srlABE are not induced by mannitol, but added bc they do transport mannitol
    # plot_names = [x+'1010' for x in gene_names]

    def _get_filtered_data(gene_names=None, return_data=False):
        symbols, sample_data, seq_data = load_data_exclude_rewiring()
        if gene_names == None:
            gene_names = symbols
        _, exp = get_gene_exp(symbols, seq_data, gene_names=gene_names)
        components, vectorized_samples = test_treatments()
        plot_components = PlotComponents(components, vectorized_samples)

        exclude = [
            "biofilm",
            "stationary",
            "transition",
            "ethanol",
            "Adaptive evolution",
            "formation",
            "autoinducer-2",
            "CORM-2",
            "attachment",
            "R1drd19 plasmid",
            "maturation",
            "heat shock",
            "PenG",
            "zinc (II) oxide",
            "norfloxacin",
        ]
        # PenG makes L-form colonies, norfloxacin kills cells, zinc (II) oxide was that outlier magnetic field study (maybe put back)
        filtered_exp, filtered_components, filtered_samples = (
            plot_components.exclude_component(exp, exclude)
        )

        return filtered_exp, filtered_components, filtered_samples, gene_names

    def plot_genes_filtered_components(gene_names, title, exp_comp, control_comp):
        """
        Plots the expression of genes in gene_names, only including samples
        that have been filtered. Plots them also side by side with including
        or excluding each component in "components".
        """
        filtered_exp, filtered_components, filtered_samples, _ = _get_filtered_data(
            gene_names
        )
        plot_components = PlotComponents(filtered_components, filtered_samples)
        plot_components.plot_exp_control_components(
            filtered_exp,
            title,
            gene_names,
            exp_comp=exp_comp,
            control_comp=control_comp,
            vectorized_samples=filtered_samples,
        )

    # plot_genes_filtered_components(['lolB', 'chaC'], "lolB_chaC_filtered", 'fumarate', 'arginine')
    # import ipdb; ipdb.set_trace()
    # dcuRS = ['dcuR', 'dcuS']
    # plot_genes_filtered_components(rbs_regulon, "rbsR_regulon_filtered", 'adenine', 'arginine')

    def plot_genes_filtered(gene_names, title):
        """
        Plots the expression of genes in gene_names, only including samples
        that have been filtered.
        """
        filtered_exp, *_ = _get_filtered_data(gene_names)
        hist_from_exp(filtered_exp, gene_names, title)

    # plot_genes_filtered(["pepQ"], "pepQ")
    # plot_genes_filtered_components(["msrA", "ytfT"], "check_znO", "zinc (II) oxide", "arginine")
    # plot_genes_filtered_components(["msrA", "ytfT"], "check_bicarbonate", "bicarbonate", "arginine")
    # plot_genes_filtered_components(["lolB", "dcuR"], "check_fumarate", "fumarate", "arginine")
    # plot_genes_filtered_components(["potE", "znuA", 'nuoM', 'nuoG'], "check_homopipes", "HOMOPIPES", "pH5.5")

    # #
    # ecocyc = EcocycReg(BASE_DIR)
    # ecocyc.init_transcr_reg_data()
    # ecocyc.one_peak_transcr_reg_info(symbols, os.path.join(SAMPLE_REG_DIR, "one_peak_genes_v5_3"),
    #                                  os.path.join(SAMPLE_REG_DIR, "ecocyc_reg_tfs_annotated"),
    #                                  "",
    #                                  os.path.join(SAMPLE_REG_DIR, "new_one_peak_genes_reg_plot_v5_3"))

    # symbols, sample_data, seq_data, vectorized_samples, total_components = load_data_exclude_rewiring()

    # plot_components = PlotComponents(total_components, vectorized_samples)
    # plot_components.write_second_peak_data(symbols, seq_data,
    #                                        os.path.join(SAMPLE_REG_DIR, "p_values_v6"),
    #                                        os.path.join(SAMPLE_REG_DIR, "diff_means_v6"),
    #                                        os.path.join(SAMPLE_REG_DIR, "second_peak_results_v6.json"))
    # import ipdb; ipdb.set_trace()

    # plot_components.write_one_peak_genes(
    #     seq_data,
    #     symbols,
    #     os.path.join(SAMPLE_REG_DIR, "p_values_v5"),
    #     os.path.join(SAMPLE_REG_DIR, "diff_means_v5"),
    #     1.0,
    #     -15,
    #     0.65,
    #     50,
    #     os.path.join(SAMPLE_REG_DIR, "one_peak_genes_v5_3")
    # )
    # plot_components.write_stats_for_low_spread_genes(seq_data, symbols, os.path.join(SAMPLE_REG_DIR, "mwu_u_stats_v4"),
    #                                                 os.path.join(SAMPLE_REG_DIR, "mwu_p_values_v4"),
    #                                                  os.path.join(SAMPLE_REG_DIR, "mwu_diff_means_v4"))
    # import ipdb; ipdb.set_trace()
    # plot_components.write_freq_components(os.path.join(SAMPLE_REG_DIR, "freq_components_v6"))
    # plot_components.detect_reg(seq_data, symbols, os.path.join(SAMPLE_REG_DIR, "p_values_v6"),
    #                         os.path.join(SAMPLE_REG_DIR, "t_statistics_v6"),
    #                          os.path.join(SAMPLE_REG_DIR, "diff_means_v6"))

    import ipdb

    idpb.set_trace()
    # plot_components.write_stats_for_low_spread_genes(seq_data, symbols, os.path.join(SAMPLE_REG_DIR, "mwu_u_stats_v4"),
    #                                                 os.path.join(SAMPLE_REG_DIR, "mwu_p_values_v4"))
    # fumarate_reg_genes = ['dctA', 'fumB', 'dcuB', 'frdA', 'frdB', 'frdC', 'frdD']

    # tdc_operon = ['tdcA', 'tdcB', 'tdcC', 'tdcD', 'tdcE', 'tdcF']
    # nar_genes = ['narG', 'narH', 'narJ', 'narI']
    # nar_other_genes = ['narZ', 'narY', 'narW', 'narV']
    # moa_operon = ['moaA', 'moaB', 'moaC', 'moaD', 'moaE']
    # deo_operon = ['deoA', 'deoB', 'deoC', 'deoD']
    # gene_exp_hist(deo_operon, symbols, seq_data, sample_data, "Deo_operon")
    # _, exp = get_gene_exp(symbols, seq_data, gene_names=moa_operon)
    # _, exp1 = get_gene_exp(symbols, seq_data, gene_names=nar_genes)
    # _, exp2 = get_gene_exp(symbols, seq_data, gene_names=nar_other_genes)
    # plot_components.plot_exp_control_components(exp, "test_fumarate_low_p_value", ['ymfN', 'cysB', 'rcsB'],
    #                                           'fumarate', 'arginine')
    # plot_components.plot_exp_control_components(exp, "fumarate_reg_genes", fumarate_reg_genes,
    #                                            "fumarate", "arginine")
    # plot_components.plot_exp_control_components(exp1, "narGHJI_operon", nar_genes,
    #                                            "nitrate-anaerobic", "anaerobic")
    # plot_components.plot_exp_control_components(exp2, "narZYWV_operon", nar_other_genes,
    #                                             "glycerophosphate, nitrate-aerobic, selenium", "anaerobic")
    # plot_components.plot_exp_control_components(exp, "moa_operon", moa_operon,
    #                                            "DMSO", "molybdate")
    # TODO: so what to do about weird cases like moaA, where for DMSO it has one at 5 and one at 9, so it's not really
    # a lower mean seems more like outliers in data?

    # TODO: tdc operon is catabolite repressed, main point is that its in anaerobic WITHOUT glucose
    #  narGHJI shows a big effect, but maybe its not statistically
    # significant bc only 2 samples?
    # TODO: narZYWV is weird, nitrate-aerobic (K medium) gives it at much less altho it should be more maybe?
    # although nitrate-aerobic does have relatively low levels of nitrate.
    # Also, anaerobic is supposed to repress it, but doesn't rly show that repressive effect.
    # Maybe check what has the rly low p-value for that one?
    # TODO: look at samples for minimum p-values
    # TODO: look at the histograms for genes that are regulated by both nitrate and anaerobic, etc.

    # coliNet = coliNetData()
    # colinet_genes = coliNet.gene_names
    # # NOTE: includes genes that are not regulators themselves but are in the same operon as regulators
    # colinet_genes_in_regulator = coliNet.gene_in_regulator_operon
    #
    # ecocyc = EcocycReg(BASE_DIR)
    # ecomac_colinet_names = ecocyc.ecocyc_to_ecomac_names(colinet_genes, symbols)
    #
    # colinet_reg_components = coliNet.gene_reg_components

    # plot_components.detect_reg(seq_data, symbols, os.path.join(SAMPLE_REG_DIR, "p_values_v5"),
    #                          os.path.join(SAMPLE_REG_DIR, "t_statistics_v5"),
    #                           os.path.join(SAMPLE_REG_DIR, "diff_means_v5"))

    # plot_components.plot_stats_for_high_freq_components(
    #     50, os.path.join(SAMPLE_REG_DIR, "p_values_v5"),
    #     ecomac_colinet_names, colinet_reg_components,
    #     colinet_genes_in_regulator,
    #     os.path.join(SAMPLE_REG_DIR, "plot_high_freq_p_values_with_special_v5_3"),
    #     os.path.join(SAMPLE_REG_DIR, "write_coliNet_comp_vs_min_p_comp_high_frq_v5_3")
    # )
    # plot_components.plot_stats_for_low_freq_components(
    #     50, os.path.join(SAMPLE_REG_DIR, "diff_means_v5"),
    #     ecomac_colinet_names, colinet_reg_components,
    #     colinet_genes_in_regulator,
    #     os.path.join(SAMPLE_REG_DIR, "plot_low_freq_diff_means_with_special_v5_3"),
    #     os.path.join(SAMPLE_REG_DIR, "write_coliNet_comp_vs_max_dm_comp_low_freq_v5_3"))

    # Idea: if we look at all genes,
    # So what are we looking for? A way to say, these genes are one-peak bc for any component, some statistic is less than
    # some cutoff. How we validate this is by saying, look at these confirmed regulation coliNet gene-component pairs,
    # their statistic is greater than the cutoff.

    # TODO: so, one-peak criteria(?): a gene must have standard deviation below say 0.8, then for low-freq components (<50 samples?) it must
    # have diff-means less than 1.0 for all components (or, > some standard-dev ratio?), and then for high-freq components (>50 samples) it must have a high p-value for all components
    # from Welch's t-test.
    # So, to assess a cutoff for low-freq components, we want coliNet regulated genes to generally have their gene-component statistic greater than that.
    # And for high-freq components, also want their Welch's t-test p-values to be greater than that.
    # So make a plot for high-freq components now that compares coliNet-reg gene-component pairs to overall, and to max p-value among high component pairs?

    # plot_components.plot_statistics_with_special(seq_data, symbols, ecomac_colinet_names, colinet_reg_components,
    #                                              colinet_genes_in_regulator,
    #                                              os.path.join(SAMPLE_REG_DIR, "p_valuesv3"),
    #                                              os.path.join(SAMPLE_REG_DIR, "t_statsv3"),
    #                                              os.path.join(SAMPLE_REG_DIR, "diff_means_v5"),
    #                                              os.path.join(SAMPLE_REG_DIR, "stats_plot_with_coliNet_v5"),
    #                                              os.path.join(SAMPLE_REG_DIR, "by_comp_stats_plot_with_coliNet_v5"))

    # TODO: paraquat (adds superoxide) for the EZ Rich defined medium case, J. Blanchard, they also said minimal effect on
    # growth rate which is nice. # TODO: that would change coliNet stuff too
    # TODO: whether to take out the Y. Yang BHI agar part maybe because it might be at stationary phase? how to check?

    # TODO: curate the nitrate concentrations in the media, bc only four samples have nitrate, but 18 other samples have nitrate
    # in media, maybe that's making it show less of a change in p-values &diff_means?? also there's the difference between anaerobic and not anaerobic
    # So now, for each colinet gene, by which ligand do we expect it to be perturbed?
    # And then can look p-values, t-stats, and diff-means for those?
    # TODO: need to fix M. Lin JM 109, the salt shock for the irrE ones, and also whether to keep glyphosate?
    # TODO: C. Reigstad, later time points of aerobic/anaerobic were from stationary phase, so probably want to remove those
    # TODO: check argP, for difference between PU... and GM... as being the two sets of microarrays? and if that's the case,
    # think about what to do for more broad cases? (may be important!)
    # TODO: partial least squares regression, think(??)
    # TODO: is sulfate in rich media?
    # TODO: nirC looks not regulated but nirBD do, might nirC have its own promoter or smth? Check with REND-seq or smth?
    import ipdb

    ipdb.set_trace()

    # genes_reg_by_x('argR', symbols, os.path.join(SAMPLE_REG_DIR, "one_peak_genes2"),
    #                os.path.join(SAMPLE_REG_DIR, "one_peak_reg_by_argR2"))
    # filtered_exp, filtered_components, filtered_samples, gene_names = _get_filtered_data()
    # plot_components = PlotComponents(filtered_components, filtered_samples)
    # plot_components.candidate_two_peak(gene_names, filtered_exp, os.path.join(SAMPLE_REG_DIR, "all_regs2"),
    #                                   os.path.join(SAMPLE_REG_DIR, "two_peak_all_regs"))

    # plot_components.identify_two_peak(os.path.join(SAMPLE_REG_DIR, "two_peak_all_regs"),
    #                                  os.path.join(SAMPLE_REG_DIR, "two_peak_genes"))
    # two_peak_reg_analyze(gene_names, os.path.join(SAMPLE_REG_DIR, "two_peak_genes"),
    #                     os.path.join(SAMPLE_REG_DIR, "two_peak_num_reg_hist"),
    #                     os.path.join(SAMPLE_REG_DIR, "two_peak_reg_data"))
    # plot_components.detect_reg(filtered_exp, gene_names, os.path.join(SAMPLE_REG_DIR, "p_values2"),
    #                           os.path.join(SAMPLE_REG_DIR, "t_statistics2"),
    #                           os.path.join(SAMPLE_REG_DIR, "p_t_plot2"),
    #                           os.path.join(SAMPLE_REG_DIR, "all_regs2"),
    #                           os.path.join(SAMPLE_REG_DIR, "no_reg_genes2"))
    # plot_components.classify_genes_by_reg(os.path.join(SAMPLE_REG_DIR, "p_t_plot2"),
    #                                       os.path.join(SAMPLE_REG_DIR, "p_values2"),
    #                                       os.path.join(SAMPLE_REG_DIR, "t_statistics2"),
    #                                       os.path.join(SAMPLE_REG_DIR, "all_regs2"),
    #                                        os.path.join(SAMPLE_REG_DIR, "no_reg_genes2"))
    # comp_null_standard_dev_vs_reg(gene_names, os.path.join(SAMPLE_REG_DIR, "no_reg_genes2"),
    #                                filtered_exp, os.path.join(SAMPLE_REG_DIR, "standard_dev_vs_reg2"))
    # write_one_peak_genes(gene_names, os.path.join(SAMPLE_REG_DIR, "no_reg_genes2"),
    #                       filtered_exp, os.path.join(SAMPLE_REG_DIR, "one_peak_genes2"))
    # one_peak_num_reg_histogram(gene_names, os.path.join(SAMPLE_REG_DIR, "one_peak_genes2"),
    #                             os.path.join(SAMPLE_REG_DIR, "one_peak_num_reg_hist2"))
    # one_peak_reg_info(gene_names, os.path.join(SAMPLE_REG_DIR, "one_peak_genes2"),
    #                    os.path.join(SAMPLE_REG_DIR, "one_peak_reg_info2"),
    #                    os.path.join(SAMPLE_REG_DIR, "one_peak_reg_plot2"))

    import ipdb

    ipdb.set_trace()
    # get_candidate_one_peak(filtered_exp, gene_names, os.path.join(OUTPUT_DIR, "one_peak_standard_dev_0.7_twofold"),
    # method='standard_dev')
    # Use a different method to get one-peak
    # Make the rankings plot
    # Make the histogram plot
    # Do parameter sensitivity

    dpiAB_regulon = ["dpiA", "dpiB", "citC", "citD", "citE", "citF", "citX", "citG"]
    # plot_genes_filtered(dpiAB_regulon, "dpiAB_regulon_filtered")
    # plot_genes_filtered_components(dpiAB_regulon, "dpiAB_regulon_filtered", "anaerobic", "arginine")

    # sigma_factors = ['rpoD', 'rpoE', 'rpoN', 'rpoH', 'rpoS', 'fliA']
    # chbR_regulon = ['chbR', 'chbA', 'chbB', 'chbF', 'chbC', 'chbG']

    # crp_investigation(symbols)

    # one_peak_reg_info(symbols, os.path.join(OUTPUT_DIR, 'one_peak_reg_by_crp_annotated'),
    #                  '', '')
    # ulaR_regulon = ['ulaR', 'ulaA', 'ulaC', 'ulaB', 'ulaD', 'ulaE', 'ulaF']
    # gene_exp_hist(['dcuR', 'dcuS'], symbols, seq_data, sample_data, "dcuRS")

    # one_peak_num_reg_histogram(symbols, os.path.join(OUTPUT_DIR, "one_peak_95%_2fold"), os.path.join(
    # OUTPUT_DIR, "one_peak_num_reg_hist"))

    # one_peak_reg_info(symbols, os.path.join(OUTPUT_DIR, "one_peak_95%_2fold_no_sigma"),
    #                  os.path.join(OUTPUT_DIR, "one_peak_reg_info_no_sigma"),
    #                  os.path.join(OUTPUT_DIR, "one_peak_regulators_compare_no_sigma"))

    # TODO: so for all these genes, we want to classify by: 1. is it no regulators at all,
    # 2. is it autoregulated only, 3. does it have other regulators (3a no autoreg, 3b yes autoreg),
    # For the class 3, we can look into the other regulators. Can say: 1. only Fis, Lrp, etc.
    # 2. Includes real other regulators.

    # in_range_vectors = vectorized_samples[in_range]
    # valine_idx = np.where(components=='valine')
    # adenine_idx = np.where(components=='adenine')[0][0]
    # no_adenine_mask = (in_range_vectors[:, adenine_idx] == 0.)
    # no_adenine_samples = {k: samples[k][no_adenine_mask] for k in samples.keys()}
    # adenine_samples = {k: samples[k][~no_adenine_mask] for k in samples.keys()}
    # import ipdb; ipdb.set_trace()
    # plot_components = PlotComponents(components, vectorized_samples)
    # plot_components.plot_components(exp, 'ilvY_regulon_components', gene_names,
    #                                components=['valine', 'glucose'])
    # plot_components.get_enriched()
    # plot_components.plot_components(exp[0], "arg_Icomponents")
    # plot_components.plot_enriched(exp[0], (0, 8), 'metHlow', combos=2)

    # peak_finder = PeakDetection()
    # peak_finder.gene_data(gene_names, exp, "neg_autoreg_peaks", "neg_autoreg_peaks.txt")
    # peak_finder.plot_genes(exp, gene_names, "neg_autoreg_genes_num_reg", num_regulators=num_regulators)
    # peak_finder.plot_genes(exp, gene_names, "argRRegulon")
    # peaks = peak_finder.detect_peaks_multigene(exp, plot_names=plot_names)
    # peaks = peak_finder.detect_peaks(exp[0], plot_name='araJ1010')
    # peak_finder.gene_data(symbols, exp, 'test_all_peaks', 'test_all_peaks.txt')

    import ipdb

    ipdb.set_trace()
    # TODO: fix the norfloxacin ones that actually do have arabinose
    # TODO: make a fcn that can plot for any gene given any components
    # TODO: deal with the times, like if time is -300, then you shouldn't
    # actually include the component!
    # TODO: LB doens't have glucose

    # TODO: look at these AraA, figure out how to work with the samples and do smth with it!

    # peak_finder = PeakDetection()
    # peaks = peak_finder.detect_peaks_multigene(exp, plot_names=plot_names)
    # peaks = peak_finder.detect_peaks(exp[0], plot_name='ygeA1010')
    # peak_finder.gene_data(symbols, exp, 'all_peaks_nopathogen', 'all_peaks_nopathogen.txt')
    # TODO: maybe do it using widths instead of standard devs, maybe normalized
    # by area of the peak or smth (but that's also less "empirical" maybe??)?
    # peak_finder.gene_data(symbols, exp, "all_peaks", "all_peaks_data.txt")
    # TODO: plot some representative examples from the scatterplot in all_peaks,
    # to see if you can get a sense of what they look like? In particular,
    # one with both very low, a couple average ones, one with a highish and lowish,
    # one with a lowish and highish, and one with both high or one high and one
    # low to show the extremes.
    # TODO: get a list of genes that you might call the category?
    import ipdb

    ipdb.set_trace()
    # def cluster_genes(gene_names, plot_name):
    #     gene_names, gene_exp = get_gene_exp(symbols, seq_data, gene_names=gene_names)
    # #hist = HistogramPreparations()
    #     seq_align = HistogramAlignment()
    #     scores, dn1 = seq_align.cluster_dendrogram(gene_exp, gene_names, os.path.join(OUTPUT_DIR, plot_name+"Dendrogram"))
    #     gene_exp_hist(dn1['ivl'][::-1], symbols, seq_data, sample_data, plot_name+"Histograms")
    #     return scores
    #
    # scores = cluster_genes(symbols[:50], "TestBigDendrogramSeqAlign")
    # hist.hist_of_hist(seq_data, "Hist_of_hist")

    # gene_names, gene_exp = get_gene_exp(symbols, seq_data, gene_names=symbols)
    # metrics = SimilarityMetrics()
    # leaves = metrics.cluster_genes(gene_exp, gene_names, "all_cluster")
    # heatmap = MakeHeatmaps()
    # heatmap.heatmap(gene_exp, "all_cluster_heatmap ", leaves)
    #
    # leaves = metrics.cluster_genes(gene_exp, gene_names, "all_cluster_heilinger", metric="Heilinger")
    # heatmap.heatmap(gene_exp, "all_cluster_heatmap_heilinger", leaves)
    # llr_genes = get_genes_from_file('gmm_llr_rankings_no_rewiring')
    # llr_genes = llr_genes[-500:]
    # candidates = get_genes_from_file("candidate_genes.txt")
    # plot_histograms(symbols, seq_data, candidates, "low_std_heatmap")
    # plot_gmm_llr(symbols, seq_data, "gene_histograms_gmm_llr")
    # import ipdb; ipdb.set_trace()
    # plot_exp(seq_data, [0, 100, 1000])
    # make_table(sample_data, sample_data)
    # compare_tail_stimulon_enrichment("recB", 8, symbols, seq_data, sample_data)
    # test_treatments()
    # synonyms = load_gene_synonyms()
    # b_numbers, symbols = load_gene_synonyms()
    # candidates = get_low_z_candidates()

    # ecocyc.find_gene_regulation(symbols)
    # ecocyc.update_synonyms(symbols)
    # all_genes = set(ecocyc.all_regulated_genes)
    # all_genes.update(set(ecocyc.regulation.keys()))
    # ecocyc.categorize_genes(symbols)
    # plot_gene_profiles(symbols, candidates[:50], seq_data, sample_data, ecocyc)
    # ecocyc.compare_category_statistics(symbols, candidates)
    # make_binary_table(sample_data, sample_data)
    # plot_rRNAs(symbols, seq_data, sample_data)

    # averaged_exp, types = average_exp(seq_data, sample_data)
    # gene_names_high_llr = ['rrlG', 'rrsH', 'uxuR', 'rrfH', 'ygaQ_3', "thrS", 'iadA', "rplX", "yjiE",
    #                       "yjiP", "rrrD", "yjiR", "fusA", "yahJ", "hsdM", "ybdN", "yfdQ", "ybeT", "citE", "rplN",
    #                       "yjiX", "setA", "ygeM"]

    # test_classify(sample_data)
    # gene_names = ['yejH', 'yfgD', 'degQ', 'yeaN', 'helD', 'ulaD', 'nudL', 'csdL', 'yqhC', 'tag']

    # compare_tail_stimulon_classify(gene_names, symbols, seq_data, sample_data)
    # gene_names = ['trpR', 'trpA', 'dnaA', 'mtr', 'treB', 'yaiA', 'rplL']
    # plot_mean_std(symbols, symbols, seq_data, "all_genes_mean_std")
    # bootstrap_max_normal(2198, 10000)
    # plot_max_z_score_for_candidates(symbols, seq_data, 'candidate_genes.txt')
    # get_candidate_genes(symbols, os.path.join(OUTPUT_DIR, "std_prob_plot.txt"), os.path.join(OUTPUT_DIR, "candidate_genes.txt"))
    # calc_with_moments(seq_data, symbols, symbols, plot_file="momentsNew", write_file="momentsNew.txt")
    # std_and_prob_plot(seq_data, symbols, symbols, plot_file='std_prob_plot', write_file='std_prob_plot.txt')

    # rank_by_std_ll(seq_data, symbols, symbols, plot_file="all_std_ll", write_file='all_std_ll.txt')
    # gene_names = ['yojI', 'ygeR', 'ybjK', 'uvrC', 'yeaN', 'dsbC', 'cmoB', 'yejA', "ybaK", 'murJ'] # low range
    # gene_names = ['yejH', 'yfgD', 'degQ', 'yeaN', 'helD', 'ulaD', 'nudL', 'csdL', 'yqhC', 'tag']
    # compare_tail_stimulons(['treB'], symbols, seq_data, sample_data)
    # for i in range(10):
    #    score, p_value, _ = compare_tail_stimulons(['treB'], symbols, seq_data, sample_data)
    #    scores.append(score)
    #    p_values.append(p_value)
    # yejH, yfgD, ulaD
    # yejH: no known regulation; yfgD: nothing known; ulaD: regulated by CRP
    # degQ: no known regulation; yeaN: regulated by nimR; helD: no known regulation; nudL: no known regulation;
    # csdL: no known regulation; yqhC: regulated by glaR (aquino17) maybe autoregulation, tag (no known regulation)
    # TODO: which tag? (neither have regulation)
    # TODO: test out this algorithm
    # TODO: regulon db thing with the low-std genes
    # TODO: for prob plot, or kurtosis/skewness, for test to normal, get threshold for what is candidate-unregulated
    # TODO: make a plot for the tables of samples that sees the distribution, and maybe the classifier scheme too?
    #  and best to see the correlations too?
    # TODO: is p-value too good to be true?? it's bc there's lots of internal correlations
    # TODO: problem of e.g. "15" being a treatment value, it's just a time but no actual treatment being made,
    #  you might group together things that shouldn't be grouped together?
    # TODO: some type of "validation", e.g., do all LBs tend to be lower than M9s for treB, etc.: incorporate into the statistic
    # TODO: maybe try out using the choosing probability instead
    # TODO: make a statistic for when to identify two groups as significantly different

    # TODO: make kurtosis and skewness and r^2 plots, and get a count for how many are below some level of both
    # TODO: use classifier scheme for the low-std genes
    # Idea: first about kurtosis/skewness/r^2, likelihood doesn't work, then show plots, then show
    # 2d plots, and how many are below some value. Then want to do smth like how many genes regulated,
    # ask how to get that. Also show the normal plots overlaid.
    # Then, used 6 features instead of binary (author, strain (some r just like a minor mutation), media,
    # treatments (whether to include time or not include time, issue above, also quite combinatorial),
    # gene, and gr. Tried to make a score for how diff two things are, tried similarity score,
    # settled on current "score": what fraction can be classified correctly, which is just for each feature
    # type you're considering, how uneven they are. And you can do it for the 6 features, or any combination
    # of them (supposedly, with all combined, you'll get a perfect score). Then can do bootstrapping based
    # on randomly drawn samples to get a p-value. Tried on treB, got that media, author, and treatment
    # were p=0.0, and media tells a clear story (M9C, LB, LB+homopipes rich media are high, others are low),
    # and author or treatment completely determine media and so that explains them, and also strain
    # was p=0.7 tho still p high score like 0.9, and gene was p=0.004ish, score around 0.93?
    # Two problems I can for-see: 1. False positives due to chance. When u get small enough samples per category,
    # there's a high chance they'll all fall within a certain expression group, though we try to get an idea
    # for this based on the p-values. Maybe it's concerning that strain gets 0.9 score already?
    # and this is exarcebated by the fact that samples within a category are highly correlated
    # in expression even if the category
    # has nothing to do with the causative reason, bc whatever the causative agent is, they're p similar.
    # 2. Within a category, they're correlated to a causative agent, bc the data points in the category
    # have correlation with the causative agent. If this causative agent is associated with something
    # that's actually in our features, it's possible to identify this error bc then we can see the high
    # correlation between some of our features and others. Even if there's no pairwise correlation,
    # there might be higher level correlation, especially significant if the causative features is also
    # spread across many types in our categorization. And so we might get a lot of false positives,
    # where we get a high score just because the category is correlated with something else that is causative.
    # But then again, it's also possible that indeed it's bc of treatments, and half of them satisfy some
    # property, and half of them don't. Maybe then u see whether one group has a low amount of treatments
    # or smth?

    # We can detect simple pairwise correlations (e.g. between media and treatment) by using those pie charts
    # and I'm sure there's a metric for it, asymmetric info smth.
    # Questions then: 1. obviously, how to deal with this? (is it possible to somehow keep the correlation
    # in our bootstrapping to see)?
    # 2. the idea about splitting into different actual
    # chemicals? pro is that features r more independent, con is that true causative reason may be more
    # complex than simply just whether a media has smth, etc. 3. Make a statistic for two groups different
    # 4. are there any other things we can use? thought about the probability thing, but that still
    # suffers from the correlation thing, and random chance with small values still.
    # 5. Some type of validation?

    # Also, the score thing was kinda able to detect the low-std genes (three genes were significant, two
    # were rly significant, one of them was shown to be regulated, the other was unknown; the seven
    # that were not significant, only one of them was actually regualted), however it showed that treB
    # was rly insignificant.

    # gene_exp_hist(['recB', 'recD'], symbols, seq_data, sample_data, title="recBD test")
    # rank_by_spread(symbols, symbols, seq_data, sample_data, output_name="all_genes_spread_rank", output_plot_name="all_genes_spread_corr", avg_method=None)
    # data = get_gene_spread_data(gene_names, os.path.join(OUTPUT_DIR, "all_genes_spread_rank.txt"))
    # plot_gene_spread_data(os.path.join(OUTPUT_DIR, "all_genes_spread_rank.txt"), "spread_properties")

    # gene_exp_hist(gene_names_high_llr, symbols, seq_data, sample_data, title="histogram_noavg_high_llr", avg_method=None)

    # _, exp = get_gene_exp(symbols, seq_data, gene_names=['treB'])
    # samples_plot(symbols, gene_names, seq_data, sample_data, avg_method="media")
    # classify_samples(sample_data, None)
    # rank, _, _, _ = make_gene_rankings(seq_data, symbols, gene_names, "ggm_llr_test")
    # rank, difference, ll_two_cluster, ll_one_cluster = make_gene_rankings(seq_data, symbols, symbols, "gmm_llr_minus_std_rankings")

    # outlier_ranks, outlier_zs = rank_by_outliers(seq_data, gene_names, gene_names)
    # for name in gene_names:
    #    name, expression = get_gene_exp(symbols, seq_data, gene_name=name)
    #    _, avg_exp = get_gene_exp(symbols, averaged_exp, gene_name=name)
    #    gene_exp_hist(name, expression)
    #    gene_exp_hist(name, avg_exp, avg_method='media')
    # x = run_gmm(expression)
    import ipdb

    ipdb.set_trace()
    # for i in range(10):
    #    name, expression = get_gene_exp(symbols, seq_data, gene_idx=i)
    #    gene_exp_hist(name, expression)

# So we want ways to tell if a gene is likely a transcription factor, being controlled in some way.
# If the gene
