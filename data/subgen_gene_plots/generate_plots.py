"""
This script creates a bar graph showing the number of antibiotic response genes and
the number of all genes that are sub-generational vs. generational. We define
generational genes to be those that are transcribed at least once per generation.

antibiotic_response_genes.txt was created by listing out all the genes EcoCyc
considers related to antibiotic response in E. Coli
(https://ecocyc.org/ECOLI/NEW-IMAGE?type=ECOCYC-CLASS&object=GO:0046677). ompF and
ompC (porins) were also included in this txt file even though they were not listed
on EcoCyc.
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle

# Number of genes considered generational in the wcEcoli Science paper (Fig 4c) minus one
CUTOFF_INDEX = 1546
# The RNA synthesization probabilites are from self.rnaSynthProb in the transcript_initiation process of the release
# version of wcEcoli
RELEASE_RNA_PROB_PATH = 'data/subgen_gene_plots/release_rna_probs.txt'
# rnas.tsv file is from the release version of wcEcoli
RELEASE_RNAS_TSV_PATH = 'data/subgen_gene_plots/release_rnas.tsv'
RESPONSE_GENES_PATH = 'data/subgen_gene_plots/antibiotic_response_genes.txt'
RNAS_TSV_PATH = 'reconstruction/ecoli/flat/rnas.tsv'
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'
TU_TO_INDEX_PATH = 'data/marA_binding/TU_id_to_index.json'


def main():
    with open(SIM_DATA_PATH, 'rb') as sim_data_file:
        sim_data = cPickle.load(sim_data_file)
    # Basal TU probabilities in vivarium-ecoli
    basal_probs = sim_data.process.transcription_regulation.basal_prob
    with open(TU_TO_INDEX_PATH) as json_file:
        tu_to_index = json.load(json_file)
    tu_to_basal_prob = {tu.split('[')[0]: basal_probs[index] for tu, index in tu_to_index.items()}
    mrna_obj_to_mrna_name = {}
    with open(RNAS_TSV_PATH) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tsv_file:
            if len(line) > 1 and line[3] == 'mRNA':
                mrna_obj_to_mrna_name[line[0]] = line[1]
    # TUs are a subset of RNA object ids. Only include the TUs of mRNAs.
    mrna_to_basal_prob = {mrna_obj_to_mrna_name[tu]: basal_prob for tu, basal_prob in tu_to_basal_prob.items()
                          if tu in mrna_obj_to_mrna_name.keys()}

    # Transcription probabilities for all RNAs in vivarum-ecoli
    mrna_probs = sorted(mrna_to_basal_prob.values(), reverse=True)
    with open(RESPONSE_GENES_PATH) as response_genes_file:
        response_genes = response_genes_file.read().split('\n')
    # All response gene names are the same as their corresponding RNA names. Only include genes that encode mRNAs.
    #response_gene_probs = sorted([mrna_to_basal_prob[response_gene] for response_gene in response_genes
    #                       if response_gene in mrna_to_basal_prob.keys()], reverse=True)
    mrna_to_basal_prob_response = {
        mrna: prob
        for mrna, prob in mrna_to_basal_prob.items()
        if mrna in response_genes
    }

    # Get the basal probabilities from the release version of wcEcoli for the next section.
    with open(RELEASE_RNA_PROB_PATH) as f:
        release_rna_probs = f.read().splitlines()
        for i in range(len(release_rna_probs)):
            # RNA probabilities are in the same order as the corresponding RNAs are listed in release_rnas.tsv
            release_rna_probs[i] = np.float64(release_rna_probs[i])

    # Here we find the probability cutoff separating generational and sub-generational genes. We do this by finding
    # the probability of the least likely to be expressed generational gene in wcEcoli and using that as the cutoff.
    release_prob_gene_tuples = []
    with open(RELEASE_RNAS_TSV_PATH) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for index, line in enumerate(tsv_file):
            if line[3] == 'mRNA':
                gene_id = line[11]
                release_prob_gene_tuples.append((release_rna_probs[index], gene_id))
    release_prob_gene_tuples = sorted(release_prob_gene_tuples, reverse=True)
    cutoff_tuple = release_prob_gene_tuples[CUTOFF_INDEX]
    generational_prob_cutoff = cutoff_tuple[0]

    table = [
        ('gene', 'basal_transcription_prob', 'antibiotic_response',
            'subgenerational'),
    ]
    num_subgen = 0
    num_subgen_response = 0
    num_gen = 0
    num_gen_response = 0
    for mrna, prob in mrna_to_basal_prob.items():
        subgen = prob < generational_prob_cutoff
        response = mrna in response_genes
        row = mrna, prob, response, subgen
        table.append(row)

        if subgen:
            num_subgen += 1
            if response:
                num_subgen_response += 1
        else:
            num_gen += 1
            if response:
                num_gen_response += 1

    data = {
        ('All Genes', 'gray'): (
            ('Generational', num_gen),
            ('Subgenerational', num_subgen),
        ),
        ('Antibiotic Response Genes', 'black'): (
            ('Generational', num_gen_response),
            ('Subgenerational', num_subgen_response),
        ),
    }

    fig, ax = plt.subplots(figsize=(9, 2))
    for (label, color), series in data.items():
        y, width = zip(*series)
        bars = ax.barh(y, width, color=color, label=label)
        ax.bar_label(bars)
    ax.legend()
    fig.tight_layout()
    fig.savefig('out/subgen_antibiotic_response_genes.png')

    with open('out/subgen_antibiotic_response_genes.csv', 'w') as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow(row)


if __name__ == '__main__':
    main()
