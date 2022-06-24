"""
This script creates a bar graph showing the number of antibiotic response genes and
the number of all genes that are sub-generational vs. generational. We define
sub-generational genes to be those that are transcribed at least once per
generation.

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

CUTOFF_INDEX = 1546  # Number of genes considered sub-generational in wcEcli minus one
FUNCTIONAL_GENES = ['ampC', 'acrA', 'acrB', 'tolC', 'ompF', 'ompC', 'marA', 'marR', 'mrcA', 'mrcB']
RELEASE_RNA_PROB_PATH = 'data/subgen_gene_plots/release_rna_probs.txt'
RELEASE_RNAS_TSV_PATH = 'data/subgen_gene_plots/release_rnas.tsv'
RESPONSE_GENES_PATH = 'data/subgen_gene_plots/antibiotic_response_genes.txt'
RNAS_TSV_PATH = 'reconstruction/ecoli/flat/rnas.tsv'
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'
TU_TO_INDEX_PATH = 'ecoli/experiments/marA_binding/TU_id_to_index.json'


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
    mrna_probs = sorted(list(mrna_to_basal_prob.values()), reverse=True)
    with open(RESPONSE_GENES_PATH) as response_genes_file:
        response_genes = response_genes_file.read().split('\n')
    # All response gene names are the same as their corresponding RNA names. Only include genes that encode mRNAs.
    response_gene_probs = sorted([mrna_to_basal_prob[response_gene] for response_gene in response_genes
                           if response_gene in mrna_to_basal_prob.keys()], reverse=True)

    # Get the basal probabilities from the release version of wcEcoli for the next section.
    with open(RELEASE_RNA_PROB_PATH) as f:
        release_rna_probs = f.read().splitlines()
        for i in range(len(release_rna_probs)):
            # RNA probabilities are in the same order as the corresponding RNAs are listed in release_rnas.tsv
            release_rna_probs[i] = np.float64(release_rna_probs[i])

    # Here we find the probability cutoff separating generational and sub-generational genes. We do this by finding
    # the probability of the most likely to be expressed sub-generational gene in wcEcoli and using that as the cutoff.
    release_prob_gene_tuples = []
    index = 0
    with open(RELEASE_RNAS_TSV_PATH) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tsv_file:
            if line[3] == 'mRNA':
                gene_id = line[11]
                release_prob_gene_tuples.append((release_rna_probs[index], gene_id))
            index += 1
    release_prob_gene_tuples = sorted(release_prob_gene_tuples, reverse=True)
    cutoff_tuple = release_prob_gene_tuples[CUTOFF_INDEX]
    generational_prob_cutoff = cutoff_tuple[0]

    def calc_num_sub(probs):
        """
        Calculates the number of sub-generational genes in a sorted list of
        (prob, gene_id) tuples.
        """
        num_sub_gen = 0
        for prob in probs:
            if prob < generational_prob_cutoff:
                break
            num_sub_gen += 1
        return num_sub_gen
    num_all_sub_gen = calc_num_sub(mrna_probs)
    num_all_generational = len(mrna_probs) - num_all_sub_gen
    num_response_sub_gen = calc_num_sub(response_gene_probs)
    num_response_generational = len(response_gene_probs) - num_response_sub_gen

    data = {'All: Generational': num_all_generational, 'All: Sub': num_all_sub_gen,
            'Response: Generational': num_response_generational, 'Response: Sub': num_response_sub_gen}
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(list(data.keys()), list(data.values()))
    ax.bar_label(bars)
    plt.title("Number of Generational and Sub-Generational Genes")
    plt.savefig('out/prob_bars.png')


if __name__ == '__main__':
    main()
