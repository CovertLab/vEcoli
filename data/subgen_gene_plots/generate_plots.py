"""
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

CUTOFF_INDEX = 1546
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'
TU_TO_INDEX_PATH = 'ecoli/experiments/marA_binding/TU_id_to_index.json'
RNAS_TSV_PATH = 'reconstruction/ecoli/flat/rnas.tsv'
RESPONSE_GENES_FILE_PATH = 'data/subgen_gene_plots/antibiotic_response_genes.txt'
RNA_SYNTH_PROB_PATH = 'data/subgen_gene_plots/release_rna_probs.txt'
RELEASE_RNAS_TSV_PATH = 'data/subgen_gene_plots/release_rnas.tsv'
# From wcEcoli paper
FUNCTIONAL_GENES = ['ampC', 'acrA', 'acrB', 'tolC', 'ompF', 'ompC', 'marA', 'marR', 'mrcA', 'mrcB']


def main():
    with open(SIM_DATA_PATH, 'rb') as sim_data_file:
        sim_data = cPickle.load(sim_data_file)
    basal_probs = sim_data.process.transcription_regulation.basal_prob
    with open(TU_TO_INDEX_PATH) as json_file:
        tu_to_index = json.load(json_file)
    tu_to_basal_prob = {tu.split('[')[0]: basal_probs[index] for tu, index in tu_to_index.items()}
    rna_obj_to_name = {}
    with open(RNAS_TSV_PATH) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tsv_file:
            if len(line) > 1:
                rna_obj_to_name[line[0]] = line[1]
    # TUs are a subset of RNA object ids
    rna_to_basal_prob = {rna_obj_to_name[tu]: basal_prob for tu, basal_prob in tu_to_basal_prob.items()}

    with open(RESPONSE_GENES_FILE_PATH) as response_genes_file:
        response_genes = response_genes_file.read().split('\n')
    r_genes_to_basal_prob = {}
    # All response gene names are the same as their corresponding rna names
    for r_gene in response_genes:
        r_genes_to_basal_prob[r_gene] = rna_to_basal_prob[r_gene]
    assert(len(r_genes_to_basal_prob.keys()) == len(response_genes))

    rna_prob_tuples = []
    for rna in rna_to_basal_prob.keys():
        rna_prob_tuples.append((rna_to_basal_prob[rna], rna))
    rna_prob_tuples = sorted(rna_prob_tuples, reverse=True)
    r_gene_prob_tuples = []
    for gene_id in r_genes_to_basal_prob.keys():
        r_gene_prob_tuples.append((r_genes_to_basal_prob[gene_id], gene_id))
    r_gene_prob_tuples = sorted(r_gene_prob_tuples, reverse=True)

    # Get the basal probabilities from the release version of wcEcoli for the next section.
    with open(RNA_SYNTH_PROB_PATH) as f:
        rna_synth_probs = f.read().splitlines()
        for i in range(len(rna_synth_probs)):
            rna_synth_probs[i] = np.float64(rna_synth_probs[i])

    # Here we find the probability cutoff separating generational and sub-generational genes. We do this by finding
    # the probability of the most likely to be expressed sub-generational gene in wcEcoli and using that as the cutoff.
    prob_gene_tuples = []
    index = 0
    with open(RELEASE_RNAS_TSV_PATH) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        for line in tsv_file:
            if line[3] == 'mRNA':
                gene_id = line[11]
                prob_gene_tuples.append((rna_synth_probs[index], gene_id))
            index += 1
    prob_gene_tuples = sorted(prob_gene_tuples, reverse=True)
    cutoff_tuple = prob_gene_tuples[CUTOFF_INDEX]
    generational_prob_cutoff = cutoff_tuple[0]

    num_all_sub_gen = 0
    for prob, gene_id in rna_prob_tuples:
        if prob < generational_prob_cutoff:
            break
        num_all_sub_gen += 1
    num_all_generational = len(rna_prob_tuples) - num_all_sub_gen

    num_r_sub_gen = 0
    for prob, gene_id in r_gene_prob_tuples:
        if prob < generational_prob_cutoff:
            break
        num_r_sub_gen += 1
    num_r_generational = len(r_gene_prob_tuples) - num_r_sub_gen

    data = {'All: Generational': num_all_generational, 'All: Sub': num_all_sub_gen,
            'Response: Generational': num_r_generational, 'Response: Sub': num_r_sub_gen}
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(list(data.keys()), list(data.values()))
    ax.bar_label(bars)
    plt.title("Number of Generational and Sub-Generational Genes")
    plt.savefig('data/subgen_gene_plots/prob_bars.png')


if __name__ == '__main__':
    main()
