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
from six.moves import cPickle

SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'
TU_TO_INDEX_PATH = 'ecoli/experiments/marA_binding/TU_id_to_index.json'
RNAS_TSV_PATH = 'reconstruction/ecoli/flat/rnas.tsv'
RESPONSE_GENES_FILE_PATH = 'data/subgen_gene_plots/antibiotic_response_genes.txt'
# From wcEcoli paper
PERCENT_GENERATIONAL = 0.355
FUNCTIONAL_GENES = ['ampC', 'acrA', 'acrB', 'tolC', 'ompF', 'ompC', 'marA', 'marR', 'mrcA', 'mrcB']

def main():
    with open(SIM_DATA_PATH, 'rb') as sim_data_file:
        sim_data = cPickle.load(sim_data_file)
    basal_probs = sim_data.process.transcription_regulation.basal_prob
    with open(TU_TO_INDEX_PATH) as json_file:
        tu_to_index = json.load(json_file)
    tu_to_basal_prob = {tu.split('[')[0]: basal_probs[index] for tu, index in tu_to_index.items()}

    tu_to_gene = {}
    with open(RNAS_TSV_PATH) as file:
        tsv_file = csv.reader(file, delimiter='\t')
        past_header = False
        for line in tsv_file:
            if line[0] == 'id':
                past_header = True
            elif past_header:
                tu_to_gene[line[0]] = line[1]

    gene_to_basal_prob = {tu_to_gene[tu]: basal_prob for tu, basal_prob in tu_to_basal_prob.items()}
    with open(RESPONSE_GENES_FILE_PATH) as response_genes_file:
        response_genes = response_genes_file.read().split('\n')
    r_genes_to_basal_prob = {}
    for r_gene in response_genes:
        r_genes_to_basal_prob[r_gene] = gene_to_basal_prob[r_gene]
    assert(len(r_genes_to_basal_prob.keys()) == len(response_genes))

    gene_prob_tuples = []
    for gene in gene_to_basal_prob.keys():
        gene_prob_tuples.append((gene_to_basal_prob[gene], gene))
    gene_prob_tuples = sorted(gene_prob_tuples, reverse=True)
    r_gene_prob_tuples = []
    for gene in r_genes_to_basal_prob.keys():
        r_gene_prob_tuples.append((r_genes_to_basal_prob[gene], gene))
    r_gene_prob_tuples = sorted(r_gene_prob_tuples, reverse=True)
    generational_index = int(len(gene_prob_tuples) * PERCENT_GENERATIONAL)
    num_generational = generational_index
    num_sub_gen = len(gene_prob_tuples) - num_generational
    generational_prob_cutoff = gene_prob_tuples[generational_index][0]
    num_r_generational = 0
    for prob, gene in r_gene_prob_tuples:
        if prob < generational_prob_cutoff:
            break
        num_r_generational += 1
    num_r_sub_gen = len(r_gene_prob_tuples) - num_r_generational
    functional_generational = []
    functional_sub = []
    for gene in FUNCTIONAL_GENES:
        if gene_to_basal_prob[gene] < generational_prob_cutoff:
            functional_sub.append(gene)
        else:
            functional_generational.append(gene)
    print('Functional: Sub -> ' + str(functional_sub) + '\n'
          'Functional: Generational -> ' + str(functional_generational))

    plt.plot(sorted(list(r_genes_to_basal_prob.values())))
    plt.title("Response Genes' Basal Probabilities (Sorted)")
    plt.xlabel("Gene Index")
    plt.savefig('data/subgen_gene_plots/r_genes_basal_probs.png')
    plt.close()

    data = {'All: Generational': num_generational, 'All: Sub': num_sub_gen,
            'Response: Generational': num_r_generational, 'Response: Sub': num_r_sub_gen}
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh(list(data.keys()), list(data.values()))
    ax.bar_label(bars)
    plt.title("Number of Generational and Sub-Generational Genes")
    plt.savefig('data/subgen_gene_plots/prob_bars.png')


if __name__ == '__main__':
    main()
