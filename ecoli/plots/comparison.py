
import os

from matplotlib import pyplot as plt
import numpy as np


def mrna_scatter_comparison(data, config, out_dir='out'):
    filename = 'mrna_comparison'
    sim_data = config['sim_data']

    # Get the names of mRNAs from the KB
    is_mRNA = sim_data.process.transcription.rna_data['is_mRNA']
    mRNA_ids = sim_data.process.transcription.rna_data['id'][is_mRNA]

    # get mRNA data
    bulk = data['bulk']
    mRNA_data = {
        mol_id: series
        for mol_id, series in bulk.items()
        if mol_id in mRNA_ids}

    # Read final mRNA counts
    counts = np.array([series[-1] for series in mRNA_data.values()])

    # get expected counts
    expectedCountsArbitrary = sim_data.process.transcription.rna_expression[sim_data.condition][is_mRNA]
    expectedCounts = expectedCountsArbitrary / expectedCountsArbitrary.sum() * counts.sum()

    maxLine = 1.1 * max(expectedCounts.max(), counts.max())
    plt.plot([0, maxLine], [0, maxLine], '--r')
    plt.plot(expectedCounts, counts, 'o', markeredgecolor='k', markerfacecolor='none')

    plt.xlabel("Expected RNA count (scaled to total)")
    plt.ylabel("Actual RNA count (at final time step)")

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.savefig(fig_path + '.png', bbox_inches='tight')


def fluxome_scatter_comparison(data, config, out_dir='out'):
    filename = 'fluxome'
    sim_data = config['sim_data']
    pass


def protein_counts_scatter_comparison(data, config, out_dir='out'):
    filename = 'protein_counts'
    sim_data = config['sim_data']
    pass


def production_rate_plot(data, config, out_dir='out'):
    filename = 'production_rate'
    sim_data = config['sim_data']
    pass

