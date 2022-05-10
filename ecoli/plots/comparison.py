
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import constants

from wholecell.utils import units

AVOGADRO = constants.N_A #* 1 / units.mol


def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


def mass_from_counts_array(counts, mw):
    return np.array([mass_from_count(count, mw) for count in counts])


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



# colors for mass_fraction_summary
COLORS_256 = [ # From colorbrewer2.org, qualitative 8-class set 1
    [228,26,28],
    [55,126,184],
    [77,175,74],
    [152,78,163],
    [255,127,0],
    [255,255,51],
    [166,86,40],
    [247,129,191]
    ]

COLORS = [
    [colorValue/255. for colorValue in color]
    for color in COLORS_256
    ]

def sum_lists_to_array(lists):
    summed_list = np.array([])
    for list in lists:
        if summed_list.any():
            summed_list += np.array(list)
        else:
            summed_list = np.array(list)
    return summed_list


def get_masses_from_sim(data, sim_data):
    """ aggregate output data to masses """

    ## Get ids from sim_data
    protein_ids = sim_data.process.translation.monomer_data['id']
    RNA_ids = sim_data.process.transcription.rna_data["id"]
    tRNA_ids = RNA_ids[sim_data.process.transcription.rna_data['is_tRNA']]
    mRNA_ids = RNA_ids[sim_data.process.transcription.rna_data['is_mRNA']]
    rRNA_ids = RNA_ids[sim_data.process.transcription.rna_data['is_rRNA']]
    # dna_ids

    smallMolecule_ids = []
    reaction_stoich = sim_data.process.metabolism.reaction_stoich
    for reaction_id, stoich_dict in reaction_stoich.items():
        for metabolite, stoich in stoich_dict.items():
            # Add metabolites that were not encountered
            if metabolite not in smallMolecule_ids:
                smallMolecule_ids.append(metabolite)

    ## Get molecule weights
    # molecule weight is converted to femtograms/mol
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data['id']
    molecular_weights = {
        molecule_id: sim_data.getter.get_mass([molecule_id]).asNumber(units.fg / units.mol)[0]
        for molecule_id in bulk_ids}

    # unique molecule weights
    unique_masses = {}
    uniqueMoleculeMasses = sim_data.internal_state.unique_molecule.unique_molecule_masses
    for (id_, mass) in zip(uniqueMoleculeMasses["id"], uniqueMoleculeMasses["mass"]):
        unique_masses[id_] = (mass / sim_data.constants.n_avogadro).asNumber(units.fg)

    ## Get the time series
    listeners = data['listeners']
    bulk = data['bulk']
    cell = np.array(listeners['mass']['cell_mass'])
    cell_dry = np.array(listeners['mass']['dry_mass'])

    # Convert to masses
    protein = sum_lists_to_array(
        [mass_from_counts_array(bulk[mol_id], molecular_weights.get(mol_id))
         for mol_id in protein_ids])
    rna = sum_lists_to_array(
        [mass_from_counts_array(bulk[mol_id], molecular_weights.get(mol_id))
         for mol_id in RNA_ids])
    rRna = sum_lists_to_array(
        [mass_from_counts_array(bulk[mol_id], molecular_weights.get(mol_id))
         for mol_id in rRNA_ids])
    tRna = sum_lists_to_array(
        [mass_from_counts_array(bulk[mol_id], molecular_weights.get(mol_id))
         for mol_id in tRNA_ids])
    mRna = sum_lists_to_array(
        [mass_from_counts_array(bulk[mol_id], molecular_weights.get(mol_id))
         for mol_id in mRNA_ids])
    smallMolecules = sum_lists_to_array(
        [mass_from_counts_array(bulk[mol_id], molecular_weights.get(mol_id))
         for mol_id in smallMolecule_ids if mol_id in bulk])
    # dna = sum_lists_to_array([bulk[mol_id] for mol_id in dna_ids])

    return {
        'cell': cell,
        'cell_dry': cell_dry,
        'protein': protein,
        'rna': rna,
        'rRna': rRna,
        'tRna': tRna,
        'mRna': mRna,
        'smallMolecules': smallMolecules,
    }



def mass_fractions_summary(data, config, out_dir='out'):
    filename = 'mass_fractions_summary'
    sim_data = config['sim_data']

    t = np.array(data['time']) / 60.

    # get masses
    masses = get_masses_from_sim(data, sim_data)
    cell = masses['cell']
    protein = masses['protein']
    rRna = masses['rRna']
    tRna = masses['tRna']
    mRna = masses['mRna']
    smallMolecules = masses['smallMolecules']
    # dna = masses['dna']

    masses = np.vstack([
        protein,
        rRna,
        tRna,
        mRna,
        # dna,
        smallMolecules,
    ]).T
    fractions = (masses / cell[:, None]).mean(axis=0)

    mass_labels = ["Protein", "rRNA", "tRNA", "mRNA", "Small Mol.s"]
    # mass_labels = ["Protein", "rRNA", "tRNA", "mRNA", "DNA", "Small Mol.s"]
    legend = [
                 '{} ({:.3f})'.format(label, fraction)
                 for label, fraction in zip(mass_labels, fractions)
             ] + ['Total dry mass']

    # make the plot
    plt.figure(figsize=(8.5, 11))
    plt.gca().set_prop_cycle('color', COLORS)

    plt.plot(t, masses / masses[0, :], linewidth=2)
    plt.plot(t, cell / cell[0], color='k', linestyle=':')

    plt.title("Biomass components (average fraction of total dry mass in parentheses)")
    plt.xlabel("Time (min)")
    plt.ylabel("Mass (normalized by t = 0 min)")
    plt.legend(legend, loc="best")

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(hspace=0.75)
    plt.savefig(fig_path + '.png', bbox_inches='tight')



def mass_fractions(data, config, out_dir='out'):
    filename = 'mass_fractions'
    sim_data = config['sim_data']

    t = np.array(data['time']) / 60.

    # get masses
    masses = get_masses_from_sim(data, sim_data)
    cell = masses['cell']
    cellDry = masses['cell_dry']
    protein = masses['protein']
    rna = masses['rna']
    smallMolecules = masses['smallMolecules']
    # dna = masses['dna']

    # make the plot
    plt.figure(figsize=(8.5, 15))
    n_subplots = 6
    second_axis_color = 'g'

    plt.subplot(n_subplots, 1, 1)
    plt.plot(t / 60., cell, linewidth=2)
    plt.plot([t[0] / 60., t[-1] / 60.], [2 * cell[0], 2 * cell[0]], 'r--')
    plt.ylabel("Total Mass (fg)")
    plt.title("Total Mass Final:Initial = %0.2f" % (cell[-1] / cell[0]), fontsize=8)

    plt.subplot(n_subplots, 1, 2)
    plt.plot(t / 60., cellDry, linewidth=2)
    plt.plot([t[0] / 60., t[-1] / 60.], [2 * cellDry[0], 2 * cellDry[0]], 'r--')
    plt.ylabel("Dry Mass (fg)")
    plt.title("Dry Mass Final:Initial = %0.2f" % (cellDry[-1] / cellDry[0]), fontsize=8)

    ax = plt.subplot(n_subplots, 1, 3)
    plt.plot(t / 60., protein, linewidth=2)
    plt.plot([t[0] / 60., t[-1] / 60.], [2 * protein[0], 2 * protein[0]], "r--")
    plt.ylabel("Protein Mass (fg)")
    plt.title("Total Protein Mass Final:Initial = %0.2f\nAverage dry mass fraction: %0.3f"
              % (protein[-1] / protein[0], np.mean(protein / cellDry)), fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t / 60., protein / cellDry, second_axis_color)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_yticks(ax2.get_ylim())
    ax2.set_ylabel('Fraction of dry mass', color=second_axis_color)

    ax = plt.subplot(n_subplots, 1, 4)
    plt.plot(t / 60., rna, linewidth=2)
    plt.plot([t[0] / 60., t[-1] / 60.], [2 * rna[0], 2 * rna[0]], "r--")
    plt.ylabel("RNA Mass (fg)")
    plt.title("Total RNA Mass Final:Initial = %0.2f\nAverage dry mass fraction: %0.3f"
              % (rna[-1] / rna[0], np.mean(rna / cellDry)), fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t / 60., rna / cellDry, second_axis_color)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_yticks(ax2.get_ylim())
    ax2.set_ylabel('Fraction of dry mass', color=second_axis_color)

    ax = plt.subplot(n_subplots, 1, 5)
    plt.plot(t / 60., smallMolecules, linewidth=2)
    plt.plot([t[0] / 60., t[-1] / 60.], [2 * smallMolecules[0], 2 * smallMolecules[0]], "r--")
    plt.ylabel("Small molecules (fg)")
    plt.title("Total Small Molecule Mass Final:Initial = %0.2f\nAverage dry mass fraction: %0.3f"
              % (smallMolecules[-1] / smallMolecules[0], np.mean(smallMolecules / cellDry)), fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(t / 60., smallMolecules / cellDry, second_axis_color)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_yticks(ax2.get_ylim())
    ax2.set_ylabel('Fraction of dry mass', color=second_axis_color)

    # ax = plt.subplot(n_subplots, 1, 6)
    # plt.plot(t / 60., dna, linewidth=2)
    # plt.plot([t[0] / 60., t[-1] / 60.], [2 * dna[0], 2 * dna[0]], "r--")
    # plt.xlabel("Time (min)")
    # plt.ylabel("DNA Mass (fg)")
    # plt.title("Total DNA Mass Final:Initial = %0.2f\nAverage dry mass fraction: %0.3f"
    #           % (dna[-1] / dna[0], np.mean(dna / cellDry)), fontsize=8)
    # ax2 = ax.twinx()
    # ax2.plot(t / 60., dna / cellDry, second_axis_color)
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax2.set_yticks(ax2.get_ylim())
    # ax2.set_ylabel('Fraction of dry mass', color=second_axis_color)

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(hspace=0.75)
    plt.savefig(fig_path + '.png', bbox_inches='tight')
