"""
Plot the Voronoi diagram of mass fractions
"""

import os
from six.moves import cPickle
import pickle
import numpy as np
from matplotlib import pyplot as plt

from wholecell.utils import units
from wholecell.utils.voronoi_plot_main import VoronoiMaster

from ecoli.composites.ecoli_master import run_ecoli
from ecoli.analysis.tablereader import TableReader
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH



SEED = 0  # random seed


class Plot:

    def __init__(self, data, sim_data=None, out_dir='out/analysis'):
        self.data = data
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.sim_data = sim_data
        if not sim_data:
            with open(SIM_DATA_PATH, 'rb') as sim_data_file:
                self.sim_data = pickle.load(sim_data_file)

        self.do_plot(self.data)

    def do_plot(self, data):

        np.random.seed(SEED)

        # Load data
        mass = TableReader("Mass", data)
        bulk_molecules = TableReader("BulkMolecules", data)
        bulk_molecule_counts = bulk_molecules.readColumn("counts")
        bulk_molecule_idx = {name: i for i, name in enumerate(bulk_molecules.readColumn("objectNames"))}
        nAvogadro = self.sim_data.constants.n_avogadro

        # lipids and polyamines
        def find_mass_molecule_group(group_id):
            temp_ids = getattr(self.sim_data.molecule_groups, str(group_id))
            temp_indexes = np.array([bulk_molecule_idx[temp] for temp in temp_ids])
            temp_counts = bulk_molecule_counts[:, temp_indexes]
            temp_mw = self.sim_data.getter.get_masses(temp_ids)
            return (units.dot(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

        lipid = find_mass_molecule_group('lipids')
        polyamines = find_mass_molecule_group('polyamines')

        # LPS, murein, and glycogen
        def find_mass_single_molecule(molecule_id):
            temp_id = getattr(self.sim_data.molecule_ids, str(molecule_id))
            temp_index = bulk_molecule_idx[temp_id]
            temp_counts = bulk_molecule_counts[:, temp_index]
            temp_mw = self.sim_data.getter.get_mass(temp_id)
            return (units.multiply(temp_counts, temp_mw) / nAvogadro).asNumber(units.fg)

        lps = find_mass_single_molecule('LPS')
        murein = find_mass_single_molecule('murein')
        glycogen = find_mass_single_molecule('glycogen')

        # other cell components
        protein = mass.readColumn("proteinMass")
        rna = mass.readColumn("rnaMass")
        tRna = mass.readColumn("tRnaMass")
        rRna = mass.readColumn("rRnaMass")
        mRna = mass.readColumn("mRnaMass")
        miscRna = rna - (tRna + rRna + mRna)
        dna = mass.readColumn("dnaMass")
        smallMolecules = mass.readColumn("smallMoleculeMass")
        metabolites = smallMolecules - (lipid + lps + murein + polyamines + glycogen)

        # create dictionary
        dic_initial = {
            'nucleic_acid': {
                'DNA': dna[0],
                'mRNA': mRna[0],
                'miscRNA': miscRna[0],
                'rRNA': rRna[0],
                'tRNA': tRna[0],
            },
            'metabolites': {
                'LPS': lps[0],
                'glycogen': glycogen[0],
                'lipid': lipid[0],
                'metabolites': metabolites[0],
                'peptidoglycan': murein[0],
                'polyamines': polyamines[0],
            },
            'protein': protein[0],
        }
        dic_final = {
            'nucleic_acid': {
                'DNA': dna[-1],
                'mRNA': mRna[-1],
                'miscRNA': miscRna[-1],
                'rRNA': rRna[-1],
                'tRNA': tRna[-1],
            },
            'metabolites': {
                'LPS': lps[-1],
                'glycogen': glycogen[-1],
                'lipid': lipid[-1],
                'metabolites': metabolites[-1],
                'peptidoglycan': murein[-1],
                'polyamines': polyamines[-1],
            },
            'protein': protein[-1],
        }

        # create the plot
        vm = VoronoiMaster()
        vm.plot([[dic_initial, dic_final]],
                title=[["Initial biomass components", "Final biomass components"]],
                ax_shape=(1, 2), chained=True)

        # TODO -- this can go into a general method in an analysis base class
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'voronoi_mass_fraction_summary.png'))
        plt.close("all")
        return vm


def run_plot():
    data = run_ecoli(total_time=10)
    Plot(data)


# python ecoli/analysis/mass_fractions_voronoi.py
if __name__ == "__main__":
    run_plot()
