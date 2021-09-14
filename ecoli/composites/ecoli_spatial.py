'''
========================
E. coli spatial composite
========================

This composite is intended to run a spatial model of E. coli, where each
cellular compartment is designated as a node. These nodes are connected by
edges which specify how cellular compartments interface with each other.
Ecoli master processes could theoretically be run within each node to run
a spatially defined model of E. coli. Currently, this composite is only
set up to run a diffusion model on three nodes, the two cytosol poles and the
nucleoid, which are connected by two edges between the poles and the nucleoid.
Furthermore, since vivarium-ecoli is not yet fully migrated, this composite
initializes molecule counts from a snapshot of wcEcoli at t=1,000 s. This model
can be run with bulk molecules, polyribosomes, or both together. If only
polyribosomes are of interest, it is recommended to not include bulk
molecules for faster runtime.

If polyribosomes are included, one of three assumptions must be input to
define how polyribosomes' hydrodynamic radius is calculated. Each of these
assumption have significant limitations, however they likely indicate a
plausible range of polyribosomes' sizes.These assumptions are as follows:

    - `spherical`: This assumes that polyribosomes are spherical proteins and
    calculates the hydrodynamic radius from the total molecular weight of the
    mRNA molecules, attached ribosomes, and attached polypeptides. This is
    the default of the diffusion network process that is called here (and is
    the assumption for all bulk molecules); further details can be found in
    `diffusion_network.py` in `vivarium-cell`.
        * :math:`r_p = 0.515*MW^{0.392}`
        * :math:`MW` = molecular weight
        - Ref: Schuwirth et al., Science (2005)
    - `mrna`: This assumes that polyribosomes are solely the mRNA molecule and
    calculates the hydrodynamic radius of the mRNA molecule from the
    nucleotide count.
        * :math:`r_p = 5.5*N^{1/3}`
        * :math:`N` = number of nucleotides in mRNA
        - Ref: Hyeon et al., J Chem Phys. (2006)
    -`linear`: This assumes that polyribosomes are solely the ribosomes and
    calculates the radius to be half the length of the summed sizes of
    ribosomes. This assumption does not have a reference
        * :math:`r_p = \\frac{n_ribosome * ribosome_size}{2}`

Since all molecules are treated as concentrations in the diffusion network
process, polyribosomes are bucketed into groups defined by the number of
ribosomes attached to each mRNA molecule.

This test case uses a mesh size of 50 nm, which is used by the diffusion
network process to scale diffusion constants to represent the impact that
a meshgrid formed by DNA in the nucleoid has on bulk molecule and polyribosome
diffusion.

    - Ref: Xiang et al., bioRxiv (2020)

Other `vivarium-cell` processes are also intended to be compatible with this
composite, but are unfinished or have not been incorporated. These processes
are `growth_rate.py` and `spatial_geometry.py`.

'''
import argparse
import numpy as np
import math
from wholecell.utils import units
from scipy import constants

from six.moves import cPickle

from vivarium.core.composer import Composer
from vivarium.core.composition import simulate_composer

# processes
from ecoli.processes.diffusion_network import DiffusionNetwork

# plots
from ecoli.plots.ecoli_spatial_plots import (
    plot_NT_availability,
    plot_single_molecule_diff,
    plot_nucleoid_diff,
    plot_large_molecules,
    plot_polyribosomes_diff,
    plot_molecule_characterizations,
)

from ecoli.states.wcecoli_state import get_state_from_file, MASSDIFFS

SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'
RIBOSOME_SIZE = 21      # in nm


class EcoliSpatial(Composer):

    defaults = {
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'nodes': [],
        'edges': {},
        'mesh_size': 50,       # in nm
        'radii': {},
        'temp': 310.15,     # in K
    }

    def __init__(self, config):
        super().__init__(config)
        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed=self.seed)
        self.sim_data_path = self.config['sim_data_path']
        self.nodes = self.config['nodes']
        self.edges = self.config['edges']
        self.mesh_size = self.config['mesh_size']
        self.time_step = self.config['time_step']
        self.radii = self.config['radii']
        self.temp = self.config['temp']

        # load sim_data
        with open(self.sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data['id']

        # molecular weight is converted to femtograms
        self.bulk_molecular_weights = {
            molecule_id: (sim_data.getter.get_mass(molecule_id) / constants.N_A).asNumber(
                units.fg / units.mol)
            for molecule_id in bulk_ids}

        # unique molecule masses
        self.unique_masses = {}
        unique_molecular_masses = sim_data.internal_state.unique_molecule.unique_molecule_masses
        for (id_, mass) in zip(unique_molecular_masses["id"], unique_molecular_masses["mass"]):
            self.unique_masses[id_] = sum((mass / sim_data.constants.n_avogadro).asNumber(units.fg))

    def initial_state(self, config):
        # initialize variables
        include_bulk = config['include_bulk']
        include_polyribosomes = config['include_polyribosomes']
        initial_state = get_state_from_file(path='data/wcecoli_t1000.json')
        bulk = {}

        if include_bulk:
            bulk = initial_state['bulk']
        else:
            self.bulk_molecular_weights = {}
        if include_polyribosomes:
            polyribosome_assumption = config['polyribosome_assumption']
            polyribosomes, polyribosomes_mw, polyribosomes_radii = add_polyribosomes(
                initial_state['unique'], self.unique_masses, polyribosome_assumption)
            bulk.update(polyribosomes)
            self.bulk_molecular_weights.update(polyribosomes_mw)
            self.radii.update(polyribosomes_radii)


        # Buckets half of cytosol labeled molecules into each pole
        cytosol_front = {
            mol_id: (math.ceil(value / 2) if '[c]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        cytosol_rear = {
            mol_id: (math.floor(value / 2) if '[c]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        nucleoid = {
            mol_id: (value if '[n]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }

        self.nodes['cytosol_front']['molecules'] = cytosol_front
        self.nodes['nucleoid']['molecules'] = nucleoid
        self.nodes['cytosol_rear']['molecules'] = cytosol_rear

        initial_spatial_state = self.nodes
        return initial_spatial_state

    def generate_processes(self, config):
        diffusion_config = {
            'nodes': list(self.nodes.keys()),
            'edges': self.edges,
            'mw': self.bulk_molecular_weights,
            'mesh_size': self.mesh_size,
            'time_step': self.time_step,
            'radii': self.radii,
            'temp': self.temp,
        }
        return {
            'diffusion_network': DiffusionNetwork(diffusion_config)
        }

    def generate_topology(self, config):
        return {
            'diffusion_network': {
                node: (node,) for node in config['nodes']
            },
        }


def add_polyribosomes(unique, unique_masses, polyribosome_assumption, save_output=False):
        # pull polyribosome building blocks from unique initial state
        is_full_transcript_rna = np.asarray([unique['RNA'][
                                                 unique_index]['is_full_transcript']
                                             for unique_index in
                                             unique['RNA'].keys()])
        is_mrna = np.asarray([unique['RNA'][
                                  unique_index]['is_mRNA']
                              for unique_index in
                              unique['RNA'].keys()])

        unique_index_rna = np.asarray(list(unique['RNA'].keys()))
        mrna_index_ribosomes = np.asarray(
            [unique['active_ribosome'][unique_index]['mRNA_index']
             for unique_index in unique['active_ribosome'].keys()])

        # This line removes ghost mrna indexes
        mrna_index_ribosomes = mrna_index_ribosomes[
            np.where(np.isin(mrna_index_ribosomes, unique_index_rna) == 1)]

        # This removes non-mrna molecules
        mrna_index_ribosomes = [mrna_index_ribosomes[i]
                                for i in np.where(np.isin(
                mrna_index_ribosomes, unique_index_rna[is_mrna]))[0]]

        # Calculates number of ribosomes on each unique mrna molecule
        mrna_index_ribosome_on_full_mrna = [mrna_index_ribosomes[i]
                                            for i in np.where(np.isin(
                mrna_index_ribosomes, unique_index_rna[is_full_transcript_rna]))[0]]
        n_ribosomes_on_full_mrna = np.bincount(mrna_index_ribosome_on_full_mrna)

        # Calculates mRNA length and NT availability for ribosomes
        mrna_length = np.asarray([unique['RNA'][
                                      str(unique_index)]['transcript_length']
                                  for unique_index in
                                  mrna_index_ribosome_on_full_mrna])
        avg_NT_per_ribosome = np.asarray([
            mrna_length[i] / n_ribosomes_on_full_mrna[mrna_index_ribosome_on_full_mrna[i]]
            for i in np.unique(mrna_index_ribosome_on_full_mrna, return_index=True)[1]
            if n_ribosomes_on_full_mrna[mrna_index_ribosome_on_full_mrna[i]] != 0
        ])
        if save_output:
            plot_NT_availability(avg_NT_per_ribosome)

        # Defines buckets for unique polyribosomes to be combined into
        groups = ['polyribosome_1[c]', 'polyribosome_2[c]', 'polyribosome_3[c]',
                  'polyribosome_4[c]', 'polyribosome_5[c]', 'polyribosome_6[c]',
                  'polyribosome_7[c]', 'polyribosome_8[c]', 'polyribosome_9[c]',
                  'polyribosome_>=10[c]']

        # Initializes variables for key information about each polyribosome group
        n_ribosomes_per_mrna_by_group = np.zeros((len(groups), len(n_ribosomes_on_full_mrna)))
        n_polyribosome_by_group = np.zeros(len(groups))
        avg_n_ribosome_by_group = np.zeros(len(groups))
        polyribosome_mass_by_group = np.zeros(len(groups))
        avg_mrna_mass = np.zeros(len(groups))
        avg_mrna_length = np.zeros(len(groups))
        avg_peptide_length = np.zeros(len(groups))
        polyribosomes = {}
        mw = {}
        radii = {}

        # Separates polyribosomes into subgroups based off of number of ribosomes on mrna
        for i in range(len(groups) - 1):
            n_ribosomes_per_mrna_by_group[i, :] = np.multiply(
                n_ribosomes_on_full_mrna,
                n_ribosomes_on_full_mrna == (i + 1))
        n_ribosomes_per_mrna_by_group[(len(groups) - 1), :] = np.multiply(
            n_ribosomes_on_full_mrna,
            n_ribosomes_on_full_mrna >= 10)

        # Calculates properties of polyribosome groups and adds to bulk molecules
        for i, group in enumerate(groups):
            n_polyribosome_by_group[i] = sum(
                [n_ribosomes_per_mrna_by_group[i, :] > 0][0])
            avg_n_ribosome_by_group[i] = np.average(
                n_ribosomes_per_mrna_by_group[i, :][
                    [n_ribosomes_per_mrna_by_group[i, :] > 0][0]
                ])
            group_idx = np.where(n_ribosomes_per_mrna_by_group[i, :] > 0)[0]
            avg_mrna_mass[i] = np.average([
                unique['RNA'][str(unique_index)]['submass'][MASSDIFFS['massDiff_mRNA']]
                for unique_index in group_idx])
            avg_mrna_length[i] = np.average([
                unique['RNA'][str(unique_index)]['transcript_length']
                for unique_index in group_idx
            ])
            avg_peptide_length[i] = sum([
                unique['active_ribosome'][unique_index][
                    'peptide_length']
                for unique_index in unique['active_ribosome'].keys()
                if unique['active_ribosome'][unique_index][
                       'mRNA_index'] in group_idx]) / n_polyribosome_by_group[i]
            polyribosome_mass_by_group[i] = avg_mrna_mass[i] + avg_n_ribosome_by_group[
                i] * unique_masses['active_ribosome'] + avg_peptide_length[
                                                i] * 1.82659422 * 10 ** -7

            # Recalculates polyribosome size per input assumption
            if polyribosome_assumption == 'linear':
                radii[group] = (avg_n_ribosome_by_group[i] * RIBOSOME_SIZE) / 2
            if polyribosome_assumption == 'mrna':
                radii[group] = 5.5 * avg_mrna_length[i] ** 0.3333

            # Adds polyribosomes to bulk molecules and molecular weights
            polyribosomes[group] = n_polyribosome_by_group[i]
            mw[group] = polyribosome_mass_by_group[i]

        return polyribosomes, mw, radii


def test_spatial_ecoli(
    polyribosome_assumption='spherical',    # choose from 'mrna', 'linear', or 'spherical'
    total_time=60,  # in seconds
):
    ecoli_config = {
        'nodes': {
            'cytosol_front': {
                'length': 0.5,      # in um
                'volume': 0.25,     # in um^3
                'molecules': {},
            },
            'nucleoid': {
                'length': 1.0,
                'volume': 0.5,
                'molecules': {},
            },
            'cytosol_rear': {
                'length': 0.5,
                'volume': 0.25,
                'molecules': {},
            },
        },
        'edges': {
            '1': {
                'nodes': ['cytosol_front', 'nucleoid'],
                'cross_sectional_area': np.pi * 0.3 ** 2,       # in um^2
                'mesh': True,
            },
            '2': {
                'nodes': ['nucleoid', 'cytosol_rear'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
                'mesh': True,
            },
        },
        'mesh_size': 50,        # in nm
        'time_step': 1,
    }

    ecoli = EcoliSpatial(ecoli_config)

    initial_config = {
        'include_bulk': False,
        'include_polyribosomes': True,
        'polyribosome_assumption': polyribosome_assumption,
    }

    settings = {
        'total_time': total_time,       # in s
        'initial_state': ecoli.initial_state(initial_config)}

    data = simulate_composer(ecoli, settings)
    return ecoli, initial_config, data


def run_spatial_ecoli(
    polyribosome_assumption='linear'  # choose from 'mrna', 'linear', or 'spherical'
):
    ecoli, initial_config, output = test_spatial_ecoli(
        polyribosome_assumption=polyribosome_assumption,
        total_time=5*60,
    )
    mesh_size = ecoli.config['mesh_size']
    nodes = ecoli.config['nodes']
    plot_molecule_characterizations(ecoli, initial_config)
    if initial_config['include_bulk']:
        mol_ids = ['ADHE-CPLX[c]', 'CPLX0-3962[c]', 'CPLX0-3953[c]']
        plot_large_molecules(output, mol_ids, mesh_size, nodes)
    if initial_config['include_polyribosomes']:
        filename = 'out/polyribosome_diffusion_' + initial_config['polyribosome_assumption']
        plot_polyribosomes_diff(output, mesh_size, nodes, filename)
    if initial_config['include_polyribosomes'] and not initial_config['include_bulk']:
        plot_nucleoid_diff(output, nodes, initial_config['polyribosome_assumption'])


# Helper functions
def array_from(d):
    return np.array(list(d.values()))


def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}


def main():
    parser = argparse.ArgumentParser(description='ecoli spatial')
    parser.add_argument('-mrna', '-m', action='store_true', default=False, help='mrna assumption')
    parser.add_argument('-linear', '-l', action='store_true', default=False, help='linear assumption')
    parser.add_argument('-spherical', '-s', action='store_true', default=False, help='spherical assumption')
    args = parser.parse_args()

    if args.mrna:
        run_spatial_ecoli('mrna')
    if args.linear:
        run_spatial_ecoli('linear')
    if args.spherical:
        run_spatial_ecoli('spherical')

if __name__ == '__main__':
    main()
