
import numpy as np
import math
from wholecell.utils import units
from scipy import constants
import matplotlib.pyplot as plt

from six.moves import cPickle

from vivarium.core.process import Generator
from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.experiment import pp

# processes
# from vivarium_cell.processes.growth_rate import GrowthRate
from vivarium_cell.processes.diffusion_network import DiffusionNetwork

from ecoli.composites.ecoli_master import get_state_from_file

#TODO: change this before merging
SIM_DATA_PATH = '../wcEcoli/out/for_vivarium/kb/simData.cPickle'
RIBOSOME_SIZE = 21      # in nm


class EcoliSpatial(Generator):

    defaults = {
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'nodes': {
        },
        'edges': {
        },
        'mesh_size': 100,
        'radii': {},
    }

    def __init__(self, config):
        super(EcoliSpatial, self).__init__(config)
        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed=self.seed)
        self.sim_data_path = self.config['sim_data_path']
        self.nodes = self.config['nodes']
        self.edges = self.config['edges']
        self.mesh_size = self.config['mesh_size']
        self.time_step = self.config['time_step']
        self.radii = self.config['radii']

        # load sim_data
        with open(self.sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data['id']

        # molecule weight is converted to femtograms
        self.bulk_molecular_weights = {
            molecule_id: (sim_data.getter.get_mass([molecule_id]) / constants.N_A).asNumber(
                units.fg / units.mol)[0]
            for molecule_id in bulk_ids}

        # unique molecule masses
        self.unique_masses = {}
        unique_molecular_masses = sim_data.internal_state.unique_molecule.unique_molecule_masses
        for (id_, mass) in zip(unique_molecular_masses["id"], unique_molecular_masses["mass"]):
            self.unique_masses[id_] = sum((mass / sim_data.constants.n_avogadro).asNumber(units.fg))
        # import ipdb; ipdb.set_trace()




    def initial_state(self, config=None):

        initial_state = get_state_from_file(path='data/wcecoli_t1000.json')
        bulk = initial_state['bulk']

        is_full_transcript_RNA = np.asarray([initial_state['unique']['RNA'][
                                      unique_index]['is_full_transcript']
                                  for unique_index in
                                  initial_state['unique']['RNA'].keys()])
        unique_index_RNA = np.asarray(list(initial_state['unique']['RNA'].keys()))
        mRNA_index_ribosomes = np.asarray(
            [initial_state['unique']['active_ribosome'][unique_index]['mRNA_index']
             for unique_index in initial_state['unique']['active_ribosome'].keys()])

        n_ribosomes_on_full_mRNA = np.bincount(
            [mRNA_index_ribosomes[i]
             for i in np.where(np.isin(
            mRNA_index_ribosomes, unique_index_RNA[is_full_transcript_RNA]))[0]])

        groups = ['polyribosome_1[c]', 'polyribosome_2[c]', 'polyribosome_3[c]',
                  'polyribosome_4[c]','polyribosome_5[c]', 'polyribosome_6[c]',
                  'polyribosome_7[c]', 'polyribosome_8[c]', 'polyribosome_9[c]',
                  'polyribosome_>=10[c]']

        n_ribosomes_per_mRNA_by_group = np.zeros((len(groups), len(n_ribosomes_on_full_mRNA)))
        n_polyribosome_by_group = np.zeros(len(groups))
        avg_n_ribosome_by_group = np.zeros(len(groups))
        polyribosome_mass_by_group = np.zeros(len(groups))
        avg_mRNA_mass = np.zeros(len(groups))

        # Breakdown polyribosomes into subgroups based off of number of ribosomes on mRNA
        for i in range(len(groups)-1):
            n_ribosomes_per_mRNA_by_group[i, :] = np.multiply(
                n_ribosomes_on_full_mRNA,
                n_ribosomes_on_full_mRNA == (i+1))
        n_ribosomes_per_mRNA_by_group[(len(groups)-1), :] = np.multiply(
            n_ribosomes_on_full_mRNA,
            n_ribosomes_on_full_mRNA >= 10)

        # Use this for faster runtime to only look at polyribosome behavior
        bulk = {}
        self.bulk_molecular_weights = {}

        for i, group in enumerate(groups):
            n_polyribosome_by_group[i] = sum(
                [n_ribosomes_per_mRNA_by_group[i, :] > 0][0])
            avg_n_ribosome_by_group[i] = np.average(
                n_ribosomes_per_mRNA_by_group[i, :][
                    [n_ribosomes_per_mRNA_by_group[i, :] > 0][0]
                ])
            avg_mRNA_mass[i] = np.average([
                initial_state['unique']['RNA'][str(unique_index)]['massDiff_mRNA']
                for unique_index in np.where(
                    n_ribosomes_per_mRNA_by_group[i, :] > 0)[0]])
            polyribosome_mass_by_group[i] = avg_mRNA_mass[i] + avg_n_ribosome_by_group[
                i] * self.unique_masses['active_ribosome']

            # Optional assumption that polyribosomes are completely linear
            # and defined by ribosome size
            self.radii[group] = (avg_n_ribosome_by_group[i] * RIBOSOME_SIZE)/2

            # Add polyribosomes to bulk molecules and molecular weights
            bulk[group] = n_polyribosome_by_group[i]
            self.bulk_molecular_weights[group] = polyribosome_mass_by_group[i]

        # TODO: Make this not hardcoded
        cytosol_front = {
            mol_id: (math.ceil(value / 2) if '[c]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        cytosol_rear = {
            mol_id: (math.floor(value / 2) if '[c]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        # this is currently all 0s
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
            # this is the initial parameters
            'nodes': list(self.nodes.keys()),
            'edges': self.edges,
            'mw': self.bulk_molecular_weights,
            'mesh_size': self.mesh_size,
            'time_step': self.time_step,
            'radii': self.radii,
        }
        return {
            'diffusion_network': DiffusionNetwork(diffusion_config)
        }

    def generate_topology(self, config):
        # connect ports
        return {
            'diffusion_network': {
                node: (node,) for node in config['nodes']
            },
        }



def test_spatial_ecoli():
    ecoli_config = {
        'nodes': {
            'cytosol_front': {
                'length': 0.75,
                'volume': 0.25,
                'molecules': {}
            },
            'nucleoid': {
                'length': 0.75,
                'volume': 0.5,
                'molecules': {}
            },
            'cytosol_rear': {
                'length': 0.75,
                'volume': 0.25,
                'molecules': {}
            },
        },
        'edges': {
            '1': {
                'nodes': ['cytosol_front', 'nucleoid'],
                # TODO: calculate this from width of cell
                'cross_sectional_area': np.pi * 0.3 ** 2,
            },
            '2': {
                'nodes': ['nucleoid', 'cytosol_rear'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
            },
        },
        'mesh_size': 50,
        'time_step': 1,
    }
    ecoli = EcoliSpatial(ecoli_config)

    settings = {
        'total_time': 5*60,
        'initial_state': ecoli.initial_state()}

    data = simulate_compartment_in_experiment(ecoli, settings)

    # TODO -- add assert here for the test to check

    return ecoli_config, data


def run_spatial_ecoli():
    ecoli_config, output = test_spatial_ecoli()
    mesh_size = ecoli_config['mesh_size']
    nodes = ecoli_config['nodes']
    # plot_single_molecule_diff(output, 'CPLX0-3962[c]')
    plot_output(output)
    # plot_single_molecule_diff(output, 'ADHE-CPLX[c]')
    plot_polyribosomes_diff(output, mesh_size, nodes)


def plot_output(output):
    plt.figure()
    cyt_f = array_from(output['cytosol_front']['molecules'])
    nuc = array_from(output['nucleoid']['molecules'])
    colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3',
              '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
    for i, mol in enumerate(output['cytosol_front']['molecules'].keys()):
        # if cyt_f[i][-1] - nuc[i][-1] > 1:
        plt.plot(output['time'], np.divide(cyt_f[i], cyt_f[i][0]), color=colors[i], linestyle='dashed')
        plt.plot(output['time'], np.divide(nuc[i], cyt_f[i][0]), color=colors[i])
            # print(mol)
    plt.xlabel('time (s)')
    plt.ylabel('Number of molecules')
    plt.title('Brownian diffusion with nucleoid mesh of 40 nm')
    plt.legend(['Cytosol front', 'Nucleoid', 'Cytosol rear'])
    out_file = 'out/simulation.png'
    plt.savefig(out_file)


def plot_single_molecule_diff(output, mol_id):
    plt.figure()
    plt.plot(output['time'], np.divide(
        output['cytosol_front']['molecules'][mol_id],
        output['cytosol_front']['volume'][0]))
    plt.plot(output['time'], np.divide(
        output['nucleoid']['molecules'][mol_id],
        output['nucleoid']['volume'][0]))
    plt.plot(output['time'], np.divide(
        output['cytosol_rear']['molecules'][mol_id],
        output['cytosol_rear']['volume'][0]))
    plt.xlabel('time (s)')
    plt.ylabel(r'Concentration (molecules / $\mu m^3$)')
    plt.title(f'Diffusion of {mol_id} over compartments with 50 nm mesh')
    plt.legend(['Cytosol front', 'Nucleoid', 'Cytosol rear'])
    out_file = 'out/single_molecule.png'
    plt.savefig(out_file)


def plot_polyribosomes_diff(output, mesh_size, nodes):
    fig = plt.figure()
    groups = ['polyribosome_1[c]', 'polyribosome_2[c]', 'polyribosome_3[c]',
                  'polyribosome_4[c]','polyribosome_5[c]', 'polyribosome_6[c]',
                  'polyribosome_7[c]', 'polyribosome_8[c]', 'polyribosome_9[c]',
                  'polyribosome_>=10[c]']
    colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3',
              '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
    time = np.divide(output['time'], 60)
    # time = output['time']
    for i, mol_id in enumerate(groups):
        plt.plot(time, np.divide(
            output['cytosol_front']['molecules'][mol_id],
            output['cytosol_front']['molecules'][mol_id][0]), linestyle='dashed',
            color=colors[i], label=str(mol_id + ' in pole'))
        plt.plot(time, np.divide(np.divide(
            output['nucleoid']['molecules'][mol_id],
            output['cytosol_front']['molecules'][mol_id][0]),
            nodes['nucleoid']['volume']/nodes['cytosol_front']['volume']),
                 color=colors[i], label=str(mol_id + ' in nucleoid'))
    plt.xlabel('time (min)')
    plt.ylabel('Normalized concentration (% total concentration)')
    plt.title(f'Diffusion of polyribosomes with mesh of {str(mesh_size)} nm')
    # lgd = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    out_file = 'out/polyribosomes.png'
    plt.tight_layout()
    # plt.savefig(out_file, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.savefig(out_file, dpi=300)

# Helper functions
def array_from(d):
    return np.array(list(d.values()))


def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}

if __name__ == '__main__':
    run_spatial_ecoli()
