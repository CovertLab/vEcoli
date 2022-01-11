"""
======================
Tests for Ecoli Master
======================
"""

import os
import pytest

from vivarium.core.engine import Engine
from vivarium.core.control import run_library_cli

from ecoli.plots.snapshots import plot_snapshots, format_snapshot_data
from ecoli.plots.snapshots_video import make_video
from ecoli.composites.ecoli_configs import ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY
from ecoli.composites.ecoli_master import Ecoli, COUNT_THRESHOLD
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

# tests
from ecoli.library.schema import get_domain_index_to_daughter
from migration.migration_utils import scalar_almost_equal


@pytest.mark.slow
def test_division(
        agent_id='1',
        total_time=60
):
    """tests that a cell can be divided and keep running"""

    # get initial mass from Ecoli composer
    initial_state = Ecoli({}).initial_state({'initial_state_file': 'vivecoli_t1840'})

    # make a new composer under an embedded path
    config = {
        'divide': True,
        'agent_id': agent_id,
        'division': {
            'threshold': 2220},  # fg
    }
    agent_path = ('agents', agent_id)
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate(path=agent_path)

    # make and run the experiment
    experiment = Engine(
        processes=ecoli_composite.processes,
        steps=ecoli_composite.steps,
        flow=ecoli_composite.flow,
        topology=ecoli_composite.topology,
        initial_state={'agents': {agent_id: initial_state}},
    )
    experiment.update(total_time)

    # retrieve output
    output = experiment.emitter.get_data()

    # get the states of the daughter cells and the mother cell
    daughter_states = []
    for timestep in output:
        if len(output[timestep]['agents'].keys()) == 2:
            d1 = list(output[timestep]['agents'].keys())[0]
            daughter_states.append(output[timestep]['agents'][d1])
            d2 = list(output[timestep]['agents'].keys())[1]
            daughter_states.append(output[timestep]['agents'][d2])
            if timestep == 0.0:
                mother_state = initial_state
            else:
                mother_idx = list(output[timestep - 2.0]['agents'].keys())[0]
                mother_state = output[timestep - 2.0]['agents'][mother_idx]
            break

    # compare the counts of bulk molecules between the mother and daughters
    for bulk_molecule in mother_state['bulk']:
        if mother_state['bulk'][bulk_molecule] > COUNT_THRESHOLD:
            assert (scalar_almost_equal(mother_state['bulk'][bulk_molecule],
                                        daughter_states[0]['bulk'][bulk_molecule] +
                                        daughter_states[1]['bulk'][bulk_molecule],
                                        custom_threshold=0.1))

    # compare the counts of unique molecules between the mother and daughters
    idx_to_d = get_domain_index_to_daughter(mother_state['unique']['chromosome_domain'])
    for key in mother_state['unique']:
        num_divided = 0
        if key == 'promoter' or key == 'oriC' or key == 'DnaA_box' or key == 'chromosomal_segment' \
                or key == 'full_chromosome' or key == 'active_replisome':
            for unique_molecule in mother_state['unique'][key]:
                if idx_to_d[0][mother_state['unique'][key][unique_molecule]['domain_index']] != -1:
                    num_divided += 1
        elif key == 'RNA':
            for rna in mother_state['unique']['RNA']:
                if mother_state['unique']['RNA'][rna]['is_full_transcript']:
                    num_divided += 1
                else:
                    rnap_index = mother_state['unique']['RNA'][rna]['RNAP_index']
                    if idx_to_d[0][mother_state['unique']['active_RNAP'][rnap_index]['domain_index']] != -1:
                        num_divided += 1
        elif key == 'active_RNAP':
            for rnap in mother_state['unique']['active_RNAP']:
                if idx_to_d[0][mother_state['unique']['active_RNAP'][rnap]['domain_index']] != -1:
                    num_divided += 1
        elif key == 'active_ribosome':
            for ribosome in mother_state['unique']['active_ribosome']:
                mrna_index = mother_state['unique']['active_ribosome'][ribosome]['mRNA_index']
                if mother_state['unique']['RNA'][mrna_index]['is_full_transcript']:
                    num_divided += 1
                else:
                    rnap_index = mother_state['unique']['RNA'][mrna_index]['RNAP_index']
                    if idx_to_d[0][mother_state['unique']['active_RNAP'][rnap_index]['domain_index']] != -1:
                        num_divided += 1
        elif key == 'chromosome_domain':
            num_divided = len(mother_state['unique']['chromosome_domain'].keys()) - 1
        assert (scalar_almost_equal(num_divided,
                len(daughter_states[0]['unique'][key]) +
                len(daughter_states[1]['unique'][key]),
                custom_threshold=0.1))

    # asserts
    final_agents = output[total_time]['agents'].keys()
    print(f"initial agent id: {agent_id}")
    print(f"final agent ids: {final_agents}")
    assert len(final_agents) == 2


def test_division_topology():
    """test that the topology is correctly dividing"""
    timestep = 2

    # get initial mass from Ecoli composer
    initial_state = Ecoli({}).initial_state({'initial_state_file': 'vivecoli_t1840'})
    initial_mass = initial_state['listeners']['mass']['cell_mass']
    division_mass = initial_mass + 4.5
    print(f"DIVIDE AT {division_mass} fg")

    # make a new composer under an embedded path
    agent_id = '0'
    config = {
        'divide': True,
        'agent_id': agent_id,
        'division': {
            'threshold': division_mass},  # fg
        'seed': 1,
    }
    agent_path = ('agents', agent_id)
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate(path=agent_path)

    # make experiment
    experiment = Engine(
        processes=ecoli_composite.processes,
        steps=ecoli_composite.steps,
        flow=ecoli_composite.flow,
        topology=ecoli_composite.topology,
        initial_state={'agents': {agent_id: initial_state}},
    )

    full_topology = experiment.state.get_topology()
    mother_topology = full_topology['agents'][agent_id].copy()

    # update one time step at a time until division
    while len(full_topology['agents']) <= 1:
        experiment.update(timestep)
        full_topology = experiment.state.get_topology()

    # assert that the daughter topologies are the same as the mother topology
    daughter_ids = list(full_topology['agents'].keys())
    for daughter_id in daughter_ids:
        daughter_topology = full_topology['agents'][daughter_id]
        assert daughter_topology == mother_topology


def test_ecoli_generate():
    ecoli_composer = Ecoli({})
    ecoli_composite = ecoli_composer.generate()

    # asserts to ecoli_composite['processes'] and ecoli_composite['topology']
    assert all('_requester' in k or
               '_evolver' in k or
               k == 'allocator' or
               isinstance(v, ECOLI_DEFAULT_PROCESSES[k])
               for k, v in ecoli_composite['processes'].items())
    assert all(ECOLI_DEFAULT_TOPOLOGY[k] == v
               for k, v in ecoli_composite['topology'].items()
               if k in ECOLI_DEFAULT_TOPOLOGY)


def test_lattice_lysis(plot=False):
    """
    Run plots:
    '''
    > python ecoli/composites/ecoli_master_tests.py -n 4 -o plot=True
    '''

    ANTIBIOTIC_KEY = 'nitrocefin'
    PUMP_KEY = 'TRANS-CPLX-201[s]'
    PORIN_KEY = 'porin'
    BETA_LACTAMASE_KEY = 'EG10040-MONOMER[p]'

    TODO: connect glucose! through local_field
    """
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'lysis.json')
    # sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'spatial.json')
    sim.total_time = 60
    data = sim.run()

    if plot:
        plot_spatial_snapshots(data, sim, experiment_dir='ecoli_lysis')


def plot_spatial_snapshots(data, sim, experiment_dir='ecoli_test'):
    out_dir = os.path.join('out', 'experiments', experiment_dir)
    os.makedirs(out_dir, exist_ok=True)

    bounds = sim.config['spatial_environment_config']['multibody']['bounds']

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=5,
        out_dir=out_dir,
        filename=f"snapshots")

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type='fields',
        out_dir=out_dir,
        filename='video',
    )



test_library = {
    '1': test_division,
    '2': test_division_topology,
    '3': test_ecoli_generate,
    '4': test_lattice_lysis,
}

# run experiments in test_library from the command line with:
# python ecoli/composites/ecoli_master_tests.py -n [experiment id]
if __name__ == '__main__':
    run_library_cli(test_library)
