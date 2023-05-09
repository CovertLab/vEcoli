"""
======================
Tests for Ecoli Master
======================
"""

import os
import numpy as np
import pytest

from vivarium.core.engine import Engine
from vivarium.core.control import run_library_cli

from ecoli.plots.snapshots import plot_snapshots, format_snapshot_data
from ecoli.plots.snapshots_video import make_video
from ecoli.composites.ecoli_configs import ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY
from ecoli.composites.ecoli_master import Ecoli, COUNT_THRESHOLD
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

# tests
from migration.migration_utils import scalar_almost_equal


@pytest.mark.slow
def test_division(
        agent_id='1',
        total_time=4
):
    """tests that a cell can be divided and keep running"""

    # get initial mass from Ecoli composer
    sim = EcoliSim.from_file()
    sim.config['initial_state_file'] = 'vivecoli_t2600'
    sim.config['divide'] = True
    sim.config['agent_id'] = agent_id
    # Set threshold so that mother divides after first timestep
    sim.config['division_threshold'] = 667.8
    sim.config['total_time'] = 4
    # Ensure unique 
    sim.config['emit_paths'] = [
        ('agents', str(agent_id), 'unique',), 
        ('agents', str(agent_id), 'bulk',),
        ('agents', str(agent_id), 'listeners',)]
    sim.build_ecoli()
    initial_state = sim.generated_initial_state
    # Override initially saved division threshold
    initial_state['agents'][str(agent_id)].pop('division_threshold')

    sim.run()

    # retrieve output
    output = sim.ecoli_experiment.emitter.get_data()
    mother_state = next(iter(output[2]['agents'].values()))
    sim_state = sim.ecoli_experiment.state.get_value()
    daughter_states = list(sim_state['agents'].values())

    # compare the counts of bulk molecules between the mother and daughters
    # this is not exact because the mother grew slightly in the timestep
    # after its last emit but before being split into two daughter cells
    assert np.allclose(mother_state['bulk'], daughter_states[0]['bulk'][
        'count'] + daughter_states[1]['bulk']['count'], rtol=0.001, atol=250)

    # compare the counts of unique molecules between the mother and daughters
    for name, mols in mother_state['unique'].items():
        d1_state = daughter_states[0]['unique'][name]
        d2_state = daughter_states[1]['unique'][name]
        m_state = np.array(mols)
        entryState_col = np.where(np.array(
            d1_state.dtype.names) == '_entryState')[0]
        n_mother = m_state[entryState_col].sum()
        n_daughter = d1_state['_entryState'].sum() + \
            d2_state['_entryState'].sum()
        if name == 'chromosome_domain':
            # Chromosome domain 0 is lost after division because
            # it has been fully split into child domains 1 and 2
            n_daughter += 1
        assert np.isclose(n_mother, n_daughter, rtol=0.01), \
            f'{name}: mother has {n_mother}, daughters have {n_daughter}'

    # Assert that no RNA is in both daughters.
    daughter1_rnas = set(daughter_states[0]['unique']['RNA']['unique_index'])
    daughter2_rnas = set(daughter_states[1]['unique']['RNA']['unique_index'])
    assert not (daughter1_rnas & daughter2_rnas)

    # asserts
    final_agents = output[total_time]['agents'].keys()
    print(f"initial agent id: {agent_id}")
    print(f"final agent ids: {final_agents}")
    assert len(final_agents) == 2


def test_division_topology():
    """test that the topology is correctly dividing"""
    timestep = 2

    # get initial mass from Ecoli composer
    initial_state = Ecoli({}).initial_state({'initial_state_file': 'vivecoli_t2600'})
    initial_mass = initial_state['listeners']['mass']['dry_mass']
    division_mass = initial_mass + 0.5
    print(f"DIVIDE AT {division_mass} fg")

    # make a new composer under an embedded path
    agent_id = '0'
    config = {
        'divide': True,
        'agent_id': agent_id,
        'division_threshold': division_mass,  # fg
        'seed': 1,
    }
    agent_path = ('agents', agent_id)
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate(path=agent_path)
    
    # Get shared process instances for partitioned processes
    process_states = {
        process.parameters['process'].name: (process.parameters['process'],)
        for process in ecoli_composite.processes['agents']['0'].values()
        if 'process' in process.parameters
    }
    initial_state['process'] = process_states

    # make experiment
    experiment = Engine(
        processes=ecoli_composite.processes,
        steps=ecoli_composite.steps,
        flow=ecoli_composite.flow,
        topology=ecoli_composite.topology,
        initial_state={'agents': {agent_id: initial_state}},
    )
    # Clean up unnecessary references
    experiment.initial_state = None
    del initial_state, process_states, ecoli_composer, ecoli_composite

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
    sim.total_time = 10
    sim.build_ecoli()
    sim.run()
    data = sim.query()

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
