"""
======================
Tests for Ecoli Master
======================
"""

import os
import numpy as np
import pytest
import warnings

from vivarium.core.engine import Engine
from vivarium.core.control import run_library_cli

from ecoli.plots.snapshots import plot_snapshots, format_snapshot_data
from ecoli.plots.snapshots_video import make_video
from ecoli.composites.ecoli_configs import ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from migration.migration_utils import recursive_compare


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
    sim.config['total_time'] = total_time
    # Ensure unique molecules are emitted
    sim.config['emit_paths'] = [
        ('agents', str(agent_id), 'unique',), 
        ('agents', str(agent_id), 'bulk',),
        ('agents', str(agent_id), 'listeners',)]
    sim.build_ecoli()
    initial_state = sim.generated_initial_state
    # Override initially saved division threshold
    initial_state['agents'][str(agent_id)].pop('division_threshold', None)

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
        # Assert that no unique mol is in both daughters
        assert not (set(d1_state['unique_index']) & 
            set(d2_state['unique_index']))

    # asserts
    final_agents = output[total_time]['agents'].keys()
    print(f"initial agent id: {agent_id}")
    print(f"final agent ids: {final_agents}")
    assert len(final_agents) == 2


def test_division_topology():
    """test that the topology is correctly dividing"""
    timestep = 2
    agent_id = '0'
    # get initial mass from Ecoli composer
    sim = EcoliSim.from_file()
    sim.config['seed'] = 1
    sim.config['initial_state_file'] = 'vivecoli_t2600'
    sim.config['divide'] = True
    sim.config['agent_id'] = agent_id
    # Ensure unique molecules are emitted
    sim.config['emit_paths'] = [
        ('agents', agent_id, 'unique',), 
        ('agents', agent_id, 'bulk',),
        ('agents', agent_id, 'listeners',)]
    sim.build_ecoli()
    initial_state = sim.generated_initial_state
    # Set threshold so that mother divides after first timestep
    initial_state['agents'][agent_id]['division_threshold'] = initial_state[
        'agents'][agent_id]['listeners']['mass']['dry_mass'] + 0.5
    division_mass = initial_state['agents'][agent_id]['division_threshold']
    print(f"DIVIDE AT {division_mass} fg")

    # make the experiment
    experiment_config = {
        'processes': sim.ecoli.processes,
        'steps': sim.ecoli.steps,
        'flow': sim.ecoli.flow,
        'topology': sim.ecoli.topology,
        'initial_state': sim.generated_initial_state,
    }

    # Since unique numpy updater is an class method, internal
    # deepcopying in vivarium-core causes this warning to appear
    warnings.filterwarnings("ignore",
        message="Incompatible schema assignment at ")
    sim.ecoli_experiment = Engine(**experiment_config)

    # Only emit designated stores
    sim.ecoli_experiment.state.set_emit_values([tuple()], False)
    sim.ecoli_experiment.state.set_emit_values(
        sim.config['emit_paths'],
        True,
    )
    
    # Clean up unnecessary references
    sim.generated_initial_state = None
    sim.ecoli_experiment.initial_state = None

    full_topology = sim.ecoli_experiment.state.get_topology()
    mother_topology = full_topology['agents'][agent_id].copy()

    # update one time step at a time until division
    while len(full_topology['agents']) <= 1:
        sim.ecoli_experiment.update(timestep)
        full_topology = sim.ecoli_experiment.state.get_topology()
    sim.ecoli_experiment.end()

    # assert that the daughter topologies are the same as the mother topology
    daughter_ids = list(full_topology['agents'].keys())
    for daughter_id in daughter_ids:
        daughter_topology = full_topology['agents'][daughter_id]
        assert daughter_topology == mother_topology


def test_ecoli_generate():
    sim = EcoliSim.from_file()
    sim.build_ecoli()

    # asserts to ecoli_composite['processes'] and ecoli_composite['topology']
    assert all('_requester' in k or
               '_evolver' in k or
               k == 'allocator' or
               isinstance(v, ECOLI_DEFAULT_PROCESSES[k])
               for k, v in sim.ecoli['processes'].items())
    for k, v in sim.ecoli['topology'].items():
        proc_name = k.split('_requester')[0].split('_evolver')[0]
        # Clock topology is not registered in registry
        if proc_name == 'clock':
            pass 
        elif proc_name in ECOLI_DEFAULT_TOPOLOGY:
            # Ignore partitioning-related keys
            assert recursive_compare(ECOLI_DEFAULT_TOPOLOGY[proc_name], v,
                ignore_keys={'request', 'evolvers_ran', 'process', 'allocate'})


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
    sim.process_configs.update({'clock': {'time_step': 2}})
    sim.build_ecoli()
    # Add beta-lactam to bulk store
    initial_state = sim.generated_initial_state
    initial_state['agents']['0']['bulk'] = np.append(
        initial_state['agents']['0']['bulk'], np.array(
            [('beta-lactam', 0),
             ('hydrolyzed-beta-lactam', 0)],
            dtype=initial_state['agents']['0']['bulk'].dtype))
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
