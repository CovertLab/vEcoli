'''Composite for simulations with EngineProcess cells in an environment.

.. note::

    This composite requires a config with the spatial environment
    enabled.
'''

import copy

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.serialize import serialize_value, deserialize_value
from vivarium.library.dict_utils import deep_merge

from ecoli.experiments.ecoli_master_sim import (
    EcoliSim,
    SimConfig,
    get_git_revision_hash,
    get_git_status,
    report_profiling,
)
from ecoli.library.logging import write_json
from ecoli.library.sim_data import RAND_MAX
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.processes.engine_process import EngineProcess
from ecoli.processes.environment.field_timeline import FieldTimeline
from ecoli.composites.environment.lattice import Lattice


class EcoliEngineProcess(Composer):

    defaults = {
        'agent_id': '0',
        'initial_cell_state': {},
        'seed': 0,
        'tunnel_out_schemas': {},
        'stub_schemas': {},
        'parallel': False,
        'ecoli_sim_config': {},
        'divide': False,
        'division_threshold': 0,
        'division_variable': tuple(),
        'reports': tuple(),
    }

    def generate_processes(self, config):
        agent_id = config['agent_id']
        self.ecoli_sim = EcoliSim({
            **config['ecoli_sim_config'],
            'seed': config['seed'],
            'agent_id': agent_id,
            'divide': False,  # Division is handled by EngineProcess.
            'spatial_environment': False,
        })
        self.ecoli_sim.build_ecoli()
        initial_inner_state = (
            config['initial_cell_state']
            or self.ecoli_sim.initial_state)

        cell_process_config = {
            'agent_id': agent_id,
            'composer': self,
            'composite': self.ecoli_sim.ecoli,
            'initial_inner_state': initial_inner_state,
            'tunnels_in': dict({
                f'{"-".join(path)}_tunnel': path
                for path in config['reports']
            }),
            'tunnel_out_schemas': config['tunnel_out_schemas'],
            'stub_schemas': config['stub_schemas'],
            'seed': (config['seed'] + 1) % RAND_MAX,
            'divide': config['divide'],
            'division_threshold': config['division_threshold'],
            'division_variable': config['division_variable'],
            '_parallel': config['parallel'],
        }
        cell_process = EngineProcess(cell_process_config)
        return {
            'cell_process': cell_process,
        }

    def generate_topology(self, config):
        topology = {
            'cell_process': {
                'agents': ('..',),
                'fields_tunnel': ('..', '..', 'fields'),
                'dimensions_tunnel': ('..', '..', 'dimensions'),
            },
        }
        for path in config['reports']:
            topology['cell_process'][f'{"-".join(path)}_tunnel'] = path
        return topology


def colony_save_states(engine, config):
    """
    Runs the simulation while saving the states of the colony at specific timesteps to jsons.
    """
    for time in config["save_times"]:
        if time > config["total_time"]:
            raise ValueError(
                f'Config contains save_time ({time}) > total '
                f'time ({config["total_time"]})')
    for i in range(len(config["save_times"])):
        if i == 0:
            time_to_next_save = config["save_times"][i]
        else:
            time_to_next_save = config["save_times"][i] - config["save_times"][i - 1]
        # Run engine to next save point
        engine.update(time_to_next_save)
        time_elapsed = config["save_times"][i]

        # Save the full state of the super-simulation
        def not_a_process(value):
            return not isinstance(value, Process)
        state = engine.state.get_value(condition=not_a_process)
        state_to_save = copy.deepcopy(state)

        del state_to_save['agents']  # Replace 'agents' with agent states
        state_to_save['agents'] = {}
        for agent_id in state['agents']:
            # Get internal state from the EngineProcess sub-simulation
            cell_state = state['agents'][agent_id]['cell_process'][0].sim.state.get_value(condition=not_a_process)
            del cell_state['environment']['exchange_data']  # Can't save, but will be restored when loading state
            state_to_save['agents'][agent_id] = cell_state

        state_to_save = serialize_value(state_to_save)
        write_json('data/colony_t' + str(time_elapsed) + '.json', state_to_save)
        print('Finished saving the state at t = ' + str(time_elapsed))

    # Finish running the simulation
    time_remaining = config["total_time"] - config["save_times"][-1]
    if time_remaining:
        engine.update(time_remaining)


def run_simulation():
    config = SimConfig()
    config.update_from_cli()

    tunnel_out_schemas = {}
    stub_schemas = {}
    if config['spatial_environment']:
        # Generate environment composite.
        environment_composer = Lattice(
            config['spatial_environment_config'])
        environment_composite = environment_composer.generate()
        field_timeline = FieldTimeline(
            config['spatial_environment_config']['field_timeline'])
        environment_composite.merge(
            processes={'field_timeline': field_timeline},
            topology={
                'field_timeline': {
                    port: tuple(path)
                    for port, path in config[
                        'spatial_environment_config'
                    ]['field_timeline_topology'].items()
                },
            },
        )

        diffusion_schema = environment_composite.processes[
            'diffusion'].get_schema()
        multibody_schema = environment_composite.processes[
            'multibody'].get_schema()
        tunnel_out_schemas['fields_tunnel'] = diffusion_schema['fields']
        tunnel_out_schemas['dimensions_tunnel'] = diffusion_schema[
            'dimensions']
        stub_schemas['diffusion'] = {
            ('boundary',): diffusion_schema['agents']['*']['boundary'],
        }
        stub_schemas['multibody'] = {
            ('boundary',): multibody_schema['agents']['*']['boundary'],
        }

    base_config = {
        'agent_id': config['agent_id'],
        'tunnel_out_schemas': tunnel_out_schemas,
        'stub_schemas': stub_schemas,
        'parallel': config['parallel'],
        'ecoli_sim_config': config.to_dict(),
        'divide': config['divide'],
        'division_threshold': config['division']['threshold'],
        'division_variable': ('listeners', 'mass', 'cell_mass'),
        'reports': tuple(
            tuple(path) for path in
            config.get('engine_process_reports', tuple())
        ),
        'seed': config['seed'],
    }
    composite = {}
    if 'initial_colony_file' in config.keys():
        initial_state = get_state_from_file(path=f'data/{config["initial_colony_file"]}.json')  # TODO(Matt): initial_state_file is wc_ecoli?
        initial_state = deserialize_value(initial_state)
        agent_states = initial_state['agents']
        for agent_id, agent_state in agent_states.items():
            agent_config = copy.deepcopy(base_config)
            agent_config['initial_cell_state'] = agent_state
            agent_config['agent_id'] = agent_id
            agent_composer = EcoliEngineProcess(agent_config)
            agent_composite = agent_composer.generate(path=('agents', agent_id))
            if not composite:
                composite = agent_composite
            composite.processes['agents'][agent_id] = agent_composite.processes['agents'][agent_id]
            composite.topology['agents'][agent_id] = agent_composite.topology['agents'][agent_id]
    else:
        composer = EcoliEngineProcess(base_config)
        composite = composer.generate(path=('agents', config['agent_id']))
        initial_state = composite.initial_state()

    if config['spatial_environment']:
        # Merge a lattice composite for the spatial environment.
        initial_environment = environment_composite.initial_state()
        composite.merge(environment_composite)
        initial_state = deep_merge(initial_state, initial_environment)

    metadata = config.to_dict()
    metadata.pop('initial_state', None)
    metadata['git_hash'] = get_git_revision_hash()
    metadata['git_status'] = get_git_status()

    emitter_config = {'type': config['emitter']}
    for key, value in config['emitter_arg']:
        emitter_config[key] = value
    engine = Engine(
        processes=composite.processes,
        topology=composite.topology,
        initial_state=initial_state,
        emitter=emitter_config,
        progress_bar=config['progress_bar'],
        metadata=metadata,
        profile=config['profile'],
    )

    # Save states while running if needed
    if config["save"]:
        colony_save_states(engine, config)
    else:
        engine.update(config['total_time'])
    engine.end()

    if config['profile']:
        report_profiling(engine.stats)


if __name__ == '__main__':
    run_simulation()
