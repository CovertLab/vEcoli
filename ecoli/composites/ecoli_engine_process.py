'''Composite for simulations with EngineProcess cells in an environment.

.. note::

    This composite requires a config with the spatial environment
    enabled.
'''

import copy
import sys

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.library.topology import get_in, assoc_path
from vivarium.library.dict_utils import deep_merge_check

from ecoli.experiments.ecoli_master_sim import (
    EcoliSim,
    SimConfig,
    get_git_revision_hash,
    get_git_status,
    report_profiling,
)
from ecoli.library.schema import bulk_schema
from ecoli.library.sim_data import RAND_MAX
from ecoli.processes.engine_process import EngineProcess
from ecoli.processes.environment.field_timeline import FieldTimeline
from ecoli.processes.listeners.mass_listener import MassListener
from ecoli.processes.shape import Shape
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

    composer = EcoliEngineProcess({
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
    })
    composite = composer.generate(path=('agents', config['agent_id']))
    initial_state = {
        'agents': {
            config['agent_id']: composite.initial_state()
        },
    }

    if config['spatial_environment']:
        # Merge a lattice composite for the spatial environment.
        initial_environment = environment_composite.initial_state()
        composite.merge(environment_composite)
        initial_state = deep_merge_check(initial_state, initial_environment)

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
    # Assert that nothing got wired into `null`.
    assert not engine.state.get_path(
        ('agents', config['agent_id'], 'cell_process')
    ).value.sim.state.get_path(('null',)).inner

    engine.update(config['total_time'])
    engine.end()

    if config['profile']:
        report_profiling(engine.stats)

if __name__ == '__main__':
    run_simulation()
