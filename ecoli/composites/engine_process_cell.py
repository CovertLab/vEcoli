import copy
import sys

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.library.topology import get_in, assoc_path

from ecoli.experiments.ecoli_master_sim import EcoliSim, SimConfig
from ecoli.library.sim_data import RAND_MAX
from ecoli.processes.engine_process import EngineProcess
from ecoli.processes.listeners.mass_listener import MassListener


class EngineProcessCell(Composer):

    defaults = {
        'agent_id': '0',
        'initial_cell_state': {},
        'seed': 0,
        'initial_tunnel_states': {},
        'parallel': False,
        'ecoli_sim_config': {},
    }

    def generate_processes(self, config):
        agent_id = config['agent_id']
        self.ecoli_sim = EcoliSim({
            **config['ecoli_sim_config'],
            'seed': config['seed'],
        })
        self.ecoli_sim.build_ecoli()
        if config['initial_cell_state']:
            initial_inner_state = {
                'agents': {
                    agent_id: config['initial_cell_state']
                }
            }
        else:
            initial_inner_state = self.ecoli_sim.initial_state
        cell_process = EngineProcess({
            'agent_id': agent_id,
            'composer': self,
            'composite': self.ecoli_sim.ecoli,
            'initial_inner_state': initial_inner_state,
            'tunnels_in': {
                'mass_tunnel': (
                    ('agents', agent_id, 'listeners', 'mass'),
                    MassListener({
                        'submass_indices': {
                            key: None
                            for key in [
                                'rna', 'rRna', 'tRna', 'mRna', 'dna',
                                'protein', 'smallMolecule']
                        }
                    }).ports_schema()['listeners']['mass'],
                ),
            },
            'seed': (config['seed'] + 1) % RAND_MAX,
            '_parallel': config['parallel'],
        })
        return {
            'cell_process': cell_process,
        }

    def generate_topology(self, config):
        return {
            'cell_process': {
                'mass_tunnel': ('listeners', 'mass'),
                'agents': ('..',),
            },
        }

    def initial_state(self, config):
        merged_config = copy.deepcopy(self.config)
        merged_config.update(config)

        relative_mass_listener_path = ('listeners', 'mass')
        absolute_mass_listener_path = (
            'agents', merged_config['agent_id']
        ) + relative_mass_listener_path

        mass_listener_state = merged_config['initial_tunnel_states'].get(
            'mass_tunnel',
            get_in(
                self.ecoli_sim.initial_state,
                absolute_mass_listener_path
            ),
        )
        initial_state = assoc_path(
            {}, relative_mass_listener_path, mass_listener_state)
        return initial_state


def run_simulation():
    config = SimConfig()
    config.update_from_cli()
    composer = EngineProcessCell({
        'agent_id': config['agent_id'],
        'parallel': config['parallel'],
        'ecoli_sim_config': config.to_dict(),
    })
    composite = composer.generate(path=('agents', config['agent_id']))
    emitter_config = {'type': config['emitter']}
    for key, value in config['emitter_arg']:
        emitter_config[key] = value
    engine = Engine(
        processes=composite.processes,
        topology=composite.topology,
        initial_state={
            'agents': {
                config['agent_id']: composer.initial_state({}),
            },
        },
        emitter=emitter_config,
        progress_bar=config['progress_bar'],
    )
    engine.update(composer.ecoli_sim.total_time)
    engine.end()


if __name__ == '__main__':
    run_simulation()
