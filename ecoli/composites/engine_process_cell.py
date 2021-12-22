import copy
import sys

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.library.topology import get_in, assoc_path

from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.library.sim_data import RAND_MAX
from ecoli.processes.engine_process import EngineProcess


class EngineProcessCell(Composer):

    defaults = {
        'agent_id': '0',
        'initial_cell_state': {},
        'seed': 0,
    }

    def generate_processes(self, config):
        agent_id = config['agent_id']
        self.ecoli_sim = EcoliSim.from_cli([
            '--agent_id', str(agent_id), '--seed', str(config['seed']),
        ] + sys.argv[1:])
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
                    {
                        variable: {
                            '_default': 0.0,
                            '_emit': True,
                            '_updater': 'set',
                        }
                        for variable in [
                            'cell_mass', 'dry_mass', 'water_mass',
                            'rnaMass', 'rRnaMass', 'tRnaMass', 'mRnaMass',
                            'dnaMass', 'proteinMass', 'smallMoleculeMass',
                            'volume', 'proteinMassFraction',
                            'rnaMassFraction', 'growth',
                            'instantaniousGrowthRate', 'dryMassFoldChange',
                            'proteinMassFoldChange', 'rnaMassFoldChange',
                            'smallMoleculeFoldChange', 'projection_mass',
                            'cytosol_mass', 'extracellular_mass',
                            'flagellum_mass', 'membrane_mass',
                            'outer_membrane_mass', 'periplasm_mass',
                            'pilus_mass', 'inner_membrane_mass',
                        ]
                    },
                ),
            },
            'seed': (config['seed'] + 1) % RAND_MAX,
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
        mass_listener_path = (
            'agents', merged_config['agent_id'], 'listeners', 'mass')
        mass_listener_state = get_in(
            self.ecoli_sim.initial_state, mass_listener_path)
        initial_state = assoc_path({}, mass_listener_path,
            mass_listener_state)
        return initial_state


def run_simulation():
    composer = EngineProcessCell()
    composite = composer.generate(path=('agents', '0'))
    engine = Engine(
        processes=composite.processes,
        topology=composite.topology,
        initial_state=composer.initial_state({}),
        emitter='database',
        progress_bar=True,
    )
    engine.update(composer.ecoli_sim.total_time)


if __name__ == '__main__':
    run_simulation()
