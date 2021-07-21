"""
Composite model with Metabolism and Convenience Kinetics
"""

from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.library.dict_utils import deep_merge

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.processes.metabolism import Metabolism
from ecoli.processes.convenience_kinetics import ConvenienceKinetics

SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData_3.cPickle'

class ConvenienceMetabolism(Composer):

    defaults = {
        'convenience_kinetics': {},
        'metabolism': {},
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
    }

    def __init__(self, config=None):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None):
        initial_state = get_state_from_file()
        return initial_state

    def generate_processes(self, config):
        time_step = config['time_step']

        # get the parameters
        metabolism_config = self.load_sim_data.get_metabolism_config(time_step=time_step)
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])

        convenience_kinetics_config = self.load_sim_data.get_metabolism_config(time_step=time_step)

        # make the processes
        return {
            'metabolism': Metabolism(metabolism_config),
            'convenience_kinetics': ConvenienceKinetics(convenience_kinetics_config),
        }

    def generate_topology(self, config):
        topology = {
            'metabolism': {
                'metabolites': ('bulk',),
                'catalysts': ('bulk',),
                'kinetics_enzymes': ('bulk',),
                'kinetics_substrates': ('bulk',),
                'amino_acids': ('bulk',),
                'listeners': ('listeners',),
                'environment': ('environment',),
                'polypeptide_elongation': ('process_state', 'polypeptide_elongation'),
                'exchange_constants': ('fluxes',),
            },

            'convenience_kinetics': {
                'internal': ('bulk',),
                'external': ('environment',),
                'exchanges': ('null',),
                'fluxes': ('fluxes',),
            },
        }
        return topology


def test_convenience_metabolism(
        total_time=10,
        progress_bar=True,
):

    composer = ConvenienceMetabolism()

    # get initial state
    initial_state = composer.initial_state()

    # generate the composite
    ecoli = composer.generate()

    # make the experiment
    ecoli_simulation = Engine({
        'processes': ecoli.processes,
        'topology': ecoli.topology,
        'initial_state': initial_state,
        'progress_bar': progress_bar,
    })

    # run the experiment
    ecoli_simulation.update(total_time)

    # retrieve the data
    output = ecoli_simulation.emitter.get_timeseries()
    
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_convenience_metabolism()
