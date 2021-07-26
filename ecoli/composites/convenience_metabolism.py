"""
Composite model with Metabolism and Convenience Kinetics
"""

from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.library.dict_utils import deep_merge

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH, AA_MEDIA_ID
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.processes.metabolism import Metabolism
from ecoli.processes.convenience_kinetics import ConvenienceKinetics
from ecoli.processes.exchange_stub import Exchange

SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData_4.cPickle'

class ConvenienceMetabolism(Composer):

    defaults = {
        'convenience_kinetics': {},
        'metabolism': {
            'media_id': AA_MEDIA_ID
        },
        'aa':True,
        'time_step': 1.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
    }

    def __init__(self, config=None):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None, aa=False, initial_time=0):
        state = get_state_from_file(path=f'data/wcecoli_t{initial_time}.json',aa=aa)
        return state

    def generate_processes(self, config):
        time_step = config['time_step']

        # get the parameters
        metabolism_config = self.load_sim_data.get_metabolism_config(time_step=time_step,aa=config['aa'])
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])

        convenience_kinetics_config = self.load_sim_data.get_convenience_kinetics_config(time_step=time_step)
        convenience_kinetics_config = deep_merge(convenience_kinetics_config, config['convenience_kinetics'])
        # make the processes

        exchange_stub_config = {'exchanges':{
                                    'L-ALPHA-ALANINE[c]': -354950.54854027436, 
                                    'ARG[c]': -1995000.64878649314, 
                                    'ASN[c]': -145420.125747449876, 
                                    'L-ASPARTATE[c]': -197510.313049595497, 
                                    'CYS[c]': -2655.257650369328, 
                                    'GLT[c]': -234960.96412240591, 
                                    'GLN[c]': -135360.34365107281, 
                                    'GLY[c]': -279030.927189588463, 
                                    'HIS[c]': -6804.0383397819205, 
                                    'ILE[c]': -20133.64878649314, 
                                    'LEU[c]': -29177.50281392895, 
                                    'LYS[c]': -234970.572810411537, 
                                    'MET[c]': -9044.676046429828, 
                                    'PHE[c]': -11638.970629616602, 
                                    'PRO[c]': -131010.371438621174, 
                                    'SER[c]': -178830.035174111854, 
                                    'THR[c]': -197660.221948645798, 
                                    'TRP[c]': -33190.498065423848, 
                                    'TYR[c]': -95700.829053816391, 
                                    'L-SELENOCYSTEINE[c]': -100.004396763981709462,
                                    'VAL[c]': -278370.83644037988}
                                }
        return {
            'convenience_kinetics': ConvenienceKinetics(convenience_kinetics_config),
            'metabolism': Metabolism(metabolism_config),
            'exchange_stub': Exchange(exchange_stub_config),
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
                'exchange_constraints': ('fluxes',),
                'amino_acids_inside':('amino_acids_inside',),
            },

            'convenience_kinetics': {
                'external': ('environment',),
                'exchanges': (None,),
                'fluxes': ('fluxes',),
                'internal':('bulk',),
                'listeners': ('listeners',),
                'import_counts':('import_counts',)
            },

            'exchange_stub':{
                'molecules':('bulk',),
            }
        }
        return topology


def test_convenience_metabolism(
        total_time=50,
        progress_bar=True,
):  

    composer = ConvenienceMetabolism()

    # get initial state
    initial_state = composer.initial_state(aa=True)

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
    
    import ipdb; ipdb.set_trace(context=15)


if __name__ == "__main__":
    test_convenience_metabolism()
