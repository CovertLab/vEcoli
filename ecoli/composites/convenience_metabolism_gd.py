
from vivarium.core.composer import Composer, Composite
from vivarium.core.engine import Engine
from vivarium.library.dict_utils import deep_merge

from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.composites.ecoli_master import SIM_DATA_PATH

from ecoli.processes.metabolism_gd import MetabolismGD
from ecoli.processes.enzyme_kinetics import EnzymeKinetics


class MetabolismGDKinetics(Composer):
    defaults = {
        'metabolism': {},
        'kinetics': {},
        'sim_data_path': SIM_DATA_PATH,
        'seed': 0,
        'reactions': []
    }

    def __init__(self, config):
        super().__init__(config)
        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])


    def generate_processes(self, config):

        reactions = config['reactions']

        # configure metabolism
        metabolism_config = self.load_sim_data.get_metabolism_gd_config()
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])
        metabolism_config['kinetic_rates'] = reactions
        metabolism_process = MetabolismGD(metabolism_config)
        reaction_stoich_r = metabolism_config['stoichiometry_r']
        reaction_stoich = {rxn_id: stoich for rxn_id, stoich in reaction_stoich_r.items() if rxn_id in reactions}

        # configure kinetics
        kinetics_config = config['kinetics']
        kinetics_config['reactions'] = reaction_stoich
        kinetics_process = EnzymeKinetics(kinetics_config)

        return {
            'metabolism': metabolism_process,
            'kinetics': kinetics_process,
        }

    def generate_topology(self, config):
        return {
            'metabolism': MetabolismGD.topology,
            'kinetics': {
                'bulk': ('bulk',),
                'fluxes': ('rates', 'fluxes',)
            },
        }


def main(
    total_time=10,
):
    config = {
        'kinetics': {
            'kinetic_parameters': {
                'reaction1': {
                    ('bulk', 'enzyme1'): {
                        'kcat_f': 5e1,
                    }
                }
            },
        },
        'reactions': []
    }

    composer = MetabolismGDKinetics(config)
    composite = composer.generate()

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')
    # TODO -- get initial fluxes as well (?)
    initial_state['bulk'].update({
        'A': 1.0,
        'B': 1.0,
        'enzyme1': 1.0,
    })

    # make the experiment
    experiment = Engine(
        processes=composite.processes,
        topology=composite.topology,
        initial_state=initial_state,
    )

    experiment.update(total_time)

    data = experiment.emitter.get_data()




# run with: python ecoli/composites/convenience_metabolism_gd.py
if __name__ == '__main__':
    main()
