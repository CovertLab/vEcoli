from vivarium.core.process import Step
from vivarium.core.composition import simulate_process
from ecoli.library.schema import bulk_schema
from ecoli.states.wcecoli_state import get_state_from_file
from vivarium.core.engine import Engine


class PorinPermeability(Step):
    defaults = {
        'porin_ids': [],
        'diffusing_molecules': []
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.porin_ids = self.parameters['porin_ids']
        self.diffusing_molecules = self.parameters['diffusing_molecules']
        self.permeability_coefficients = self.parameters['permeability_coefficients']

    def ports_schema(self):
        return {
            'porins': bulk_schema(self.porin_ids),
            'permeabilities': {mol_id: {
                '_default': 0.0,
                '_emit': True,
                '_updater': 'set'
            } for mol_id in self.diffusing_molecules},  # Different permeability for every antibiotic
            'surface_area': {'_default': 0.0}
        }

    def next_update(self, timestep, states):
        porins = states['porins']
        surface_area = states['surface_area']
        # Math here
        permeabilities = {}  # TODO (Matt): for every diffusing molecule, one permeability value for the entire cell
        return {'permeabilities': permeabilities}


def main():
    sim_time = 10

    # Calculate permeability by porin type
    # Average of the porin counts during the simulation divided by the average surface area during the simulation
    # Calculate these averages outside of this process

    # ompc porin count about 50,000 halfway through
    #

    parameters = {
        'porin_ids': ['EG10670', 'EG10671', 'EG10729'],
        'diffusing_molecules': [],
        'permeability_coefficients': {}
    }
    process = PorinPermeability(parameters)

    initial_state = get_state_from_file(path='data/vivecoli_t1840.json')
    initial_state['boundary']['surface_area'] = 6.75  # 4.5 microns squared at the very beginning, 9 at end
    # Create a new save state that has surface area
    sim = Engine(processes={'porin_permeability': process},
                 topology={
                     'porin_permeability':{
                         'porins': ('bulk',),
                         'permeabilities': ('permeabilities',),
                         'surface_area': ('boundary', 'surface_area')
                     },
                 },
                 initial_state=initial_state)
    sim.update(sim_time)
    data = sim.emitter.get_data()


if __name__ == '__main__':
    main()
