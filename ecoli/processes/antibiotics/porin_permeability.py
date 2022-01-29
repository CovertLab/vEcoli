from vivarium.core.process import Step
from vivarium.core.composition import simulate_process
from ecoli.library.schema import bulk_schema
from ecoli.states.wcecoli_state import get_state_from_file
from vivarium.core.engine import Engine

SA_AVERAGE = 6.22200939450696
OMPC_CONCENTRATION_PERM = 0.003521401200296894
OMPF_CONCENTRATION_PERM = 0.01195286573132685

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
        permeability_coefficients = self.permeability_coefficients
        cell_permeability = 0
        for porin_id in porins:
            cell_permeability += (porins[porin_id] / surface_area) * permeability_coefficients[porin_id]
        permeabilities = {'cephaloridine': cell_permeability}  # TODO (Matt): for every diffusing molecule, one permeability value for the entire cell
        return {'permeabilities': permeabilities}


def main():
    sim_time = 10

    parameters = {
        'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
        'diffusing_molecules': ['cephaloridine'],  # Temporary
        'permeability_coefficients': {'CPLX0-7533[o]': OMPC_CONCENTRATION_PERM,
                                      'CPLX0-7534[o]': OMPF_CONCENTRATION_PERM}
    }
    process = PorinPermeability(parameters)

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    sim = Engine(processes={'porin_permeability': process},
                 topology={
                     'porin_permeability': {
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
