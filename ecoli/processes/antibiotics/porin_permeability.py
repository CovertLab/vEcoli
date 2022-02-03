from ecoli.library.schema import bulk_schema
from ecoli.states.wcecoli_state import get_state_from_file
from vivarium.core.engine import Engine
from vivarium.core.process import Step
from vivarium.library.units import units

SA_AVERAGE = 6.22200939450696
CEPH_OMPC_CON_PERM = 0.003521401200296894 * 1e-5 * units.cm * units.micron * units.micron / units.sec
CEPH_OMPF_CON_PERM = 0.01195286573132685 * 1e-5 * units.cm * units.micron * units.micron / units.sec


class PorinPermeability(Step):
    defaults = {
        'porin_ids': [],
        'diffusing_molecules': [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.porin_ids = self.parameters['porin_ids']
        self.diffusing_molecules = self.parameters['diffusing_molecules']

    def ports_schema(self):
        return {
            'porins': bulk_schema(self.porin_ids),
            'permeabilities': {mol_id: {
                '_default': 1e-5 * units.cm / units.sec,
                '_emit': True,
                '_updater': 'set'
            } for mol_id in self.diffusing_molecules},  # Different permeability for every antibiotic
            'surface_area': {'_default': 0.0}
        }

    def next_update(self, timestep, states):
        porins = states['porins']
        surface_area = states['surface_area'] * units.micron * units.micron
        permeabilities = {}
        for molecule in self.diffusing_molecules:
            cell_permeability = 0
            for porin_id in self.diffusing_molecules[molecule].keys():
                cell_permeability += (porins[porin_id] / surface_area) * self.diffusing_molecules[molecule][porin_id]
            permeabilities[molecule] = cell_permeability
        return {'permeabilities': permeabilities}


def main():
    sim_time = 10

    parameters = {
        'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
        'diffusing_molecules': {
            'cephaloridine': {
                'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
            }
        },
    }
    process = PorinPermeability(parameters)

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    sim = Engine(processes={'porin_permeability': process},
                 topology={
                     'porin_permeability': {
                         'porins': ('bulk',),
                         'permeabilities': ('boundary', 'permeabilities',),
                         'surface_area': ('boundary', 'surface_area',)
                     },
                 },
                 initial_state=initial_state)
    sim.update(sim_time)
    data = sim.emitter.get_data()


if __name__ == '__main__':
    main()
