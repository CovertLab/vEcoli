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
        permeability_coefficients = self.permeability_coefficients
        cell_permeability = 0
        for porin_id in porins:
            cell_permeability += (porins[porin_id] / surface_area) * permeability_coefficients[porin_id]
        permeabilities = {'cephaloridine': cell_permeability}  # TODO (Matt): for every diffusing molecule, one permeability value for the entire cell
        return {'permeabilities': permeabilities}


def main():
    sim_time = 10

    # Calculate permeability by porin type
    # Average of the porin counts during the simulation divided by the average surface area during the simulation
    # Calculate these averages outside of this process

    from vivarium.core.emitter import (
        data_from_database, get_local_client, timeseries_from_data)
    data, conf = data_from_database('9cc838ec-7d6e-11ec-b2e8-1e00312eb299',
                                    get_local_client("localhost", "27017", "simulations"),
                                    query=[('bulk', 'EG10670-MONOMER[o]'),
                                           ('bulk', 'EG10671-MONOMER[o]'),
                                           ('boundary', 'surface_area')])
    data = timeseries_from_data(data)

    sa_sum = 0
    ompc_sum = 0
    ompf_sum = 0
    sa_len = len(data['boundary']['surface_area'])
    ompc_len = len(data['bulk']['EG10670-MONOMER[o]'])
    ompf_len = len(data['bulk']['EG10671-MONOMER[o]'])
    for i in range(sa_len):
        sa_sum += data['boundary']['surface_area'][i]
    for i in range(ompc_len):
        ompc_sum += data['bulk']['EG10670-MONOMER[o]'][i]
    for i in range(ompf_len):
        ompf_sum += data['bulk']['EG10671-MONOMER[o]'][i]
    sa_average = sa_sum / sa_len
    ompc_average = ompc_sum / ompc_len  # ompc porin count about 50,000 halfway through on ecocyc
    ompf_average = ompf_sum / ompf_len  # ompf porin count about 71,798 halfway through on ecocyc
    import ipdb; ipdb.set_trace()

    ompc_concentration = ompc_average / sa_average
    ompf_concentration = ompf_average / sa_average
    # Is volume-based concentration more appropriate?
    ompc_permeability = 4.5 / ompc_concentration  # porin concentration permeability coefficient
    ompf_permeability = 52.6 / ompf_concentration

    parameters = {
        'porin_ids': ['EG10670-MONOMER[o]', 'EG10671-MONOMER[o]'],
        'diffusing_molecules': ['cephaloridine'],  # Temporary
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
