from vivarium.core.process import Process
from vivarium.library.units import units


class Exchange(Process):
    """ Exchange
    A minimal exchange process that moves molecules between two ports.
    """
    defaults = {
        'molecules': [],
        'uptake_rate': {},
        'secrete_rate': {},
        'default_uptake_rate': 1e-1,
        'default_secrete_rate': 1e-4,
        # calculated in deriver_globals assuming mass = 1000 fg, density = 1100 g/L
        'mmol_to_counts': 547467.342,  # units.L / units.mmol
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.uptake_rate = self.parameters['uptake_rate']
        self.secrete_rate = self.parameters['secrete_rate']

        for mol_id in self.parameters['molecules']:
            if mol_id not in self.uptake_rate:
                self.uptake_rate[mol_id] = self.parameters['default_uptake_rate']
            if mol_id not in self.secrete_rate:
                self.secrete_rate[mol_id] = self.parameters['default_secrete_rate']

    def ports_schema(self):
        return {
            'exchange': {
                mol_id: {'_default': 0}
                for mol_id in self.parameters['molecules']},
            'external': {
                mol_id: {
                    '_default': 0.0,
                    '_emit': True}
                for mol_id in self.parameters['molecules']},
            'internal': {
                mol_id: {
                    '_default': 0.0,
                    '_emit': True}
                for mol_id in self.parameters['molecules']}}

    def next_update(self, timestep, states):
        external_molecules = states['external']
        internal_molecules = states['internal']

        delta_in = {
            mol_id: mol_ex * self.uptake_rate[mol_id] -
                    internal_molecules[mol_id] * self.secrete_rate[mol_id] * timestep
            for mol_id, mol_ex in external_molecules.items()}

        # convert delta concentrations to exchange counts
        # assumes concentrations in mmol/L
        exchange_counts = {}
        for molecule, concentration in delta_in.items():
            exchange_counts[molecule] = -int(concentration * self.parameters['mmol_to_counts'])

        return {
            'internal': delta_in,
            'exchange': exchange_counts}
