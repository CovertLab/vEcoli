"""
MetabolismGD
"""

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import bulk_schema, array_from

from wholecell.utils import units

from ecoli.library.fba_gd import GradientDescentFba, FbaResult


COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
CONVERSION_UNITS = MASS_UNITS * TIME_UNITS / VOLUME_UNITS
GDCW_BASIS = units.mmol / units.g / units.h

USE_KINETICS = True


class MetabolismGD(Process):
    name = 'ecoli-metabolism-gradient-descent'

    defaults = {
        'stoichiometry': [],
        'reaction_catalysts': [],
        'catalyst_ids': [],
        'media_id': 'minimal',
        'objective_type': 'homeostatic',
        'nutrients': [],
        'cell_density': 1100 * units.g / units.L,

        # 'get_import_constraints': lambda u, c, p: (u, c, []),
        # 'nutrientToDoublingTime': {},
        # 'use_trna_charging': False,
        # 'include_ppgpp': False,
        # 'aa_names': [],
        # 'current_timeline': None,
        # 'media_id': 'minimal',
        # 'condition': 'basal',
        # 'nutrients': [],
        # 'metabolism': {},
        # 'non_growth_associated_maintenance': 8.39 * units.mmol / (units.g * units.h),
        # 'avogadro': 6.02214076e+23 / units.mol,
        # 'cell_density': 1100 * units.g / units.L,
        # 'dark_atp': 33.565052868380675 * units.mmol / units.g,
        # 'cell_dry_mass_fraction': 0.3,
        # 'get_biomass_as_concentrations': lambda doubling_time: {},
        # 'ppgpp_id': 'ppgpp',
        # 'get_ppGpp_conc': lambda media: 0.0,
        # 'exchange_data_from_media': lambda media: [],
        # 'get_mass': lambda exchanges: [],
        # 'doubling_time': 44.0 * units.min,
        # 'amino_acid_ids': [],
        # 'seed': 0,
    }

    def __init__(self, parameters):
        super().__init__(parameters)


        # TODO -- extract the FBA configuration from the metabolism dataclass object
        # metabolism = self.parameters['metabolism']
        stoichiometry = self.parameters['stoichiometry']
        reaction_catalysts = self.parameters['reaction_catalysts']
        catalyst_ids = self.parameters['catalyst_ids']
        media_id = self.parameters['media_id']
        objective_type = self.parameters['objective_type']
        nutrients = parameters['nutrients'] # "minimal"? same as media_id, two different entries in sim_data


        import ipdb;
        #ipdb.set_trace()

        # TODO -- get list of exchanges, should be working
        exchange_molecules = set()
        exchanges = parameters['exchange_data_from_media'](media_id)
        exchange_molecules.update(exchanges['externalExchangeMolecules'])
        exchange_molecules = list(sorted(exchange_molecules))

        # TODO -- get object dict, do something about dataclass here
        conc_dict = metabolism.concentration_updates.concentrations_based_on_nutrients(nutrients)
        doubling_time = parameters['doubling_time']
        conc_dict.update(self.getBiomassAsConcentrations(doubling_time))

        self.homeostatic_objective = dict((key, conc_dict[key].asNumber(CONC_UNITS)) for key in conc_dict)

        ## Include all concentrations that will be present in a sim for constant length listeners
        for met in self.metaboliteNamesFromNutrients:
            if met not in self.homeostatic_objective:
                self.homeostatic_objective[met] = 0.

        # Create model to use to solve metabolism updates
        self.model = GradientDescentFba(
            reactions=stoichiometry,
            exchanges=exchange_molecules,
            objective= self.homeostatic_objective,
            objectiveType=objective_type # missing objectiveParameters for kinetic models
        )


    def ports_schema(self):

        # TODO -- replace all the names with something compatible to current parameters
        return {
            'metabolites': bulk_schema(self.model.metaboliteNamesFromNutrients),
            'catalysts': bulk_schema(self.model.catalyst_ids),
            'kinetics_enzymes': bulk_schema(self.model.kinetic_constraint_enzymes),
            'kinetics_substrates': bulk_schema(self.model.kinetic_constraint_substrates),
            'amino_acids': bulk_schema(self.aa_names),

            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
                'exchange': bulk_schema(self.model.fba.getExternalMoleculeIDs()),
                'exchange_data': {
                    'unconstrained': {'_default': []},
                    'constrained': {'_default': []}}},

            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0},
                    'dry_mass': {'_default': 0.0}},

                'fba_results': {
                    'media_id': {'_default': '', '_updater': 'set'},
                    'conc_updates': {'_default': [], '_updater': 'set'},
                    'catalyst_counts': {'_default': 0, '_updater': 'set'},
                    'translation_gtp': {'_default': 0, '_updater': 'set'},
                    'coefficient': {'_default': 0.0, '_updater': 'set'},
                    'unconstrained_molecules': {'_default': [], '_updater': 'set'},
                    'constrained_molecules': {'_default': [], '_updater': 'set'},
                    'uptake_constraints': {'_default': [], '_updater': 'set'},
                    'deltaMetabolites': {'_default': [], '_updater': 'set'},
                    'reactionFluxes': {'_default': [], '_updater': 'set'},
                    'externalExchangeFluxes': {'_default': [], '_updater': 'set'},
                    'objectiveValue': {'_default': [], '_updater': 'set'},
                    'shadowPrices': {'_default': [], '_updater': 'set'},
                    'reducedCosts': {'_default': [], '_updater': 'set'},
                    'targetConcentrations': {'_default': [], '_updater': 'set'},
                    'homeostaticObjectiveValues': {'_default': [], '_updater': 'set'},
                    'kineticObjectiveValues': {'_default': [], '_updater': 'set'}},

                'enzyme_kinetics': {
                    'metaboliteCountsInit': {'_default': 0, '_updater': 'set'},
                    'metaboliteCountsFinal': {'_default': 0, '_updater': 'set'},
                    'enzymeCountsInit': {'_default': 0, '_updater': 'set'},
                    'countsToMolar': {'_default': 1.0, '_updater': 'set'},
                    'actualFluxes': {'_default': [], '_updater': 'set'},
                    'targetFluxes': {'_default': [], '_updater': 'set'},
                    'targetFluxesUpper': {'_default': [], '_updater': 'set'},
                    'targetFluxesLower': {'_default': [], '_updater': 'set'}}},

            'polypeptide_elongation': {
                'aa_count_diff': {
                    '_default': {},
                    '_emit': True},
                'gtp_to_hydrolyze': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(self, timestep, states):

        # extract the states from the ports
        metabolites = states['metabolites']
        catalysts = states['catalysts']





        # TODO -- get the states, use them to set the FBA problem.
        # Need to run set_molecule_levels and set_reaction_bounds for homeostatic solution.
        # set molecule_levels requires exchange_constraints from dataclass.
        kinetic_constraints = self.parameters['metabolism'].get_kinetic_constraints(catalysts, metabolites)

        # make the new objective
        objective = {}

        # run FBA
        solution: FbaResult = self.model.solve(
            params=kinetic_constraints,
            objective=objective,

        )

        # TODO -- extract FBA solution, and pass the update
        # solution -- changes in internal and exchange metabolites

        return {
            'metabolites': {},  # changes to internal metabolites
            'environment': {
                'exchanges': {}  # changes to external metabolites
            }
        }


def test_metabolism():
    test_config = {
        'stoichiometry': np.array([])
    }
    process = MetabolismGD(test_config)

    # TODO -- run the model
    initial_state = {
        'metabolites': {'A': 10, 'B': 100}
    }
    settings = {
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_process(process, settings)

    print(data)

if __name__ == '__main__':
    test_metabolism()
