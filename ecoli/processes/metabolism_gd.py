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
        'concentration_updates': None,
        'maintenance_reaction': {},

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

        # variables
        maintenance_reaction = self.parameters['maintenance_reaction']
        self.stoichiometry = self.parameters['stoichiometry_r']
        self.stoichiometry.append({'reaction id': 'maintenance_reaction',
                                   'stoichiometry': parameters['maintenance_reaction'],
                                   'is reversible': False})
        reaction_catalysts = self.parameters['reaction_catalysts']
        self.media_id = self.parameters['media_id']
        objective_type = self.parameters['objective_type']
        self.cell_density = self.parameters['cell_density']
        self.nAvogadro = self.parameters['avogadro']
        self.nutrient_to_doubling_time = self.parameters['nutrientToDoublingTime']
        self.ngam = parameters['non_growth_associated_maintenance']
        self.gam = parameters['dark_atp'] * parameters['cell_dry_mass_fraction']

        # new variables for the model
        self.cell_mass = None
        self.previous_mass = None
        self.reaction_fluxes = None

        # methods
        self._biomass_concentrations = {}  # type: dict
        self._getBiomassAsConcentrations = parameters['get_biomass_as_concentrations']
        concentration_updates = self.parameters['concentration_updates']
        self.exchange_constraints = self.parameters['exchange_constraints']

        # retrieve exchanged molecules
        exchange_molecules = set()
        exchanges = parameters['exchange_data_from_media'](self.media_id)
        exchange_molecules.update(exchanges['externalExchangeMolecules'])
        self.exchange_molecules = exchange_molecules
        exchange_molecules = list(sorted(exchange_molecules))  # set vs list, unify?

        # retrieve conc dict and get homeostatic objective
        conc_dict = concentration_updates.concentrations_based_on_nutrients(self.media_id)
        doubling_time = parameters['doubling_time']
        conc_dict.update(self.getBiomassAsConcentrations(doubling_time))

        self.homeostatic_objective = dict((key, conc_dict[key].asNumber(CONC_UNITS)) for key in conc_dict)



        # Create model to use to solve metabolism updates
        self.model = GradientDescentFba(
            reactions=self.stoichiometry,
            exchanges=exchange_molecules,
            objective=self.homeostatic_objective,
            objectiveType=objective_type  # missing objectiveParameters for kinetic models
        )

        # for ports schema
        self.metabolite_names_for_nutrients = self.get_port_metabolite_names(conc_dict)
        # Include all concentrations that will be present in a sim for constant length listeners
        for met in self.metabolite_names_for_nutrients:
            if met not in self.homeostatic_objective:
                self.homeostatic_objective[met] = 0.

        # for ports schema
        self.aa_names = self.parameters['aa_names']
        self.catalyst_ids = self.parameters['catalyst_ids']
        self.kinetic_constraint_enzymes = self.parameters['kinetic_constraint_enzymes']
        self.kinetic_constraint_substrates = self.parameters['kinetic_constraint_substrates']

    def get_port_metabolite_names(self, conc_dict):
        metabolite_names_from_nutrients = set()
        metabolite_names_from_nutrients.update(conc_dict)
        return list(sorted(metabolite_names_from_nutrients))

    def ports_schema(self):

        return {
            'metabolites': bulk_schema(self.metabolite_names_for_nutrients),
            'catalysts': bulk_schema(self.catalyst_ids),
            'kinetics_enzymes': bulk_schema(self.kinetic_constraint_enzymes),
            'kinetics_substrates': bulk_schema(self.kinetic_constraint_substrates),
            'amino_acids': bulk_schema(self.aa_names),

            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
                'exchange': bulk_schema(self.exchange_molecules),
                # can probably remove, identical to exchanges except tuple.
                'exchange_data': {
                    'unconstrained': {'_default': []},
                    'constrained': {'_default': []}}},

            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0},
                    'dry_mass': {'_default': 0.0}},

                # comment out. keep fluxes for seed.
                'fba_results': {
                    #     'media_id': {'_default': '', '_updater': 'set'},
                    #     'conc_updates': {'_default': [], '_updater': 'set'},
                    #     'catalyst_counts': {'_default': 0, '_updater': 'set'},
                    #     'translation_gtp': {'_default': 0, '_updater': 'set'},
                    #     'coefficient': {'_default': 0.0, '_updater': 'set'},
                    #     'unconstrained_molecules': {'_default': [], '_updater': 'set'},
                    #     'constrained_molecules': {'_default': [], '_updater': 'set'},
                    #     'uptake_constraints': {'_default': [], '_updater': 'set'},
                    #     'deltaMetabolites': {'_default': [], '_updater': 'set'},
                    'reactionFluxes': {'_default': [], '_updater': 'set'},
                    #     'externalExchangeFluxes': {'_default': [], '_updater': 'set'},
                    #     'objectiveValue': {'_default': [], '_updater': 'set'},
                    #     'shadowPrices': {'_default': [], '_updater': 'set'},
                    #     'reducedCosts': {'_default': [], '_updater': 'set'},
                    #     'targetConcentrations': {'_default': [], '_updater': 'set'},
                    #     'homeostaticObjectiveValues': {'_default': [], '_updater': 'set'},
                    #     'kineticObjectiveValues': {'_default': [], '_updater': 'set'}
                },
                #
                # 'enzyme_kinetics': {
                #     'metaboliteCountsInit': {'_default': 0, '_updater': 'set'},
                #     'metaboliteCountsFinal': {'_default': 0, '_updater': 'set'},
                #     'enzymeCountsInit': {'_default': 0, '_updater': 'set'},
                #     'countsToMolar': {'_default': 1.0, '_updater': 'set'},
                #     'actualFluxes': {'_default': [], '_updater': 'set'},
                #     'targetFluxes': {'_default': [], '_updater': 'set'},
                #     'targetFluxesUpper': {'_default': [], '_updater': 'set'},
                #     'targetFluxesLower': {'_default': [], '_updater': 'set'}}
            },

            'polypeptide_elongation': {
                'aa_count_diff': {
                    '_default': {},
                    '_emit': True},
                'gtp_to_hydrolyze': {
                    '_default': 0,
                    '_emit': True}
            }
        }

    def next_update(self, timestep, states):

        # extract the states from the ports
        metabolite_counts = states['metabolites']
        catalyst_counts = states['catalysts']
        translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']

        # kinetic_enzyme_counts = states['kinetics_enzymes'] # kinetics related
        # kinetic_substrate_counts = states['kinetics_substrates']
        # get_kinetic_constraints = self.parameters['get_kinetic_constraints']

        # might delete later?
        translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']

        if self.cell_mass is not None:
            self.previous_mass = self.cell_mass
        self.cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
        dry_mass = states['listeners']['mass']['dry_mass'] * units.fg
        current_media_id = states['environment']['media_id']
        unconstrained = states['environment']['exchange_data']['unconstrained']
        constrained = states['environment']['exchange_data']['constrained']

        cell_volume = self.cell_mass / self.cell_density
        counts_to_molar = (1 / (self.nAvogadro * cell_volume)).asUnit(CONC_UNITS)
        coefficient = dry_mass / self.cell_mass * self.cell_density * timestep * units.s

        doubling_time = self.nutrient_to_doubling_time.get(self.media_id,
                                                           self.nutrient_to_doubling_time[self.media_id], )
        conc_updates = self.getBiomassAsConcentrations(doubling_time)
        conc_updates = {met: conc.asNumber(CONC_UNITS)
                        for met, conc in conc_updates.items()}

        # are all of these needed?
        # self.set_molecule_levels(metabolite_counts, counts_to_molar,
        # coefficient, current_media_id, unconstrained, constrained, conc_updates)

        # objective update
        _, objective = self.exchange_constraints(
            self.exchange_molecules, coefficient, CONC_UNITS,
            current_media_id, unconstrained, constrained, conc_updates,
        )
        # TODO Get target flux for solver.
        current_metabolite_concentrations = {key: value*counts_to_molar for key, value in metabolite_counts.items()}
        target_delta_concentrations = {key: (objective[key]*CONC_UNITS - current_metabolite_concentrations[key])*timestep
                                       for key, value in objective.items()}


        # TODO Implement GAM.

        # calculate mass delta for GAM
        if self.previous_mass is not None:
            flux_gam = self.gam * (self.cell_mass - self.previous_mass)
        else:
            flux_gam = 0
        flux_ngam = (self.ngam * coefficient).asNumber(CONC_UNITS)
        flux_gtp = (counts_to_molar * translation_gtp).asNumber(CONC_UNITS)

        # TODO Figure out how to implement catalysis. Can come later.
        # reaction_bounds = np.inf * np.ones(len(self.reactions_with_catalyst))
        # no_rxn_mask = self.catalysis_matrix.dot(catalyst_counts) == 0
        # reaction_bounds[no_rxn_mask] = 0

        # Need to run set_molecule_levels and set_reaction_bounds for homeostatic solution.
        # set molecule_levels requires exchange_constraints from dataclass.

        # kinetic constraints
        # kinetic_constraints = get_kinetic_constraints(catalyst_counts, metabolite_counts) # kinetic

        import ipdb
        ipdb.set_trace()

        # run FBA
        solution: FbaResult = self.model.solve(
            objective=target_delta_concentrations,
            initial=self.reaction_fluxes,
            params=None,
            reaction_flux_bounds={'maintenance_reaction': [sum([flux_gam, flux_ngam, flux_gtp])-0.001,
                                                           sum([flux_gam, flux_ngam, flux_gtp])]}
        )

        self.reaction_fluxes = []

        # TODO (Niels) -- extract FBA solution, and pass the update
        # solution -- changes in internal and exchange metabolites

        return {
            'metabolites': {},  # changes to internal metabolites
            'environment': {
                'exchanges': {}  # changes to external metabolites
            }
        }

    def getBiomassAsConcentrations(self, doubling_time):
        """
        Caches the result of the sim_data function to improve performance since
        function requires computation but won't change for a given doubling_time.

        Args:
            doubling_time (float with time units): doubling time of the cell to
                get the metabolite concentrations for

        Returns:
            dict {str : float with concentration units}: dictionary with metabolite
                IDs as keys and concentrations as values

        """

        # TODO (Niels) Repeats code found in processes/metabolism.py Should think of a way to share.

        minutes = doubling_time.asNumber(units.min)  # hashable
        if minutes not in self._biomass_concentrations:
            self._biomass_concentrations[minutes] = self._getBiomassAsConcentrations(doubling_time)

        return self._biomass_concentrations[minutes]

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
