"""
MetabolismGD
"""

import numpy as np
import json

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import bulk_schema, array_from

from wholecell.utils import units
from wholecell.utils.random import stochasticRound

from ecoli.library.fba_gd import GradientDescentFba, FbaResult, TargetDmdtObjective
from ecoli.processes.registries import topology_registry


COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
CONVERSION_UNITS = MASS_UNITS * TIME_UNITS / VOLUME_UNITS
GDCW_BASIS = units.mmol / units.g / units.h

USE_KINETICS = True

NAME = 'ecoli-metabolism-gradient-descent'
TOPOLOGY = topology_registry.access('ecoli-metabolism')
TOPOLOGY['kinetic_flux_targets'] = ('rates', 'fluxes')
topology_registry.register(NAME, TOPOLOGY)


class MetabolismGD(Process):
    name = NAME
    topology = TOPOLOGY

    defaults = {
        'stoichiometry': [],
        'reaction_catalysts': [],
        'catalyst_ids': [],
        'kinetic_rates': [],  # TODO (Cyrus) -- get these passed in, these are a subset of the stoichimetry
        'media_id': 'minimal',
        'objective_type': 'homeostatic',
        'nutrients': [],
        'cell_density': 1100 * units.g / units.L,
        'concentration_updates': None,
        'maintenance_reaction': {},
    }

    def __init__(self, parameters):
        super().__init__(parameters)

        # variables
        #   print(len(self.parameters['stoichiometry_r']), len(self.parameters['stoichiometry']))\

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
        self.random_state = np.random.RandomState(seed=self.parameters['seed'])

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

        # retrieve conc dict and get homeostatic objective.
        conc_dict = concentration_updates.concentrations_based_on_nutrients(self.media_id)
        doubling_time = parameters['doubling_time']
        conc_dict.update(self.getBiomassAsConcentrations(doubling_time))

        self.homeostatic_objective = dict((key, conc_dict[key].asNumber(CONC_UNITS)) for key in conc_dict)

        json.dump(self.parameters['stoichiometry_r'], open("notebooks/test_files/stoichiometry.json", 'w'))
        json.dump(list(self.exchange_molecules), open("notebooks/test_files/exchanges.json", 'w'))
        json.dump(self.homeostatic_objective, open("notebooks/test_files/homeostatic_objective.json", 'w'))
        json.dump(reaction_catalysts, open("notebooks/test_files/reaction_catalysts.json", 'w'))

        # Create model to use to solve metabolism updates
        self.model = GradientDescentFba(
            reactions=self.stoichiometry,
            exchanges=list(self.exchange_molecules),
            target_metabolites=self.homeostatic_objective)
        self.model.add_objective('homeostatic', TargetDmdtObjective(self.model.network, self.homeostatic_objective))
        # TODO(Niels): self.model.add_objective('kinetic', ...)

        self.objective = self.homeostatic_objective


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
            'amino_acids_total': bulk_schema(self.aa_names, partition=False),
            'kinetic_flux_targets': {reaction_id: {} for reaction_id in self.parameters['kinetic_rates']},

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

                'fba_results': {
                    'estimated_fluxes': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_homeostatic_dmdt' : {'_default': {}, '_updater': 'set', '_emit': True},
                    'target_homeostatic_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_exchange_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_all_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                },
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

        kinetic_flux_targets = states['kinetic_flux_targets']  # TODO -- this feeds into the FBA problem

        # needed for kinetics
        # catalyst_counts = states['catalysts']
        # translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']
        # kinetic_enzyme_counts = states['kinetics_enzymes'] # kinetics related
        # kinetic_substrate_counts = states['kinetics_substrates']
        # get_kinetic_constraints = self.parameters['get_kinetic_constraints']

        # might delete later?
        translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']

        if self.cell_mass is not None:
            self.previous_mass = self.cell_mass
        self.cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
        dry_mass = states['listeners']['mass']['dry_mass'] * units.fg
        # current_media_id = states['environment']['media_id']
        # unconstrained = states['environment']['exchange_data']['unconstrained']
        # constrained = states['environment']['exchange_data']['constrained']

        cell_volume = self.cell_mass / self.cell_density
        counts_to_molar = (1 / (self.nAvogadro * cell_volume)).asUnit(CONC_UNITS)
        coefficient = dry_mass / self.cell_mass * self.cell_density * timestep * units.s


        # are all of these needed? sunset later.
        # doubling_time = self.nutrient_to_doubling_time.get(self.media_id,
        #                                                    self.nutrient_to_doubling_time[self.media_id], )
        # conc_updates = self.getBiomassAsConcentrations(doubling_time)
        # conc_updates = {met: conc.asNumber(CONC_UNITS)
        #                 for met, conc in conc_updates.items()}

        # self.set_molecule_levels(metabolite_counts, counts_to_molar,
        # coefficient, current_media_id, unconstrained, constrained, conc_updates)

        # objective update - i don't think this is necessary currently?
        # _, objective = self.exchange_constraints(
        #     self.exchange_molecules, coefficient, CONC_UNITS,
        #     current_media_id, unconstrained, constrained, conc_updates,
        # )

        # TODO Get target flux for solver.
        current_metabolite_concentrations = {key: value*counts_to_molar for key, value in metabolite_counts.items()}
        target_homeostatic_fluxes = {key: ((self.objective[key]*CONC_UNITS
                                            - current_metabolite_concentrations[key])/timestep).asNumber()
                                     for key, value in self.objective.items()}

        # TODO Implement GAM/NGAM/GTP. Be aware of GAM. How to scale?
        # calculate mass delta for GAM
        if self.previous_mass is not None:
            flux_gam = self.gam * (self.cell_mass - self.previous_mass)/VOLUME_UNITS
        else:
            flux_gam = 0 * CONC_UNITS
        flux_ngam = (self.ngam * coefficient)
        flux_gtp = (counts_to_molar * translation_gtp)

        total_maintenance = flux_gam + flux_ngam + flux_gtp

        # TODO (Niels) increase maintenance target weight.
        kinetic_targets = {'maintenance_reaction': total_maintenance.asNumber}

        # TODO Figure out how to implement catalysis. Can come later.
        # reaction_bounds = np.inf * np.ones(len(self.reactions_with_catalyst))
        # no_rxn_mask = self.catalysis_matrix.dot(catalyst_counts) == 0
        # reaction_bounds[no_rxn_mask] = 0

        # Need to run set_molecule_levels and set_reaction_bounds for homeostatic solution.
        # set molecule_levels requires exchange_constraints from dataclass.

        # kinetic constraints
        # kinetic_constraints = get_kinetic_constraints(catalyst_counts, metabolite_counts) # kinetic

        json.dump(target_homeostatic_fluxes, open("notebooks/test_files/target_homeostatic_fluxes.json", 'w'))
        json.dump(self.reaction_fluxes, open("notebooks/test_files/initial_reaction_fluxes.json", 'w'))

        # run FBA
        solution: FbaResult = self.model.solve(
            {'homeostatic': target_homeostatic_fluxes, 'kinetic': kinetic_targets},
            initial_velocities=self.reaction_fluxes,
            tr_solver='lsmr', max_nfev=6, ftol=0.00001, verbose=2,
            tr_options={'atol': 10 ** (-7), 'btol': 10 ** (-7), 'conlim': 10 ** (8), 'show': False}
        )

        self.reaction_fluxes = solution.velocities
        self.metabolite_dmdt = solution.dm_dt

        # updates for homeostatic targets
        homeostasis_metabolite_updates = {key: int(np.round(
            (self.metabolite_dmdt[key]*CONC_UNITS/counts_to_molar*timestep).asNumber()
        )) for key in self.objective.keys()}

        # updates for exchanges
        exchange_metabolite_updates = {key: int(np.round(
            (self.metabolite_dmdt[key] * CONC_UNITS / counts_to_molar * timestep).asNumber()
        )) for key in self.exchange_molecules}

        # calculate
        objective_counts = {key: int((self.objective[key]/counts_to_molar).asNumber())
                            - metabolite_counts[key] for key in self.objective.keys()}

        return {
            'metabolites': homeostasis_metabolite_updates,  # changes to internal metabolites
            'environment': {
                'exchanges': exchange_metabolite_updates  # changes to external metabolites
            },
            'listeners': {
                'fba_results': {
                    'estimated_fluxes': self.reaction_fluxes,
                    'estimated_homeostatic_dmdt': homeostasis_metabolite_updates,
                    'target_homeostatic_dmdt': objective_counts,
                    'estimated_exchange_dmdt': exchange_metabolite_updates,
                    'estimated_all_dmdt': self.metabolite_dmdt
                }
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
        'stoichiometry': np.array([]),
        'kinetic_rates': [],  # pass in list of reaction names that will be targeted
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
