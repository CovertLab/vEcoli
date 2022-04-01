"""
MetabolismGD
"""

import numpy as np
import json

from vivarium.core.process import Process

from ecoli.library.schema import bulk_schema

from wholecell.utils import units

from ecoli.library.fba_gd import GradientDescentFba, FbaResult, TargetDmdtObjective, \
    TargetVelocityObjective, VelocityBoundsObjective
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
# TODO (Cyrus) - Re-add when kinetics are added.
# TOPOLOGY['kinetic_flux_targets'] = ('rates', 'fluxes')
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

        # config initialization
        maintenance_reaction = self.parameters['maintenance_reaction']
        self.stoichiometry = self.parameters['stoichiometry_r']
        self.stoichiometry.append({'reaction id': 'maintenance_reaction',
                                   'stoichiometry': parameters['maintenance_reaction'],
                                   'is reversible': False,
                                   'enzyme': []})

        self.metabolite_names = self.parameters['metabolite_names']
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

        # methods from config
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
        self.kinetic_objective = [reaction['reaction id'] for reaction in self.stoichiometry if reaction['enzyme']]
        self.maintenance_objective = ['maintenance_reaction']

        self.carbon_source_active_transport = ['TRANS-RXN-157-PTSH-PHOSPHORYLATED/GLC//ALPHA-GLC-6-P/PTSH-MONOMER.52.',
                                               'TRANS-RXN-157-PTSH-PHOSPHORYLATED/GLC//D-glucopyranose-6-phosphate'
                                               '/PTSH-MONOMER.66.',
                                               'TRANS-RXN-157-PTSH-PHOSPHORYLATED/GLC//GLC-6-P/PTSH-MONOMER.46.']

        self.carbon_source_facilitated_diffusion = ['RXN0-7077-GLC/PROTON//ALPHA-GLUCOSE/PROTON.33.',
                                                    'RXN0-7077-GLC/PROTON//Glucopyranose/PROTON.33.',
                                                    'RXN0-7077-GLC/PROTON//GLC/PROTON.23.',
                                                    'TRANS-RXN0-574-GLC//GLC.9.',
                                                    'TRANS-RXN0-574-GLC//Glucopyranose.19.']

        # Create model to use to solve metabolism updates
        self.model = GradientDescentFba(
            reactions=self.stoichiometry,
            exchanges=list(self.exchange_molecules),
            target_metabolites=self.homeostatic_objective)
        self.model.add_objective('homeostatic', TargetDmdtObjective(self.model.network, self.homeostatic_objective))
        # self.model.add_objective('kinetic', TargetVelocityObjective(self.model.network, self.kinetic_objective))
        self.model.add_objective('maintenance',
                                 TargetVelocityObjective(self.model.network, self.maintenance_objective, weight=20))
        # TODO (Cyrus): self.model.add_objective('boundary', ...)
        self.model.add_objective('diffusion',
                                 TargetVelocityObjective(self.model.network,
                                                         self.carbon_source_facilitated_diffusion,
                                                         weight=5))
        self.model.add_objective("active_transport",
                                 VelocityBoundsObjective(self.model.network,
                                                         {reaction_id: (0, 0.55) for reaction_id
                                                          in self.carbon_source_active_transport}, weight=10))

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
            # TODO (Cyrus) Add internal metabolites as bulk schema.
            'metabolites': bulk_schema(self.metabolite_names_for_nutrients),
            'catalysts': bulk_schema(self.catalyst_ids),
            'kinetics_enzymes': bulk_schema(self.kinetic_constraint_enzymes),
            'kinetics_substrates': bulk_schema(self.kinetic_constraint_substrates),
            'amino_acids': bulk_schema(self.aa_names),
            'amino_acids_total': bulk_schema(self.aa_names, partition=False),
            # 'kinetic_flux_targets': {reaction_id: {} for reaction_id in self.parameters['kinetic_rates']},

            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
                'exchange': bulk_schema(self.exchange_molecules),
                # can probably remove, identical to exchanges except tuple.
                'exchange_data': {
                    'unconstrained': {'_default': []},
                    'constrained': {'_default': []}}},

            'polypeptide_elongation': {
                'aa_count_diff': {
                    '_default': {},
                    '_emit': True,
                    '_divider': 'empty_dict'},
                'gtp_to_hydrolyze': {
                    '_default': 0,
                    '_emit': True,
                    '_divider': 'zero'}
            },

            'listeners': {
                'mass': {
                    # TODO (Matt -> Cyrus): These should not be using a divider. Mass listener should run before metabolism after division.
                    'cell_mass': {'_default': 0.0,
                                  '_divider': 'split'},
                    'dry_mass': {'_default': 0.0,
                                 '_divider': 'split'}},

                'fba_results': {
                    'estimated_fluxes': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_homeostatic_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'target_homeostatic_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'target_kinetic_fluxes': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_exchange_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_all_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'maintenance_target': {'_default': {}, '_updater': 'set', '_emit': True},
                    'solution_fluxes': {'_default': {}, '_updater': 'set', '_emit': True},
                    'solution_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'solution_residuals': {'_default': {}, '_updater': 'set', '_emit': True},
                },
            },

        }

    def next_update(self, timestep, states):

        # extract the states from the ports
        current_metabolite_counts = states['metabolites']
        self.timestep = timestep

        # TODO (Cyrus) - Implement kinetic model
        # kinetic_flux_targets = states['kinetic_flux_targets']
        # needed for kinetics
        catalyst_counts = states['catalysts']
        # translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']
        # kinetic_enzyme_counts = states['kinetics_enzymes'] # kinetics related
        # kinetic_substrate_counts = states['kinetics_substrates']
        # get_kinetic_constraints = self.parameters['get_kinetic_constraints']

        # cell mass difference for calculating GAM
        if self.cell_mass is not None:
            self.previous_mass = self.cell_mass
        self.cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
        dry_mass = states['listeners']['mass']['dry_mass'] * units.fg

        cell_volume = self.cell_mass / self.cell_density
        coefficient = dry_mass / self.cell_mass * self.cell_density * timestep * units.s  # TODO (Cyrus) what's this?
        self.counts_to_molar = (1 / (self.nAvogadro * cell_volume)).asUnit(CONC_UNITS)

        #
        translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']

        if self.previous_mass is not None:
            flux_gam = self.gam * (self.cell_mass - self.previous_mass) / VOLUME_UNITS
        else:
            flux_gam = 0 * CONC_UNITS
        flux_ngam = (self.ngam * coefficient)
        flux_gtp = (self.counts_to_molar * translation_gtp)

        total_maintenance = flux_gam + flux_ngam + flux_gtp

        kinetic_targets = {}
        # # TODO (Cyrus) - Figure out how to implement catalysis. Can come later.
        for reaction in self.stoichiometry:
            if reaction['enzyme'] and sum([catalyst_counts[enzyme] for enzyme in reaction['enzyme']]) == 0:
                kinetic_targets[reaction['reaction id']] = 0

        maintenance_target = {}
        maintenance_target['maintenance_reaction'] = total_maintenance.asNumber()

        current_metabolite_concentrations = {key: value * self.counts_to_molar for key, value in
                                             current_metabolite_counts.items()}
        target_homeostatic_dmdt = {key: ((self.homeostatic_objective[key] * CONC_UNITS
                                          - current_metabolite_concentrations[key]) / timestep).asNumber()
                                   for key, value in self.homeostatic_objective.items()}

        diffusion_target = {reaction: 0 for reaction in self.carbon_source_facilitated_diffusion}

        # reaction_bounds = np.inf * np.ones(len(self.reactions_with_catalyst))
        # no_rxn_mask = self.catalysis_matrix.dot(catalyst_counts) == 0
        # reaction_bounds[no_rxn_mask] = 0

        # Need to run set_molecule_levels and set_reaction_bounds for homeostatic solution.
        # set molecule_levels requires exchange_constraints from dataclass.

        # kinetic constraints
        # kinetic_constraints = get_kinetic_constraints(catalyst_counts, metabolite_counts) # kinetic

        # run FBA
        solution: FbaResult = self.model.solve(
            {'homeostatic': target_homeostatic_dmdt,
             'kinetic': kinetic_targets,
             'maintenance': maintenance_target,
             'diffusion': diffusion_target,
             },
            initial_velocities=self.reaction_fluxes,
            tr_solver='lsmr', max_nfev=64, ftol=10 ** (-5), verbose=2, xtol=10 ** (-4),
            tr_options={'atol': 10 ** (-8), 'btol': 10 ** (-8), 'conlim': 10 ** (10), 'show': False}
        )

        self.reaction_fluxes = solution.velocities
        self.metabolite_dmdt = solution.dm_dt

        # recalculate flux concentrations to counts
        estimated_reaction_fluxes = self.concentrationToCounts(self.reaction_fluxes)
        metabolite_dmdt_counts = self.concentrationToCounts(self.metabolite_dmdt)
        target_kinetic_dmdt = self.concentrationToCounts(kinetic_targets)
        target_maintenance_flux = self.concentrationToCounts(maintenance_target)
        target_homeostatic_dmdt = self.concentrationToCounts(target_homeostatic_dmdt)

        estimated_homeostatic_dmdt = {key: metabolite_dmdt_counts[key] for key in self.homeostatic_objective.keys()}
        estimated_exchange_dmdt = {key: metabolite_dmdt_counts[key] for key in self.exchange_molecules}

        return {
            'metabolites': estimated_homeostatic_dmdt,  # changes to internal metabolites
            'environment': {
                'exchanges': estimated_exchange_dmdt  # changes to external metabolites
            },
            'listeners': {
                'fba_results': {
                    'estimated_fluxes': estimated_reaction_fluxes,
                    'estimated_homeostatic_dmdt': estimated_homeostatic_dmdt,
                    'target_homeostatic_dmdt': target_homeostatic_dmdt,
                    'target_kinetic_fluxes': target_kinetic_dmdt,
                    'estimated_exchange_dmdt': estimated_exchange_dmdt,
                    'estimated_all_dmdt': metabolite_dmdt_counts,
                    'maintenance_target': target_maintenance_flux,
                    'solution_fluxes': solution.velocities,
                    'solution_dmdt': solution.dm_dt,
                    'solution_residuals': solution.residual,
                }
            }
        }

    def concentrationToCounts(self, concentration_dict):
        return {key: int(np.round(
            (concentration_dict[key] * CONC_UNITS / self.counts_to_molar * self.timestep).asNumber()
        )) for key in concentration_dict}

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

        # TODO (Cyrus) Repeats code found in processes/metabolism.py Should think of a way to share.

        minutes = doubling_time.asNumber(units.min)  # hashable
        if minutes not in self._biomass_concentrations:
            self._biomass_concentrations[minutes] = self._getBiomassAsConcentrations(doubling_time)

        return self._biomass_concentrations[minutes]

# TODO (Cyrus) - Consider adding test with toy network.
