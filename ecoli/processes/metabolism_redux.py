"""
MetabolismRedux
"""

import numpy as np
import time
from scipy.sparse import csr_matrix

from vivarium.core.process import Process, Step

from ecoli.library.schema import bulk_schema, array_from

from wholecell.utils import units

from ecoli.processes.registries import topology_registry
import pandas as pd
import cvxpy as cp
from typing import Iterable, Mapping, Any
from dataclasses import dataclass

COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
TIME_UNITS = units.s
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS
CONVERSION_UNITS = MASS_UNITS * TIME_UNITS / VOLUME_UNITS
GDCW_BASIS = units.mmol / units.g / units.h

# TODO Do in ParCa. This is a hack to get the model to run.
BAD_RXNS = ["RXN-12440", "TRANS-RXN-121", "TRANS-RXN-300", "TRANS-RXN-8", "R15-RXN-MET/CPD-479//CPD-479/MET.25.",
            "DISULFOXRED-RXN[CCO-PERI-BAC]-MONOMER0-4152/MONOMER0-4438//MONOMER0-4438/MONOMER0-4152.71.", ]

NAME = 'ecoli-metabolism-redux'
TOPOLOGY = topology_registry.access('ecoli-metabolism')
# TODO (Cyrus) - Re-add when kinetics are added.
TOPOLOGY['first_update'] = ("deriver_skips", "metabolism-redux",)
topology_registry.register(NAME, TOPOLOGY)


class MetabolismRedux(Step):
    name = NAME
    topology = TOPOLOGY

    defaults = {
        'stoichiometry': [],
        'reaction_catalysts': [],
        'catalyst_ids': [],
        'kinetic_rates': [],  # TODO (Cyrus) -- get these passed in, these are a subset of the stoichimetry
        'media_id': 'minimal',
        'objective_type': 'homeostatic',
        'cell_density': 1100 * units.g / units.L,
        'concentration_updates': None,
        'maintenance_reaction': {},
    }

    def __init__(self, parameters):
        super().__init__(parameters)

        # config initialization
        stoichiometry_raw = self.parameters['stoichiometry']
        maintenance_reaction = self.parameters['maintenance_reaction']

        # TODO (Cyrus) - Remove this when we have a better way to handle these reactions.
        # generate carbon mistake in parca, efflux/influx proton gen, mass gen
        self.stoichiometry = []
        for i, rxn in enumerate(stoichiometry_raw):
            if rxn['reaction id'] not in BAD_RXNS:
                self.stoichiometry.append(rxn)

        self.stoichiometry.append({'reaction id': 'maintenance_reaction',
                                   'stoichiometry': maintenance_reaction, 'enzyme': []})

        self.media_id = self.parameters['media_id']
        self.cell_density = self.parameters['cell_density']
        self.nAvogadro = self.parameters['avogadro']
        self.ngam = parameters['non_growth_associated_maintenance']
        self.gam = parameters['dark_atp'] * parameters['cell_dry_mass_fraction']

        # new variables for the model
        self.cell_mass = None
        self.previous_mass = None
        self.reaction_fluxes = None

        # methods from config
        self._biomass_concentrations = {}  # type: dict
        self._getBiomassAsConcentrations = parameters['get_biomass_as_concentrations']
        concentration_updates = self.parameters['concentration_updates']
        self.exchange_constraints = self.parameters['exchange_constraints']
        self.get_kinetic_constraints = self.parameters['get_kinetic_constraints']
        self.kinetic_constraint_reactions = self.parameters['kinetic_constraint_reactions']

        # retrieve exchanged molecules
        exchange_molecules = set()
        exchanges = parameters['exchange_data_from_media'](self.media_id)
        exchange_molecules.update(exchanges['externalExchangeMolecules'])
        self.exchange_molecules = list(sorted(exchange_molecules))

        # retrieve conc dict and get homeostatic objective.
        conc_dict = concentration_updates.concentrations_based_on_nutrients(self.media_id)
        doubling_time = parameters['doubling_time']
        conc_dict.update(self.getBiomassAsConcentrations(doubling_time))

        self.homeostatic_objective = dict((key, conc_dict[key].asNumber(CONC_UNITS)) for key in conc_dict)
        self.maintenance_objective = ['maintenance_reaction']

        # Network flow initialization
        stoichiometric_matrix_dict = {item["reaction id"]: item["stoichiometry"] for item in self.stoichiometry}

        self.network_flow_model = NetworkFlowModel(reactions=stoichiometric_matrix_dict,
                                                   homeostatic_metabolites=self.homeostatic_objective,
                                                   kinetic_reactions=self.kinetic_constraint_reactions,
                                                   free_reactions='TRANS-RXN-145')

        # for ports schema # TODO (Cyrus) - Remove some of these. They are not used.
        self.metabolite_names_for_nutrients = self.get_port_metabolite_names(conc_dict)
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
                    'target_kinetic_bounds': {'_default': {}, '_updater': 'set', '_emit': False},
                    'estimated_exchange_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'estimated_intermediate_dmdt': {'_default': {}, '_updater': 'set', '_emit': True},
                    'maintenance_target': {'_default': {}, '_updater': 'set', '_emit': True},
                    'reaction_catalyst_counts': {'_default': {}, '_updater': 'set', '_emit': True},
                    'time_per_step': {'_default': 0.0, '_updater': 'set', '_emit': True},

                    'unconstrained_molecules': {'_default': [], '_updater': 'set', '_emit': True},
                    'constrained_molecules': {'_default': {}, '_updater': 'set', '_emit': True},
                },
            },
            'evolvers_ran': {'_default': True},
            'first_update': {
                '_default': True,
                '_updater': 'set',
                '_divider': {'divider': 'set_value', 'config': {'value': True}},
            }
        }

    def update_condition(self, timestep, states):
        return states['evolvers_ran']

    def next_update(self, timestep, states):
        # Skip t=0
        if states['first_update']:
            return {'first_update': False}

        # metabolites not in either set are constrained to zero uptake.
        unconstrained_uptake = states['environment']['exchange_data']['unconstrained']
        constrained_uptake = states['environment']['exchange_data']['constrained']
        self.allowed_exchange_uptake = list(unconstrained_uptake) + list(constrained_uptake.keys())
        self.disallowed_exchange_uptake = list(set(self.exchange_molecules) - set(self.allowed_exchange_uptake))
        self.exchange_molecules = list(set(self.exchange_molecules).union(set(self.allowed_exchange_uptake)))

        # extract the states from the ports
        current_metabolite_counts = states['metabolites']
        self.timestep = self.calculate_timestep(
            states)  # TODO (Cyrus) - this will break when quicker processes are added

        # TODO (Cyrus) - Implement kinetic model
        # kinetic_flux_targets = states['kinetic_flux_targets']
        # needed for kinetics
        current_catalyst_counts = states['catalysts']
        translation_gtp = states['polypeptide_elongation']['gtp_to_hydrolyze']
        kinetic_enzyme_counts = states['kinetics_enzymes']  # kinetics related
        kinetic_substrate_counts = states['kinetics_substrates']

        # cell mass difference for calculating GAM
        if self.cell_mass is not None:
            self.previous_mass = self.cell_mass
        self.cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
        dry_mass = states['listeners']['mass']['dry_mass'] * units.fg

        cell_volume = self.cell_mass / self.cell_density
        # Coefficient to convert between flux (mol/g DCW/hr) basis and concentration (M) basis
        conversion_coefficient = dry_mass / self.cell_mass * self.cell_density * timestep * units.s
        self.counts_to_molar = (1 / (self.nAvogadro * cell_volume)).asUnit(CONC_UNITS)

        # maintenance target
        if self.previous_mass is not None:
            flux_gam = self.gam * (self.cell_mass - self.previous_mass) / VOLUME_UNITS
        else:
            flux_gam = 0 * CONC_UNITS
        flux_ngam = (self.ngam * conversion_coefficient)
        flux_gtp = (self.counts_to_molar * translation_gtp)

        total_maintenance = flux_gam + flux_ngam + flux_gtp
        maintenance_target = {'maintenance_reaction': total_maintenance.asNumber()}

        # reaction catalyst counts. If no catalysts, set to one. Otherwise, set to current counts.
        reaction_catalyst_counts = {}
        for rxn in self.stoichiometry:
            if rxn['enzyme']:
                # sum of catalyst counts
                reaction_catalyst_counts[rxn['reaction id']] = sum(
                    [current_catalyst_counts[enzyme] for enzyme in rxn['enzyme']])
            else:
                reaction_catalyst_counts[rxn['reaction id']] = 1

        current_metabolite_concentrations = {str(key): value * self.counts_to_molar for key, value in
                                             current_metabolite_counts.items()}
        target_homeostatic_dmdt = {str(key): ((self.homeostatic_objective[key] * CONC_UNITS
                                               - current_metabolite_concentrations[key]) / self.timestep).asNumber()
                                   for key, value in self.homeostatic_objective.items()}

        # Need to run set_molecule_levels and set_reaction_bounds for homeostatic solution.
        # set molecule_levels requires exchange_constraints from dataclass.

        # kinetic constraints # TODO (Cyrus) eventually collect isozymes in single reactions, map enzymes to reacts
        #  via stoich instead of kinetic_constraint_reactions
        kinetic_enzyme_conc = self.counts_to_molar * array_from(kinetic_enzyme_counts)
        kinetic_substrate_conc = self.counts_to_molar * array_from(kinetic_substrate_counts)
        kinetic_constraints = self.get_kinetic_constraints(kinetic_enzyme_conc, kinetic_substrate_conc)  # kinetic
        enzyme_kinetic_boundaries = ((self.timestep * units.s) * kinetic_constraints).asNumber(CONC_UNITS).astype(float)
        enzyme_kinetic_reactions = self.kinetic_constraint_reactions

        # remove redundant kinetic reactions
        target_kinetic_bounds = {
            enzyme_kinetic_reactions[i]: (enzyme_kinetic_boundaries[i, 0], enzyme_kinetic_boundaries[i, 2])
            for i in range(len(enzyme_kinetic_reactions)) if enzyme_kinetic_reactions[i] is not None}

        target_kinetic_values = {enzyme_kinetic_reactions[i]: enzyme_kinetic_boundaries[i, 1]
                                 for i in range(len(enzyme_kinetic_reactions)) if
                                 enzyme_kinetic_reactions[i] is not None}

        # set up network flow model exchanges and uptakes
        if not self.network_flow_model.set_up:
            self.network_flow_model.set_up_exchanges(self.exchange_molecules, self.allowed_exchange_uptake)

        # TODO (Cyrus) solve network flow problem to get fluxes
        solution: FlowResult = self.network_flow_model.solve(homeostatic_targets=target_homeostatic_dmdt,
                                                             maintenance_target=maintenance_target[
                                                                 'maintenance_reaction'],
                                                             kinetic_targets=target_kinetic_values,
                                                             reaction_catalyst_counts=reaction_catalyst_counts,
                                                             objective_weights={'secretion': 0.01, 'efficiency': 0.0001,
                                                                                'kinetics': 0.000001},
                                                             upper_flux_bound=100000000)

        self.reaction_fluxes = solution.velocities
        self.metabolite_dmdt = solution.dm_dt
        self.metabolite_exchange = solution.exchanges

        # recalculate flux concentrations to counts # TODO Put these into update
        estimated_reaction_fluxes = self.concentrationToCounts(self.reaction_fluxes)
        metabolite_dmdt_counts = self.concentrationToCounts(self.metabolite_dmdt)
        target_kinetic_flux = self.concentrationToCounts(target_kinetic_values)
        target_maintenance_flux = self.concentrationToCounts(maintenance_target)
        target_homeostatic_dmdt = self.concentrationToCounts(target_homeostatic_dmdt)
        estimated_exchange_dmdt = self.concentrationToCounts(self.metabolite_exchange)
        target_kinetic_bounds = self.concentrationToCounts(target_kinetic_bounds)

        # Include all concentrations that will be present in a sim for constant length listeners. doesn't affect fba.
        for met in self.metabolite_names_for_nutrients:
            if met not in target_homeostatic_dmdt:
                target_homeostatic_dmdt[str(met)] = 0.

        estimated_homeostatic_dmdt = {str(key): metabolite_dmdt_counts[key] for key in
                                      self.homeostatic_objective.keys()}
        intermediates = list(
            set(metabolite_dmdt_counts.keys()) - set(self.exchange_molecules) - set(self.homeostatic_objective.keys()))
        estimated_intermediate_dmdt = {str(key): metabolite_dmdt_counts[key] for key in intermediates}

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
                    'target_kinetic_fluxes': target_kinetic_flux,
                    'target_kinetic_bounds': target_kinetic_bounds,
                    'estimated_exchange_dmdt': estimated_exchange_dmdt,
                    'estimated_intermediate_dmdt': estimated_intermediate_dmdt,
                    'maintenance_target': target_maintenance_flux,
                    'reaction_catalyst_counts': reaction_catalyst_counts,
                    'time_per_step': time.time(),

                    'unconstrained_molecules': unconstrained_uptake,
                    'constrained_molecules': constrained_uptake,
                }
            }
        }

    def concentrationToCounts(self, concentration_dict):
        return {key: np.rint(
            np.dot(concentration_dict[key], (CONC_UNITS / self.counts_to_molar * self.timestep).asNumber())
        ) for key in concentration_dict}

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


@dataclass
class FlowResult:
    """Reaction velocities and dm/dt for an FBA solution, with metrics."""
    velocities: Mapping[str, float]
    dm_dt: Mapping[str, float]
    exchanges: Mapping[str, float]
    objective: float


class NetworkFlowModel:
    # TODO Documentation
    """A network flow model for estimating fluxes in the metabolic network based on network structure. Flow is mainly
    driven by precursor demand (homeostatic objective) and availability of nutrients."""

    def __init__(self,
                 reactions: Iterable[dict],
                 homeostatic_metabolites: Iterable[str],
                 kinetic_reactions: Iterable[str],
                 free_reactions: Iterable[str]):

        # pandas automatically creates S matrix from dict of dicts, then we fill zeros in remainder
        self.Sd = pd.DataFrame.from_dict(reactions, dtype=np.int8).fillna(0).astype(np.int8)
        self.n_mets, self.n_orig_rxns = self.Sd.shape

        # extract names of mets and rxns
        self.mets = list(self.Sd.index)
        self.rxns = list(self.Sd.columns)
        self.intermediates = list(set(self.mets) - set(homeostatic_metabolites))

        # steady state indices, secretion indices
        self.intermediates_idx = [self.mets.index(met) for met in self.intermediates]
        self.homeostatic_idx = [self.mets.index(met) for met in homeostatic_metabolites]
        self.kinetic_idx = [self.rxns.index(rxn) for rxn in kinetic_reactions]
        self.penalty_idx = [self.rxns.index(rxn) for rxn in self.rxns if rxn not in free_reactions]

        self.set_up = False

    def set_up_exchanges(self,
                         exchanges: Iterable[str],
                         uptakes: Iterable[str]):
        """Set up exchange reactions for the network flow model. Exchanges allow certain metabolites to have flow out of
        the system. Uptakes allow certain metabolites to also have flow into the system."""
        all_exchanges = exchanges.copy()
        all_exchanges.extend(uptakes)

        self.Se = pd.DataFrame(index=self.Sd.index)
        for met in all_exchanges:
            exch_name = met + " exchange"
            if met in uptakes:
                self.Se[exch_name] = np.zeros(self.n_mets).astype(np.int8)
                self.Se.at[met, exch_name] = 1

            self.Se[exch_name + " rev"] = np.zeros(self.n_mets).astype(np.int8)
            self.Se.at[met, exch_name + " rev"] = -1

        self.S_orig = csr_matrix(np.array(self.Sd, dtype=np.float64))
        self.S_exch = csr_matrix(np.array(self.Se, dtype=np.float64))

        _, self.n_exch_rxns = self.S_exch.shape

        self.secretion_idx = np.where(self.S_exch.sum(axis=0) == -1)[0]
        self.maintenance_idx = self.rxns.index("maintenance_reaction")  # TODO (Cyrus) - use name provided

    def solve(self,
              homeostatic_targets: Mapping[str, float],
              kinetic_targets: Mapping[str, float],
              maintenance_target: float = 0,
              reaction_catalyst_counts: Mapping[str, int] = None,
              objective_weights: Mapping[str, float] = None,
              upper_flux_bound: float = 100
              ) -> FlowResult:
        """Solve the network flow model for fluxes and dm/dt values."""

        homeostatic_arr = [[self.mets.index(met), target] for met, target in homeostatic_targets.items()]
        homeostatic_idx, homeostatic_target = np.array(homeostatic_arr, dtype=np.int64)[:, 0], np.array(
            homeostatic_arr)[:, 1]

        kinetic_array = [[self.rxns.index(rxn), target] for rxn, target in kinetic_targets.items()]
        kinetic_idx, kinetic_target = np.array(kinetic_array, dtype=np.int64)[:, 0], np.array(kinetic_array)[:, 1]

        if reaction_catalyst_counts is not None:
            catalyst_array = [[self.rxns.index(rxn), target] for rxn, target in reaction_catalyst_counts.items()]
            catalyst_idx, catalyst_target = np.array(catalyst_array, dtype=np.int64)[:, 0], np.array(catalyst_array)[:,
                                                                                            1]
            z_idx, nz_idx = catalyst_idx[catalyst_target == 0], catalyst_idx[catalyst_target != 0]

        # set up variables
        v = cp.Variable(self.n_orig_rxns)
        e = cp.Variable(self.n_exch_rxns)
        dm = self.S_orig @ v + self.S_exch @ e
        exch = self.S_exch @ e

        constr = []
        constr.append(dm[self.intermediates_idx] == 0)
        constr.append(v[self.maintenance_idx] == maintenance_target)

        if reaction_catalyst_counts is not None:
            constr.append(v[z_idx] == 0)

        # constr.append(dm[homeostatic_idx[]] == homeostatic_target)
        constr.extend([v >= 0, v <= upper_flux_bound, e >= 0, e <= upper_flux_bound])

        loss = 0
        loss += cp.norm1(dm[homeostatic_idx] - homeostatic_target)
        loss += objective_weights['secretion'] * (cp.sum(e[self.secretion_idx]))
        loss += objective_weights['efficiency'] * (cp.sum(v[self.penalty_idx]))
        # loss += objective_weights['efficiency'] * cp.sum(cp.multiply(v, 1/np.log(catalyst_target+2)))   # TODO is nz_idx necessary?
        loss += objective_weights['kinetics'] * cp.norm1(v[kinetic_idx] - kinetic_target)

        p = cp.Problem(
            cp.Minimize(loss),
            constr
        )

        p.solve(solver=cp.GLOP, verbose=False)
        if p.status != "optimal":
            raise ValueError("Network flow model of metabolism did not converge to an optimal solution.")

        velocities, dm_dt, exchanges = np.array(v.value), np.array(dm.value), np.array(exch.value)
        velocities = {self.rxns[i]: velocities[i] for i in range(len(velocities))}
        dm_dt = {self.mets[i]: dm_dt[i] for i in range(len(dm_dt))}
        exchanges = {self.mets[i]: exchanges[i] for i in range(len(exchanges))}
        objective = p.value

        return FlowResult(velocities=velocities,
                          dm_dt=dm_dt,
                          exchanges=exchanges,
                          objective=objective)

# TODO (Cyrus) - Consider adding test with toy network.
