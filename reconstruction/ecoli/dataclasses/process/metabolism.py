"""
SimulationData for metabolism process

TODO:

- improved estimate of ILE/LEU abundance or some external data point
- implement L1-norm minimization for AA concentrations
- find concentration for PI[c]
- add (d)NTP byproduct concentrations
"""
# mypy: disable-error-code=attr-defined

from copy import copy
import itertools
import re
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from unum import Unum

from numba import njit
import numpy as np
import numpy.typing as npt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ecoli.library.schema import bulk_name_to_idx, counts
from reconstruction.ecoli.dataclasses.getter_functions import (
    UNDEFINED_COMPARTMENT_IDS_TO_ABBREVS,
)
from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from reconstruction.ecoli.dataclasses.constants import Constants
from wholecell.utils import units

KINETIC_CONSTRAINT_CONC_UNITS = units.umol / units.L
K_CAT_UNITS = 1 / units.s
"""Units for k :sub:`cat` values"""
METABOLITE_CONCENTRATION_UNITS = units.mol / units.L
"""Units for metabolite concentrations"""
DRY_MASS_UNITS = units.fg
"""Units for dry mass"""

USE_ALL_CONSTRAINTS = False  # False will remove defined constraints from objective

REVERSE_TAG = " (reverse)"
REVERSE_REACTION_ID = "{{}}{}".format(REVERSE_TAG)
ENZYME_REACTION_ID = "{}__{}"

VERBOSE = False


class InvalidReactionDirectionError(Exception):
    pass


class Metabolism(object):
    """Metabolism

    Args:
            raw_data: Raw data object
            sim_data: Simulation data object

    Attributes:
            solver (str): solver ID, should match a value in modular_fba.py,
                    set by :py:meth:`~._set_solver_values`
            kinetic_objective_weight (float): weighting for the kinetic objective,
                    1-weighting for the homeostatic objective, set by
                    :py:meth:`~._set_solver_values`
            kinetic_objective_weight_in_range (float): weighting for deviations
                    from the kinetic target within min and max ranges, set by
                    :py:meth:`~._set_solver_values`
            secretion_penalty_coeff (float): penalty on secretion fluxes, set by
                    :py:meth:`~._set_solver_values`
            metabolite_charge (dict[str, int]): mapping of metabolite IDs to charge,
                    set by :py:meth:`~._add_metabolite_charge`

            concentration_updates
            conc_dict (dict):
            nutrients_to_internal_conc (dict[str, dict[str, Unum]]):

            kinetic_constraint_reactions:
            kinetic_constraint_enzymes:
            kinetic_constraint_substrates:
            _kcats:
            _saturations:
            _enzymes:
            constraint_is_kcat_only:
            _compiled_enzymes:
            _compiled_saturation:
            reaction_stoich:
            maintenance_reaction:
            reaction_catalysts:
            catalyst_ids:
            reactions_with_catalyst:
            catalysis_matrix_I:
            catalysis_matrix_J:
            catalysis_matrix_V:
            use_all_constraints:
            constraints_to_disable:
            base_reaction_ids:
            reaction_id_to_base_reaction_id:
            amino_acid_export_kms:

            transport_reactions (list[str]): transport reaction IDs in the
                    metabolic network (includes reverse reactions and reactions
                    with kinetic constraints), set by
                    :py:meth:`~._build_transport_reactions`
            aa_synthesis_pathways (dict[str, dict]): data for allosteric
                    inhibition of amino acid pathways indexed by amino acid ID with
                    location tag and nested dictionary with the following keys::

                            {'enzymes' (str): limiting/regulated enzyme ID in synthesis
                                    pathway with location tag,
                            'kcat_data' (units.Unum): kcat associated with enzyme
                                    reaction with units of 1/time,
                            'ki' (Tuple[units.Unum, units.Unum]]): lower and upper
                                    limits of KI associated with enzyme reaction with units
                                    of mol/volume}

                    Set by :py:meth:`~._build_amino_acid_pathways`
            KI_aa_synthesis (numpy.ndarray[float]): KI for each AA for synthesis
                    portion of supply (in units of
                    :py:data:`~.METABOLITE_CONCENTRATION_UNITS`), set by
                    :py:meth:`~.set_phenomological_supply_constants`
            KM_aa_export (numpy.ndarray[float]): KM for each AA for export portion
                    of supply (in units of
                    :py:data:`~.METABOLITE_CONCENTRATION_UNITS`), set by
                    :py:meth:`~.set_phenomological_supply_constants`
            fraction_supply_rate (float): fraction of AA supply that comes from
                    a base synthesis rate, set by
                    :py:meth:`~.set_phenomological_supply_constants`
            fraction_import_rate (numpy.ndarray[float]): fraction of AA supply that
                    comes from AA import if nutrients are present, set by
                    :py:meth:`~.set_phenomological_supply_constants`
            ppgpp_synthesis_reaction (str): reaction ID for ppGpp synthesis
                    (catalyzed by RelA and SpoT), set by :py:meth:`~._build_ppgpp_reactions`
            ppgpp_degradation_reaction (str): reaction ID for ppGpp degradation
                    (catalyzed by SpoT), set by :py:meth:`~._build_ppgpp_reactions`
            ppgpp_reaction_names (list[str]): names of reaction involved in ppGpp,
                    set by :py:meth:`~._build_ppgpp_reactions`
            ppgpp_reaction_metabolites (list[str]): names of metabolites in
                    ppGpp reactions, set by :py:meth:`~._build_ppgpp_reactions`
            ppgpp_reaction_stoich (numpy.ndarray[int]): 2D array with metabolites
                    on rows and reactions on columns containing the stoichiometric
                    coefficient, set by :py:meth:`~._build_ppgpp_reactions`
            aa_to_exporters (dict[str, list]): dictonary that maps aa to
                    transporters involved in export reactions. Set by
                    :py:meth:`~.set_mechanistic_export_constants`.
            aa_to_exporters_matrix (numpy.ndarray[int]): correlation matrix.
                    Columns correspond to exporting enzymes and rows to amino acids.
                    Set by :py:meth:`~.set_mechanistic_export_constants`.
            aa_exporter_names (numpy.ndarray[str]): names of all exporters. Set by
                    :py:meth:`~.set_mechanistic_export_constants`.
            aa_export_kms (numpy.ndarray[float]): kms corresponding to generic
                    transport/enzyme reactions for each AA in concentration units.
                    Set by :py:meth:`~.set_mechanistic_export_constants`.
            export_kcats_per_aa (numpy.ndarray[float]): kcats corresponding to
                    generic export reactions for each AA. Units in counts/second.
                    Set by :py:meth:`~.set_mechanistic_export_constants` and
                    :py:meth:`~.set_mechanistic_export_constants`.
            aa_to_importers (dict[str, list]): dictonary that maps aa to
                    transporters involved in import reactions. Set by
                    :py:meth:`~.set_mechanistic_uptake_constants`.
            aa_to_importers_matrix (numpy.ndarray[int]): correlation matrix.
                    Columns correspond to importing enzymes and rows to amino acids.
                    Set by :py:meth:`~.set_mechanistic_export_constants`.
            aa_importer_names (numpy.ndarray[str]): names of all importers.
                    Set by :py:meth:`~.set_mechanistic_export_constants`.
            import_kcats_per_aa (numpy.ndarray[float]): kcats corresponding
                    to generic import reactions for each AA. Units in counts/second.
                    Set by :py:meth:`~.set_mechanistic_export_constants`.
            aa_enzymes (numpy.ndarray[str]): enzyme ID with location tag for each
                    enzyme that can catalyze an amino acid pathway with
                    :py:attr:`~.enzyme_to_amino_acid` mapping these to each amino acid.
                    Set by :py:meth:`~.set_mechanistic_supply_constants`.
            aa_kcats_fwd (numpy.ndarray[float]): forward kcat value for each
                    synthesis pathway in units of :py:data:`~.K_CAT_UNITS`, ordered by
                    amino acid molecule group. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
            aa_kcats_rev (numpy.ndarray[float]): reverse kcat value for each
                    synthesis pathway in units of :py:data:`~.K_CAT_UNITS`, ordered by
                    amino acid molecule group. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
            aa_kis (numpy.ndarray[float]): KI value for each synthesis pathway
                    in units of :py:data:`~.METABOLITE_CONCENTRATION_UNITS`, ordered
                    by amino acid molecule group. Will be inf if there is no inhibitory
                    control. Set by :py:meth:`~.set_mechanistic_supply_constants`.
            aa_upstream_kms (list[list[float]]): KM value associated with the
                    amino acid that feeds into each synthesis pathway in units of
                    :py:data:`~.METABOLITE_CONCENTRATION_UNITS`, ordered by amino
                    acid molecule group. Will be 0 if there is no upstream amino
                    acid considered. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
            aa_reverse_kms (numpy.ndarray[float]): KM value associated with the
                    amino acid in each synthesis pathway in units of
                    :py:data:`~.METABOLITE_CONCENTRATION_UNITS`, ordered by amino acid
                    molecule group. Will be inf if the synthesis pathway is not
                    reversible. Set by :py:meth:`~.set_mechanistic_supply_constants`.
            aa_upstream_mapping (numpy.ndarray[int]): index of the upstream amino
                    acid that feeds into each synthesis pathway, ordered by amino
                    acid molecule group. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
            enzyme_to_amino_acid (numpy.ndarray[float]): relationship mapping from
                    aa_enzymes to amino acids (n enzymes, m amino acids).  Will
                    contain a 1 if the enzyme associated with the row can catalyze
                    the pathway for the amino acid associated with the column.
                    Set by :py:meth:`~.set_mechanistic_supply_constants`.
            aa_forward_stoich (numpy.ndarray[float]): relationship mapping from
                    upstream amino acids to downstream amino acids (n upstream,
                    m downstream).  Will contain a -1 if the amino acid associated
                    with the row is required for synthesis of the amino acid
                    associated with the column. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
            aa_reverse_stoich (numpy.ndarray[float]): relationship mapping from
                    upstream amino acids to downstream amino acids (n downstream,
                    m upstream).  Will contain a -1 if the amino acid associated
                    with the row is produced through a reverse reaction from
                    the amino acid associated with the column. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
            aa_import_kis (numpy.ndarray[float]): inhibition constants for amino
                    acid import based on the internal amino acid concentration
            specific_import_rates (numpy.ndarray[float]): import rates expected
                    in rich media conditions for each amino acid normalized by dry
                    cell mass in units of :py:data:`~.K_CAT_UNITS` /
                    :py:data:`~.DRY_MASS_UNITS`, ordered by amino acid molecule group.
                    Set by :py:meth:`~.set_mechanistic_supply_constants`.
            max_specific_import_rates (numpy.ndarray[float]): max import rates
                    for each amino acid without including internal concentration
                    inhibition normalized by dry cell mass in units of
                    :py:data:`~.K_CAT_UNITS` / :py:data:`~.DRY_MASS_UNITS`, ordered by
                    amino acid molecule group. Set by
                    :py:meth:`~.set_mechanistic_supply_constants`.
    """

    def __init__(self, raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"):
        self._set_solver_values(sim_data.constants)
        self._build_biomass(raw_data, sim_data)
        self._build_metabolism(raw_data, sim_data)
        self._build_ppgpp_reactions(raw_data, sim_data)
        self._build_transport_reactions(raw_data, sim_data)
        self._build_amino_acid_pathways(raw_data, sim_data)
        self._add_metabolite_charge(raw_data)

    def _add_metabolite_charge(self, raw_data: KnowledgeBaseEcoli):
        """
        Compiles all metabolite charges.

        Args:
                raw_data: Raw data object

        Attributes set:
                - :py:attr:`~.metabolite_charge`
        """
        self.metabolite_charge = {}
        for met in raw_data.metabolites:
            self.metabolite_charge[met["id"]] = met["molecular_charge"]

    def _set_solver_values(self, constants: Constants):
        """
        Sets values to be used in the FBA solver.

        Attributes set:
                - :py:attr:`~.solver`
                - :py:attr:`~.kinetic_objective_weight`
                - :py:attr:`~.secretion_penalty_coeff`
        """

        self.solver = "glpk-linear"
        if "linear" in self.solver:
            self.kinetic_objective_weight = (
                constants.metabolism_kinetic_objective_weight_linear
            )
        else:
            self.kinetic_objective_weight = (
                constants.metabolism_kinetic_objective_weight_quadratic
            )
        self.kinetic_objective_weight_in_range = (
            constants.metabolism_kinetic_objective_weight_in_range
        )
        self.secretion_penalty_coeff = constants.secretion_penalty_coeff

    def _build_biomass(
        self, raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"
    ):
        """
        Calculates metabolite concentration targets.

        Args:
                raw_data: Raw data object
                sim_data: Simulation data object

        Attributes set:
                - :py:attr:`~.concentration_updates`
                - :py:attr:`~.conc_dict`
                - :py:attr:`~.nutrients_to_internal_conc`
        """
        wildtypeIDs = set(entry["molecule id"] for entry in raw_data.biomass)

        # Create vector of metabolite target concentrations

        # Since the data only covers certain metabolites, we need to rationally
        # expand the dataset to include the other molecules in the biomass
        # function.

        # First, load in metabolites that do have concentrations, then assign
        # compartments according to those given in the biomass objective.  Or,
        # if there is no compartment, assign it to the cytoplasm.

        concentration_sources = [
            "Park Concentration",
            "Lempp Concentration",
            "Kochanowski Concentration",
            "Sander Concentration",
        ]
        excluded = {
            "Park Concentration": {
                "GLT",  # Steady state concentration reached with tRNA charging is much lower than Park
                "THR",  # Attenuation needs concentration to be lower to match validation data
                "VAL",  # Synthesis pathway kcat needs concentration to be lower and closer to KI
            },
            "Lempp Concentration": {
                "ATP",  # TF binding does not solve with average concentration
                "VAL",  # Synthesis pathway kcat needs concentration to be lower and closer to KI
            },
            "Kochanowski Concentration": {
                "ATP",  # TF binding does not solve with average concentration
            },
            "Sander Concentration": {
                "GLT",  # Steady state concentration reached with tRNA charging is much lower than Sander
            },
        }
        metaboliteIDs = []
        metaboliteConcentrations = []

        wildtypeIDtoCompartment = {
            wildtypeID[:-3]: wildtypeID[-3:] for wildtypeID in wildtypeIDs
        }  # this assumes biomass reaction components only exist in a single compartment

        for row in raw_data.metabolite_concentrations:
            metabolite_id = row["Metabolite"]
            if not sim_data.getter.is_valid_molecule(metabolite_id):
                if VERBOSE:
                    print(
                        "Metabolite concentration for unknown molecule: {}".format(
                            metabolite_id
                        )
                    )
                continue

            # Use average of both sources
            # TODO (Travis): geometric mean?
            conc = np.nanmean(
                [
                    row[source].asNumber(METABOLITE_CONCENTRATION_UNITS)
                    for source in concentration_sources
                    if metabolite_id not in excluded.get(source, set())
                ]
            )

            # Check that a value was in the datasets being used
            if not np.isfinite(conc):
                if VERBOSE:
                    print(
                        "No concentration in active datasets for {}".format(
                            metabolite_id
                        )
                    )
                continue

            if metabolite_id in wildtypeIDtoCompartment:
                metaboliteIDs.append(
                    metabolite_id + wildtypeIDtoCompartment[metabolite_id]
                )
            else:
                metaboliteIDs.append(metabolite_id + "[c]")

            metaboliteConcentrations.append(conc)

        # CYS/SEL: concentration based on other amino acids
        aaConcentrations = []
        for aaIndex, aaID in enumerate(sim_data.amino_acid_code_to_id_ordered.values()):
            if aaID in metaboliteIDs:
                metIndex = metaboliteIDs.index(aaID)
                aaConcentrations.append(metaboliteConcentrations[metIndex])
        aaSmallestConc = min(aaConcentrations)

        metaboliteIDs.append("CYS[c]")
        metaboliteConcentrations.append(aaSmallestConc)

        metaboliteIDs.append("L-SELENOCYSTEINE[c]")
        metaboliteConcentrations.append(aaSmallestConc)

        # DGTP: set to smallest of all other DNTP concentrations
        dntpConcentrations = []
        for dntpIndex, dntpID in enumerate(sim_data.molecule_groups.dntps):
            if dntpID in metaboliteIDs:
                metIndex = metaboliteIDs.index(dntpID)
                dntpConcentrations.append(metaboliteConcentrations[metIndex])
        dntpSmallestConc = min(dntpConcentrations)

        metaboliteIDs.append("DGTP[c]")
        metaboliteConcentrations.append(dntpSmallestConc)

        # H: from reported pH
        hydrogenConcentration = 10 ** (-sim_data.constants.pH)

        metaboliteIDs.append(sim_data.molecule_ids.proton)
        metaboliteConcentrations.append(hydrogenConcentration)

        # PPI
        ppi_conc = sim_data.constants.ppi_concentration.asNumber(
            METABOLITE_CONCENTRATION_UNITS
        )
        metaboliteIDs.append(sim_data.molecule_ids.ppi)
        metaboliteConcentrations.append(ppi_conc)

        metaboliteIDs.append("Pi[c]")
        metaboliteConcentrations.append(ppi_conc)

        # include metabolites that are part of biomass
        d = sim_data.mass.getBiomassAsConcentrations(sim_data.doubling_time)
        for key, value in d.items():
            metaboliteIDs.append(key)
            metaboliteConcentrations.append(
                value.asNumber(METABOLITE_CONCENTRATION_UNITS)
            )

        # Load relative metabolite changes
        relative_changes: dict[str, dict[str, float]] = {}
        for row in raw_data.relative_metabolite_concentrations:
            met = row["Metabolite"]
            met_id = met + wildtypeIDtoCompartment.get(met, "[c]")

            # AA concentrations are determined through charging
            if met_id in sim_data.molecule_groups.amino_acids:
                continue

            # Get relative metabolite change in each media condition
            for col, value in row.items():
                # Skip the ID column and minimal column (only has values of 1)
                # or skip invalid values
                if col == "Metabolite" or col == "minimal" or not np.isfinite(value):
                    continue

                if col not in relative_changes:
                    relative_changes[col] = {}
                relative_changes[col][met_id] = value

        ## Add manually curated values for other media
        for (
            media,
            data,
        ) in sim_data.adjustments.relative_metabolite_concentrations_changes.items():
            if media not in relative_changes:
                relative_changes[media] = {}
            for met, change in data.items():
                if met not in relative_changes[media]:
                    relative_changes[media][met] = change

        # save concentrations as class variables
        unique_ids, counts = np.unique(metaboliteIDs, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(
                "Multiple concentrations for metabolite(s): {}".format(
                    ", ".join(unique_ids[counts > 1])
                )
            )

        # TODO (Travis): only pass raw_data and sim_data and create functions to load absolute and relative concentrations
        conc_dict = dict(
            zip(
                metaboliteIDs,
                METABOLITE_CONCENTRATION_UNITS * np.array(metaboliteConcentrations),
            )
        )
        all_metabolite_ids = {met["id"] for met in raw_data.metabolites}
        linked_metabolites = self._build_linked_metabolites(raw_data, conc_dict)
        self.concentration_updates = ConcentrationUpdates(
            conc_dict,
            relative_changes,
            raw_data.equilibrium_reactions,
            sim_data.external_state.exchange_dict,
            all_metabolite_ids,
            linked_metabolites,
        )
        self.conc_dict = self.concentration_updates.concentrations_based_on_nutrients(
            "minimal"
        )
        self.nutrients_to_internal_conc = {}
        self.nutrients_to_internal_conc["minimal"] = self.conc_dict.copy()

    def _build_linked_metabolites(
        self, raw_data: KnowledgeBaseEcoli, conc_dict: dict[str, Unum]
    ) -> dict[str, dict[str, Any]]:
        """
        Calculates ratio between linked metabolites to keep it constant
        throughout a simulation.

        Args:
                raw_data: Raw data object
                conc_dict: Mapping of metabolite IDs to homeostatic concentrations
                        calculated by :py:meth:`~_build_biomass`

        Returns:
                Mapping from a linked metabolite to its lead
                metabolite and concentration ratio to be maintained::

                                {'lead' (str): metabolite to link the concentration to,
                                'ratio' (float): ratio to multiply the lead concentration by}
        """

        linked_metabolites = {}
        for row in raw_data.linked_metabolites:
            lead = row["Lead metabolite"]
            linked = row["Linked metabolite"]
            ratio = units.strip_empty_units(conc_dict[linked] / conc_dict[lead])

            linked_metabolites[linked] = {"lead": lead, "ratio": ratio}

        return linked_metabolites

    def _build_metabolism(
        self, raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"
    ):
        """
        Build the matrices/vectors for metabolism (FBA)
        Reads in and stores reaction and kinetic constraint information

        Args:
                raw_data: Raw data object
                sim_data: Simulation data object

        Attributes set:
                - :py:attr:`~.kinetic_constraint_reactions`
                - :py:attr:`~.kinetic_constraint_enzymes`
                - :py:attr:`~.kinetic_constraint_substrates`
                - :py:attr:`~._kcats`
                - :py:attr:`~._saturations`
                - :py:attr:`~._enzymes`
                - :py:attr:`~.constraint_is_kcat_only`
                - :py:attr:`~._compiled_enzymes`
                - :py:attr:`~._compiled_saturation`
                - :py:attr:`~.reaction_stoich`
                - :py:attr:`~.maintenance_reaction`
                - :py:attr:`~.reaction_catalysts`
                - :py:attr:`~.catalyst_ids`
                - :py:attr:`~.reactions_with_catalyst`
                - :py:attr:`~.catalysis_matrix_I`
                - :py:attr:`~.catalysis_matrix_J`
                - :py:attr:`~.catalysis_matrix_V`
                - :py:attr:`~.use_all_constraints`
                - :py:attr:`~.constraints_to_disable`
                - :py:attr:`~.base_reaction_ids`
                - :py:attr:`~.reaction_id_to_base_reaction_id`
                - :py:attr:`~.amino_acid_export_kms`
        """
        (
            base_rxn_ids,
            reaction_stoich,
            reversible_reactions,
            catalysts,
            rxn_id_to_base_rxn_id,
        ) = self.extract_reactions(raw_data, sim_data)

        # Load kinetic reaction constraints from raw_data
        known_metabolites = set(self.conc_dict)
        raw_constraints = self.extract_kinetic_constraints(
            raw_data,
            sim_data,
            stoich=reaction_stoich,
            catalysts=catalysts,
            known_metabolites=known_metabolites,
        )

        # Make modifications from kinetics data
        (
            constraints,
            reaction_stoich,
            catalysts,
            reversible_reactions,
            rxn_id_to_base_rxn_id,
        ) = self._replace_enzyme_reactions(
            raw_constraints,
            reaction_stoich,
            catalysts,
            reversible_reactions,
            rxn_id_to_base_rxn_id,
        )

        # Create symbolic kinetic equations
        (
            self.kinetic_constraint_reactions,
            self.kinetic_constraint_enzymes,
            self.kinetic_constraint_substrates,
            self._kcats,
            self._saturations,
            self._enzymes,
            self.constraint_is_kcat_only,
        ) = self._lambdify_constraints(constraints)
        self._compiled_enzymes: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None
        self._compiled_saturation: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None

        # TODO: move this to a sim_data analysis script
        if VERBOSE:
            print("\nSummary of included metabolism kinetics:")
            print(
                "Reactions with kinetics: {}".format(
                    len(self.kinetic_constraint_reactions)
                )
            )
            print(
                "Enzymes with kinetics: {}".format(len(self.kinetic_constraint_enzymes))
            )
            print(
                "Metabolites in kinetics: {}".format(
                    len(self.kinetic_constraint_substrates)
                )
            )
            print(
                "Number of kcat values: {}".format(
                    len([k for c in constraints.values() for k in c["kcat"]])
                )
            )
            print(
                "Number of saturation terms: {}".format(
                    len([s for c in constraints.values() for s in c["saturation"]])
                )
            )

        # Verify no substrates with unknown concentrations have been added
        unknown = {
            m for m in self.kinetic_constraint_substrates if m not in known_metabolites
        }
        if unknown:
            raise ValueError(
                "Unknown concentration for {}. Need to remove"
                " kinetics saturation term.".format(", ".join(unknown))
            )

        # Extract data
        reactions_with_catalyst = sorted(catalysts)
        catalyst_ids = sorted({c for all_cat in catalysts.values() for c in all_cat})

        # Create catalysis matrix (to be used in the simulation)
        catalysisMatrixI = []
        catalysisMatrixJ = []
        catalysisMatrixV = []

        for row, reaction in enumerate(reactions_with_catalyst):
            for catalyst in catalysts[reaction]:
                col = catalyst_ids.index(catalyst)
                catalysisMatrixI.append(row)
                catalysisMatrixJ.append(col)
                catalysisMatrixV.append(1)

        catalysisMatrixI = np.array(catalysisMatrixI)
        catalysisMatrixJ = np.array(catalysisMatrixJ)
        catalysisMatrixV = np.array(catalysisMatrixV)

        # Properties for FBA reconstruction
        self.reaction_stoich = reaction_stoich
        # TODO (ggsun): add this as a raw .tsv file
        self.maintenance_reaction = {
            "ATP[c]": -1,
            "WATER[c]": -1,
            "ADP[c]": +1,
            "Pi[c]": +1,
            "PROTON[c]": +1,
        }

        # Properties for catalysis matrix (to set hard bounds)
        self.reaction_catalysts = catalysts
        self.catalyst_ids = catalyst_ids
        self.reactions_with_catalyst = reactions_with_catalyst
        self.catalysis_matrix_I = catalysisMatrixI
        self.catalysis_matrix_J = catalysisMatrixJ
        self.catalysis_matrix_V = catalysisMatrixV

        # Properties for setting flux targets
        self.use_all_constraints = USE_ALL_CONSTRAINTS
        self.constraints_to_disable = [
            rxn["disabled reaction"] for rxn in raw_data.disabled_kinetic_reactions
        ]

        # Properties for conversion of fluxes to those for base reaction IDs
        self.base_reaction_ids = base_rxn_ids
        self.reaction_id_to_base_reaction_id = rxn_id_to_base_rxn_id

        self.amino_acid_export_kms = raw_data.amino_acid_export_kms

    def _build_ppgpp_reactions(
        self, raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"
    ):
        """
        Creates structures for ppGpp reactions for use in polypeptide_elongation.

        Args:
                raw_data: Raw data object
                sim_data: Simulation data object

        Attributes set:
                - :py:attr:`~.ppgpp_synthesis_reaction`
                - :py:attr:`~.ppgpp_degradation_reaction`
                - :py:attr:`~.ppgpp_reaction_names`
                - :py:attr:`~.ppgpp_reaction_metabolites`
                - :py:attr:`~.ppgpp_reaction_stoich`
        """

        self.ppgpp_synthesis_reaction = "GDPPYPHOSKIN-RXN"
        self.ppgpp_degradation_reaction = "PPGPPSYN-RXN"

        self.ppgpp_reaction_names = [
            self.ppgpp_synthesis_reaction,
            self.ppgpp_degradation_reaction,
        ]

        self.ppgpp_reaction_metabolites = []

        # Indices (i: metabolite, j: reaction) and values (v: stoichiometry)
        # for sparse reaction matrix
        metabolite_indices: dict[str, int] = {}
        new_index = 0
        rxn_i = []
        rxn_j = []
        rxn_v = []

        # Record sparse indices in the matrix
        for j, rxn in enumerate(self.ppgpp_reaction_names):
            for met, stoich in self.reaction_stoich[rxn].items():
                idx = metabolite_indices.get(met, new_index)

                if idx == new_index:
                    metabolite_indices[met] = new_index
                    self.ppgpp_reaction_metabolites.append(met)
                    new_index += 1

                rxn_i.append(idx)
                rxn_j.append(j)
                rxn_v.append(stoich)

        # Assemble matrix based on indices
        # new_index is number of metabolites, j+1 is number of reactions
        self.ppgpp_reaction_stoich = np.zeros((new_index, j + 1), dtype=np.int32)
        self.ppgpp_reaction_stoich[rxn_i, rxn_j] = rxn_v

    def _build_transport_reactions(
        self, raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"
    ):
        """
        Creates list of transport reactions that are included in the
        reaction network.

        Args:
                raw_data: Raw data object
                sim_data: Simulation data object

        Attributes set:
                - :py:attr:`~.transport_reactions`
        """
        transport_reactions = [
            rxn_id
            for rxn_id, stoich in self.reaction_stoich.items()
            if self._is_transport_rxn(stoich)
        ]

        self.transport_reactions = transport_reactions

    def _build_amino_acid_pathways(
        self, raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"
    ):
        """
        Creates mapping between enzymes and amino acid pathways with
        allosteric inhibition feedback from the amino acid.

        Args:
                raw_data: Raw data object
                sim_data: Simulation data object

        Attributes set:
                - :py:attr:`~.aa_synthesis_pathways`
        """

        self.aa_synthesis_pathways = {}
        cytoplasm_tag = "[c]"

        for row in raw_data.amino_acid_pathways:
            data: dict[str, Any] = {}
            data["enzymes"] = [
                e + sim_data.getter.get_compartment_tag(e) for e in row["Enzymes"]
            ]
            data["reverse enzymes"] = [
                e + sim_data.getter.get_compartment_tag(e)
                for e in row["Reverse enzymes"]
            ]
            data["kcat_data"] = 0 / units.s if units.isnan(row["kcat"]) else row["kcat"]
            if units.isnan(row["KI, lower bound"]) or units.isnan(
                row["KI, lower bound"]
            ):
                data["ki"] = None
            else:
                data["ki"] = (row["KI, lower bound"], row["KI, upper bound"])
            data["upstream"] = {
                k + cytoplasm_tag: v for k, v in row["Upstream amino acids"].items()
            }
            data["reverse"] = {
                k + cytoplasm_tag: v for k, v in row["Reverse amino acids"].items()
            }
            data["km, upstream"] = {
                k + cytoplasm_tag: v for k, v in row["KM, upstream"].items()
            }
            data["km, reverse"] = row["KM, reverse"]
            data["km, degradation"] = (
                np.inf * units.mol / units.L
                if units.isnan(row["KM, degradation"])
                else row["KM, degradation"]
            )
            data["downstream"] = {
                k + cytoplasm_tag: v for k, v in row["Downstream amino acids"].items()
            }
            self.aa_synthesis_pathways[row["Amino acid"] + cytoplasm_tag] = data

        self.aa_synthesis_pathway_adjustments: dict[str, dict[str, float]] = {}
        for row in raw_data.adjustments.amino_acid_pathways:
            # Read data from row
            aa = row["Amino acid"] + cytoplasm_tag
            parameter = row["Parameter"]
            factor = row["Factor"]

            # Store adjustments to be used later
            adjustments = self.aa_synthesis_pathway_adjustments.get(aa, {})
            adjustments[parameter] = factor
            self.aa_synthesis_pathway_adjustments[aa] = adjustments

        self.amino_acid_uptake_rates = {}
        for row in raw_data.amino_acid_uptake_rates:
            rates = {}
            rates["uptake"] = row["Uptake"]
            rates["LB"] = row["Uptake, LB"]
            rates["UB"] = row["Uptake, UB"]
            self.amino_acid_uptake_rates[row["Amino acid"]] = rates

    def get_kinetic_constraints(self, enzymes: Unum, substrates: Unum) -> Unum:
        """
        Allows for dynamic code generation for kinetic constraint calculation
        for use in Metabolism process. Inputs should be unitless but the order
        of magnitude should match the kinetics parameters (umol/L/s).

        If trying to pickle sim_data object after function has been called,
        _compiled_enzymes and _compiled_saturation might not be able to be pickled.
        See __getstate__(), __setstate__() comments on PR 111 to address.

        Returns np.array of floats of the kinetic constraint target for each
        reaction with kinetic parameters

        Args:
                enzymes: concentrations of enzymes associated with kinetic
                        constraints (mol / volume units)
                substrates: concentrations of substrates associated with kinetic
                        constraints (mol / volume units)

        Returns:
                Array of dimensions (n reactions, 3) where each row contains the
                min, mean and max kinetic constraints for each reaction with kinetic
                constraints (mol / volume / time units)
        """

        if self._compiled_enzymes is None:
            self._compiled_enzymes = eval("lambda e: {}".format(self._enzymes))
        if self._compiled_saturation is None:
            self._compiled_saturation = eval("lambda s: {}".format(self._saturations))

        # Strip units from args
        enzs = enzymes.asNumber(KINETIC_CONSTRAINT_CONC_UNITS)
        subs = substrates.asNumber(KINETIC_CONSTRAINT_CONC_UNITS)

        capacity = np.array(self._compiled_enzymes(enzs))[:, None] * self._kcats
        saturation = np.array(
            [[min(v), sum(v) / len(v), max(v)] for v in self._compiled_saturation(subs)]
        )

        return KINETIC_CONSTRAINT_CONC_UNITS * K_CAT_UNITS * capacity * saturation

    def exchange_constraints(
        self,
        exchangeIDs,
        coefficient,
        targetUnits,
        media_id,
        unconstrained,
        constrained,
        concModificationsBasedOnCondition=None,
    ):
        """
        Called during Metabolism process
        Returns the homeostatic objective concentrations based on the current nutrients
        Returns levels for external molecules available to exchange based on the current nutrients
        """

        newObjective = self.concentration_updates.concentrations_based_on_nutrients(
            imports=unconstrained.union(constrained),
            media_id=media_id,
            conversion_units=targetUnits,
        )
        if concModificationsBasedOnCondition is not None:
            newObjective.update(concModificationsBasedOnCondition)

        externalMoleculeLevels = np.zeros(len(exchangeIDs), np.float64)

        for index, moleculeID in enumerate(exchangeIDs):
            if moleculeID in unconstrained:
                externalMoleculeLevels[index] = np.inf
            elif moleculeID in constrained:
                externalMoleculeLevels[index] = (
                    constrained[moleculeID] * coefficient
                ).asNumber(targetUnits)
            else:
                externalMoleculeLevels[index] = 0.0

        return externalMoleculeLevels, newObjective

    def set_phenomological_supply_constants(self, sim_data: "SimulationDataEcoli"):
        """
        Sets constants to determine amino acid supply during translation.  Used
        with aa_supply_scaling() during simulations but supply can
        alternatively be determined mechanistically.  This approach may require
        manually adjusting constants (fraction_supply_inhibited and
        fraction_supply_exported) but has less variability related to gene
        expression and regulation.

        Args:
                sim_data: simulation data

        Attributes set:
                - :py:attr:`~.KI_aa_synthesis`
                - :py:attr:`~.KM_aa_export`
                - :py:attr:`~.fraction_supply_rate`
                - :py:attr:`~.fraction_import_rate`

        Assumptions:
                - Each internal amino acid concentration in 'minimal_plus_amino_acids'
                  media is not lower than in 'minimal' media

        TODO (Travis):
                Better handling of concentration assumption
        """

        aa_ids = sim_data.molecule_groups.amino_acids
        conc = self.concentration_updates.concentrations_based_on_nutrients

        aa_conc_basal = np.array(
            [
                conc(media_id="minimal")[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                for aa in aa_ids
            ]
        )
        aa_conc_aa_media = np.array(
            [
                conc(media_id="minimal_plus_amino_acids")[aa].asNumber(
                    METABOLITE_CONCENTRATION_UNITS
                )
                for aa in aa_ids
            ]
        )

        # Lower concentrations might produce strange rates (excess supply or
        # negative import when present externally) and constants so raise
        # to double check the implementation
        if not np.all(aa_conc_basal <= aa_conc_aa_media):
            aas = np.array(aa_ids)[np.where(aa_conc_basal > aa_conc_aa_media)]
            raise ValueError(
                "Check that amino acid concentrations should be lower in amino acid media for {}".format(
                    aas
                )
            )

        f_inhibited = sim_data.constants.fraction_supply_inhibited
        f_exported = sim_data.constants.fraction_supply_exported

        # Assumed units of METABOLITE_CONCENTRATION_UNITS for KI and KM
        self.KI_aa_synthesis = f_inhibited * aa_conc_basal / (1 - f_inhibited)
        self.KM_aa_export = (1 / f_exported - 1) * aa_conc_aa_media
        self.fraction_supply_rate = (
            1 - f_inhibited + aa_conc_basal / (self.KM_aa_export + aa_conc_basal)
        )
        self.fraction_import_rate = 1 - (
            self.fraction_supply_rate
            + 1 / (1 + aa_conc_aa_media / self.KI_aa_synthesis)
            - f_exported
        )

    def aa_supply_scaling(
        self, aa_conc: Unum, aa_present: Unum
    ) -> npt.NDArray[np.float64]:
        """
        Called during polypeptide_elongation process
        Determine amino acid supply rate scaling based on current amino acid
        concentrations.

        Args:
                aa_conc: internal concentration for each amino acid (ndarray[float])
                aa_present: whether each amino acid is in the
                        external environment or not (ndarray[bool])

        Returns:
                Scaling for the supply of each amino acid with
                higher supply rate if >1, lower supply rate if <1
        """

        aa_conc = aa_conc.asNumber(METABOLITE_CONCENTRATION_UNITS)

        aa_supply = self.fraction_supply_rate
        aa_import = aa_present * self.fraction_import_rate
        aa_synthesis = 1 / (1 + aa_conc / self.KI_aa_synthesis)
        aa_export = aa_conc / (self.KM_aa_export + aa_conc)
        supply_scaling = aa_supply + aa_import + aa_synthesis - aa_export

        return supply_scaling

    def get_aa_to_transporters_mapping_data(
        self, sim_data: "SimulationDataEcoli", export: bool = False
    ) -> tuple[dict[str, list], npt.NDArray[np.float64], npt.NDArray[np.str_]]:
        """
        Creates a dictionary that maps amino acids with their transporters.
        Based on this dictionary, it creates a correlation matrix with rows
        as AA and columns as transporters.

        Args:
                sim_data: simulation data
                export: if True, the parameters calculated are for mechanistic
                        export instead of uptake

        Returns:
                3-element tuple containing

                        - aa_to_transporters: dictonary that maps aa to
                          transporters involved in transport reactions
                        - aa_to_transporters_matrix: correlation matrix.
                          Columns correspond to transporter enzymes and rows to
                          amino acids
                        - aa_transporters_names: names of all transporters
        """

        def matches_direction(direction):
            if export:
                return direction < 0
            else:
                return direction > 0

        # Mapping aminoacids to their transporters
        # CYS does not have any uptake reaction, so we initialize the dict with it to ensure
        # the presence of the 21 AAs
        # TODO (Santiago): Reversible reactions?
        aa_to_transporters: dict[str, list[str]] = {"CYS[c]": []}
        for reaction in self.transport_reactions:
            for aa in sim_data.molecule_groups.amino_acids:
                if aa in self.reaction_stoich[reaction] and matches_direction(
                    self.reaction_stoich[reaction][aa]
                ):
                    if aa not in aa_to_transporters:
                        aa_to_transporters[aa] = []
                    aa_to_transporters[aa] += self.reaction_catalysts.get(reaction, [])

        aa_to_transporters = {
            aa: aa_to_transporters[aa] for aa in sim_data.molecule_groups.amino_acids
        }

        c = 0
        transporters_to_idx = {}
        for aa, transporters in aa_to_transporters.items():
            for transporter in transporters:
                if transporter not in transporters_to_idx:
                    transporters_to_idx[transporter] = c
                    c += 1

        aa_to_transporters_matrix = [[0]] * len(aa_to_transporters)

        for i, trnspts in enumerate(aa_to_transporters.values()):
            temp = [0] * len(transporters_to_idx)
            for tr in trnspts:
                temp[transporters_to_idx[tr]] = 1
            aa_to_transporters_matrix[i] = temp

        aa_transporters_names = list(transporters_to_idx.keys())

        return (
            aa_to_transporters,
            np.array(aa_to_transporters_matrix),
            np.array(aa_transporters_names),
        )

    def set_mechanistic_export_constants(
        self,
        sim_data: "SimulationDataEcoli",
        cell_specs: dict[str, dict],
        basal_container: np.ndarray,
    ):
        """
        Calls get_aa_to_transporters_mapping_data() for AA export, which calculates
        the total amount of export transporter counts per AA. Kcats are calculated using
        the same exchange rates as for uptake and transporter counts. Missing KMs are calculated
        based on present KMs. This is done by calculating the average factor for
        KMs compared to estimated concentration (av_factor = sum(KM / concentration) / n_aa_with_kms).
        ** KM = av_factor * concentration


        Args:
                sim_data: simulation data
                cell_specs: mapping from condition to calculated cell properties
                basal_container: average initial bulk molecule counts in the basal
                        condition (structured Numpy array, see :ref:`bulk`)

        Attributes set:
                - :py:attr:`~.aa_to_exporters`
                - :py:attr:`~.aa_to_exporters_matrix`
                - :py:attr:`~.aa_exporter_names`
                - :py:attr:`~.aa_export_kms`
                - :py:attr:`~.export_kcats_per_aa`
        """

        self.aa_to_exporters, self.aa_to_exporters_matrix, self.aa_exporter_names = (
            self.get_aa_to_transporters_mapping_data(sim_data, export=True)
        )

        aa_names = sim_data.molecule_groups.amino_acids
        aa_idx = bulk_name_to_idx(aa_names, basal_container["id"])
        counts_to_molar = (
            sim_data.constants.cell_density / cell_specs["basal"]["avgCellDryMassInit"]
        ) / sim_data.constants.n_avogadro
        aa_conc = {
            aa: counts * counts_to_molar
            for aa, counts in zip(aa_names, counts(basal_container, aa_idx))
        }
        aa_with_km = {}

        # Calculate average factor to estimate missing KMs
        coeff_estimate_kms = 0.0
        for export_kms in self.amino_acid_export_kms:
            aa_with_km[export_kms["Amino Acid"]] = export_kms["KM"]
            coef_per_aa = 0
            for km in export_kms["KM"].values():
                coef_per_aa += (
                    km.asUnit(METABOLITE_CONCENTRATION_UNITS)
                    / aa_conc[export_kms["Amino Acid"]]
                )
            coeff_estimate_kms += coef_per_aa / len(export_kms["KM"])
        coeff_estimate_kms = coeff_estimate_kms / len(self.amino_acid_export_kms)

        # Calculate estimated KMs for each AA
        single_kms = {}
        for aa in aa_names:
            if aa in aa_with_km:
                single_kms[aa] = np.mean(list(aa_with_km[aa].values()))
            else:
                single_kms[aa] = coeff_estimate_kms * aa_conc[aa]

        self.aa_export_kms = np.array(
            [single_kms[aa].asNumber(METABOLITE_CONCENTRATION_UNITS) for aa in aa_names]
        )

    def set_mechanistic_uptake_constants(
        self,
        sim_data: "SimulationDataEcoli",
        cell_specs: dict[str, dict],
        with_aa_container: np.ndarray,
    ):
        """
        Based on the matrix calculated in get_aa_to_transporters_mapping_data(),
        we calculate the total amount of transporter counts per AA.

        Args:
                sim_data: simulation data
                cell_specs: mapping from condition to calculated cell properties
                with_aa_container: average initial bulk molecule counts in the
                        ``with_aa`` condition (structured Numpy array, see :ref:`bulk`)

        Attributes set:
                - :py:attr:`~.aa_to_importers`
                - :py:attr:`~.aa_to_importers_matrix`
                - :py:attr:`~.aa_importer_names`
                - :py:attr:`~.import_kcats_per_aa`
                - :py:attr:`~.export_kcats_per_aa`

        TODO:
                - Include external amino acid concentrations and KM values
        """

        aa_names = sim_data.molecule_groups.amino_acids
        aa_idx = bulk_name_to_idx(aa_names, with_aa_container["id"])
        counts_to_molar = (
            sim_data.constants.cell_density
            / cell_specs["with_aa"]["avgCellDryMassInit"]
        ) / sim_data.constants.n_avogadro
        aa_conc = counts(with_aa_container, aa_idx) * counts_to_molar.asNumber(
            METABOLITE_CONCENTRATION_UNITS
        )
        exchange_rates = self.specific_import_rates * cell_specs["with_aa"][
            "avgCellDryMassInit"
        ].asNumber(units.fg)

        self.aa_to_importers, self.aa_to_importers_matrix, self.aa_importer_names = (
            self.get_aa_to_transporters_mapping_data(sim_data)
        )

        aa_importer_idx = bulk_name_to_idx(
            self.aa_importer_names, with_aa_container["id"]
        )
        importer_counts = counts(with_aa_container, aa_importer_idx)
        aa_exporter_idx = bulk_name_to_idx(
            self.aa_exporter_names, with_aa_container["id"]
        )
        exporter_counts = counts(with_aa_container, aa_exporter_idx)
        counts_per_aa_import = self.aa_to_importers_matrix.dot(importer_counts)
        counts_per_aa_export = self.aa_to_exporters_matrix.dot(exporter_counts)

        # Solve for the two unknown kcats with the calculated net exchange rate
        # in rich media conditions and the assumption that import and export
        # rates are equal at the export KM based on how the export KM values
        # were curated.
        # Import will decrease and export will increase with higher amino acids
        # for stable amino acid concentrations.
        import_saturation_in_rich = 1 / (1 + aa_conc / self.aa_import_kis)
        export_saturation_in_rich = 1 / (1 + self.aa_export_kms / aa_conc)
        import_saturation_at_km = 1 / (1 + self.aa_export_kms / self.aa_import_kis)
        export_saturation_at_km = 0.5

        import_capacity_at_km = counts_per_aa_import * import_saturation_at_km
        export_capacity_at_km = counts_per_aa_export * export_saturation_at_km
        with np.errstate(divide="ignore", invalid="ignore"):
            import_vs_export_kcat = export_capacity_at_km / import_capacity_at_km
            kcat_export = exchange_rates / (
                import_vs_export_kcat * counts_per_aa_import * import_saturation_in_rich
                - counts_per_aa_export * export_saturation_in_rich
            )
            kcat_export[~np.isfinite(kcat_export)] = 0
            kcat_import = import_vs_export_kcat * kcat_export
            kcat_import[~np.isfinite(kcat_import)] = 0

        if np.any(kcat_export < 0) or np.any(kcat_import < 0):
            raise ValueError(
                "Could not solve for positive transport kcat."
                " Check assumptions or amino acid concentrations compared to KMs."
            )

        self.export_kcats_per_aa = kcat_export
        self.import_kcats_per_aa = kcat_import

    def set_mechanistic_supply_constants(
        self,
        sim_data: "SimulationDataEcoli",
        cell_specs: dict[str, dict],
        basal_container: np.ndarray,
        with_aa_container: np.ndarray,
    ):
        """
        Sets constants to determine amino acid supply during translation.  Used
        with amino_acid_synthesis() and amino_acid_import() during simulations
        but supply can alternatively be determined phenomologically.  This
        approach is more detailed and should better respond to environmental
        changes and perturbations but has more variability related to gene
        expression and regulation.

        Args:
                sim_data: simulation data
                cell_specs: mapping from condition to calculated cell properties
                basal_container: average initial bulk molecule counts in the basal
                        condition (structured Numpy array, see :ref:`bulk`)
                with_aa_container: average initial bulk molecule counts in the
                        ``with_aa`` condition

        Sets class attributes:
                - :py:attr:`~.aa_enzymes`
                - :py:attr:`~.aa_kcats_fwd`
                - :py:attr:`~.aa_kcats_rev`
                - :py:attr:`~.aa_kis`
                - :py:attr:`~.aa_upstream_kms`
                - :py:attr:`~.aa_reverse_kms`
                - :py:attr:`~.aa_upstream_mapping`
                - :py:attr:`~.enzyme_to_amino_acid`
                - :py:attr:`~.aa_forward_stoich`
                - :py:attr:`~.aa_reverse_stoich`
                - :py:attr:`~.aa_import_kis`
                - :py:attr:`~.specific_import_rates`
                - :py:attr:`~.max_specific_import_rates`

        Assumptions:

                - Only one reaction is limiting in an amino acid pathway (typically
                  the first and one with KI) and the kcat for forward or reverse
                  directions will apply to all enzymes that can catalyze that step
                - kcat for reverse and degradation reactions is the same (each amino
                  acid only has reverse or degradation at this point but that could
                  change with modifications to the amino_acid_pathways flat file)

        TODO:

                - Search for new kcat/KM values in literature or use metabolism_kinetics.tsv
                - Consider multiple reaction steps
                - Include mulitple amino acid inhibition on importers (currently
                  amino acids only inhibit their own import but some transporters
                  import multiple amino acids and will be inhibited by all of the
                  amino acids for the import of each amino acid)
        """

        aa_ids = sim_data.molecule_groups.amino_acids
        n_aas = len(aa_ids)
        self.aa_to_index = {aa: i for i, aa in enumerate(aa_ids)}
        conc = self.concentration_updates.concentrations_based_on_nutrients

        # Measured data used as targets for calculations
        measured_uptake_rates = {}
        fwd_kcat_targets = {}
        for aa in aa_ids:
            aa_no_tag = aa[:-3]
            measured_uptake_rates[aa_no_tag] = (
                self.amino_acid_uptake_rates[aa_no_tag]["uptake"].asNumber(
                    units.mmol / units.g / units.h
                )
                if aa_no_tag in self.amino_acid_uptake_rates
                else 0
            )
            if (
                kcat := self.aa_synthesis_pathways[aa]["kcat_data"].asNumber(
                    K_CAT_UNITS
                )
            ) > 0:
                fwd_kcat_targets[aa_no_tag] = kcat
        default_fwd_target = np.mean(list(fwd_kcat_targets.values()))

        # Allosteric inhibition constants to match required supply rate
        basal_rates = (
            sim_data.translation_supply_rate["minimal"]
            * cell_specs["basal"]["avgCellDryMassInit"]
            * sim_data.constants.n_avogadro
        ).asNumber(K_CAT_UNITS)
        with_aa_rates = (
            sim_data.translation_supply_rate["minimal_plus_amino_acids"]
            * cell_specs["with_aa"]["avgCellDryMassInit"]
            * sim_data.constants.n_avogadro
        ).asNumber(K_CAT_UNITS)
        basal_supply_mapping = dict(zip(aa_ids, basal_rates))
        with_aa_supply_mapping = dict(zip(aa_ids, with_aa_rates))
        aa_enzymes = []
        enzyme_to_aa_fwd = []
        enzyme_to_aa_rev = []
        aa_kcats_fwd = {}
        aa_kcats_rev = {}
        aa_kis = {}
        upstream_aas_for_km = {}
        aa_upstream_kms = {}
        aa_reverse_kms = {}
        aa_degradation_kms = {}
        fwd_rates: dict[str, float] = {}
        rev_rates: dict[str, float] = {}
        deg_rates: dict[str, float] = {}
        calculated_uptake_rates = {}
        minimal_conc = conc("minimal")
        with_aa_conc = conc("minimal_plus_amino_acids")

        # Get order of amino acids to calculate parameters for to ensure that
        # parameters that are dependent on other amino acids are run after
        # those calculations have completed
        self.aa_forward_stoich = np.eye(n_aas)
        self.aa_reverse_stoich = np.eye(n_aas)
        dependencies: dict[str, set[str]] = {}
        for aa in aa_ids:
            for downstream_aa in self.aa_synthesis_pathways[aa]["downstream"]:
                if units.isfinite(
                    self.aa_synthesis_pathways[downstream_aa]["km, degradation"]
                ):
                    dependencies.setdefault(aa, set()).add(downstream_aa)

            # Convert individual supply calculations to overall supply based on dependencies
            # via dot product (self.aa_forward_stoich @ supply)
            for upstream_aa, stoich in self.aa_synthesis_pathways[aa][
                "upstream"
            ].items():
                self.aa_forward_stoich[
                    self.aa_to_index[upstream_aa], self.aa_to_index[aa]
                ] = -stoich
                dependencies.setdefault(upstream_aa, set()).add(aa)

            for reverse_aa, stoich in self.aa_synthesis_pathways[aa]["reverse"].items():
                self.aa_reverse_stoich[
                    self.aa_to_index[reverse_aa], self.aa_to_index[aa]
                ] = -stoich
                dependencies.setdefault(reverse_aa, set()).add(aa)

        ordered_aa_ids: list[str] = []
        for _ in aa_ids:  # limit number of iterations number of amino acids in case there are cyclic links
            for aa in sorted(set(aa_ids) - set(ordered_aa_ids)):
                for downstream_aa in dependencies.get(aa, set()):
                    if downstream_aa not in ordered_aa_ids:
                        break
                else:
                    ordered_aa_ids.append(aa)
        if len(ordered_aa_ids) != n_aas:
            raise RuntimeError(
                "Could not determine amino acid order to calculate dependencies first."
                " Make sure there are no cyclical pathways for amino acids that can degrade."
            )

        for amino_acid in ordered_aa_ids:
            data = self.aa_synthesis_pathways[amino_acid]
            fwd_enzymes = data["enzymes"]
            fwd_enzymes_basal_idx = bulk_name_to_idx(fwd_enzymes, basal_container["id"])
            fwd_enzymes_basal = counts(basal_container, fwd_enzymes_basal_idx).sum()
            fwd_enzymes_with_aa_idx = bulk_name_to_idx(
                fwd_enzymes, with_aa_container["id"]
            )
            fwd_enzymes_with_aa = counts(
                with_aa_container, fwd_enzymes_with_aa_idx
            ).sum()
            rev_enzymes = data["reverse enzymes"]
            rev_enzymes_basal_idx = bulk_name_to_idx(rev_enzymes, basal_container["id"])
            rev_enzymes_basal = counts(basal_container, rev_enzymes_basal_idx).sum()
            rev_enzymes_with_aa_idx = bulk_name_to_idx(
                rev_enzymes, with_aa_container["id"]
            )
            rev_enzymes_with_aa = counts(
                with_aa_container, rev_enzymes_with_aa_idx
            ).sum()

            aa_conc_basal = minimal_conc[amino_acid]
            aa_conc_with_aa = with_aa_conc[amino_acid]
            if data["ki"] is None:
                ki = np.inf * units.mol / units.L
            else:
                # Get largest dynamic range possible given the range of measured KIs
                lower_limit, upper_limit = data["ki"]
                if aa_conc_basal < lower_limit:
                    ki = lower_limit
                elif aa_conc_basal > upper_limit:
                    ki = upper_limit
                else:
                    ki = aa_conc_basal
            upstream_aa = [aa for aa in data["upstream"]]
            km_conc_basal = METABOLITE_CONCENTRATION_UNITS * np.array(
                [
                    minimal_conc[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                    for aa in upstream_aa
                ]
            )
            km_conc_with_aa = METABOLITE_CONCENTRATION_UNITS * np.array(
                [
                    with_aa_conc[aa].asNumber(METABOLITE_CONCENTRATION_UNITS)
                    for aa in upstream_aa
                ]
            )
            kms_upstream = data["km, upstream"]
            kms = METABOLITE_CONCENTRATION_UNITS * np.array(
                [
                    kms_upstream.get(aa, minimal_conc[aa]).asNumber(
                        METABOLITE_CONCENTRATION_UNITS
                    )
                    for aa in upstream_aa
                ]
            )  # TODO: better way to fill in this missing data
            if data["reverse"]:
                if units.isnan(data["km, reverse"]):
                    km_reverse = (
                        minimal_conc[amino_acid] * 10
                    )  # TODO: better way to fill in this missing data
                else:
                    km_reverse = data["km, reverse"]
            # TODO: remove this if separating reverse and deg enzymes
            elif (
                data["reverse enzymes"]
                and not units.isfinite(data["km, degradation"])
                and amino_acid != "L-SELENOCYSTEINE[c]"
            ):
                km_reverse = (
                    minimal_conc[amino_acid] * 10
                )  # TODO: better way to fill in this missing data
            else:
                km_reverse = np.inf * units.mol / units.L
            km_degradation = data["km, degradation"]

            # Make required adjustments in order to get positive kcats and import rates
            for parameter, factor in self.aa_synthesis_pathway_adjustments.get(
                amino_acid, {}
            ).items():
                if parameter == "ki":
                    ki *= factor
                elif parameter == "km_degradation":
                    km_degradation *= factor
                elif parameter == "km_reverse":
                    km_reverse *= factor
                elif parameter == "kms":
                    kms *= factor
                else:
                    raise ValueError(
                        f"Unexpected parameter adjustment ({parameter}) for {amino_acid}."
                    )

            if units.isfinite(km_reverse) and units.isfinite(km_degradation):
                raise ValueError(
                    "Currently cannot have a reverse and degradation KM for amino acid"
                    f" synthesis pathways ({amino_acid}).  Consider a method to solve for separate"
                    " kcats and implement matching saturation calculation in amino_acid_synthesis()"
                    " and calc_kcats()"
                )

            def calc_kcats(
                aa_conc_basal,
                km_conc_basal,
                aa_conc_with_aa,
                km_conc_with_aa,
                kms,
                km_reverse,
                km_degradation,
                ki,
                uptake_rate,
            ):
                # Calculate kcat value to ensure sufficient supply to double
                fwd_fraction_basal = units.strip_empty_units(
                    1
                    / (1 + aa_conc_basal / ki)
                    * np.prod(1 / (1 + kms / km_conc_basal))
                )
                rev_fraction_basal = units.strip_empty_units(
                    1
                    / (
                        1
                        + km_reverse
                        / aa_conc_basal
                        * (1 + aa_conc_basal / km_degradation)
                    )
                )
                deg_fraction_basal = units.strip_empty_units(
                    1
                    / (
                        1
                        + km_degradation
                        / aa_conc_basal
                        * (1 + aa_conc_basal / km_reverse)
                    )
                )
                loss_fraction_basal = rev_fraction_basal + deg_fraction_basal
                fwd_capacity_basal = fwd_enzymes_basal * fwd_fraction_basal
                rev_capacity_basal = rev_enzymes_basal * loss_fraction_basal

                fwd_fraction_with_aa = units.strip_empty_units(
                    1
                    / (1 + aa_conc_with_aa / ki)
                    * np.prod(1 / (1 + kms / km_conc_with_aa))
                )
                rev_fraction_with_aa = units.strip_empty_units(
                    1
                    / (
                        1
                        + km_reverse
                        / aa_conc_with_aa
                        * (1 + aa_conc_with_aa / km_degradation)
                    )
                )
                deg_fraction_with_aa = units.strip_empty_units(
                    1
                    / (
                        1
                        + km_degradation
                        / aa_conc_with_aa
                        * (1 + aa_conc_with_aa / km_reverse)
                    )
                )
                loss_fraction_with_aa = rev_fraction_with_aa + deg_fraction_with_aa
                fwd_capacity_with_aa = fwd_enzymes_with_aa * fwd_fraction_with_aa
                rev_capacity_with_aa = rev_enzymes_with_aa * loss_fraction_with_aa

                supply_basal = basal_supply_mapping[amino_acid]
                supply_with_aa = with_aa_supply_mapping[amino_acid]
                downstream_basal = 0
                downstream_with_aa = 0
                for i, stoich in enumerate(
                    self.aa_forward_stoich[self.aa_to_index[amino_acid], :]
                ):
                    if stoich < 0:
                        downstream_aa = aa_ids[i]
                        downstream_basal += -stoich * fwd_rates[downstream_aa][0]
                        downstream_with_aa += -stoich * fwd_rates[downstream_aa][1]
                for i, stoich in enumerate(
                    self.aa_reverse_stoich[self.aa_to_index[amino_acid], :]
                ):
                    if stoich < 0:
                        downstream_aa = aa_ids[i]
                        downstream_basal += stoich * rev_rates[downstream_aa][0]
                        downstream_with_aa += stoich * rev_rates[downstream_aa][1]
                uptake = (
                    units.mmol
                    / units.g
                    / units.h
                    * uptake_rate
                    * cell_specs["with_aa"]["avgCellDryMassInit"]
                ).asNumber(units.count * K_CAT_UNITS)

                balance_basal = supply_basal + downstream_basal
                balance_with_aa = supply_with_aa + downstream_with_aa - uptake
                A = np.array(
                    [
                        [fwd_capacity_basal, -rev_capacity_basal],
                        [fwd_capacity_with_aa, -rev_capacity_with_aa],
                    ]
                )
                b = np.array([balance_basal, balance_with_aa])
                try:
                    kcat_fwd, kcat_rev = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    if VERBOSE:
                        print(
                            f"Warning: could not solve directly for {amino_acid} kcats - switching to least squares"
                        )
                    kcat_fwd, kcat_rev = np.linalg.lstsq(A, b, rcond=None)[0]

                fwd_rate = (
                    kcat_fwd * fwd_enzymes_basal * fwd_fraction_basal,
                    kcat_fwd * fwd_enzymes_with_aa * fwd_fraction_with_aa,
                )
                rev_rate = (
                    kcat_rev * rev_enzymes_basal * rev_fraction_basal,
                    kcat_rev * rev_enzymes_with_aa * rev_fraction_with_aa,
                )
                deg_rate = (
                    kcat_rev * rev_enzymes_basal * deg_fraction_basal,
                    kcat_rev * rev_enzymes_with_aa * deg_fraction_with_aa,
                )

                return kcat_fwd, kcat_rev, fwd_rate, rev_rate, deg_rate, uptake

            # Fit forward and reverse kcats by adjusting the uptake rate
            def objective(aa, uptake, kcat_fwd, kcat_rev):
                aa = aa[:-3]
                diffs = np.array(
                    [
                        measured_uptake_rates[aa] - uptake,
                        fwd_kcat_targets.get(aa, default_fwd_target) - kcat_fwd,
                        0
                        if units.isfinite(km_reverse)
                        else kcat_rev,  # no penalty if reverse, minimize if degradation
                    ]
                )
                weights = np.array([1000, 1, 1])
                return np.linalg.norm(weights * diffs)

            # TODO (travis): use a more rigorous method to fit the kcats (eg. gradient descent)
            # It would be better to include all amino acids in an objective and solve for all
            # kcats iteratively instead of solving for kcats for a single amino acid and then the next
            best_objective = None
            kcat_fwd = None
            n_factors = 500
            if VERBOSE:
                print("uptake:")
            for factor in np.logspace(-1, 1, n_factors):
                results = calc_kcats(
                    aa_conc_basal,
                    km_conc_basal,
                    aa_conc_with_aa,
                    km_conc_with_aa,
                    kms,
                    km_reverse,
                    km_degradation,
                    ki,
                    factor * measured_uptake_rates[amino_acid[:-3]],
                )

                new_kcat_fwd, new_kcat_rev, *_ = results
                if VERBOSE:
                    print(f"\t{factor:.2f}:\t{new_kcat_fwd:5.1f}\t{new_kcat_rev:5.1f}")
                if new_kcat_fwd >= 0 and new_kcat_rev >= 0:
                    new_objective = objective(
                        amino_acid,
                        factor * measured_uptake_rates[amino_acid[:-3]],
                        new_kcat_fwd,
                        new_kcat_rev,
                    )
                    if best_objective is None or new_objective < best_objective:
                        kcat_fwd, kcat_rev, fwd_rate, rev_rate, deg_rate, uptake = (
                            results
                        )
                        data["kcat"] = kcat_fwd * K_CAT_UNITS
                        best_objective = new_objective

            # Vary input parameters for kcat calculations for debugging purposes
            if VERBOSE:
                print("KMs:")
                for factor in np.logspace(-1, 1, n_factors):
                    new_kcat_fwd, new_kcat_rev, *_ = calc_kcats(
                        aa_conc_basal,
                        km_conc_basal,
                        aa_conc_with_aa,
                        km_conc_with_aa,
                        factor * kms,
                        km_reverse,
                        km_degradation,
                        ki,
                        measured_uptake_rates[amino_acid[:-3]],
                    )

                    print(f"\t{factor:.2f}:\t{new_kcat_fwd:5.1f}\t{new_kcat_rev:5.1f}")

                print("km_reverse:")
                for factor in np.logspace(-1, 1, n_factors):
                    new_kcat_fwd, new_kcat_rev, *_ = calc_kcats(
                        aa_conc_basal,
                        km_conc_basal,
                        aa_conc_with_aa,
                        km_conc_with_aa,
                        kms,
                        factor * km_reverse,
                        km_degradation,
                        ki,
                        measured_uptake_rates[amino_acid[:-3]],
                    )

                    print(f"\t{factor:.2f}:\t{new_kcat_fwd:5.1f}\t{new_kcat_rev:5.1f}")

                print("km_degradation:")
                for factor in np.logspace(-1, 1, n_factors):
                    new_kcat_fwd, new_kcat_rev, *_ = calc_kcats(
                        aa_conc_basal,
                        km_conc_basal,
                        aa_conc_with_aa,
                        km_conc_with_aa,
                        kms,
                        km_reverse,
                        factor * km_degradation,
                        ki,
                        measured_uptake_rates[amino_acid[:-3]],
                    )

                    print(f"\t{factor:.2f}:\t{new_kcat_fwd:5.1f}\t{new_kcat_rev:5.1f}")

                print("ki:")
                for factor in np.logspace(-1, 1, n_factors):
                    new_kcat_fwd, new_kcat_rev, *_ = calc_kcats(
                        aa_conc_basal,
                        km_conc_basal,
                        aa_conc_with_aa,
                        km_conc_with_aa,
                        kms,
                        km_reverse,
                        km_degradation,
                        factor * ki,
                        measured_uptake_rates[amino_acid[:-3]],
                    )

                    print(f"\t{factor:.2f}:\t{new_kcat_fwd:5.1f}\t{new_kcat_rev:5.1f}")
                print(f"*** {amino_acid}: {kcat_fwd:5.1f} {kcat_rev:5.1f} ***")

            if kcat_fwd is None:
                raise ValueError(
                    "Could not find positive forward and reverse"
                    f" kcat for {amino_acid}. Run with VERBOSE to check input"
                    " parameters like KM and KI or check concentrations."
                )

            aa_enzymes += fwd_enzymes + rev_enzymes
            enzyme_to_aa_fwd += [amino_acid] * len(fwd_enzymes) + [None] * len(
                rev_enzymes
            )
            enzyme_to_aa_rev += [None] * len(fwd_enzymes) + [amino_acid] * len(
                rev_enzymes
            )
            aa_kcats_fwd[amino_acid] = kcat_fwd
            aa_kcats_rev[amino_acid] = kcat_rev
            aa_kis[amino_acid] = ki.asNumber(METABOLITE_CONCENTRATION_UNITS)
            upstream_aas_for_km[amino_acid] = upstream_aa
            aa_upstream_kms[amino_acid] = kms.asNumber(METABOLITE_CONCENTRATION_UNITS)
            aa_reverse_kms[amino_acid] = km_reverse.asNumber(
                METABOLITE_CONCENTRATION_UNITS
            )
            aa_degradation_kms[amino_acid] = km_degradation.asNumber(
                METABOLITE_CONCENTRATION_UNITS
            )
            fwd_rates[amino_acid] = fwd_rate
            rev_rates[amino_acid] = rev_rate
            deg_rates[amino_acid] = deg_rate
            calculated_uptake_rates[amino_acid] = uptake

        self.aa_enzymes = np.unique(aa_enzymes)
        self.aa_kcats_fwd = np.array([aa_kcats_fwd[aa] for aa in aa_ids])
        self.aa_kcats_rev = np.array([aa_kcats_rev[aa] for aa in aa_ids])
        self.aa_kis = np.array([aa_kis[aa] for aa in aa_ids])
        self.aa_reverse_kms = np.array([aa_reverse_kms[aa] for aa in aa_ids])
        self.aa_degradation_kms = np.array([aa_degradation_kms[aa] for aa in aa_ids])

        # Import inhibition of transporters
        rich_conc = np.array(
            [with_aa_conc[aa].asNumber(METABOLITE_CONCENTRATION_UNITS) for aa in aa_ids]
        )
        self.aa_import_kis: np.ndarray = (
            rich_conc.copy()
        )  # Assume this conc is the inhibition constant: TODO: find KIs
        saturation = 1 / (1 + rich_conc / self.aa_import_kis)
        self.specific_import_rates: np.ndarray = np.array(
            [calculated_uptake_rates[aa] for aa in aa_ids]
        ) / cell_specs["with_aa"]["avgCellDryMassInit"].asNumber(DRY_MASS_UNITS)
        self.max_specific_import_rates = self.specific_import_rates / saturation

        # KMs for upstream amino acids
        upstream_kms = [aa_upstream_kms[aa] for aa in aa_ids]
        upstream_aas = [upstream_aas_for_km[aa] for aa in aa_ids]
        self.aa_upstream_kms = np.zeros((n_aas, n_aas))
        for i, (kms, aas) in enumerate(zip(upstream_kms, upstream_aas)):
            for km, aa in zip(kms, aas):
                self.aa_upstream_kms[i, self.aa_to_index[aa]] = km

        # Convert enzyme counts to an amino acid basis via dot product (counts @ self.enzyme_to_amino_acid)
        self.enzyme_to_amino_acid_fwd = np.zeros((len(self.aa_enzymes), n_aas))
        self.enzyme_to_amino_acid_rev = np.zeros((len(self.aa_enzymes), n_aas))
        enzyme_mapping = {e: i for i, e in enumerate(self.aa_enzymes)}
        aa_mapping = {a: i for i, a in enumerate(aa_ids)}
        for enzyme, fwd, rev in zip(aa_enzymes, enzyme_to_aa_fwd, enzyme_to_aa_rev):
            if fwd is not None:
                self.enzyme_to_amino_acid_fwd[
                    enzyme_mapping[enzyme], aa_mapping[fwd]
                ] = 1
            if rev is not None:
                self.enzyme_to_amino_acid_rev[
                    enzyme_mapping[enzyme], aa_mapping[rev]
                ] = 1

        # Concentrations for reference in analysis plot
        conversion = (
            sim_data.constants.cell_density
            / sim_data.constants.n_avogadro
            * sim_data.mass.cell_dry_mass_fraction
        )
        aa_enzymes_basal_idx = bulk_name_to_idx(self.aa_enzymes, basal_container["id"])
        basal_counts = counts(basal_container, aa_enzymes_basal_idx)
        aa_enzymes_with_aa_idx = bulk_name_to_idx(
            self.aa_enzymes, with_aa_container["id"]
        )
        with_aa_counts = counts(with_aa_container, aa_enzymes_with_aa_idx)
        self.aa_supply_enzyme_conc_with_aa = (
            conversion * with_aa_counts / cell_specs["with_aa"]["avgCellDryMassInit"]
        )
        self.aa_supply_enzyme_conc_basal = (
            conversion * basal_counts / cell_specs["basal"]["avgCellDryMassInit"]
        )

        # Check calculations that could end up negative
        neg_idx = np.where(self.max_specific_import_rates < 0)[0]
        if len(neg_idx):
            bad_aas = ", ".join([aa_ids[idx] for idx in neg_idx])
            print(f"{self.max_specific_import_rates = }")
            raise ValueError(
                f"Import rate was determined to be negative for {bad_aas}."
                " Check input parameters like supply and synthesis or enzyme expression."
            )

    def get_pathway_enzyme_counts_per_aa(
        self, enzyme_counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the counts of enzymes for forward and reverse reactions in the
        amino acid synthesis network based on all of the enzymes used in the
        network.  Useful to get the counts to pass to amino_acid_synthesis()
        from counts based on self.aa_enzymes.

        Args:
                enzyme_counts: counts of all enzymes in the amino acid network

        Returns:
                2-element tuple containing
                        - counts_per_aa_fwd: counts of enzymes for the forward reaction
                          for each amino acid
                        - counts_per_aa_rev: counts of enzymes for the reverse reaction
                          for each amino acid
        """

        counts_per_aa_fwd = enzyme_counts @ self.enzyme_to_amino_acid_fwd
        counts_per_aa_rev = enzyme_counts @ self.enzyme_to_amino_acid_rev
        return counts_per_aa_fwd, counts_per_aa_rev

    def amino_acid_synthesis(
        self,
        counts_per_aa_fwd: npt.NDArray[np.int64],
        counts_per_aa_rev: npt.NDArray[np.int64],
        aa_conc: Union[units.Unum, npt.NDArray[np.float64]],
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Calculate the net rate of synthesis for amino acid pathways (can be
        negative with reverse reactions).

        Args:
                counts_per_aa_fwd: counts for enzymes in forward reactions for
                        each amino acid
                counts_per_aa_rev: counts for enzymes in loss reactions for each
                        amino acid
                aa_conc: concentrations of each amino acid with mol/volume units

        Returns:
                3-element tuple containing

                        - synthesis: net rate of synthesis for each amino acid pathway.
                          array is unitless but represents counts of amino acid per second
                        - forward_fraction: saturated fraction for forward reactions
                        - loss_fraction: saturated fraction for loss reactions

        .. note::
                Currently does not match saturation terms used in calc_kcats since
                it assumes only a reverse or degradation KM exists for simpler calculations
        """

        # Convert to appropriate arrays
        if units.hasUnit(aa_conc):
            aa_conc = aa_conc.asNumber(METABOLITE_CONCENTRATION_UNITS)

        return amino_acid_synthesis_jit(
            counts_per_aa_fwd,
            counts_per_aa_rev,
            aa_conc,
            self.aa_upstream_kms,
            self.aa_kis,
            self.aa_reverse_kms,
            self.aa_degradation_kms,
            self.aa_forward_stoich,
            self.aa_kcats_fwd,
            self.aa_reverse_stoich,
            self.aa_kcats_rev,
        )

    def amino_acid_export(
        self,
        aa_transporters_counts: npt.NDArray[np.int64],
        aa_conc: Union[units.Unum, npt.NDArray[np.float64]],
        mechanistic_uptake: bool,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the rate of amino acid export.

        Args:
                aa_transporters_counts: counts of each transporter
                aa_conc: concentrations of each amino acid with mol/volume units
                mechanistic_uptake: if true, the uptake is calculated based on transporters

        Returns:
                Rate of export for each amino acid (unitless but
                represents counts of amino acid per second)
        """
        if units.hasUnit(aa_conc):
            aa_conc = aa_conc.asNumber(METABOLITE_CONCENTRATION_UNITS)

        return amino_acid_export_jit(
            aa_transporters_counts,
            aa_conc,
            mechanistic_uptake,
            self.aa_to_exporters_matrix,
            self.export_kcats_per_aa,
            self.aa_export_kms,
        )

    def amino_acid_import(
        self,
        aa_in_media: npt.NDArray[np.bool_],
        dry_mass: units.Unum,
        internal_aa_conc: Union[units.Unum, npt.NDArray[np.float64]],
        aa_transporters_counts: npt.NDArray[np.int64],
        mechanistic_uptake: bool,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the rate of amino acid uptake.

        Args:
                aa_in_media: bool for each amino acid being present in current media
                dry_mass: current dry mass of the cell, with mass units
                internal_aa_conc: internal concentrations of amino acids
                aa_transporters_counts: counts of each transporter
                mechanistic_uptake: if true, the uptake is calculated based on
                        transporters

        Returns:
                Rate of uptake for each amino acid (unitless but
                represents counts of amino acid per second)
        """

        if units.hasUnit(internal_aa_conc):
            internal_aa_conc = internal_aa_conc.asNumber(METABOLITE_CONCENTRATION_UNITS)

        dry_mass = dry_mass.asNumber(DRY_MASS_UNITS)
        return amino_acid_import_jit(
            aa_in_media,
            dry_mass,
            internal_aa_conc,
            aa_transporters_counts,
            mechanistic_uptake,
            self.aa_import_kis,
            self.aa_to_importers_matrix,
            self.import_kcats_per_aa,
            self.max_specific_import_rates,
        )

    def get_amino_acid_conc_conversion(self, conc_units):
        return units.strip_empty_units(conc_units / METABOLITE_CONCENTRATION_UNITS)

    @staticmethod
    def extract_reactions(
        raw_data: KnowledgeBaseEcoli, sim_data: "SimulationDataEcoli"
    ) -> tuple[
        list[str],
        dict[str, dict[str, int]],
        list[str],
        dict[str, list[str]],
        dict[str, str],
    ]:
        """
        Extracts reaction data from raw_data to build metabolism reaction
        network with stoichiometry, reversibility and enzyme catalysts.

        Args:
                raw_data: knowledge base data
                sim_data: simulation data

        Returns:
                5-element tuple containing

                        - base_rxn_ids: list of base reaction IDs from which reaction
                          IDs were derived from
                        - reaction_stoich: stoichiometry of metabolites for each reaction::

                                {reaction ID: {metabolite ID with location tag: stoichiometry}}

                        - reversible_reactions: reaction IDs for reactions that have a
                          reverse complement, does not have reverse tag
                        - reaction_catalysts: enzyme catalysts for each reaction with known
                          catalysts, likely a subset of reactions in stoich::

                                {reaction ID: enzyme IDs with location tag}

                        - rxn_id_to_base_rxn_id: mapping from reaction IDs to the IDs of
                          the base reactions they were derived from::

                                {reaction ID: base ID}
        """
        compartment_ids_to_abbreviations = {
            comp["id"]: comp["abbrev"] for comp in raw_data.compartments
        }
        compartment_ids_to_abbreviations.update(UNDEFINED_COMPARTMENT_IDS_TO_ABBREVS)

        valid_directions = {"L2R", "R2L", "BOTH"}
        forward_directions = {"L2R", "BOTH"}
        reverse_directions = {"R2L", "BOTH"}

        metabolite_ids = {met["id"] for met in cast(Any, raw_data).metabolites}

        # Build mapping from each complexation subunit to all downstream
        # complexes containing the subunit, including itself
        # Start by building mappings from subunits to complexes that are
        # directly formed from the subunit through a single reaction
        subunit_id_to_parent_complexes: dict[str, list[str]] = {}

        for comp_reaction in itertools.chain(
            cast(Any, raw_data).complexation_reactions,
            cast(Any, raw_data).equilibrium_reactions,
        ):
            complex_id = None

            # Find ID of complex
            for mol_id, coeff in comp_reaction["stoichiometry"].items():
                if coeff > 0:
                    complex_id = mol_id
                    break

            assert complex_id is not None

            # Map each subunit to found complex
            for mol_id, coeff in comp_reaction["stoichiometry"].items():
                if mol_id == complex_id or mol_id in metabolite_ids:
                    continue
                elif mol_id in subunit_id_to_parent_complexes:
                    subunit_id_to_parent_complexes[mol_id].append(complex_id)
                else:
                    subunit_id_to_parent_complexes[mol_id] = [complex_id]

        # Recursive function that returns a list of all downstream complexes
        # containing the given subunit, including itself
        def get_all_complexes(subunit_id):
            all_downstream_complex_ids = [subunit_id]

            if subunit_id not in subunit_id_to_parent_complexes:
                return all_downstream_complex_ids

            # Get downstream complexes of all parent complexes
            for parent_complex_id in subunit_id_to_parent_complexes[subunit_id]:
                all_downstream_complex_ids.extend(get_all_complexes(parent_complex_id))

            # Remove duplicates
            return sorted(set(all_downstream_complex_ids))

        subunit_id_to_all_downstream_complexes = {
            subunit_id: get_all_complexes(subunit_id)
            for subunit_id in subunit_id_to_parent_complexes.keys()
        }

        # Initialize variables to store reaction information
        all_base_rxns = set()
        reaction_stoich = {}
        reversible_reactions = []
        reaction_catalysts = {}
        rxn_id_to_base_rxn_id = {}

        # Load and parse reaction information from raw_data
        for reaction in cast(Any, raw_data).metabolic_reactions:
            reaction_id = reaction["id"]
            stoich = reaction["stoichiometry"]
            direction = reaction["direction"]

            if len(stoich) <= 1:
                raise Exception(
                    "Invalid biochemical reaction: {}, {}".format(reaction_id, stoich)
                )

            if direction not in valid_directions:
                raise InvalidReactionDirectionError(
                    f"The direction {direction} given for reaction {reaction_id} is invalid."
                )

            forward = direction in forward_directions
            reverse = direction in reverse_directions

            def convert_compartment_tags(met_id):
                new_met_id = met_id

                for comp_id, comp_abbrev in compartment_ids_to_abbreviations.items():
                    new_met_id = new_met_id.replace(f"[{comp_id}]", f"[{comp_abbrev}]")

                return new_met_id

            # All protein complexes that contain an enzyme subunit are assumed
            # to retain the enzyme's catalytic activity
            catalysts_for_this_rxn = []
            all_potential_catalysts = []
            for catalyst in reaction["catalyzed_by"]:
                all_potential_catalysts.extend(
                    subunit_id_to_all_downstream_complexes.get(catalyst, [catalyst])
                )

            for catalyst in sorted(set(all_potential_catalysts)):
                if sim_data.getter.is_valid_molecule(catalyst):
                    catalysts_with_loc = catalyst + sim_data.getter.get_compartment_tag(
                        catalyst
                    )
                    catalysts_for_this_rxn.append(catalysts_with_loc)
                # If we don't have the catalyst in our reconstruction, drop it
                else:
                    if VERBOSE:
                        print(
                            "Skipping catalyst {} for {} since it is not in the model".format(
                                catalyst, reaction_id
                            )
                        )

            # Get base reaction ID of this reaction
            # If reaction ID does not end with a dot, the given reaction ID is
            # already a base reaction ID
            if reaction_id[-1] != ".":
                base_reaction_id = reaction_id
            # If reaction ID ends with a dot, find the base reaction ID based
            # on the following rules (provided by EcoCyc):
            #   The parsing instructions for obtaining the base rxn-ID are, effectively:
            #   1: Check whether the full rxn-ID ends with a dot.
            #   2: If there is no dot at the end, it is already a base rxn-ID.
            #       Otherwise, find the position of a second dot, to the left of the dot at the end.
            #   3: Extract the string between the last dot and the second to last dot.
            #       If this intervening string consists of only digits, then convert this string to an integer.
            #       In this case, this rxn-ID stands for a generic rxn, and has extra suffixes
            #       that need to be trimmed off, to retrieve the base rxn-ID.
            #       The extracted integer indicates the length of the suffixes to be trimmed off.
            #   4: To find the end position of the base rxn-ID, subtract the integer (obtained by 3: )
            #       from the position of the second dot (obtained by 2: ) .
            #   5: Retrieve the base rxn-ID, which is the substring from the very left (position 0)
            #       to the end position (obtained by 4: ) .
            else:
                reaction_id_split = reaction_id[:-1].split(".")
                suffix_length = int(reaction_id_split[-1])
                base_reaction_id = ".".join(reaction_id_split[:-1])[:-suffix_length]

            if forward:
                reaction_stoich[reaction_id] = {
                    convert_compartment_tags(moleculeID): stoichCoeff
                    for moleculeID, stoichCoeff in stoich.items()
                }
                if len(catalysts_for_this_rxn) > 0:
                    reaction_catalysts[reaction_id] = catalysts_for_this_rxn
                rxn_id_to_base_rxn_id[reaction_id] = base_reaction_id

            if reverse:
                reverse_reaction_id = REVERSE_REACTION_ID.format(reaction_id)
                reaction_stoich[reverse_reaction_id] = {
                    convert_compartment_tags(moleculeID): -stoichCoeff
                    for moleculeID, stoichCoeff in stoich.items()
                }
                if len(catalysts_for_this_rxn) > 0:
                    reaction_catalysts[reverse_reaction_id] = list(
                        catalysts_for_this_rxn
                    )
                rxn_id_to_base_rxn_id[reverse_reaction_id] = base_reaction_id

            if forward and reverse:
                reversible_reactions.append(reaction_id)

            if base_reaction_id not in all_base_rxns:
                all_base_rxns.add(base_reaction_id)

        base_rxn_ids = sorted(list(all_base_rxns))

        return (
            base_rxn_ids,
            reaction_stoich,
            reversible_reactions,
            reaction_catalysts,
            rxn_id_to_base_rxn_id,
        )

    @staticmethod
    def match_reaction(
        stoich: dict[str, dict[str, int]],
        catalysts: dict[str, list[str]],
        rxn_to_match: str,
        enz: str,
        mets: list[str],
        direction: Optional[str] = None,
    ) -> list[str]:
        """
        Matches a given reaction (rxn_to_match) to reactions that exist in
        stoich given that enz is known to catalyze the reaction and mets are
        reactants in the reaction. Can perform a fuzzy reaction match since
        rxn_to_match just needs to be part of the actual reaction name to match
        specific instances of a reaction.
        (eg. rxn_to_match="ALCOHOL-DEHYDROG-GENERIC-RXN" can match
        "ALCOHOL-DEHYDROG-GENERIC-RXN-ETOH/NAD//ACETALD/NADH/PROTON.30.").

        Args:
                stoich: stoichiometry of metabolites for each reaction::

                        {reaction ID: {metabolite ID with location tag: stoichiometry}}

                catalysts: enzyme catalysts for each reaction with known catalysts,
                        likely a subset of reactions in stoich::

                        {reaction ID: enzyme IDs with location tag}

                rxn_to_match: reaction ID from kinetics to match to existing reactions
                enz: enzyme ID with location tag
                mets: metabolite IDs with no location tag from kinetics
                direction: reaction directionality, ``'forward'`` or ``'reverse'``
                        or ``None``

        Returns:
                Matched reaction IDs in stoich
        """

        # Mapping to handle instances of metabolite classes in kinetics
        # Keys: specific molecules in kinetics file
        # Values: class of molecules in reactions file that contain the key
        class_mets = {
            "RED-THIOREDOXIN-MONOMER": "Red-Thioredoxin",
            "RED-THIOREDOXIN2-MONOMER": "Red-Thioredoxin",
            "RED-GLUTAREDOXIN": "Red-Glutaredoxins",
            "GRXB-MONOMER": "Red-Glutaredoxins",
            "GRXC-MONOMER": "Red-Glutaredoxins",
            "OX-FLAVODOXIN1": "Oxidized-flavodoxins",
            "OX-FLAVODOXIN2": "Oxidized-flavodoxins",
        }

        # Match full reaction name from partial reaction in kinetics. Must
        # also match metabolites since there can be multiple reaction instances.
        match = False
        match_candidates = []
        if rxn_to_match in stoich:
            match_candidates.append(rxn_to_match)
        else:
            for long_rxn, long_mets in stoich.items():
                if rxn_to_match in long_rxn and not long_rxn.endswith(REVERSE_TAG):
                    match = True
                    stripped_enzs = {e[:-3] for e in catalysts.get(long_rxn, [])}
                    stripped_mets = {m[:-3] for m in long_mets}
                    if (
                        np.all([class_mets.get(m, m) in stripped_mets for m in mets])
                        and enz in stripped_enzs
                    ):
                        match_candidates.append(long_rxn)

        if len(match_candidates) == 0:
            if VERBOSE:
                if match:
                    print(
                        "Partial reaction match: {} {} {} {} {}".format(
                            rxn_to_match, enz, stripped_enzs, mets, stripped_mets
                        )
                    )
                else:
                    print("No reaction match: {}".format(rxn_to_match))

        # Determine direction of kinetic reaction from annotation or
        # metabolite stoichiometry.
        rxn_matches = []
        for rxn in match_candidates:
            reverse_rxn = REVERSE_REACTION_ID.format(rxn)
            reverse_rxn_exists = reverse_rxn in stoich
            if direction:
                reverse = direction == "reverse"
            else:
                s = {k[:-3]: v for k, v in stoich.get(rxn, {}).items()}
                direction_ = np.unique(
                    np.sign([s.get(class_mets.get(m, m), 0) for m in mets])
                )
                if len(direction_) == 0 and not reverse_rxn_exists:
                    reverse = False
                elif len(direction_) != 1 or direction_[0] == 0:
                    if VERBOSE:
                        print(
                            "Conflicting directionality: {} {} {}".format(
                                rxn, mets, direction_
                            )
                        )
                    continue
                else:
                    reverse = direction_[0] > 0

            # Verify a reverse reaction exists in the model
            if reverse:
                if reverse_rxn_exists:
                    rxn_matches.append(reverse_rxn)
                    continue
                else:
                    if VERBOSE:
                        print("No reverse reaction: {} {}".format(rxn, mets))
                    continue

            rxn_matches.append(rxn)

        return sorted(rxn_matches)

    @staticmethod
    def temperature_adjusted_kcat(
        kcat: Unum, temp: Union[float, str] = ""
    ) -> npt.NDArray[np.float64]:
        """
        Args:
                kcat: enzyme turnover number(s) (1 / time)
                temp: temperature of measurement, defaults to 25 if ''

        Returns:
                Temperature adjusted kcat values, in units of 1/s
        """

        if isinstance(temp, str):
            temp = 25
        return 2 ** ((37.0 - temp) / 10.0) * kcat.asNumber(K_CAT_UNITS)

    @staticmethod
    def _construct_default_saturation_equation(
        mets: list[str], kms: list[float], kis: list[float], known_mets: Iterable[str]
    ) -> str:
        """
        Args:
                mets: metabolite IDs with location tag for KM and KI
                        parameters ordered to match order of kms then kis
                kms: KM parameters associated with mets
                kis: KI parameters associated with mets
                known_mets: metabolite IDs with location tag with known
                        concentrations

        Returns:
                Saturation equation with metabolites to replace delimited
                by double quote (e.g. "metabolite")
        """

        # Check input dimensions
        n_params = len(kms) + len(kis)
        if n_params == 0:
            return "1"
        if n_params != len(mets):
            if VERBOSE:
                print("Saturation parameter mismatch: {} {} {}".format(mets, kms, kis))
            return "1"

        terms = []
        # Add KM terms
        for m, k in zip(mets, kms):
            if m in known_mets:
                terms.append('1+{}/"{}"'.format(k, m))
            elif VERBOSE:
                print("Do not have concentration for {} with KM={}".format(m, k))
        # Add KI terms
        for m, k in zip(mets[len(kms) :], kis):
            if m in known_mets:
                terms.append('1+"{}"/{}'.format(m, k))
            elif VERBOSE:
                print("Do not have concentration for {} with KI={}".format(m, k))

        # Enclose groupings if being multiplied together
        if len(terms) > 1:
            terms[0] = "(" + terms[0]
            terms[-1] += ")"
        elif len(terms) == 0:
            return "1"

        return "1/({})".format(")*(".join(terms))

    @staticmethod
    def _extract_custom_constraint(
        constraint: dict[str, Any],
        reactant_tags: dict[str, str],
        product_tags: dict[str, str],
        known_mets: set[str],
    ) -> tuple[Optional[npt.NDArray[np.float64]], list[str]]:
        """
        Args:
                constraint: values defining a kinetic constraint::

                        {'customRateEquation' (str): mathematical representation of
                                rate (must contain 'kcat*E'),
                        'customParameterVariables' (dict[str, str]): mapping of
                                variable names in the rate equation to metabolite IDs
                                without location tags (must contain 'E'),
                        'customParameterConstants' (list[str]): constant strings
                                in the rate equation that correspond to values (must
                                contain 'kcat'),
                        'customParameterConstantValues' (list[float]): values for
                                each of the constant strings,
                        'Temp' (float or ''): temperature of measurement}

                reactant_tags: mapping of molecule IDs without a location tag to
                        molecule IDs with a location tag for all reactants
                product_tags: mapping of molecule IDs without a location tag to
                        molecule IDs with a location tag for all products
                known_mets: molecule IDs with a location tag for molecules with
                        known concentrations

        Returns:
                2-element tuple containing

                        - kcats: temperature adjusted kcat value, in units of 1/s
                        - saturation: saturation equation with metabolites to replace
                          delimited by double quote (eg. "metabolite")
        """

        equation = constraint["customRateEquation"]
        variables = constraint["customParameterVariables"]
        constant_keys = constraint["customParameterConstants"]
        constant_values = constraint["customParameterConstantValues"]
        temp = constraint["Temp"]

        # Need to have these in the constraint
        kcat_str = "kcat"
        enzyme_str = "E"
        capacity_str = "{}*{}".format(kcat_str, enzyme_str)

        # Need to replace these symbols in equations
        symbol_sub = {
            "^": "**",
        }

        # Make sure kcat exists
        if kcat_str not in constant_keys:
            if VERBOSE:
                print(
                    "Missing {} in custom constants: {}".format(kcat_str, constant_keys)
                )
            return None, []

        custom_kcat = (
            1 / units.s * np.array([constant_values[constant_keys.index(kcat_str)]])
        )
        kcats = Metabolism.temperature_adjusted_kcat(custom_kcat, temp)

        # Make sure equation can be parsed, otherwise just return kcat
        if enzyme_str not in variables:
            if VERBOSE:
                print(
                    "Missing enzyme key ({}) in custom variables: {}".format(
                        enzyme_str, variables
                    )
                )
            return kcats, []
        if capacity_str not in equation:
            if VERBOSE:
                print(
                    "Expected to find {} in custom equation: {}".format(
                        capacity_str, equation
                    )
                )
            return kcats, []
        if len(constant_keys) != len(constant_values):
            if VERBOSE:
                print(
                    "Mismatch between constants: {} {}".format(
                        constant_keys, constant_values
                    )
                )
            return kcats, []

        variables_with_tags = {
            k: reactant_tags.get(v, product_tags.get(v, None))
            for k, v in variables.items()
            if k != enzyme_str and (v in reactant_tags or v in product_tags)
        }

        # Substitute values into custom equations
        ## Replace terms with known constant values or sim molecule IDs with concentrations
        custom_subs = {k: str(v) for k, v in zip(constant_keys, constant_values)}
        custom_subs.update(
            {
                k: '"{}"'.format(v)
                for k, v in variables_with_tags.items()
                if v in known_mets
            }
        )

        ## Remove capacity to get only saturation term
        new_equation = equation.replace(capacity_str, "1")

        ## Tokenize equation to terms and symbols
        parsed_variables = re.findall(r"\w*", new_equation)[
            :-1
        ]  # Remove trailing empty match
        ## Ensure valid input of known variables or a float term
        for v in parsed_variables:
            if not (v == "" or v in custom_subs):
                try:
                    float(v)
                except ValueError:
                    if VERBOSE:
                        print(
                            "Unknown value encountered in custom equation {}: {}".format(
                                equation, v
                            )
                        )
                    return kcats, []
        parsed_symbols = re.findall(r"\W", new_equation)
        tokenized_equation = np.array(parsed_variables)
        symbol_idx_mask = tokenized_equation == ""

        ## Verify tokenized equation matches original before replacements
        tokenized_equation[symbol_idx_mask] = parsed_symbols
        if "".join(tokenized_equation) != new_equation:
            if VERBOSE:
                print("Error parsing custom equation: {}".format(equation))
            return kcats, []

        ## Perform replacement of symbols
        tokenized_equation[symbol_idx_mask] = [
            symbol_sub.get(s, s) for s in parsed_symbols
        ]

        # Reconstruct saturation equation with replacements
        saturation = [
            "".join([custom_subs.get(token, token) for token in tokenized_equation])
        ]

        return kcats, saturation

    @staticmethod
    def extract_kinetic_constraints(
        raw_data: KnowledgeBaseEcoli,
        sim_data: "SimulationDataEcoli",
        stoich: Optional[dict[str, dict[str, int]]] = None,
        catalysts: Optional[dict[str, list[str]]] = None,
        known_metabolites: Optional[Set[str]] = None,
    ) -> dict[tuple[str, str], dict[str, list[Any]]]:
        """
        Load and parse kinetic constraint information from raw_data

        Args:
                raw_data: knowledge base data
                sim_data: simulation data
                stoich: stoichiometry of metabolites for each reaction (if ``None``,
                        data is loaded from ``raw_data`` and ``sim_data``)::

                        {reaction ID: {metabolite ID with location tag: stoichiometry}}

                catalysts: enzyme catalysts for each reaction with known catalysts,
                        likely a subset of reactions in ``stoich`` (if ``None``, data
                        is loaded from ``raw_data`` and ``sim_data``::

                        {reaction ID: enzyme IDs with location tag}

                known_metabolites: metabolites with known concentrations

        Returns:
                Valid kinetic constraints for each reaction/enzyme pair::

                        {(reaction ID, enzyme with location tag): {
                                'kcat': kcat values (list[float]),
                                'saturation': saturation equations (list[str])
                        }}
        """

        # Load data for optional args if needed
        if stoich is None or catalysts is None:
            _, loaded_stoich, _, loaded_catalysts, _ = Metabolism.extract_reactions(
                raw_data, sim_data
            )

            if stoich is None:
                stoich = loaded_stoich
            if catalysts is None:
                catalysts = loaded_catalysts

        known_metabolites_ = set() if known_metabolites is None else known_metabolites

        constraints: dict[tuple[str, str], dict[str, list]] = {}
        for constraint in cast(Any, raw_data).metabolism_kinetics:
            rxn = constraint["reactionID"]
            enzyme = constraint["enzymeID"]
            metabolites = constraint["substrateIDs"]
            direction = constraint["direction"]
            kms = list(constraint["kM"].asNumber(KINETIC_CONSTRAINT_CONC_UNITS))
            kis = list(constraint["kI"].asNumber(KINETIC_CONSTRAINT_CONC_UNITS))
            n_reactants = len(metabolites) - len(kis)
            matched_rxns = Metabolism.match_reaction(
                stoich, catalysts, rxn, enzyme, metabolites[:n_reactants], direction
            )

            for matched_rxn in matched_rxns:
                # Ensure enzyme catalyzes reaction in model
                enzymes_tag_conversion = {
                    e[:-3]: e for e in catalysts.get(matched_rxn, [])
                }
                if enzyme not in enzymes_tag_conversion:
                    if VERBOSE:
                        print("{} does not catalyze {}".format(enzyme, matched_rxn))
                    continue

                # Update metabolites with a location tag from the reaction
                # First look in reactants but some products can inhibit
                reactant_tags = {
                    k[:-3]: k for k, v in stoich[matched_rxn].items() if v < 0
                }
                product_tags = {
                    k[:-3]: k for k, v in stoich[matched_rxn].items() if v > 0
                }
                mets_with_tag = [
                    reactant_tags.get(met, product_tags.get(met, ""))
                    for met in metabolites
                    if met in reactant_tags or met in product_tags
                ]
                if len(mets_with_tag) != len(metabolites):
                    # Warn if verbose but no continue since we can still use kcat
                    if VERBOSE:
                        print(
                            "Could not match all metabolites: {} {}".format(
                                metabolites, mets_with_tag
                            )
                        )

                # Extract kcat and saturation parameters
                if constraint["rateEquationType"] == "custom":
                    kcats, saturation = Metabolism._extract_custom_constraint(
                        constraint, reactant_tags, product_tags, known_metabolites_
                    )
                    if kcats is None:
                        continue
                else:
                    kcats = Metabolism.temperature_adjusted_kcat(
                        constraint["kcat"], constraint["Temp"]
                    )
                    if len(kcats) > 1:
                        if len(kcats) != len(kms) or len(kms) != len(mets_with_tag):
                            if VERBOSE:
                                print(
                                    "Could not align kcats and kms: {} {} {} {}".format(
                                        rxn, kcats, kms, mets_with_tag
                                    )
                                )
                            continue

                        saturation = [
                            Metabolism._construct_default_saturation_equation(
                                [m], [km], [], known_metabolites_
                            )
                            for m, km in zip(mets_with_tag, kms)
                        ]
                    else:
                        saturation = [
                            Metabolism._construct_default_saturation_equation(
                                mets_with_tag, kms, kis, known_metabolites_
                            )
                        ]

                    saturation = [s for s in saturation if s != "1"]

                # Add new kcats and saturation terms for the enzymatic reaction
                key = (matched_rxn, enzymes_tag_conversion[enzyme])
                entries = constraints.get(key, {})
                entries["kcat"] = entries.get("kcat", []) + list(kcats)
                entries["saturation"] = entries.get("saturation", []) + saturation
                constraints[key] = entries

        return constraints

    @staticmethod
    def _replace_enzyme_reactions(
        constraints: dict[tuple[str, str], dict[str, list[Any]]],
        stoich: dict[str, dict[str, int]],
        rxn_catalysts: dict[str, list[str]],
        reversible_rxns: list[str],
        rxn_id_to_compiled_id: dict[str, str],
    ) -> tuple[
        dict[str, Any],
        dict[str, dict[str, int]],
        dict[str, list[str]],
        list[str],
        dict[str, str],
    ]:
        """
        Modifies reaction IDs in data structures to duplicate reactions with
        kinetic constraints and multiple enzymes.

        Args:
                constraints: valid kinetic constraints for each reaction/enzyme pair::

                        {(reaction ID, enzyme with location tag): {
                                'kcat': kcat values (list[float]),
                                'saturation': saturation equations (list[str])
                        }}

                stoich: stoichiometry of metabolites for each reaction (if None, data
                        is loaded from raw_data and sim_data)::

                        {reaction ID: {metabolite ID with location tag: stoichiometry}}

                rxn_catalysts: enzyme catalysts for each reaction with known catalysts,
                        likely a subset of reactions in stoich (if None, data is loaded
                        from raw_data and sim_data)::

                        {reaction ID: enzyme IDs with location tag}

                reversible_rxns: reaction IDs for reactions that have a reverse
                        complement, does not have reverse tag
                rxn_id_to_compiled_id: mapping from reaction IDs to the IDs of the
                        original reactions they were derived from

        Returns:
                5-element tuple containing

                        - new_constraints: valid kinetic constraints for each reaction::

                                {reaction ID: {
                                        'enzyme': enzyme catalyst (str),
                                        'kcat': kcat values (list[float]),
                                        'saturation': saturation equations (list[str])
                                }}

                        - stoich: stoichiometry of metabolites for each reaction with
                          updated reactions for enzyme catalyzed kinetic reactions::

                                {reaction ID: {metabolite ID with location tag: stoichiometry}}

                        - rxn_catalysts: enzyme catalysts for each reaction with known
                          catalysts, likely a subset of reactions in stoich with
                          updated reactions for enzyme catalyzed kinetic reactions::

                                {reaction ID: enzyme IDs with location tag}

                        - reversible_rxns: reaction IDs for reactions that have a reverse
                          complement with updated reactions for enzyme catalyzed kinetic
                          reactions, does not have reverse tag
                        - rxn_id_to_compiled_id: mapping from reaction IDs to the IDs
                          of the original reactions they were derived from, with updated
                          reactions for enzyme catalyzed kinetic reactions
        """

        new_constraints = {}

        n_catalysts = {rxn: len(catalysts) for rxn, catalysts in rxn_catalysts.items()}

        # Split out reactions that are kinetically constrained and that have
        # more than one enzyme that catalyzes the reaction
        for (rxn, enzyme), constraint in constraints.items():
            if n_catalysts[rxn] > 1:
                # Create new reaction name with enzyme appended to the end
                if rxn.endswith(REVERSE_TAG):
                    new_rxn = REVERSE_REACTION_ID.format(
                        ENZYME_REACTION_ID.format(rxn[: -len(REVERSE_TAG)], enzyme[:-3])
                    )
                else:
                    new_rxn = ENZYME_REACTION_ID.format(rxn, enzyme[:-3])

                # Add the new reaction to appropriate lists and dicts
                stoich[new_rxn] = copy(stoich[rxn])
                rxn_catalysts[new_rxn] = [enzyme]
                if rxn in reversible_rxns:
                    reversible_rxns.append(new_rxn)

                # Remove enzyme from old reaction and remove old reaction if no
                # more enzyme catalysts
                rxn_catalysts[rxn].pop(rxn_catalysts[rxn].index(enzyme))

                if len(rxn_catalysts[rxn]) == 0:
                    stoich.pop(rxn)
                    rxn_catalysts.pop(rxn)
                    if rxn in reversible_rxns:
                        reversible_rxns.pop(reversible_rxns.index(rxn))
            else:
                new_rxn = rxn

            rxn_id_to_compiled_id[new_rxn] = rxn_id_to_compiled_id[rxn]

            # noinspection PyTypeChecker
            new_constraints[new_rxn] = dict(constraints[(rxn, enzyme)], enzyme=enzyme)

        return (
            new_constraints,
            stoich,
            rxn_catalysts,
            reversible_rxns,
            rxn_id_to_compiled_id,
        )

    @staticmethod
    def _lambdify_constraints(
        constraints: dict[str, Any],
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        npt.NDArray[np.float64],
        str,
        str,
        npt.NDArray[np.bool_],
    ]:
        """
        Creates str representations of kinetic terms to be used to create
        kinetic constraints that are returned with getKineticConstraints().

        Args:
                constraints: valid kinetic constraints for each reaction::

                        {reaction ID: {
                                'enzyme': enzyme catalyst (str),
                                'kcat': kcat values (list[float]),
                                'saturation': saturation equations (list[str])
                        }}

        Returns:
                7-element tuple containing

                        - rxns: sorted reaction IDs for reactions with a kinetic
                          constraint
                        - enzymes: sorted enzyme IDs for enzymes that catalyze a
                          kinetic reaction
                        - substrates: sorted substrate IDs for substrates that are
                          needed for kinetic saturation terms
                        - all_kcats: (n rxns, 3) min, mean and max kcat value for
                          each reaction
                        - all_saturations: sympy str representation of a list of
                          saturation terms (eg. '[s[0] / (1 + s[0]), 2 / (2 + s[1])]')
                        - all_enzymes: sympy str representation of enzymes for each
                          reaction (e.g. '[e[0], e[2], e[1]]')
                        - constraint_is_kcat_only: True if reaction only has kcat
                          values and no saturation terms
        """

        # Ordered lists of constraint related IDs
        rxns = sorted(constraints)
        enzymes = sorted({c["enzyme"] for c in constraints.values()})
        substrates = sorted(
            {
                match.strip('"')
                for c in constraints.values()
                for s in c["saturation"]
                for match in re.findall('".+?"', s)
            }
        )

        # Mapping to replace molecule IDs with generic list strings
        enzyme_sub = {e: "e[{}]".format(i) for i, e in enumerate(enzymes)}
        substrate_sub = {
            '"{}"'.format(s): "s[{}]".format(i) for i, s in enumerate(substrates)
        }

        # Mapping to replace generic list strings with sympy variables.
        # Need separate mapping from above because sympy handles '[]' as indexing
        # so location tags are not parsed properly.
        enzyme_symbols = {
            "e": [sp.symbols("e[{}]".format(i)) for i in range(len(enzymes))]
        }
        substrate_symbols = {
            "s": [sp.symbols("s[{}]".format(i)) for i in range(len(substrates))]
        }

        # Values to return
        all_kcats = np.zeros((len(rxns), 3))
        all_saturations = []
        all_enzymes = []
        constraint_is_kcat_only = []

        # Extract out data from each constraint
        for i, rxn in enumerate(rxns):
            kcats = constraints[rxn]["kcat"]
            saturation = constraints[rxn]["saturation"]
            enzyme = constraints[rxn]["enzyme"]

            # Parse saturation equations into sympy format
            # If no saturation data is known, assume open range from 0 to 1
            saturations = []
            for sat in saturation:
                if sat == "1":
                    continue

                for token, replace in substrate_sub.items():
                    sat = sat.replace(token, replace)
                saturations.append(parse_expr(sat, local_dict=substrate_symbols))
            if len(saturations) == 0:
                saturations = [0, 1]
                constraint_is_kcat_only.append(True)
            else:
                constraint_is_kcat_only.append(False)

            # Save values for this constraint
            all_kcats[i, :] = [np.min(kcats), np.mean(kcats), np.max(kcats)]
            all_saturations.append(saturations)
            all_enzymes.append(
                parse_expr(enzyme_sub[enzyme], local_dict=enzyme_symbols)
            )

        # Convert to str to save as class attr to be executed
        all_saturations = str(all_saturations)
        all_enzymes = str(all_enzymes)
        constraint_is_kcat_only = np.array(constraint_is_kcat_only)

        return (
            rxns,
            enzymes,
            substrates,
            all_kcats,
            all_saturations,
            all_enzymes,
            constraint_is_kcat_only,
        )

    def _is_transport_rxn(self, stoich: dict[str, int]) -> bool:
        """
        Determines if the metabolic reaction with a given stoichiometry is a
        transport reactions that transports metabolites between different
        compartments. A metabolic reaction is considered to be a transport
        reaction if the substrate set and the product share the same metabolite
        tagged into different compartments.

        Args:
                stoich: Stoichiometry of the metabolic reaction::

                        {
                        metabolite ID (str): stoichiometric coefficient (int)
                        }

        Returns:
                True if given stoichiometry is a transport reaction
        """
        is_transport_rxn = False

        # Get IDs of all substrates and products
        substrates, products = [], []
        for mol_id, coeff in stoich.items():
            if coeff < 0:
                substrates.append(mol_id)
            else:
                products.append(mol_id)

        # Get mapping from IDs to IDs without compartments
        substrates_tagged_to_no_tag = {
            mol_id: re.sub(r"\[.*]", "", mol_id) for mol_id in substrates
        }
        products_tagged_to_no_tag = {
            mol_id: re.sub(r"\[.*]", "", mol_id) for mol_id in products
        }

        overlap_no_tag = set(substrates_tagged_to_no_tag.values()) & set(
            products_tagged_to_no_tag.values()
        )

        for mol_id_no_tag in list(overlap_no_tag):
            substrates_tagged = [
                mol_tagged
                for mol_tagged in substrates
                if substrates_tagged_to_no_tag[mol_tagged] == mol_id_no_tag
            ]
            products_tagged = [
                mol_tagged
                for mol_tagged in products
                if products_tagged_to_no_tag[mol_tagged] == mol_id_no_tag
            ]

            overlap_tagged = set(substrates_tagged) & set(products_tagged)

            # Tag reaction as a transport reaction if there is no overlap
            # between those substrates and products with locations included
            if len(overlap_tagged) == 0:
                is_transport_rxn = True
                break

        return is_transport_rxn


@njit
def np_apply_along_axis(func1d, axis, arr):
    if arr.ndim != 2:
        raise RuntimeError("Array must have 2 dimensions.")
    if axis not in [0, 1]:
        raise RuntimeError("Axis must be 0 or 1.")
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit
def np_prod(array, axis):
    return np_apply_along_axis(np.prod, axis, array)


@njit
def amino_acid_synthesis_jit(
    counts_per_aa_fwd,
    counts_per_aa_rev,
    aa_conc,
    aa_upstream_kms,
    aa_kis,
    aa_reverse_kms,
    aa_degradation_kms,
    aa_forward_stoich,
    aa_kcats_fwd,
    aa_reverse_stoich,
    aa_kcats_rev,
):
    km_saturation = np_prod(1 / (1 + aa_upstream_kms / aa_conc), axis=1)

    # Determine saturation fraction for reactions
    forward_fraction = 1 / (1 + aa_conc / aa_kis) * km_saturation
    reverse_fraction = 1 / (1 + aa_reverse_kms / aa_conc)
    deg_fraction = 1 / (1 + aa_degradation_kms / aa_conc)
    loss_fraction = reverse_fraction + deg_fraction

    # Calculate synthesis rate
    synthesis = (
        aa_forward_stoich.astype(np.float64)
        @ (aa_kcats_fwd * counts_per_aa_fwd * forward_fraction).astype(np.float64)
        - aa_reverse_stoich.astype(np.float64)
        @ (aa_kcats_rev * counts_per_aa_rev * reverse_fraction).astype(np.float64)
        - aa_kcats_rev * counts_per_aa_rev * deg_fraction
    )

    return synthesis, forward_fraction, loss_fraction


@njit
def amino_acid_export_jit(
    aa_transporters_counts,
    aa_conc,
    mechanistic_uptake,
    aa_to_exporters_matrix,
    export_kcats_per_aa,
    aa_export_kms,
):
    if mechanistic_uptake:
        # Export based on mechanistic model
        trans_counts_per_aa = aa_to_exporters_matrix.astype(
            np.float64
        ) @ aa_transporters_counts.astype(np.float64)
        export_rates = (
            export_kcats_per_aa * trans_counts_per_aa / (1 + aa_export_kms / aa_conc)
        )
    else:
        # Export is lumped with specific uptake rates in amino_acid_import
        # and not dependent on internal amino acid concentrations or
        # explicitly considered here
        export_rates = np.zeros(len(aa_conc))
    return export_rates


@njit
def amino_acid_import_jit(
    aa_in_media,
    dry_mass,
    internal_aa_conc,
    aa_transporters_counts,
    mechanistic_uptake,
    aa_import_kis,
    aa_to_importers_matrix,
    import_kcats_per_aa,
    max_specific_import_rates,
):
    saturation = 1 / (1 + internal_aa_conc / aa_import_kis)
    if mechanistic_uptake:
        # Uptake based on mechanistic model
        counts_per_aa = aa_to_importers_matrix.astype(
            np.float64
        ) @ aa_transporters_counts.astype(np.float64)
        import_rates = import_kcats_per_aa * counts_per_aa
    else:
        import_rates = max_specific_import_rates * dry_mass

    return import_rates * saturation * aa_in_media


# Class used to update metabolite concentrations based on the current nutrient conditions
class ConcentrationUpdates(object):
    def __init__(
        self,
        concDict,
        relative_changes,
        equilibriumReactions,
        exchange_data_dict,
        all_metabolite_ids,
        linked_metabolites,
    ):
        self.units = units.getUnit(list(concDict.values())[0])
        self.default_concentrations_dict = dict(
            (key, concDict[key].asNumber(self.units)) for key in concDict
        )
        self.exchange_fluxes = self._exchange_flux_present(exchange_data_dict)
        self.relative_changes = relative_changes
        self._all_metabolite_ids = all_metabolite_ids
        self.linked_metabolites = linked_metabolites

        # factor of internal amino acid increase if amino acids present in nutrients
        self.molecule_scale_factors = {
            "L-ALPHA-ALANINE[c]": 2.0,
            "ARG[c]": 2.0,
            "ASN[c]": 2.0,
            "L-ASPARTATE[c]": 2.0,
            "CYS[c]": 2.0,
            "GLT[c]": 2.0,
            "GLN[c]": 2.0,
            "GLY[c]": 2.0,
            "HIS[c]": 2.0,
            "ILE[c]": 2.0,
            "LEU[c]": 2.0,
            "LYS[c]": 2.0,
            "MET[c]": 2.0,
            "PHE[c]": 2.0,
            "PRO[c]": 2.0,
            "SER[c]": 2.0,
            "THR[c]": 2.0,
            "TRP[c]": 2.0,
            "TYR[c]": 2.0,
            "L-SELENOCYSTEINE[c]": 2.0,
            "VAL[c]": 2.0,
        }

        self.molecule_set_amounts = self._add_molecule_amounts(
            equilibriumReactions, self.default_concentrations_dict
        )

    # return adjustments to concDict based on nutrient conditions
    def concentrations_based_on_nutrients(
        self,
        media_id: Optional[str] = None,
        imports: Optional[str] = None,
        conversion_units: Optional[units.Unum] = None,
    ) -> dict[str, Any]:
        if conversion_units:
            conversion = self.units.asNumber(conversion_units)
        else:
            conversion = self.units

        if imports is None and media_id is not None:
            imports = self.exchange_fluxes[media_id]

        concentrationsDict = self.default_concentrations_dict.copy()

        metaboliteTargetIds = sorted(concentrationsDict.keys())
        concentrations = conversion * np.array(
            [concentrationsDict[k] for k in metaboliteTargetIds]
        )
        concDict = dict(zip(metaboliteTargetIds, concentrations))

        if imports is not None:
            # For faster conversions than .asNumber(conversion_units) for each setAmount
            if conversion_units:
                conversion_to_no_units = conversion_units.asUnit(self.units)

            # Adjust for measured concentration changes in different media
            if media_id in self.relative_changes:
                for mol_id, conc_change in self.relative_changes[media_id].items():
                    if mol_id in concDict:
                        concDict[mol_id] *= conc_change

            for moleculeName, setAmount in self.molecule_set_amounts.items():
                if (
                    moleculeName in imports
                    and (
                        moleculeName[:-3] + "[c]" not in self.molecule_scale_factors
                        or moleculeName == "L-SELENOCYSTEINE[c]"
                    )
                ) or (
                    moleculeName in self.molecule_scale_factors
                    and moleculeName[:-3] + "[p]" in imports
                ):
                    if conversion_units:
                        setAmount = (setAmount / conversion_to_no_units).asNumber()
                    concDict[moleculeName] = setAmount

        for met, linked in self.linked_metabolites.items():
            concDict[met] = concDict[linked["lead"]] * linked["ratio"]

        return concDict

    def _exchange_flux_present(
        self, exchange_data: dict[str, Any]
    ) -> dict[str, set[str]]:
        """
        Caches the presence of exchanges in each media condition based on
        exchange_data to set concentrations in concentrations_based_on_nutrients().

        Args:
                exchange_data: dictionary of exchange data for all media conditions
                        with keys::

                                {importUnconstrainedExchangeMolecules (dict[str, set[str]]):
                                        exchange molecules (with location tag) for each media key
                                        that do not have an upper bound on their flux,
                                importConstrainedExchangeMolecules (dict[str,
                                        dict[str, float with mol/mass/time units]]): constrained
                                        molecules (with location tag) for each media key with upper
                                        bound flux constraints}

        Returns:
                Sets of molecules IDs (with location tags) that can be imported
                for each media ID
        """

        exchange_fluxes = {}
        for media, env in exchange_data.items():
            exchange_fluxes[media] = {mol for mol, conc in env.items() if conc > 0}

        return exchange_fluxes

    def _add_molecule_amounts(self, equilibriumReactions, concDict):
        moleculeSetAmounts = {}
        for reaction in equilibriumReactions:
            # We only want to do this for species with standard Michaelis-Menten kinetics initially
            if len(reaction["stoichiometry"]) != 3:
                continue

            moleculeName = [
                mol_id
                for mol_id in reaction["stoichiometry"].keys()
                if mol_id in self._all_metabolite_ids
            ][0]
            amountToSet = 1e-4
            moleculeSetAmounts[moleculeName + "[p]"] = amountToSet * self.units
            moleculeSetAmounts[moleculeName + "[c]"] = amountToSet * self.units

        for moleculeName, scaleFactor in self.molecule_scale_factors.items():
            moleculeSetAmounts[moleculeName] = (
                scaleFactor * concDict[moleculeName] * self.units
            )
        return moleculeSetAmounts
