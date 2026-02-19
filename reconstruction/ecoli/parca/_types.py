"""
Input/Output dataclasses for all ParCa pipeline stages.

All dataclasses live in this single module to avoid circular imports
between stage modules that share types.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ============================================================================
# Stage 2: input_adjustments
# ============================================================================


@dataclass
class InputAdjustmentsInput:
    """Input data extracted from sim_data for the input_adjustments stage."""

    debug: bool

    # Translation efficiencies
    monomer_ids: np.ndarray
    translation_efficiencies: np.ndarray  # .copy() of translation_efficiencies_by_monomer
    translation_eff_adjustments: dict  # {protein_id: multiplier}
    balanced_translation_groups: list  # list of lists of protein ids

    # RNA expression
    rna_ids: np.ndarray
    cistron_ids: np.ndarray
    basal_rna_expression: np.ndarray  # .copy() of rna_expression["basal"]
    rna_expression_adjustments: dict  # {mol_id: multiplier}
    cistron_id_to_rna_indexes: Dict[str, np.ndarray]  # pre-computed mapping

    # Degradation rates
    rna_deg_rates: np.ndarray  # .copy() of rna_data.struct_array["deg_rate"]
    cistron_deg_rates: np.ndarray  # .copy() of cistron_data.struct_array["deg_rate"]
    rna_deg_rate_adjustments: dict  # {mol_id: multiplier}
    protein_deg_rates: np.ndarray  # .copy() of monomer_data.struct_array["deg_rate"]
    protein_deg_rate_adjustments: dict  # {protein_id: multiplier}

    # TF conditions (for debug filtering)
    tf_to_active_inactive_conditions: dict


@dataclass
class InputAdjustmentsOutput:
    """Output data to merge back into sim_data after input_adjustments."""

    translation_efficiencies: np.ndarray
    basal_rna_expression: np.ndarray
    rna_deg_rates: np.ndarray
    cistron_deg_rates: np.ndarray
    protein_deg_rates: np.ndarray
    tf_to_active_inactive_conditions: Optional[dict] = None  # only set if debug=True


# ============================================================================
# Stage 3: basal_specs
# ============================================================================


@dataclass
class BasalSpecsInput:
    """Input data for the basal_specs stage.

    NOTE: This stage requires sim_data process objects for the
    expressionConverge loop, Km fitting (setKmCooperativeEndoRNonLinearRNAdecay),
    ppGpp expression setting, and maintenance cost fitting.  These are accessed
    through the mutable ``sim_data_ref``.  The compute function WILL mutate
    sim_data_ref (mass values, expression arrays, Km values, etc.) because
    downstream sub-functions within the stage depend on earlier mutations.
    A future refactoring will decompose these dependencies.
    """

    variable_elongation_transcription: bool
    variable_elongation_translation: bool
    disable_ribosome_capacity_fitting: bool
    disable_rnapoly_capacity_fitting: bool
    cache_dir: Optional[str]

    # Mutable reference to sim_data — needed by expressionConverge,
    # setKmCooperativeEndoRNonLinearRNAdecay, fitMaintenanceCosts, and
    # set_ppgpp_expression.  Will be decomposed in later refactoring.
    sim_data_ref: Any


@dataclass
class BasalSpecsOutput:
    """Output data to merge back into cell_specs after basal_specs.

    NOTE: sim_data mutations are applied inside compute_basal_specs
    via sim_data_ref (mass, expression, Km, darkATP, ppGpp).
    merge_output only writes cell_specs["basal"].
    """

    # cell_specs["basal"] contents
    conc_dict: dict
    expression: np.ndarray
    synth_prob: np.ndarray
    fit_cistron_expression: np.ndarray
    doubling_time: Any  # units
    avg_cell_dry_mass_init: Any  # units
    fit_avg_soluble_target_mol_mass: Any  # units
    bulk_container: np.ndarray


# ============================================================================
# Stage 4: tf_condition_specs
# ============================================================================


@dataclass
class TfConditionSpecsInput:
    """Input data for the tf_condition_specs stage.

    NOTE: This stage requires sim_data process objects for
    expressionConverge and expressionFromConditionAndFoldChange.
    sim_data_ref is passed as a mutable reference because compute
    updates sim_data expression dicts between TF-specific fitting and
    combined-condition fitting.
    """

    variable_elongation_transcription: bool
    variable_elongation_translation: bool
    disable_ribosome_capacity_fitting: bool
    disable_rnapoly_capacity_fitting: bool
    cpus: int

    # Mutable reference — expressionConverge reads deeply, and
    # expression dicts are updated between sub-steps.
    sim_data_ref: Any


@dataclass
class TfConditionSpecsConditionOutput:
    """Computed results for a single condition (TF or combined)."""

    condition_label: str
    conc_dict: dict
    expression: np.ndarray
    synth_prob: np.ndarray
    cistron_expression: Optional[np.ndarray]  # None for basal
    fit_cistron_expression: np.ndarray
    doubling_time: Any  # units
    avg_cell_dry_mass_init: Any  # units
    fit_avg_soluble_target_mol_mass: Any  # units
    bulk_container: np.ndarray


@dataclass
class TfConditionSpecsOutput:
    """Output data to merge back into sim_data/cell_specs after tf_condition_specs.

    NOTE: sim_data expression dict mutations are applied during compute
    via sim_data_ref.  merge_output only writes cell_specs entries.
    """

    condition_outputs: List[TfConditionSpecsConditionOutput]


# ============================================================================
# Stage 5: fit_condition
# ============================================================================


@dataclass
class FitConditionConditionInput:
    """Per-condition data extracted from cell_specs for fit_condition."""

    condition_label: str
    nutrients: str
    expression: np.ndarray
    conc_dict: dict
    avg_cell_dry_mass_init: Any  # units
    doubling_time: Any  # units


@dataclass
class FitConditionInput:
    """Input data for the fit_condition stage.

    NOTE: This stage requires sim_data process objects for stochastic
    simulation (complexation, equilibrium, two-component system).
    These are passed as read-only references in ``sim_data_ref`` and
    will be fully extracted in a future refactoring.  The compute
    function does NOT mutate sim_data or cell_specs.
    """

    conditions: List[FitConditionConditionInput]
    cpus: int

    # Read-only reference to sim_data, needed by calculateBulkDistributions
    # and calculateTranslationSupply. Will be decomposed further in later
    # refactoring passes.
    sim_data_ref: Any


@dataclass
class FitConditionConditionOutput:
    """Computed results for a single condition from fitCondition."""

    condition_label: str
    bulk_average_container: np.ndarray
    bulk_deviation_container: np.ndarray
    protein_monomer_average_container: np.ndarray
    protein_monomer_deviation_container: np.ndarray
    translation_aa_supply: Any  # units array


@dataclass
class FitConditionOutput:
    """Output data to merge back into sim_data/cell_specs after fit_condition."""

    condition_outputs: List[FitConditionConditionOutput]

    # Nutrient -> translation supply rate (first occurrence per nutrient)
    translation_supply_rate: Dict[str, Any]


# ============================================================================
# Stage 6: promoter_binding
# ============================================================================


@dataclass
class PromoterBindingInput:
    """Input data for the promoter_binding stage.

    NOTE: This stage requires deep reads into sim_data process objects
    (equilibrium, two_component_system, replication, transcription_regulation)
    and cell_specs (bulkAverageContainer, doubling_time, avgCellDryMassInit).
    Both are passed as mutable/read references.  The compute function WILL
    mutate sim_data_ref (pPromoterBound, rna_synth_prob).
    """

    sim_data_ref: Any
    cell_specs_ref: dict  # read-only reference for promoter fitting


@dataclass
class PromoterBindingOutput:
    """Output data to merge back into cell_specs after promoter_binding.

    NOTE: sim_data mutations (pPromoterBound, rna_synth_prob) are applied
    during compute via sim_data_ref.  merge_output only writes cell_specs.
    """

    r_vector: np.ndarray
    r_columns: dict  # G_col_name_to_index mapping


# ============================================================================
# Stage 7: adjust_promoters
# ============================================================================


@dataclass
class AdjustPromotersInput:
    """Input data for the adjust_promoters stage.

    NOTE: This stage requires deep reads into sim_data process objects
    (equilibrium, metabolism, transcription_regulation) and cell_specs
    (bulkAverageContainer, avgCellDryMassInit, r_vector, r_columns).
    The compute function WILL mutate sim_data_ref (molecule_set_amounts,
    equilibrium reverse rates via fitLigandConcentrations).
    """

    sim_data_ref: Any
    cell_specs_ref: dict  # read-only reference


@dataclass
class AdjustPromotersOutput:
    """Output data to merge back into sim_data after adjust_promoters.

    NOTE: sim_data mutations from fitLigandConcentrations (molecule_set_amounts,
    equilibrium reverse rates) are applied during compute via sim_data_ref.
    merge_output writes basal_prob and delta_prob to sim_data.
    """

    basal_prob: np.ndarray
    delta_prob: dict  # {"deltaI": ..., "deltaJ": ..., "deltaV": ..., "shape": ...}


# ============================================================================
# Stage 8: set_conditions
# ============================================================================


@dataclass
class SetConditionsConditionInput:
    """Pre-computed data for a single condition in set_conditions."""

    condition_label: str
    nutrients: str
    has_perturbations: bool
    doubling_time: Any  # units

    # Concentration data (pre-computed from sim_data methods)
    target_molecule_ids: list  # sorted(concDict)
    target_molecule_concentrations: Any  # units array (mol/L)
    molecular_weights: Any  # units array (g/mol)

    # Mass rescaling constants
    non_small_molecule_initial_cell_mass: Any  # units
    avg_cell_to_initial_cell_conversion_factor: float
    cell_density: Any  # units
    n_avogadro: Any  # units

    # From cell_specs
    bulk_container: np.ndarray  # copy of bulkContainer
    avg_cell_dry_mass_init_old: Any  # current value for verbose logging

    # RNA synth prob for this condition
    rna_synth_prob: np.ndarray

    # Growth rate parameters (pre-computed)
    fraction_active_rnap: Any
    rnap_elongation_rate: Any
    ribosome_elongation_rate: Any
    fraction_active_ribosome: Any


@dataclass
class SetConditionsInput:
    """Input data extracted from sim_data/cell_specs for the set_conditions stage."""

    conditions: List[SetConditionsConditionInput]

    # RNA type masks (shared across all conditions)
    is_mRNA: np.ndarray
    is_tRNA: np.ndarray
    is_rRNA: np.ndarray
    includes_ribosomal_protein: np.ndarray
    includes_RNAP: np.ndarray

    verbose: int


@dataclass
class SetConditionsConditionOutput:
    """Computed results for a single condition."""

    condition_label: str
    avg_cell_dry_mass_init: Any  # units
    fit_avg_soluble_pool_mass: Any  # units
    bulk_container: np.ndarray  # updated bulk container


@dataclass
class SetConditionsOutput:
    """Output data to merge back into sim_data/cell_specs after set_conditions."""

    # sim_data dicts (keyed by nutrients)
    rnaSynthProbFraction: Dict[str, dict]
    rnapFractionActiveDict: Dict[str, Any]
    rnaSynthProbRProtein: Dict[str, np.ndarray]
    rnaSynthProbRnaPolymerase: Dict[str, np.ndarray]
    rnaPolymeraseElongationRateDict: Dict[str, Any]
    expectedDryMassIncreaseDict: Dict[str, Any]
    ribosomeElongationRateDict: Dict[str, Any]
    ribosomeFractionActiveDict: Dict[str, Any]

    # cell_specs updates per condition
    condition_outputs: List[SetConditionsConditionOutput]


# ============================================================================
# Stage 9: final_adjustments
# ============================================================================


@dataclass
class FinalAdjustmentsInput:
    """Input data for the final_adjustments stage.

    NOTE: This stage calls deep sim_data methods (calculate_attenuation,
    adjust_*_ppgpp_expression, set_*_constants, set_ppgpp_kinetics_parameters)
    that read from and mutate sim_data process objects extensively.
    cell_specs is read by calculate_attenuation and set_mechanistic_* methods.
    All mutations happen on sim_data_ref; there are no cell_specs writes.
    """

    sim_data_ref: Any
    cell_specs_ref: dict  # read-only reference


@dataclass
class FinalAdjustmentsOutput:
    """Output data after final_adjustments.

    NOTE: All mutations are applied during compute via sim_data_ref.
    merge_output is a no-op since there are no cell_specs writes and
    all sim_data modifications are done by deep process-object methods.
    """

    pass  # No extractable outputs — all writes go to sim_data_ref
