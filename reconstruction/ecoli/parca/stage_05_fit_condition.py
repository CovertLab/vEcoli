"""
Stage 5: fit_condition — Calculate bulk distributions and translation
supply rates for every condition.

Three public functions:
    extract_input(sim_data, cell_specs, **kwargs) -> FitConditionInput
    compute_fit_condition(inp) -> FitConditionOutput
    merge_output(sim_data, cell_specs, out)

For each condition this stage:
1. Instantiates N_SEEDS cells with RNA/protein counts drawn from mass-
   based distributions
2. Runs complexation, equilibrium, and two-component-system processes
   to steady state
3. Collects mean/std of all bulk molecule counts
4. Computes translation amino-acid supply rates

NOTE: calculateBulkDistributions and calculateTranslationSupply use
sim_data process objects (equilibrium solver, two-component system,
stochastic complexation) that have not yet been decomposed into pure
data.  They are accessed through the read-only ``sim_data_ref`` in
FitConditionInput.  A future refactoring will extract these into
explicit data.
"""

from stochastic_arrow import StochasticSystem
import numpy as np

from ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import units
from wholecell.utils.fitting import normalize

from reconstruction.ecoli.parca._shared import (
    apply_updates,
    netLossRateFromDilutionAndDegradationProtein,
    proteinDistributionFrommRNA,
    totalCountFromMassesAndRatios,
    totalCountIdDistributionRNA,
    totalCountIdDistributionProtein,
)
from reconstruction.ecoli.parca._types import (
    FitConditionConditionInput,
    FitConditionConditionOutput,
    FitConditionInput,
    FitConditionOutput,
)

# Constants (mirrored from fit_sim_data_1.py)
N_SEEDS = 10
VERBOSE = 1


# ============================================================================
# Extract / Merge
# ============================================================================


def extract_input(sim_data, cell_specs, **kwargs) -> FitConditionInput:
    """Pull per-condition data from cell_specs and a read-only sim_data ref."""
    condition_inputs = []
    for condition_label in sorted(cell_specs):
        spec = cell_specs[condition_label]
        nutrients = sim_data.conditions[condition_label]["nutrients"]
        condition_inputs.append(
            FitConditionConditionInput(
                condition_label=condition_label,
                nutrients=nutrients,
                expression=spec["expression"],
                conc_dict=spec["concDict"],
                avg_cell_dry_mass_init=spec["avgCellDryMassInit"],
                doubling_time=spec["doubling_time"],
            )
        )

    return FitConditionInput(
        conditions=condition_inputs,
        cpus=kwargs.get("cpus", 1),
        sim_data_ref=sim_data,
    )


def merge_output(sim_data, cell_specs, out: FitConditionOutput):
    """Write computed results back into sim_data and cell_specs."""
    for cond_out in out.condition_outputs:
        spec = cell_specs[cond_out.condition_label]
        spec["bulkAverageContainer"] = cond_out.bulk_average_container
        spec["bulkDeviationContainer"] = cond_out.bulk_deviation_container
        spec["proteinMonomerAverageContainer"] = cond_out.protein_monomer_average_container
        spec["proteinMonomerDeviationContainer"] = cond_out.protein_monomer_deviation_container
        spec["translation_aa_supply"] = cond_out.translation_aa_supply

    for nutrients, supply in out.translation_supply_rate.items():
        sim_data.translation_supply_rate[nutrients] = supply


# ============================================================================
# Sub-functions (use sim_data read-only)
# ============================================================================


def calculateBulkDistributions(
    sim_data, expression, concDict, avgCellDryMassInit, doubling_time
):
    """
    Find distributions of copy numbers for macromolecules by instantiating
    N_SEEDS cells, forming complexes, and iterating equilibrium/two-component
    system processes to steady state.

    Returns:
        (bulkAverageContainer, bulkDeviationContainer,
         proteinMonomerAverageContainer, proteinMonomerDeviationContainer)
    """
    totalCount_RNA, ids_rnas, distribution_RNA = totalCountIdDistributionRNA(
        sim_data, expression, doubling_time
    )
    totalCount_protein, ids_protein, distribution_protein = (
        totalCountIdDistributionProtein(sim_data, expression, doubling_time)
    )
    ids_complex = sim_data.process.complexation.molecule_names
    ids_equilibrium = sim_data.process.equilibrium.molecule_names
    ids_twoComponentSystem = sim_data.process.two_component_system.molecule_names
    ids_metabolites = sorted(concDict)
    conc_metabolites = (units.mol / units.L) * np.array(
        [concDict[key].asNumber(units.mol / units.L) for key in ids_metabolites]
    )
    allMoleculesIDs = sorted(
        set(ids_rnas)
        | set(ids_protein)
        | set(ids_complex)
        | set(ids_equilibrium)
        | set(ids_twoComponentSystem)
        | set(ids_metabolites)
    )

    complexationStoichMatrix = sim_data.process.complexation.stoich_matrix().astype(
        np.int64, order="F"
    )

    cellDensity = sim_data.constants.cell_density
    cellVolume = avgCellDryMassInit / cellDensity / sim_data.mass.cell_dry_mass_fraction

    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data.struct_array["id"]
    bulkContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", int)],
    )

    rna_idx = bulk_name_to_idx(ids_rnas, bulkContainer["id"])
    protein_idx = bulk_name_to_idx(ids_protein, bulkContainer["id"])
    complexation_molecules_idx = bulk_name_to_idx(ids_complex, bulkContainer["id"])
    equilibrium_molecules_idx = bulk_name_to_idx(ids_equilibrium, bulkContainer["id"])
    two_component_system_molecules_idx = bulk_name_to_idx(
        ids_twoComponentSystem, bulkContainer["id"]
    )
    metabolites_idx = bulk_name_to_idx(ids_metabolites, bulkContainer["id"])
    all_molecules_idx = bulk_name_to_idx(allMoleculesIDs, bulkContainer["id"])

    allMoleculeCounts = np.empty((N_SEEDS, len(allMoleculesIDs)), np.int64)
    proteinMonomerCounts = np.empty((N_SEEDS, len(ids_protein)), np.int64)

    for seed in range(N_SEEDS):
        bulkContainer["count"][all_molecules_idx] = 0
        bulkContainer["count"][rna_idx] = totalCount_RNA * distribution_RNA
        bulkContainer["count"][protein_idx] = (
            totalCount_protein * distribution_protein
        )

        proteinMonomerCounts[seed, :] = counts(bulkContainer, protein_idx)
        complexationMoleculeCounts = counts(bulkContainer, complexation_molecules_idx)

        # Form complexes
        time_step = 2**31
        complexation_rates = sim_data.process.complexation.rates
        system = StochasticSystem(complexationStoichMatrix.T, random_seed=seed)
        complexation_result = system.evolve(
            time_step, complexationMoleculeCounts, complexation_rates
        )

        updatedCompMoleculeCounts = complexation_result["outcome"]
        bulkContainer["count"][complexation_molecules_idx] = updatedCompMoleculeCounts

        metDiffs = np.inf * np.ones_like(counts(bulkContainer, metabolites_idx))
        nIters = 0

        while np.linalg.norm(metDiffs, np.inf) > 1:
            random_state = np.random.RandomState(seed)
            metCounts = conc_metabolites * cellVolume * sim_data.constants.n_avogadro
            metCounts.normalize()
            metCounts.checkNoUnit()
            bulkContainer["count"][metabolites_idx] = metCounts.asNumber().round()

            rxnFluxes, _ = sim_data.process.equilibrium.fluxes_and_molecules_to_SS(
                bulkContainer["count"][equilibrium_molecules_idx],
                cellVolume.asNumber(units.L),
                sim_data.constants.n_avogadro.asNumber(1 / units.mol),
                random_state,
                jit=False,
            )
            bulkContainer["count"][equilibrium_molecules_idx] += np.dot(
                sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
                rxnFluxes.astype(np.int64),
            )
            assert np.all(bulkContainer["count"][equilibrium_molecules_idx] >= 0)

            _, moleculeCountChanges = (
                sim_data.process.two_component_system.molecules_to_ss(
                    bulkContainer["count"][two_component_system_molecules_idx],
                    cellVolume.asNumber(units.L),
                    sim_data.constants.n_avogadro.asNumber(1 / units.mmol),
                )
            )

            bulkContainer["count"][two_component_system_molecules_idx] += (
                moleculeCountChanges.astype(np.int64)
            )

            metDiffs = (
                bulkContainer["count"][metabolites_idx]
                - metCounts.asNumber().round()
            )

            nIters += 1
            if nIters > 100:
                raise Exception("Equilibrium reactions are not converging!")

        allMoleculeCounts[seed, :] = counts(bulkContainer, all_molecules_idx)

    # Build output containers
    bulkAverageContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", np.float64)],
    )
    bulkDeviationContainer = np.array(
        [mol_data for mol_data in zip(bulk_ids, np.zeros(len(bulk_ids)))],
        dtype=[("id", bulk_ids.dtype), ("count", np.float64)],
    )
    monomer_ids = sim_data.process.translation.monomer_data["id"]
    proteinMonomerAverageContainer = np.array(
        [mol_data for mol_data in zip(monomer_ids, np.zeros(len(monomer_ids)))],
        dtype=[("id", monomer_ids.dtype), ("count", np.float64)],
    )
    proteinMonomerDeviationContainer = np.array(
        [mol_data for mol_data in zip(monomer_ids, np.zeros(len(monomer_ids)))],
        dtype=[("id", monomer_ids.dtype), ("count", np.float64)],
    )

    bulkAverageContainer["count"][all_molecules_idx] = allMoleculeCounts.mean(0)
    bulkDeviationContainer["count"][all_molecules_idx] = allMoleculeCounts.std(0)
    proteinMonomerAverageContainer["count"] = proteinMonomerCounts.mean(0)
    proteinMonomerDeviationContainer["count"] = proteinMonomerCounts.std(0)

    return (
        bulkAverageContainer,
        bulkDeviationContainer,
        proteinMonomerAverageContainer,
        proteinMonomerDeviationContainer,
    )


def calculateTranslationSupply(
    sim_data, doubling_time, bulkContainer, avgCellDryMassInit
):
    """
    Compute supply rates of amino acids to translation given doubling time.

    Returns:
        translation_aa_supply (units array of mol/(mass*time))
    """
    aaCounts = sim_data.process.translation.monomer_data["aa_counts"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)
    nAvogadro = sim_data.constants.n_avogadro

    molAAPerGDCW = units.sum(
        aaCounts * np.tile(proteinCounts.reshape(-1, 1), (1, 21)), axis=0
    ) * ((1 / (units.aa * nAvogadro)) * (1 / avgCellDryMassInit))

    translation_aa_supply = molAAPerGDCW * np.log(2) / doubling_time

    return translation_aa_supply


def _fit_single_condition(sim_data, spec, condition):
    """
    Fit a single condition: find bulk distributions and translation supply.

    This is the worker function called for each condition (possibly in
    parallel via multiprocessing).

    Args:
        sim_data: read-only SimulationDataEcoli
        spec: dict with 'expression', 'concDict', 'avgCellDryMassInit',
              'doubling_time'
        condition: condition label string

    Returns:
        {condition: updated spec dict}
    """
    if VERBOSE > 0:
        print("Fitting condition {}".format(condition))

    (
        bulkAverageContainer,
        bulkDeviationContainer,
        proteinMonomerAverageContainer,
        proteinMonomerDeviationContainer,
    ) = calculateBulkDistributions(
        sim_data,
        spec["expression"],
        spec["concDict"],
        spec["avgCellDryMassInit"],
        spec["doubling_time"],
    )
    spec["bulkAverageContainer"] = bulkAverageContainer
    spec["bulkDeviationContainer"] = bulkDeviationContainer
    spec["proteinMonomerAverageContainer"] = proteinMonomerAverageContainer
    spec["proteinMonomerDeviationContainer"] = proteinMonomerDeviationContainer

    spec["translation_aa_supply"] = calculateTranslationSupply(
        sim_data,
        spec["doubling_time"],
        spec["proteinMonomerAverageContainer"],
        spec["avgCellDryMassInit"],
    )

    return {condition: spec}


# ============================================================================
# Main compute function
# ============================================================================


def compute_fit_condition(inp: FitConditionInput) -> FitConditionOutput:
    """
    Run fitCondition for each condition and collect results.

    Does not mutate sim_data or cell_specs. All results are returned
    via FitConditionOutput.

    NOTE: sim_data_ref is accessed read-only through the input for
    calculateBulkDistributions and calculateTranslationSupply.
    """
    sim_data = inp.sim_data_ref

    # Build spec dicts for each condition (working copies)
    working_specs = {}
    for cond in inp.conditions:
        working_specs[cond.condition_label] = {
            "expression": cond.expression,
            "concDict": cond.conc_dict,
            "avgCellDryMassInit": cond.avg_cell_dry_mass_init,
            "doubling_time": cond.doubling_time,
        }

    # Run fitCondition for each condition (possibly in parallel)
    conditions = [cond.condition_label for cond in inp.conditions]
    args = [
        (sim_data, working_specs[condition], condition)
        for condition in conditions
    ]
    apply_updates(_fit_single_condition, args, conditions, working_specs, inp.cpus)

    # Collect per-condition outputs
    condition_outputs = []
    for cond in inp.conditions:
        spec = working_specs[cond.condition_label]
        condition_outputs.append(
            FitConditionConditionOutput(
                condition_label=cond.condition_label,
                bulk_average_container=spec["bulkAverageContainer"],
                bulk_deviation_container=spec["bulkDeviationContainer"],
                protein_monomer_average_container=spec[
                    "proteinMonomerAverageContainer"
                ],
                protein_monomer_deviation_container=spec[
                    "proteinMonomerDeviationContainer"
                ],
                translation_aa_supply=spec["translation_aa_supply"],
            )
        )

    # Build translation_supply_rate dict (first occurrence per nutrient)
    translation_supply_rate = {}
    for cond in inp.conditions:
        nutrients = cond.nutrients
        if nutrients not in translation_supply_rate:
            spec = working_specs[cond.condition_label]
            translation_supply_rate[nutrients] = spec["translation_aa_supply"]

    return FitConditionOutput(
        condition_outputs=condition_outputs,
        translation_supply_rate=translation_supply_rate,
    )
