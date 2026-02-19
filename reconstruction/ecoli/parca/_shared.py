"""
Shared utility functions across ParCa pipeline stages.

This module contains:
1. Pure math functions (no sim_data dependency) — used by multiple stages
2. sim_data-reading functions — still take sim_data as a parameter,
   used by expressionConverge and other multi-stage functions.
   These will be decomposed into pure functions in future refactoring.

All functions were originally defined in fit_sim_data_1.py.
"""

import traceback
from typing import Callable

import numpy as np

from ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import parallelization, units
from wholecell.utils.fitting import (
    masses_and_counts_for_homeostatic_target,
    normalize,
)

# Fitting parameters (used by expressionConverge)
FITNESS_THRESHOLD = 1e-9
MAX_FITTING_ITERATIONS = 200
VERBOSE = 1


def totalCountFromMassesAndRatios(totalMass, individualMasses, distribution):
    """
    Determine expected total counts for a group of molecules to achieve
    a total mass with a given distribution of individual molecules.

    Math:
        Total mass = dot(mass, count)
        f = count / Total counts
        Total mass = Total counts * dot(mass, f)
        Total counts = Total mass / dot(mass, f)

    Args:
        totalMass: float with mass units
        individualMasses: array of floats with mass units
        distribution: array of floats, normalized to 1

    Returns:
        float: total counts (does not need to be a whole number)
    """
    assert np.allclose(np.sum(distribution), 1)
    counts = 1 / units.dot(individualMasses, distribution) * totalMass
    return units.strip_empty_units(counts)


def proteinDistributionFrommRNA(
    distribution_mRNA, translation_efficiencies, netLossRate
):
    """
    Compute protein distribution from mRNA distribution at steady state.

    dP_i / dt = k * M_i * e_i - P_i * Loss_i
    At steady state: P_i = k * M_i * e_i / Loss_i
    Normalized over all i.

    Args:
        distribution_mRNA: array of floats, normalized to 1
        translation_efficiencies: array of floats, normalized to 1
        netLossRate: array of floats with units of 1/time

    Returns:
        array of floats for protein distribution, normalized to 1
    """
    assert np.allclose(np.sum(distribution_mRNA), 1)
    assert np.allclose(np.sum(translation_efficiencies), 1)
    distributionUnnormed = (
        1 / netLossRate * distribution_mRNA * translation_efficiencies
    )
    distributionNormed = distributionUnnormed / units.sum(distributionUnnormed)
    distributionNormed.normalize()
    distributionNormed.checkNoUnit()

    return distributionNormed.asNumber()


def mRNADistributionFromProtein(
    distribution_protein, translation_efficiencies, netLossRate
):
    """
    Compute mRNA distribution from protein distribution at steady state.

    dP_i / dt = k * M_i * e_i - P_i * Loss_i
    At steady state: M_i = Loss_i * P_i / (k * e_i)
    Normalized over all i.

    Args:
        distribution_protein: array of floats, normalized to 1
        translation_efficiencies: array of floats, normalized to 1
        netLossRate: array of floats with units of 1/time

    Returns:
        array of floats for mRNA distribution, normalized to 1
    """
    assert np.allclose(np.sum(distribution_protein), 1)
    distributionUnnormed = netLossRate * distribution_protein / translation_efficiencies
    distributionNormed = distributionUnnormed / units.sum(distributionUnnormed)
    distributionNormed.normalize()
    distributionNormed.checkNoUnit()

    return distributionNormed.asNumber()


def calculateMinPolymerizingEnzymeByProductDistribution(
    productLengths, elongationRates, netLossRate, productCounts
):
    """
    Compute the number of ribosomes required to maintain steady state.

    R = sum over i ((L_i / e_r) * k_loss_i * P_i)

    Args:
        productLengths: array with units of amino_acids
        elongationRates: array with units of amino_acid/time
        netLossRate: array with units of 1/time
        productCounts: array of floats

    Returns:
        float with dimensionless units
    """
    nPolymerizingEnzymeNeeded = units.sum(
        productLengths / elongationRates * netLossRate * productCounts
    )
    return nPolymerizingEnzymeNeeded


def calculateMinPolymerizingEnzymeByProductDistributionRNA(
    productLengths, elongationRates, netLossRate
):
    """
    Compute the number of RNA polymerases required to maintain steady state of mRNA.

    RNAp = sum over i ((L_i / e_r) * k_loss_i)

    Args:
        productLengths: array with units of nucleotides
        elongationRates: array with units of nucleotide/time
        netLossRate: array with units of 1/time

    Returns:
        float with dimensionless units
    """
    nPolymerizingEnzymeNeeded = units.sum(
        productLengths / elongationRates * netLossRate
    )
    return nPolymerizingEnzymeNeeded


def netLossRateFromDilutionAndDegradationProtein(doublingTime, degradationRates):
    """
    Compute total loss rate (degradation + dilution) for proteins.

    Args:
        doublingTime: float with units of time
        degradationRates: array with units of 1/time

    Returns:
        array with units of 1/time
    """
    return np.log(2) / doublingTime + degradationRates


def netLossRateFromDilutionAndDegradationRNA(
    doublingTime, totalEndoRnaseCountsCapacity, Km, rnaConc, countsToMolar
):
    """
    Compute total loss rate (degradation + dilution) for RNA using
    Michaelis-Menten kinetics with competitive inhibition.

    Args:
        doublingTime: float with units of time
        totalEndoRnaseCountsCapacity: float with units of 1/time
        Km: array with units of mol/volume
        rnaConc: array with units of mol/volume
        countsToMolar: float with units of mol/volume

    Returns:
        array with units of 1/time
    """
    fracSaturated = rnaConc / Km / (1 + units.sum(rnaConc / Km))
    rnaCounts = (1 / countsToMolar) * rnaConc

    return (np.log(2) / doublingTime) * rnaCounts + (
        totalEndoRnaseCountsCapacity * fracSaturated
    )


def netLossRateFromDilutionAndDegradationRNALinear(
    doublingTime, degradationRates, rnaCounts
):
    """
    Compute total loss rate (degradation + dilution) for RNA, linear model.

    Args:
        doublingTime: float with units of time
        degradationRates: array with units of 1/time
        rnaCounts: array of floats

    Returns:
        array with units of 1/time
    """
    return (np.log(2) / doublingTime + degradationRates) * rnaCounts


def rescale_mass_for_soluble_metabolites(
    bulk_container,
    target_molecule_ids,
    target_molecule_concentrations,
    molecular_weights,
    non_small_molecule_initial_cell_mass,
    avg_cell_to_initial_cell_conversion_factor,
    cell_density,
    n_avogadro,
):
    """
    Adjust cell mass to accommodate target small molecule concentrations.

    Pure version of rescaleMassForSolubleMetabolites — takes pre-extracted
    data instead of sim_data.

    Args:
        bulk_container: numpy structured array with 'id' and 'count' columns
            (will be modified in place — caller should pass a copy if needed)
        target_molecule_ids: sorted list of molecule ID strings
        target_molecule_concentrations: units array of concentrations (mol/L)
        molecular_weights: units array of molecular weights (g/mol)
        non_small_molecule_initial_cell_mass: mass of protein + RNA + DNA
            divided by avg_cell_to_initial_cell_conversion_factor
        avg_cell_to_initial_cell_conversion_factor: float
        cell_density: cell density with units
        n_avogadro: Avogadro's number with units

    Returns:
        (new_avg_cell_dry_mass_init, fit_avg_soluble_target_mol_mass)
    """
    masses_to_add, counts_to_add = masses_and_counts_for_homeostatic_target(
        non_small_molecule_initial_cell_mass,
        target_molecule_concentrations,
        molecular_weights,
        cell_density,
        n_avogadro,
    )

    target_molecule_idx = bulk_name_to_idx(target_molecule_ids, bulk_container["id"])
    bulk_container["count"][target_molecule_idx] = counts_to_add

    # Remove water from dry mass calculation
    water_idx = target_molecule_ids.index("WATER[c]")
    small_molecule_dry_mass = units.hstack(
        (masses_to_add[:water_idx], masses_to_add[water_idx + 1:])
    )

    new_avg_cell_dry_mass_init = (
        non_small_molecule_initial_cell_mass + units.sum(small_molecule_dry_mass)
    )
    fit_avg_soluble_target_mol_mass = (
        units.sum(small_molecule_dry_mass)
        * avg_cell_to_initial_cell_conversion_factor
    )

    return new_avg_cell_dry_mass_init, fit_avg_soluble_target_mol_mass


# ============================================================================
# sim_data-reading functions used by multiple stages
# ============================================================================
# These functions still take sim_data as a parameter.  They are shared
# across stages 3 (basal_specs), 4 (tf_condition_specs), and 5
# (fit_condition).  Future refactoring will decompose them into pure
# functions with explicit inputs.


def totalCountIdDistributionRNA(sim_data, expression, doubling_time):
    """
    Calculate total RNA counts, IDs, and distribution from expression.

    Returns:
        (total_count_RNA, ids_rnas, distribution_RNA)
    """
    transcription = sim_data.process.transcription
    ids_rnas = transcription.rna_data["id"]
    total_mass_RNA = (
        sim_data.mass.get_component_masses(doubling_time)["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    mws = transcription.rna_data["mw"]
    is_rRNA = transcription.rna_data["is_rRNA"]
    is_tRNA = transcription.rna_data["is_tRNA"]
    mws[is_rRNA] = transcription.rna_data["rRNA_mw"][is_rRNA]
    mws[is_tRNA] = transcription.rna_data["tRNA_mw"][is_tRNA]
    individual_masses_RNA = mws / sim_data.constants.n_avogadro

    distribution_RNA = normalize(expression)

    total_count_RNA = totalCountFromMassesAndRatios(
        total_mass_RNA, individual_masses_RNA, distribution_RNA
    )

    return total_count_RNA, ids_rnas, distribution_RNA


def totalCountIdDistributionProtein(sim_data, expression, doubling_time):
    """
    Calculate total protein counts, IDs, and distribution from expression.

    Returns:
        (total_count_protein, ids_protein, distribution_protein)
    """
    ids_protein = sim_data.process.translation.monomer_data["id"]
    total_mass_protein = (
        sim_data.mass.get_component_masses(doubling_time)["proteinMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    individual_masses_protein = (
        sim_data.process.translation.monomer_data["mw"]
        / sim_data.constants.n_avogadro
    )

    mRNA_cistron_expression = (
        sim_data.process.transcription.cistron_tu_mapping_matrix.dot(expression)[
            sim_data.process.transcription.cistron_data["is_mRNA"]
        ]
    )
    distribution_transcripts_by_protein = normalize(
        sim_data.relation.monomer_to_mRNA_cistron_mapping().dot(
            mRNA_cistron_expression
        )
    )

    translation_efficiencies_by_protein = normalize(
        sim_data.process.translation.translation_efficiencies_by_monomer
    )
    degradationRates = sim_data.process.translation.monomer_data["deg_rate"]

    netLossRate_protein = netLossRateFromDilutionAndDegradationProtein(
        doubling_time, degradationRates
    )

    distribution_protein = proteinDistributionFrommRNA(
        distribution_transcripts_by_protein,
        translation_efficiencies_by_protein,
        netLossRate_protein,
    )

    total_count_protein = totalCountFromMassesAndRatios(
        total_mass_protein, individual_masses_protein, distribution_protein
    )

    return total_count_protein, ids_protein, distribution_protein


def setInitialRnaExpression(sim_data, expression, doubling_time):
    """
    Set initial RNA expression based on mass fractions and distributions.

    For rRNA the counts are set based on mass, for tRNA based on mass
    and Dong 1996 data, and for mRNA based on mass and relative abundance.

    Returns:
        expression (array) - adjusted RNA expression, normalized to 1
    """
    n_avogadro = sim_data.constants.n_avogadro
    transcription = sim_data.process.transcription
    cistron_data = transcription.cistron_data
    rna_data = transcription.rna_data
    get_average_copy_number = sim_data.process.replication.get_average_copy_number
    rna_mw = rna_data["mw"]
    rna_rRNA_mw = rna_data["rRNA_mw"]
    rna_tRNA_mw = rna_data["tRNA_mw"]
    rna_coord = rna_data["replication_coordinate"]

    is_rRNA = rna_data["is_rRNA"]
    is_tRNA = rna_data["is_tRNA"]
    is_mRNA = rna_data["is_mRNA"]

    all_RNA_ids = rna_data["id"]
    ids_rRNA = all_RNA_ids[is_rRNA]
    ids_mRNA = all_RNA_ids[is_mRNA]
    ids_tRNA = all_RNA_ids[is_tRNA]
    ids_tRNA_cistrons = cistron_data["id"][cistron_data["is_tRNA"]]

    # Get mass fractions
    initial_rna_mass = (
        sim_data.mass.get_component_masses(doubling_time)["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    ppgpp = sim_data.growth_rate_parameters.get_ppGpp_conc(doubling_time)
    rna_fractions = transcription.get_rna_fractions(ppgpp)
    total_mass_rRNA = initial_rna_mass * rna_fractions["rRNA"]
    total_mass_tRNA = initial_rna_mass * rna_fractions["tRNA"]
    total_mass_mRNA = initial_rna_mass * rna_fractions["mRNA"]

    individual_masses_rRNA = rna_rRNA_mw[is_rRNA] / n_avogadro
    individual_masses_tRNA = rna_tRNA_mw[is_tRNA] / n_avogadro
    individual_masses_mRNA = rna_mw[is_mRNA] / n_avogadro

    # rRNA: equal per-copy transcription probabilities
    tau = doubling_time.asNumber(units.min)
    coord_rRNA = rna_coord[is_rRNA]
    n_avg_copy_rRNA = get_average_copy_number(tau, coord_rRNA)
    distribution_rRNA = normalize(n_avg_copy_rRNA)

    total_count_rRNA = totalCountFromMassesAndRatios(
        total_mass_rRNA, individual_masses_rRNA, distribution_rRNA
    )
    counts_rRNA = total_count_rRNA * distribution_rRNA

    # Subtract tRNA mass expressed from rRNA TUs
    tRNA_masses_in_each_rRNA = rna_tRNA_mw[is_rRNA] / n_avogadro
    total_mass_tRNA_in_rRNAs = units.dot(counts_rRNA, tRNA_masses_in_each_rRNA)
    total_mass_tRNA -= total_mass_tRNA_in_rRNAs

    # tRNA distribution from Dong 1996
    tRNA_distribution = sim_data.mass.get_trna_distribution(doubling_time)
    tRNA_id_to_dist = {
        trna_id: dist
        for (trna_id, dist) in zip(
            tRNA_distribution["id"], tRNA_distribution["molar_ratio_to_16SrRNA"]
        )
    }
    distribution_tRNA_cistrons = np.zeros(len(ids_tRNA_cistrons))
    for i, tRNA_id in enumerate(ids_tRNA_cistrons):
        distribution_tRNA_cistrons[i] = tRNA_id_to_dist[tRNA_id]

    tRNA_expressed_from_rRNA_mask = transcription.cistron_tu_mapping_matrix.dot(
        is_rRNA
    )[cistron_data["is_tRNA"]].astype(bool)
    distribution_tRNA_cistrons[tRNA_expressed_from_rRNA_mask] = 0
    distribution_tRNA_cistrons = normalize(distribution_tRNA_cistrons)

    distribution_tRNA_including_transcripts, _ = transcription.fit_trna_expression(
        distribution_tRNA_cistrons
    )

    is_hybrid = rna_data["is_rRNA"][rna_data["includes_tRNA"]]
    distribution_tRNA = distribution_tRNA_including_transcripts[~is_hybrid]
    distribution_tRNA = normalize(distribution_tRNA)

    total_count_tRNA = totalCountFromMassesAndRatios(
        total_mass_tRNA, individual_masses_tRNA, distribution_tRNA
    )
    counts_tRNA = total_count_tRNA * distribution_tRNA

    # mRNA: mass and relative abundances
    distribution_mRNA = normalize(expression[is_mRNA])
    total_count_mRNA = totalCountFromMassesAndRatios(
        total_mass_mRNA, individual_masses_mRNA, distribution_mRNA
    )
    counts_mRNA = total_count_mRNA * distribution_mRNA

    # Build expression container
    rRNA_idx = bulk_name_to_idx(ids_rRNA, all_RNA_ids)
    tRNA_idx = bulk_name_to_idx(ids_tRNA, all_RNA_ids)
    mRNA_idx = bulk_name_to_idx(ids_mRNA, all_RNA_ids)
    rna_expression_container = np.zeros(len(all_RNA_ids), dtype=np.float64)
    rna_expression_container[rRNA_idx] = counts_rRNA
    rna_expression_container[tRNA_idx] = counts_tRNA
    rna_expression_container[mRNA_idx] = counts_mRNA

    expression = normalize(rna_expression_container)

    return expression


def createBulkContainer(sim_data, expression, doubling_time):
    """
    Create a bulk container tracking counts of all bulk molecules.

    Returns:
        bulkContainer (np.ndarray) - structured array with 'id' and 'count'
    """
    total_count_RNA, ids_rnas, distribution_RNA = totalCountIdDistributionRNA(
        sim_data, expression, doubling_time
    )
    total_count_protein, ids_protein, distribution_protein = (
        totalCountIdDistributionProtein(sim_data, expression, doubling_time)
    )

    ids_molecules = sim_data.internal_state.bulk_molecules.bulk_data["id"]

    bulkContainer = np.array(
        [mol_data for mol_data in zip(ids_molecules, np.zeros(len(ids_molecules)))],
        dtype=[("id", ids_molecules.dtype), ("count", np.float64)],
    )

    counts_RNA = total_count_RNA * distribution_RNA
    rna_idx = bulk_name_to_idx(ids_rnas, bulkContainer["id"])
    bulkContainer["count"][rna_idx] = counts_RNA

    counts_protein = total_count_protein * distribution_protein
    protein_idx = bulk_name_to_idx(ids_protein, bulkContainer["id"])
    bulkContainer["count"][protein_idx] = counts_protein

    return bulkContainer


def setRibosomeCountsConstrainedByPhysiology(
    sim_data, bulkContainer, doubling_time, variable_elongation_translation
):
    """
    Set counts of ribosomal protein subunits based on three constraints:
    (1) Expected protein distribution doubles in one cell cycle
    (2) Measured rRNA mass fractions
    (3) Expected ribosomal protein subunit counts based on RNA expression data

    Modifies bulkContainer in place.
    """
    active_fraction = sim_data.growth_rate_parameters.get_fraction_active_ribosome(
        doubling_time
    )

    ribosome_30S_subunits = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s30_full_complex
    )["subunitIds"]
    ribosome_50S_subunits = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s50_full_complex
    )["subunitIds"]
    ribosome_30S_stoich = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s30_full_complex
    )["subunitStoich"]
    ribosome_50S_stoich = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.s50_full_complex
    )["subunitStoich"]

    monomer_ids = set(sim_data.process.translation.monomer_data["id"])

    def remove_rRNA(subunit_ids, subunit_stoich):
        is_protein = np.array(
            [(subunit_id in monomer_ids) for subunit_id in subunit_ids]
        )
        return (subunit_ids[is_protein], subunit_stoich[is_protein])

    ribosome_30S_subunits, ribosome_30S_stoich = remove_rRNA(
        ribosome_30S_subunits, ribosome_30S_stoich
    )
    ribosome_50S_subunits, ribosome_50S_stoich = remove_rRNA(
        ribosome_50S_subunits, ribosome_50S_stoich
    )

    # CONSTRAINT 1: protein distribution doubling
    proteinLengths = units.sum(
        sim_data.process.translation.monomer_data["aa_counts"], axis=1
    )
    proteinDegradationRates = sim_data.process.translation.monomer_data["deg_rate"]
    protein_idx = bulk_name_to_idx(
        sim_data.process.translation.monomer_data["id"], bulkContainer["id"]
    )
    proteinCounts = counts(bulkContainer, protein_idx)

    netLossRate_protein = netLossRateFromDilutionAndDegradationProtein(
        doubling_time, proteinDegradationRates
    )

    elongation_rates = sim_data.process.translation.make_elongation_rates(
        None,
        sim_data.growth_rate_parameters.get_ribosome_elongation_rate(
            doubling_time
        ).asNumber(units.aa / units.s),
        1,
        variable_elongation_translation,
    )

    nRibosomesNeeded = np.ceil(
        calculateMinPolymerizingEnzymeByProductDistribution(
            proteinLengths, elongation_rates, netLossRate_protein, proteinCounts
        ).asNumber(units.aa / units.s)
        / active_fraction
    )

    constraint1_ribosome30SCounts = nRibosomesNeeded * ribosome_30S_stoich
    constraint1_ribosome50SCounts = nRibosomesNeeded * ribosome_50S_stoich

    # CONSTRAINT 2: measured rRNA mass fraction
    rna_data = sim_data.process.transcription.rna_data
    rrna_idx = bulk_name_to_idx(
        rna_data["id"][rna_data["is_rRNA"]], bulkContainer["id"]
    )
    rRNA_tu_counts = counts(bulkContainer, rrna_idx)
    rRNA_cistron_counts = (
        sim_data.process.transcription.rRNA_cistron_tu_mapping_matrix.dot(
            rRNA_tu_counts
        )
    )
    rRNA_cistron_indexes = np.where(
        sim_data.process.transcription.cistron_data["is_rRNA"]
    )[0]
    rRNA_23S_counts = rRNA_cistron_counts[
        sim_data.process.transcription.cistron_data["is_23S_rRNA"][rRNA_cistron_indexes]
    ]
    rRNA_16S_counts = rRNA_cistron_counts[
        sim_data.process.transcription.cistron_data["is_16S_rRNA"][rRNA_cistron_indexes]
    ]
    rRNA_5S_counts = rRNA_cistron_counts[
        sim_data.process.transcription.cistron_data["is_5S_rRNA"][rRNA_cistron_indexes]
    ]

    massFracPredicted_30SCount = rRNA_16S_counts.sum()
    massFracPredicted_50SCount = min(rRNA_23S_counts.sum(), rRNA_5S_counts.sum())

    constraint2_ribosome30SCounts = massFracPredicted_30SCount * ribosome_30S_stoich
    constraint2_ribosome50SCounts = massFracPredicted_50SCount * ribosome_50S_stoich

    # CONSTRAINT 3: expression-based counts (already in bulkContainer)
    ribosome_30S_idx = bulk_name_to_idx(ribosome_30S_subunits, bulkContainer["id"])
    ribosome30SCounts = counts(bulkContainer, ribosome_30S_idx)
    ribosome_50S_idx = bulk_name_to_idx(ribosome_50S_subunits, bulkContainer["id"])
    ribosome50SCounts = counts(bulkContainer, ribosome_50S_idx)

    # Set to max of all constraints
    bulkContainer["count"][ribosome_30S_idx] = np.fmax(
        np.fmax(ribosome30SCounts, constraint1_ribosome30SCounts),
        constraint2_ribosome30SCounts,
    )
    bulkContainer["count"][ribosome_50S_idx] = np.fmax(
        np.fmax(ribosome50SCounts, constraint1_ribosome50SCounts),
        constraint2_ribosome50SCounts,
    )


def setRNAPCountsConstrainedByPhysiology(
    sim_data,
    bulkContainer,
    doubling_time,
    avgCellDryMassInit,
    variable_elongation_transcription,
    Km=None,
):
    """
    Set counts of RNA polymerase based on physiological constraints.

    Modifies bulkContainer in place.
    """
    rnaLengths = units.sum(
        sim_data.process.transcription.rna_data["counts_ACGU"], axis=1
    )

    rnaLossRate = None
    rna_idx = bulk_name_to_idx(
        sim_data.process.transcription.rna_data["id"], bulkContainer["id"]
    )

    if Km is None:
        rnaLossRate = netLossRateFromDilutionAndDegradationRNALinear(
            doubling_time,
            sim_data.process.transcription.rna_data["deg_rate"],
            counts(bulkContainer, rna_idx),
        )
    else:
        cellDensity = sim_data.constants.cell_density
        cellVolume = (
            avgCellDryMassInit / cellDensity / sim_data.mass.cell_dry_mass_fraction
        )
        countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

        rnaConc = countsToMolar * counts(bulkContainer, rna_idx)
        endoRNase_idx = bulk_name_to_idx(
            sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"]
        )
        endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
        kcatEndoRNase = sim_data.process.rna_decay.kcats
        totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

        rnaLossRate = netLossRateFromDilutionAndDegradationRNA(
            doubling_time,
            (1 / countsToMolar) * totalEndoRnaseCapacity,
            Km,
            rnaConc,
            countsToMolar,
        )

    elongation_rates = sim_data.process.transcription.make_elongation_rates(
        None,
        sim_data.growth_rate_parameters.get_rnap_elongation_rate(
            doubling_time
        ).asNumber(units.nt / units.s),
        1,
        variable_elongation_transcription,
    )

    nActiveRnapNeeded = calculateMinPolymerizingEnzymeByProductDistributionRNA(
        rnaLengths, elongation_rates, rnaLossRate
    ).asNumber(units.nt / units.s)

    nRnapsNeeded = np.ceil(
        nActiveRnapNeeded
        / sim_data.growth_rate_parameters.get_fraction_active_rnap(doubling_time)
    )

    rnapIds = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.full_RNAP
    )["subunitIds"]
    rnapStoich = sim_data.process.complexation.get_monomers(
        sim_data.molecule_ids.full_RNAP
    )["subunitStoich"]
    minRnapSubunitCounts = nRnapsNeeded * rnapStoich

    rnap_idx = bulk_name_to_idx(rnapIds, bulkContainer["id"])

    if np.any(minRnapSubunitCounts < 0):
        raise ValueError("RNAP protein counts must be positive.")

    bulkContainer["count"][rnap_idx] = minRnapSubunitCounts


def fitExpression(sim_data, bulkContainer, doubling_time, avgCellDryMassInit, Km=None):
    """
    Determine expression and synthesis probabilities for RNA molecules.

    Modifies bulkContainer counts of RNA and proteins.

    Returns:
        (expression, synth_prob, fit_cistron_expression, cistron_expression_res)
    """
    transcription = sim_data.process.transcription
    translation = sim_data.process.translation
    translation_efficiencies_by_protein = normalize(
        translation.translation_efficiencies_by_monomer
    )
    degradation_rates_protein = translation.monomer_data["deg_rate"]
    net_loss_rate_protein = netLossRateFromDilutionAndDegradationProtein(
        doubling_time, degradation_rates_protein
    )
    avg_cell_fraction_mass = sim_data.mass.get_component_masses(doubling_time)
    total_mass_RNA = (
        avg_cell_fraction_mass["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    cistron_tu_mapping_matrix = transcription.cistron_tu_mapping_matrix

    rna_idx = bulk_name_to_idx(transcription.rna_data["id"], bulkContainer["id"])
    RNA_counts = counts(bulkContainer, rna_idx)
    rna_expression_container = normalize(RNA_counts)

    mRNA_tu_expression_frac = np.sum(
        rna_expression_container[transcription.rna_data["is_mRNA"]]
    )

    fit_cistron_expression = normalize(cistron_tu_mapping_matrix.dot(RNA_counts))
    mRNA_cistron_expression_frac = fit_cistron_expression[
        transcription.cistron_data["is_mRNA"]
    ].sum()

    protein_idx = bulk_name_to_idx(translation.monomer_data["id"], bulkContainer["id"])
    counts_protein = counts(bulkContainer, protein_idx)
    mRNA_cistron_distribution_per_protein = mRNADistributionFromProtein(
        normalize(counts_protein),
        translation_efficiencies_by_protein,
        net_loss_rate_protein,
    )

    mRNA_cistron_distribution = normalize(
        sim_data.relation.monomer_to_mRNA_cistron_mapping().T.dot(
            mRNA_cistron_distribution_per_protein
        )
    )

    fit_cistron_expression[transcription.cistron_data["is_mRNA"]] = (
        mRNA_cistron_expression_frac * mRNA_cistron_distribution
    )

    fit_tu_expression, cistron_expression_res = transcription.fit_rna_expression(
        fit_cistron_expression
    )
    fit_mRNA_tu_expression = fit_tu_expression[transcription.rna_data["is_mRNA"]]

    rna_expression_container[transcription.rna_data["is_mRNA"]] = (
        mRNA_tu_expression_frac * normalize(fit_mRNA_tu_expression)
    )
    expression = normalize(rna_expression_container)

    # Update RNA counts
    mws = transcription.rna_data["mw"]
    is_rRNA = transcription.rna_data["is_rRNA"]
    is_tRNA = transcription.rna_data["is_tRNA"]
    mws[is_rRNA] = transcription.rna_data["rRNA_mw"][is_rRNA]
    mws[is_tRNA] = transcription.rna_data["tRNA_mw"][is_tRNA]

    n_rnas = totalCountFromMassesAndRatios(
        total_mass_RNA, mws / sim_data.constants.n_avogadro, expression
    )
    bulkContainer["count"][rna_idx] = n_rnas * expression

    if Km is None:
        rnaLossRate = netLossRateFromDilutionAndDegradationRNALinear(
            doubling_time,
            transcription.rna_data["deg_rate"],
            counts(bulkContainer, rna_idx),
        )
    else:
        cellDensity = sim_data.constants.cell_density
        dryMassFraction = sim_data.mass.cell_dry_mass_fraction
        cellVolume = avgCellDryMassInit / cellDensity / dryMassFraction
        countsToMolar = 1 / (sim_data.constants.n_avogadro * cellVolume)

        endoRNase_idx = bulk_name_to_idx(
            sim_data.process.rna_decay.endoRNase_ids, bulkContainer["id"]
        )
        endoRNaseConc = countsToMolar * counts(bulkContainer, endoRNase_idx)
        kcatEndoRNase = sim_data.process.rna_decay.kcats
        totalEndoRnaseCapacity = units.sum(endoRNaseConc * kcatEndoRNase)

        rnaLossRate = netLossRateFromDilutionAndDegradationRNA(
            doubling_time,
            (1 / countsToMolar) * totalEndoRnaseCapacity,
            Km,
            countsToMolar * counts(bulkContainer, rna_idx),
            countsToMolar,
        )

    synth_prob = normalize(rnaLossRate.asNumber(1 / units.min))

    return expression, synth_prob, fit_cistron_expression, cistron_expression_res


def rescaleMassForSolubleMetabolites(sim_data, bulkMolCntr, concDict, doubling_time):
    """
    Adjust the cell's mass to accommodate target small molecule concentrations.

    This is the sim_data-reading wrapper around the pure version.  Used by
    expressionConverge (stages 3, 4).

    Modifies bulkMolCntr in place.

    Returns:
        (newAvgCellDryMassInit, fitAvgSolubleTargetMolMass)
    """
    avgCellFractionMass = sim_data.mass.get_component_masses(doubling_time)

    non_small_molecule_initial_cell_mass = (
        avgCellFractionMass["proteinMass"]
        + avgCellFractionMass["rnaMass"]
        + avgCellFractionMass["dnaMass"]
    ) / sim_data.mass.avg_cell_to_initial_cell_conversion_factor

    molar_units = units.mol / units.L

    targetMoleculeIds = sorted(concDict)
    targetMoleculeConcentrations = molar_units * np.array(
        [concDict[key].asNumber(molar_units) for key in targetMoleculeIds]
    )

    assert np.all(targetMoleculeConcentrations.asNumber(molar_units) > 0), (
        "Homeostatic dFBA objective requires non-zero (positive) concentrations"
    )

    molecular_weights = sim_data.getter.get_masses(targetMoleculeIds)

    return rescale_mass_for_soluble_metabolites(
        bulkMolCntr,
        targetMoleculeIds,
        targetMoleculeConcentrations,
        molecular_weights,
        non_small_molecule_initial_cell_mass,
        sim_data.mass.avg_cell_to_initial_cell_conversion_factor,
        sim_data.constants.cell_density,
        sim_data.constants.n_avogadro,
    )


def expressionConverge(
    sim_data,
    expression,
    concDict,
    doubling_time,
    Km=None,
    conditionKey=None,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Iteratively fit synthesis probabilities for RNA.

    Calculates initial expression based on gene expression data and makes
    adjustments to match physiological constraints for ribosome and RNAP
    counts.  Relies on fitExpression() to converge.

    Returns:
        (expression, synthProb, fit_cistron_expression, avgCellDryMassInit,
         fitAvgSolubleTargetMolMass, bulkContainer, concDict)
    """
    if VERBOSE > 0:
        print(
            f"Fitting RNA synthesis probabilities for condition {conditionKey} ...",
            end="",
        )

    for iteration in range(MAX_FITTING_ITERATIONS):
        if VERBOSE > 1:
            print("Iteration: {}".format(iteration))

        initialExpression = expression.copy()
        expression = setInitialRnaExpression(sim_data, expression, doubling_time)
        bulkContainer = createBulkContainer(sim_data, expression, doubling_time)
        avgCellDryMassInit, fitAvgSolubleTargetMolMass = (
            rescaleMassForSolubleMetabolites(
                sim_data, bulkContainer, concDict, doubling_time
            )
        )

        if not disable_rnapoly_capacity_fitting:
            setRNAPCountsConstrainedByPhysiology(
                sim_data,
                bulkContainer,
                doubling_time,
                avgCellDryMassInit,
                variable_elongation_transcription,
                Km,
            )

        if not disable_ribosome_capacity_fitting:
            setRibosomeCountsConstrainedByPhysiology(
                sim_data, bulkContainer, doubling_time, variable_elongation_translation
            )

        expression, synthProb, fit_cistron_expression, cistron_expression_res = (
            fitExpression(
                sim_data, bulkContainer, doubling_time, avgCellDryMassInit, Km
            )
        )

        degreeOfFit = np.sqrt(np.mean(np.square(initialExpression - expression)))

        if VERBOSE > 1:
            print("degree of fit: {}".format(degreeOfFit))
            print(
                f"Average cistron expression residuals: {np.linalg.norm(cistron_expression_res)}"
            )

        if degreeOfFit < FITNESS_THRESHOLD:
            print("! Fitting converged after {} iterations".format(iteration + 1))
            break

    else:
        raise Exception("Fitting did not converge")

    return (
        expression,
        synthProb,
        fit_cistron_expression,
        avgCellDryMassInit,
        fitAvgSolubleTargetMolMass,
        bulkContainer,
        concDict,
    )


def apply_updates(
    func: Callable[..., dict],
    args: list[tuple],
    labels: list[str],
    dest: dict,
    cpus: int,
):
    """
    Use multiprocessing (if cpus > 1) to apply args to a function to get
    dictionary updates for a destination dictionary.

    Args:
        func: function to call with args
        args: list of args to apply to func
        labels: label for each set of args for exception information
        dest: destination dictionary that will be updated with results
        cpus: number of cpus to use
    """
    if cpus > 1:
        print("Starting {} Parca processes".format(cpus))

        pool = parallelization.pool(cpus)
        results = {
            label: pool.apply_async(func, a)
            for label, a in zip(labels, args)
        }
        pool.close()
        pool.join()

        failed = []
        for label, result in results.items():
            if result.successful():
                dest.update(result.get())
            else:
                try:
                    result.get()
                except Exception:
                    traceback.print_exc()
                    failed.append(label)

        if failed:
            raise RuntimeError(
                "Error(s) raised for {} while using multiple processes".format(
                    ", ".join(failed)
                )
            )
        pool = None
        print("End parallel processing")
    else:
        for a in args:
            dest.update(func(*a))


def expressionFromConditionAndFoldChange(transcription, condPerturbations, tfFCs):
    """
    Adjust expression of RNA based on fold changes from basal for a condition.

    Since fold changes are reported for individual RNA cistrons, the changes
    are applied to the basal expression levels of each cistron and the
    resulting vector is mapped back to RNA expression through NNLS.

    Args:
        transcription: sim_data.process.transcription object
        condPerturbations: {cistron_id: fold_change} for genotype perturbations
        tfFCs: {cistron_id: fold_change} for transcription factor fold changes

    Returns:
        (expression, cistron_expression) — both normalized arrays
    """
    cistron_ids = transcription.cistron_data["id"]
    cistron_expression = transcription.fit_cistron_expression["basal"].copy()

    cistron_id_to_index = {
        cistron_id: i for (i, cistron_id) in enumerate(cistron_ids)
    }
    cistron_indexes = []
    cistron_fcs = []

    for cistron_id, fc_value in tfFCs.items():
        if cistron_id in condPerturbations:
            continue
        cistron_indexes.append(cistron_id_to_index[cistron_id])
        cistron_fcs.append(fc_value)

    def apply_fcs_to_expression(expression, indexes, fcs):
        fcs = [
            fc
            for (idx, fc) in sorted(
                zip(indexes, fcs), key=lambda pair: pair[0]
            )
        ]
        indexes = [
            idx
            for (idx, fc) in sorted(
                zip(indexes, fcs), key=lambda pair: pair[0]
            )
        ]

        indexes_bool = np.zeros(len(expression), dtype=bool)
        indexes_bool[indexes] = 1
        fcs = np.array(fcs)
        scaleTheRestBy = (1.0 - (expression[indexes] * fcs).sum()) / (
            1.0 - (expression[indexes]).sum()
        )
        expression[indexes_bool] *= fcs
        expression[~indexes_bool] *= scaleTheRestBy

        return expression

    cistron_expression = apply_fcs_to_expression(
        cistron_expression, cistron_indexes, cistron_fcs
    )

    # Map new cistron expression to RNA expression via NNLS
    expression, _ = transcription.fit_rna_expression(cistron_expression)
    expression = normalize(expression)

    # Apply genotype perturbations
    rna_indexes = []
    rna_fcs = []
    cistron_perturbation_indexes = []
    cistron_perturbation_values = []

    for cistron_id, perturbation_value in condPerturbations.items():
        rna_indexes_with_cistron = transcription.cistron_id_to_rna_indexes(
            cistron_id
        )
        rna_indexes.extend(rna_indexes_with_cistron)
        rna_fcs.extend([perturbation_value] * len(rna_indexes_with_cistron))
        cistron_perturbation_indexes.append(cistron_id_to_index[cistron_id])
        cistron_perturbation_values.append(perturbation_value)

    expression = apply_fcs_to_expression(expression, rna_indexes, rna_fcs)
    cistron_expression = apply_fcs_to_expression(
        cistron_expression,
        cistron_perturbation_indexes,
        cistron_perturbation_values,
    )

    return expression, cistron_expression
