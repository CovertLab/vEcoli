"""
Pure math functions shared across ParCa pipeline stages.

These functions have NO sim_data dependency and NO side effects.
They operate only on their explicit arguments (numpy arrays, units values).

Imports: numpy, wholecell.utils.units, wholecell.utils.fitting, ecoli.library.schema.
"""

import numpy as np

from ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import units
from wholecell.utils.fitting import masses_and_counts_for_homeostatic_target, normalize


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
