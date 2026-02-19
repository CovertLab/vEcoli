"""
Stage 8: set_conditions — Rescale mass for soluble metabolites and populate
condition-specific dictionaries on sim_data.

Three public functions:
    extract_input(sim_data, cell_specs, **kwargs) -> SetConditionsInput
    compute_set_conditions(inp) -> SetConditionsOutput
    merge_output(sim_data, cell_specs, out)

The stage loops over all conditions in cell_specs and:
1. Computes concentration-based mass rescaling for each condition
2. Populates per-nutrient dictionaries for synth prob fractions,
   elongation rates, and active fractions
"""

import numpy as np

from wholecell.utils import units

from reconstruction.ecoli.parca._shared import rescale_mass_for_soluble_metabolites
from reconstruction.ecoli.parca._types import (
    SetConditionsConditionInput,
    SetConditionsConditionOutput,
    SetConditionsInput,
    SetConditionsOutput,
)


# ============================================================================
# Extract / Merge
# ============================================================================


def extract_input(sim_data, cell_specs, **kwargs) -> SetConditionsInput:
    """Pull all data needed for set_conditions from sim_data and cell_specs."""
    transcription = sim_data.process.transcription
    rna_data = transcription.rna_data

    molar_units = units.mol / units.L

    condition_inputs = []
    for condition_label in sorted(cell_specs):
        condition = sim_data.conditions[condition_label]
        nutrients = condition["nutrients"]
        doubling_time = sim_data.condition_to_doubling_time[condition_label]
        spec = cell_specs[condition_label]

        # Pre-compute concentration dict
        conc_dict = sim_data.process.metabolism.concentration_updates.concentrations_based_on_nutrients(
            media_id=nutrients
        )
        conc_dict.update(
            sim_data.mass.getBiomassAsConcentrations(doubling_time)
        )

        target_molecule_ids = sorted(conc_dict)
        target_molecule_concentrations = molar_units * np.array(
            [conc_dict[key].asNumber(molar_units) for key in target_molecule_ids]
        )
        molecular_weights = sim_data.getter.get_masses(target_molecule_ids)

        # Pre-compute mass data for rescaleMassForSolubleMetabolites
        avg_cell_fraction_mass = sim_data.mass.get_component_masses(doubling_time)
        non_small_molecule_initial_cell_mass = (
            avg_cell_fraction_mass["proteinMass"]
            + avg_cell_fraction_mass["rnaMass"]
            + avg_cell_fraction_mass["dnaMass"]
        ) / sim_data.mass.avg_cell_to_initial_cell_conversion_factor

        # Pre-compute growth rate parameters
        grp = sim_data.growth_rate_parameters
        fraction_active_rnap = grp.get_fraction_active_rnap(doubling_time)
        rnap_elongation_rate = grp.get_rnap_elongation_rate(doubling_time)
        ribosome_elongation_rate = grp.get_ribosome_elongation_rate(doubling_time)
        fraction_active_ribosome = grp.get_fraction_active_ribosome(doubling_time)

        condition_inputs.append(
            SetConditionsConditionInput(
                condition_label=condition_label,
                nutrients=nutrients,
                has_perturbations=len(condition["perturbations"]) > 0,
                doubling_time=doubling_time,
                target_molecule_ids=target_molecule_ids,
                target_molecule_concentrations=target_molecule_concentrations,
                molecular_weights=molecular_weights,
                non_small_molecule_initial_cell_mass=non_small_molecule_initial_cell_mass,
                avg_cell_to_initial_cell_conversion_factor=sim_data.mass.avg_cell_to_initial_cell_conversion_factor,
                cell_density=sim_data.constants.cell_density,
                n_avogadro=sim_data.constants.n_avogadro,
                bulk_container=spec["bulkContainer"].copy(),
                avg_cell_dry_mass_init_old=spec["avgCellDryMassInit"],
                rna_synth_prob=transcription.rna_synth_prob[condition_label].copy(),
                fraction_active_rnap=fraction_active_rnap,
                rnap_elongation_rate=rnap_elongation_rate,
                ribosome_elongation_rate=ribosome_elongation_rate,
                fraction_active_ribosome=fraction_active_ribosome,
            )
        )

    return SetConditionsInput(
        conditions=condition_inputs,
        is_mRNA=rna_data["is_mRNA"],
        is_tRNA=rna_data["is_tRNA"],
        is_rRNA=rna_data["is_rRNA"],
        includes_ribosomal_protein=rna_data["includes_ribosomal_protein"],
        includes_RNAP=rna_data["includes_RNAP"],
        verbose=1,
    )


def merge_output(sim_data, cell_specs, out: SetConditionsOutput):
    """Write computed results back into sim_data and cell_specs."""
    sim_data.process.transcription.rnaSynthProbFraction = out.rnaSynthProbFraction
    sim_data.process.transcription.rnapFractionActiveDict = out.rnapFractionActiveDict
    sim_data.process.transcription.rnaSynthProbRProtein = out.rnaSynthProbRProtein
    sim_data.process.transcription.rnaSynthProbRnaPolymerase = (
        out.rnaSynthProbRnaPolymerase
    )
    sim_data.process.transcription.rnaPolymeraseElongationRateDict = (
        out.rnaPolymeraseElongationRateDict
    )
    sim_data.expectedDryMassIncreaseDict = out.expectedDryMassIncreaseDict
    sim_data.process.translation.ribosomeElongationRateDict = (
        out.ribosomeElongationRateDict
    )
    sim_data.process.translation.ribosomeFractionActiveDict = (
        out.ribosomeFractionActiveDict
    )

    for cond_out in out.condition_outputs:
        spec = cell_specs[cond_out.condition_label]
        spec["avgCellDryMassInit"] = cond_out.avg_cell_dry_mass_init
        spec["fitAvgSolublePoolMass"] = cond_out.fit_avg_soluble_pool_mass
        spec["bulkContainer"] = cond_out.bulk_container


# ============================================================================
# Pure sub-functions
# ============================================================================


def compute_synth_prob_fractions(rna_synth_prob, is_mRNA, is_tRNA, is_rRNA):
    """
    Compute the total synthesis probability for mRNA, tRNA, and rRNA.

    Args:
        rna_synth_prob: array of synthesis probabilities per RNA
        is_mRNA: boolean mask for mRNAs
        is_tRNA: boolean mask for tRNAs
        is_rRNA: boolean mask for rRNAs

    Returns:
        dict with keys "mRna", "tRna", "rRna" mapping to float sums
    """
    return {
        "mRna": float(rna_synth_prob[is_mRNA].sum()),
        "tRna": float(rna_synth_prob[is_tRNA].sum()),
        "rRna": float(rna_synth_prob[is_rRNA].sum()),
    }


# ============================================================================
# Main compute function
# ============================================================================


def compute_set_conditions(inp: SetConditionsInput) -> SetConditionsOutput:
    """
    Pure function: loop over conditions, rescale mass, populate dicts.

    No sim_data, no cell_specs, no side effects.
    """
    rnaSynthProbFraction = {}
    rnapFractionActiveDict = {}
    rnaSynthProbRProtein = {}
    rnaSynthProbRnaPolymerase = {}
    rnaPolymeraseElongationRateDict = {}
    expectedDryMassIncreaseDict = {}
    ribosomeElongationRateDict = {}
    ribosomeFractionActiveDict = {}

    condition_outputs = []

    for cond in inp.conditions:
        if inp.verbose > 0:
            print("Updating mass in condition {}".format(cond.condition_label))

        # Rescale mass for soluble metabolites
        avg_cell_dry_mass_init, fit_avg_soluble_pool_mass = (
            rescale_mass_for_soluble_metabolites(
                cond.bulk_container,
                cond.target_molecule_ids,
                cond.target_molecule_concentrations,
                cond.molecular_weights,
                cond.non_small_molecule_initial_cell_mass,
                cond.avg_cell_to_initial_cell_conversion_factor,
                cond.cell_density,
                cond.n_avogadro,
            )
        )

        if inp.verbose > 0:
            print("{} to {}".format(cond.avg_cell_dry_mass_init_old, avg_cell_dry_mass_init))

        condition_outputs.append(
            SetConditionsConditionOutput(
                condition_label=cond.condition_label,
                avg_cell_dry_mass_init=avg_cell_dry_mass_init,
                fit_avg_soluble_pool_mass=fit_avg_soluble_pool_mass,
                bulk_container=cond.bulk_container,
            )
        )

        # Populate per-nutrient dicts (only for conditions without perturbations)
        if not cond.has_perturbations:
            nutrients = cond.nutrients

            if nutrients not in rnaSynthProbFraction:
                rnaSynthProbFraction[nutrients] = compute_synth_prob_fractions(
                    cond.rna_synth_prob, inp.is_mRNA, inp.is_tRNA, inp.is_rRNA
                )

            if nutrients not in rnaSynthProbRProtein:
                rnaSynthProbRProtein[nutrients] = cond.rna_synth_prob[
                    inp.includes_ribosomal_protein
                ]

            if nutrients not in rnaSynthProbRnaPolymerase:
                rnaSynthProbRnaPolymerase[nutrients] = cond.rna_synth_prob[
                    inp.includes_RNAP
                ]

            if nutrients not in rnapFractionActiveDict:
                rnapFractionActiveDict[nutrients] = cond.fraction_active_rnap

            if nutrients not in rnaPolymeraseElongationRateDict:
                rnaPolymeraseElongationRateDict[nutrients] = cond.rnap_elongation_rate

            if nutrients not in expectedDryMassIncreaseDict:
                expectedDryMassIncreaseDict[nutrients] = avg_cell_dry_mass_init

            if nutrients not in ribosomeElongationRateDict:
                ribosomeElongationRateDict[nutrients] = cond.ribosome_elongation_rate

            if nutrients not in ribosomeFractionActiveDict:
                ribosomeFractionActiveDict[nutrients] = cond.fraction_active_ribosome

    return SetConditionsOutput(
        rnaSynthProbFraction=rnaSynthProbFraction,
        rnapFractionActiveDict=rnapFractionActiveDict,
        rnaSynthProbRProtein=rnaSynthProbRProtein,
        rnaSynthProbRnaPolymerase=rnaSynthProbRnaPolymerase,
        rnaPolymeraseElongationRateDict=rnaPolymeraseElongationRateDict,
        expectedDryMassIncreaseDict=expectedDryMassIncreaseDict,
        ribosomeElongationRateDict=ribosomeElongationRateDict,
        ribosomeFractionActiveDict=ribosomeFractionActiveDict,
        condition_outputs=condition_outputs,
    )
