"""
Stage 4: tf_condition_specs — Build cell specifications for each
transcription-factor condition and combined conditions.

Three public functions:
    extract_input(sim_data, cell_specs, **kwargs) -> TfConditionSpecsInput
    compute_tf_condition_specs(inp) -> TfConditionSpecsOutput
    merge_output(sim_data, cell_specs, out)

This stage:
1. For each TF, builds active and inactive conditions by computing
   expression from fold changes and running expressionConverge
   (parallelizable across TFs)
2. Updates sim_data expression dicts for all TF conditions
3. Builds combined conditions (with_aa, etc.) by aggregating fold changes
   from multiple TFs and running expressionConverge

NOTE: compute_tf_condition_specs mutates sim_data_ref because
buildCombinedConditionCellSpecifications depends on expression dicts
that are set during step 2.  merge_output only writes cell_specs.
"""

import numpy as np

from wholecell.utils import parallelization

from reconstruction.ecoli.parca._shared import (
    apply_updates,
    expressionConverge,
    expressionFromConditionAndFoldChange,
)
from reconstruction.ecoli.parca._types import (
    TfConditionSpecsConditionOutput,
    TfConditionSpecsInput,
    TfConditionSpecsOutput,
)


# ============================================================================
# Extract / Merge
# ============================================================================


def extract_input(sim_data, cell_specs, **kwargs) -> TfConditionSpecsInput:
    """Pull configuration from kwargs and pass sim_data as mutable ref."""
    return TfConditionSpecsInput(
        variable_elongation_transcription=kwargs.get(
            "variable_elongation_transcription", True
        ),
        variable_elongation_translation=kwargs.get(
            "variable_elongation_translation", False
        ),
        disable_ribosome_capacity_fitting=kwargs.get(
            "disable_ribosome_capacity_fitting", False
        ),
        disable_rnapoly_capacity_fitting=kwargs.get(
            "disable_rnapoly_capacity_fitting", False
        ),
        cpus=kwargs.get("cpus", 1),
        sim_data_ref=sim_data,
    )


def merge_output(sim_data, cell_specs, out: TfConditionSpecsOutput):
    """Write computed results into cell_specs.

    sim_data expression dict mutations are already applied by
    compute_tf_condition_specs via sim_data_ref.
    """
    for cond_out in out.condition_outputs:
        cell_specs[cond_out.condition_label] = {
            "concDict": cond_out.conc_dict,
            "expression": cond_out.expression,
            "synthProb": cond_out.synth_prob,
            "fit_cistron_expression": cond_out.fit_cistron_expression,
            "doubling_time": cond_out.doubling_time,
            "avgCellDryMassInit": cond_out.avg_cell_dry_mass_init,
            "fitAvgSolubleTargetMolMass": cond_out.fit_avg_soluble_target_mol_mass,
            "bulkContainer": cond_out.bulk_container,
        }
        # cistron_expression is only present for non-basal conditions
        if cond_out.cistron_expression is not None:
            cell_specs[cond_out.condition_label]["cistron_expression"] = (
                cond_out.cistron_expression
            )


# ============================================================================
# Compute
# ============================================================================


def compute_tf_condition_specs(inp: TfConditionSpecsInput) -> TfConditionSpecsOutput:
    """Run the full tf_condition_specs stage.

    This function mutates inp.sim_data_ref as a side effect because
    combined-condition fitting depends on sim_data expression dicts
    that are set after TF-specific fitting.
    """
    sim_data = inp.sim_data_ref
    cpus = parallelization.cpus(inp.cpus)

    # --- Step 1: Build per-TF condition cell specs (parallelizable) ---
    conditions = list(sorted(sim_data.tf_to_active_inactive_conditions))
    args = [
        (
            sim_data,
            tf,
            inp.variable_elongation_transcription,
            inp.variable_elongation_translation,
            inp.disable_ribosome_capacity_fitting,
            inp.disable_rnapoly_capacity_fitting,
        )
        for tf in conditions
    ]
    working_cell_specs = {}
    apply_updates(
        buildTfConditionCellSpecifications,
        args,
        conditions,
        working_cell_specs,
        cpus,
    )

    # --- Step 2: Update sim_data expression dicts from TF conditions ---
    for conditionKey in working_cell_specs:
        sim_data.process.transcription.rna_expression[conditionKey] = (
            working_cell_specs[conditionKey]["expression"]
        )
        sim_data.process.transcription.rna_synth_prob[conditionKey] = (
            working_cell_specs[conditionKey]["synthProb"]
        )
        sim_data.process.transcription.cistron_expression[conditionKey] = (
            working_cell_specs[conditionKey]["cistron_expression"]
        )
        sim_data.process.transcription.fit_cistron_expression[conditionKey] = (
            working_cell_specs[conditionKey]["fit_cistron_expression"]
        )

    # --- Step 3: Build combined condition cell specs ---
    buildCombinedConditionCellSpecifications(
        sim_data,
        working_cell_specs,
        inp.variable_elongation_transcription,
        inp.variable_elongation_translation,
        inp.disable_ribosome_capacity_fitting,
        inp.disable_rnapoly_capacity_fitting,
    )

    # --- Step 4: Collect all condition outputs ---
    condition_outputs = []
    for label, spec in sorted(working_cell_specs.items()):
        condition_outputs.append(
            TfConditionSpecsConditionOutput(
                condition_label=label,
                conc_dict=spec["concDict"],
                expression=spec["expression"],
                synth_prob=spec["synthProb"],
                cistron_expression=spec.get("cistron_expression"),
                fit_cistron_expression=spec["fit_cistron_expression"],
                doubling_time=spec["doubling_time"],
                avg_cell_dry_mass_init=spec["avgCellDryMassInit"],
                fit_avg_soluble_target_mol_mass=spec["fitAvgSolubleTargetMolMass"],
                bulk_container=spec["bulkContainer"],
            )
        )

    return TfConditionSpecsOutput(condition_outputs=condition_outputs)


# ============================================================================
# Sub-functions (ported from fit_sim_data_1.py)
# ============================================================================


def buildTfConditionCellSpecifications(
    sim_data,
    tf,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Create cell specifications for a given transcription factor by fitting
    expression for active and inactive conditions.

    Returns:
        dict {tf__active: {...}, tf__inactive: {...}}
    """
    cell_specs = {}
    for choice in ["__active", "__inactive"]:
        conditionKey = tf + choice
        conditionValue = sim_data.conditions[conditionKey]

        # Get expression from fold changes over basal
        fcData = {}
        if choice == "__active" and conditionValue != sim_data.conditions["basal"]:
            fcData = sim_data.tf_to_fold_change[tf]
        if choice == "__inactive" and conditionValue != sim_data.conditions["basal"]:
            fcDataTmp = sim_data.tf_to_fold_change[tf].copy()
            for key, value in fcDataTmp.items():
                fcData[key] = 1.0 / value
        expression, cistron_expression = expressionFromConditionAndFoldChange(
            sim_data.process.transcription,
            conditionValue["perturbations"],
            fcData,
        )

        # Get metabolite concentrations
        concDict = (
            sim_data.process.metabolism.concentration_updates
            .concentrations_based_on_nutrients(
                media_id=conditionValue["nutrients"]
            )
        )
        concDict.update(
            sim_data.mass.getBiomassAsConcentrations(
                sim_data.condition_to_doubling_time[conditionKey]
            )
        )

        cell_specs[conditionKey] = {
            "concDict": concDict,
            "expression": expression,
            "doubling_time": sim_data.condition_to_doubling_time.get(
                conditionKey, sim_data.condition_to_doubling_time["basal"]
            ),
        }

        # Fit expression
        (
            expression,
            synthProb,
            fit_cistron_expression,
            avgCellDryMassInit,
            fitAvgSolubleTargetMolMass,
            bulkContainer,
            concDict,
        ) = expressionConverge(
            sim_data,
            cell_specs[conditionKey]["expression"],
            cell_specs[conditionKey]["concDict"],
            cell_specs[conditionKey]["doubling_time"],
            sim_data.process.transcription.rna_data["Km_endoRNase"],
            conditionKey=conditionKey,
            variable_elongation_transcription=variable_elongation_transcription,
            variable_elongation_translation=variable_elongation_translation,
            disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
        )

        cell_specs[conditionKey]["expression"] = expression
        cell_specs[conditionKey]["synthProb"] = synthProb
        cell_specs[conditionKey]["cistron_expression"] = cistron_expression
        cell_specs[conditionKey]["fit_cistron_expression"] = fit_cistron_expression
        cell_specs[conditionKey]["avgCellDryMassInit"] = avgCellDryMassInit
        cell_specs[conditionKey]["fitAvgSolubleTargetMolMass"] = (
            fitAvgSolubleTargetMolMass
        )
        cell_specs[conditionKey]["bulkContainer"] = bulkContainer

    return cell_specs


def buildCombinedConditionCellSpecifications(
    sim_data,
    cell_specs,
    variable_elongation_transcription=True,
    variable_elongation_translation=False,
    disable_ribosome_capacity_fitting=False,
    disable_rnapoly_capacity_fitting=False,
):
    """
    Create cell specifications for combined conditions (with_aa, etc.)
    where multiple transcription factors are active simultaneously.

    Modifies cell_specs in place and updates sim_data expression dicts.
    """
    for conditionKey in sim_data.condition_active_tfs:
        if conditionKey == "basal":
            continue

        # Aggregate fold changes from all active/inactive TFs
        fcData = {}
        conditionValue = sim_data.conditions[conditionKey]
        for tf in sim_data.condition_active_tfs[conditionKey]:
            for gene, fc in sim_data.tf_to_fold_change[tf].items():
                fcData[gene] = fcData.get(gene, 1) * fc
        for tf in sim_data.condition_inactive_tfs[conditionKey]:
            for gene, fc in sim_data.tf_to_fold_change[tf].items():
                fcData[gene] = fcData.get(gene, 1) / fc

        expression, cistron_expression = expressionFromConditionAndFoldChange(
            sim_data.process.transcription,
            conditionValue["perturbations"],
            fcData,
        )

        # Get metabolite concentrations
        concDict = (
            sim_data.process.metabolism.concentration_updates
            .concentrations_based_on_nutrients(
                media_id=conditionValue["nutrients"]
            )
        )
        concDict.update(
            sim_data.mass.getBiomassAsConcentrations(
                sim_data.condition_to_doubling_time[conditionKey]
            )
        )

        cell_specs[conditionKey] = {
            "concDict": concDict,
            "expression": expression,
            "doubling_time": sim_data.condition_to_doubling_time.get(
                conditionKey, sim_data.condition_to_doubling_time["basal"]
            ),
        }

        # Fit expression
        (
            expression,
            synthProb,
            fit_cistron_expression,
            avgCellDryMassInit,
            fitAvgSolubleTargetMolMass,
            bulkContainer,
            concDict,
        ) = expressionConverge(
            sim_data,
            cell_specs[conditionKey]["expression"],
            cell_specs[conditionKey]["concDict"],
            cell_specs[conditionKey]["doubling_time"],
            sim_data.process.transcription.rna_data["Km_endoRNase"],
            conditionKey=conditionKey,
            variable_elongation_transcription=variable_elongation_transcription,
            variable_elongation_translation=variable_elongation_translation,
            disable_ribosome_capacity_fitting=disable_ribosome_capacity_fitting,
            disable_rnapoly_capacity_fitting=disable_rnapoly_capacity_fitting,
        )

        cell_specs[conditionKey]["expression"] = expression
        cell_specs[conditionKey]["synthProb"] = synthProb
        cell_specs[conditionKey]["cistron_expression"] = cistron_expression
        cell_specs[conditionKey]["fit_cistron_expression"] = fit_cistron_expression
        cell_specs[conditionKey]["avgCellDryMassInit"] = avgCellDryMassInit
        cell_specs[conditionKey]["fitAvgSolubleTargetMolMass"] = (
            fitAvgSolubleTargetMolMass
        )
        cell_specs[conditionKey]["bulkContainer"] = bulkContainer

        # Update sim_data expression dicts
        sim_data.process.transcription.rna_expression[conditionKey] = (
            cell_specs[conditionKey]["expression"]
        )
        sim_data.process.transcription.rna_synth_prob[conditionKey] = (
            cell_specs[conditionKey]["synthProb"]
        )
        sim_data.process.transcription.cistron_expression[conditionKey] = (
            cell_specs[conditionKey]["cistron_expression"]
        )
        sim_data.process.transcription.fit_cistron_expression[conditionKey] = (
            cell_specs[conditionKey]["fit_cistron_expression"]
        )
