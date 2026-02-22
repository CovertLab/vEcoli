"""
Process-bigraph Step classes for the ParCa pipeline.

Each of the 9 ParCa stages is wrapped as ``process_bigraph.Step`` subclasses.
Stages 2 and 8 (pure) are decomposed into Extract → Pure Compute → Merge
triplets so that the pure compute Step has only explicit typed ports with
no ``parca_state`` dependency.

No Step declares ports named ``sim_data`` or ``cell_specs``.  Coupled stages
pass pipeline state through a registered ``'parca_state'`` type.  Pure stages
exchange data through named ``'overwrite'``-typed ports.

Sequential ordering is achieved via separate input/output store paths:
stage N reads from ``state_<N-1>`` and writes to ``state_<N>``, creating
a natural DAG that the Composite resolves without cycles.

Usage::

    from reconstruction.ecoli.parca.composite import run_parca
    sim_data = run_parca(raw_data, cpus=4, debug=True)
"""

import time

from process_bigraph import Step

from reconstruction.ecoli.parca.parca_types import ParcaState
from reconstruction.ecoli.parca._types import (
    InputAdjustmentsInput,
    InputAdjustmentsOutput,
    SetConditionsInput,
    SetConditionsOutput,
)
from reconstruction.ecoli.parca.stage_02_input_adjustments import (
    extract_input as _extract_02,
    compute_input_adjustments,
    merge_output as _merge_02,
)
from reconstruction.ecoli.parca.stage_03_basal_specs import (
    extract_input as _extract_03,
    compute_basal_specs,
    merge_output as _merge_03,
)
from reconstruction.ecoli.parca.stage_04_tf_condition_specs import (
    extract_input as _extract_04,
    compute_tf_condition_specs,
    merge_output as _merge_04,
)
from reconstruction.ecoli.parca.stage_05_fit_condition import (
    extract_input as _extract_05,
    compute_fit_condition,
    merge_output as _merge_05,
)
from reconstruction.ecoli.parca.stage_06_promoter_binding import (
    extract_input as _extract_06,
    compute_promoter_binding,
    merge_output as _merge_06,
)
from reconstruction.ecoli.parca.stage_07_adjust_promoters import (
    extract_input as _extract_07,
    compute_adjust_promoters,
    merge_output as _merge_07,
)
from reconstruction.ecoli.parca.stage_08_set_conditions import (
    extract_input as _extract_08,
    compute_set_conditions,
    merge_output as _merge_08,
)
from reconstruction.ecoli.parca.stage_09_final_adjustments import (
    extract_input as _extract_09,
    compute_final_adjustments,
    merge_output as _merge_09,
)


# ---------------------------------------------------------------------------
# Stage 1: Initialize (COUPLED)
# ---------------------------------------------------------------------------

class InitializeStep(Step):
    """Stage 1: Initialize sim_data from raw_data."""

    config_schema = {
        'basal_expression_condition': {
            '_type': 'string',
            '_default': 'M9 Glucose minus AAs',
        },
    }

    def inputs(self):
        return {
            'state': 'parca_state',
            'raw_data': 'overwrite',
        }

    def outputs(self):
        return {
            'state': 'parca_state',
        }

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        sim_data.initialize(
            raw_data=state['raw_data'],
            basal_expression_condition=self.config.get(
                'basal_expression_condition', 'M9 Glucose minus AAs'),
        )
        print(f"  Stage 1 (initialize) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data,
                                cell_specs=parca_state.cell_specs),
        }


# ---------------------------------------------------------------------------
# Stage 2: Input Adjustments — Extract / Pure / Merge
# ---------------------------------------------------------------------------

# Port schema shared by Extract (outputs) and InputAdjustmentsStep (inputs)
_STAGE_02_INPUT_PORTS = {
    'monomer_ids': 'overwrite',
    'translation_efficiencies': 'overwrite',
    'translation_eff_adjustments': 'overwrite',
    'balanced_translation_groups': 'overwrite',
    'rna_ids': 'overwrite',
    'cistron_ids': 'overwrite',
    'basal_rna_expression': 'overwrite',
    'rna_expression_adjustments': 'overwrite',
    'cistron_id_to_rna_indexes': 'overwrite',
    'rna_deg_rates': 'overwrite',
    'cistron_deg_rates': 'overwrite',
    'rna_deg_rate_adjustments': 'overwrite',
    'protein_deg_rates': 'overwrite',
    'protein_deg_rate_adjustments': 'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
}

# Port schema shared by InputAdjustmentsStep (outputs) and Merge (inputs)
_STAGE_02_OUTPUT_PORTS = {
    'translation_efficiencies': 'overwrite',
    'basal_rna_expression': 'overwrite',
    'rna_deg_rates': 'overwrite',
    'cistron_deg_rates': 'overwrite',
    'protein_deg_rates': 'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
}


class ExtractForStage2Step(Step):
    """Helper: extract fields from parca_state for the pure Stage 2 step."""

    config_schema = {
        'debug': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return dict(_STAGE_02_INPUT_PORTS)

    def update(self, state):
        parca_state = state['state']
        inp = _extract_02(parca_state.sim_data, parca_state.cell_specs,
                          debug=self.config.get('debug', False))
        return {
            'monomer_ids': inp.monomer_ids,
            'translation_efficiencies': inp.translation_efficiencies,
            'translation_eff_adjustments': inp.translation_eff_adjustments,
            'balanced_translation_groups': inp.balanced_translation_groups,
            'rna_ids': inp.rna_ids,
            'cistron_ids': inp.cistron_ids,
            'basal_rna_expression': inp.basal_rna_expression,
            'rna_expression_adjustments': inp.rna_expression_adjustments,
            'cistron_id_to_rna_indexes': inp.cistron_id_to_rna_indexes,
            'rna_deg_rates': inp.rna_deg_rates,
            'cistron_deg_rates': inp.cistron_deg_rates,
            'rna_deg_rate_adjustments': inp.rna_deg_rate_adjustments,
            'protein_deg_rates': inp.protein_deg_rates,
            'protein_deg_rate_adjustments': inp.protein_deg_rate_adjustments,
            'tf_to_active_inactive_conditions': inp.tf_to_active_inactive_conditions,
        }


class InputAdjustmentsStep(Step):
    """Stage 2: PURE — every field is an explicit typed port.

    No parca_state in inputs or outputs.  Operates entirely on
    individually-named data ports extracted by ExtractForStage2Step.
    """

    config_schema = {
        'debug': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return dict(_STAGE_02_INPUT_PORTS)

    def outputs(self):
        return dict(_STAGE_02_OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()
        inp = InputAdjustmentsInput(
            debug=self.config.get('debug', False),
            monomer_ids=state['monomer_ids'],
            translation_efficiencies=state['translation_efficiencies'],
            translation_eff_adjustments=state['translation_eff_adjustments'],
            balanced_translation_groups=state['balanced_translation_groups'],
            rna_ids=state['rna_ids'],
            cistron_ids=state['cistron_ids'],
            basal_rna_expression=state['basal_rna_expression'],
            rna_expression_adjustments=state['rna_expression_adjustments'],
            cistron_id_to_rna_indexes=state['cistron_id_to_rna_indexes'],
            rna_deg_rates=state['rna_deg_rates'],
            cistron_deg_rates=state['cistron_deg_rates'],
            rna_deg_rate_adjustments=state['rna_deg_rate_adjustments'],
            protein_deg_rates=state['protein_deg_rates'],
            protein_deg_rate_adjustments=state['protein_deg_rate_adjustments'],
            tf_to_active_inactive_conditions=state['tf_to_active_inactive_conditions'],
        )
        out = compute_input_adjustments(inp)
        print(f"  Stage 2 (input_adjustments) completed in {time.time() - t0:.1f}s")
        return {
            'translation_efficiencies': out.translation_efficiencies,
            'basal_rna_expression': out.basal_rna_expression,
            'rna_deg_rates': out.rna_deg_rates,
            'cistron_deg_rates': out.cistron_deg_rates,
            'protein_deg_rates': out.protein_deg_rates,
            'tf_to_active_inactive_conditions': out.tf_to_active_inactive_conditions,
        }


class MergeAfterStage2Step(Step):
    """Helper: merge Stage 2 outputs back into parca_state."""

    def inputs(self):
        return {
            'state': 'parca_state',
            **_STAGE_02_OUTPUT_PORTS,
        }

    def outputs(self):
        return {'state': 'parca_state'}

    def update(self, state):
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        out = InputAdjustmentsOutput(
            translation_efficiencies=state['translation_efficiencies'],
            basal_rna_expression=state['basal_rna_expression'],
            rna_deg_rates=state['rna_deg_rates'],
            cistron_deg_rates=state['cistron_deg_rates'],
            protein_deg_rates=state['protein_deg_rates'],
            tf_to_active_inactive_conditions=state['tf_to_active_inactive_conditions'],
        )
        _merge_02(sim_data, cell_specs, out)

        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
        }


# ---------------------------------------------------------------------------
# Stage 3: Basal Specs (COUPLED)
# ---------------------------------------------------------------------------

class BasalSpecsStep(Step):
    """Stage 3: Build basal cell specifications."""

    config_schema = {
        'variable_elongation_transcription': {
            '_type': 'boolean', '_default': True},
        'variable_elongation_translation': {
            '_type': 'boolean', '_default': False},
        'disable_ribosome_capacity_fitting': {
            '_type': 'boolean', '_default': False},
        'disable_rnapoly_capacity_fitting': {
            '_type': 'boolean', '_default': False},
        'cache_dir': 'string',
    }

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return {
            'state': 'parca_state',
            'conc_dict': 'overwrite',
            'expression': 'overwrite',
            'synth_prob': 'overwrite',
            'fit_cistron_expression': 'overwrite',
            'doubling_time': 'overwrite',
            'avg_cell_dry_mass_init': 'overwrite',
            'fit_avg_soluble_target_mol_mass': 'overwrite',
            'bulk_container': 'overwrite',
        }

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        inp = _extract_03(
            sim_data, cell_specs,
            variable_elongation_transcription=self.config.get(
                'variable_elongation_transcription', True),
            variable_elongation_translation=self.config.get(
                'variable_elongation_translation', False),
            disable_ribosome_capacity_fitting=self.config.get(
                'disable_ribosome_capacity_fitting', False),
            disable_rnapoly_capacity_fitting=self.config.get(
                'disable_rnapoly_capacity_fitting', False),
            cache_dir=self.config.get('cache_dir', ''),
        )
        out = compute_basal_specs(inp)
        _merge_03(sim_data, cell_specs, out)

        print(f"  Stage 3 (basal_specs) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
            'conc_dict': out.conc_dict,
            'expression': out.expression,
            'synth_prob': out.synth_prob,
            'fit_cistron_expression': out.fit_cistron_expression,
            'doubling_time': out.doubling_time,
            'avg_cell_dry_mass_init': out.avg_cell_dry_mass_init,
            'fit_avg_soluble_target_mol_mass': out.fit_avg_soluble_target_mol_mass,
            'bulk_container': out.bulk_container,
        }


# ---------------------------------------------------------------------------
# Stage 4: TF Condition Specs (COUPLED)
# ---------------------------------------------------------------------------

class TfConditionSpecsStep(Step):
    """Stage 4: Build cell specifications for each TF condition."""

    config_schema = {
        'variable_elongation_transcription': {
            '_type': 'boolean', '_default': True},
        'variable_elongation_translation': {
            '_type': 'boolean', '_default': False},
        'disable_ribosome_capacity_fitting': {
            '_type': 'boolean', '_default': False},
        'disable_rnapoly_capacity_fitting': {
            '_type': 'boolean', '_default': False},
        'cpus': {'_type': 'integer', '_default': 1},
    }

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return {
            'state': 'parca_state',
            'condition_outputs': 'overwrite',
        }

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        inp = _extract_04(
            sim_data, cell_specs,
            variable_elongation_transcription=self.config.get(
                'variable_elongation_transcription', True),
            variable_elongation_translation=self.config.get(
                'variable_elongation_translation', False),
            disable_ribosome_capacity_fitting=self.config.get(
                'disable_ribosome_capacity_fitting', False),
            disable_rnapoly_capacity_fitting=self.config.get(
                'disable_rnapoly_capacity_fitting', False),
            cpus=self.config.get('cpus', 1),
        )
        out = compute_tf_condition_specs(inp)
        _merge_04(sim_data, cell_specs, out)

        print(f"  Stage 4 (tf_condition_specs) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
            'condition_outputs': out.condition_outputs,
        }


# ---------------------------------------------------------------------------
# Stage 5: Fit Condition (COUPLED)
# ---------------------------------------------------------------------------

class FitConditionStep(Step):
    """Stage 5: Calculate bulk distributions and translation supply rates."""

    config_schema = {
        'cpus': {'_type': 'integer', '_default': 1},
    }

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return {
            'state': 'parca_state',
            'condition_outputs': 'overwrite',
            'translation_supply_rate': 'overwrite',
        }

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        inp = _extract_05(
            sim_data, cell_specs,
            cpus=self.config.get('cpus', 1),
        )
        out = compute_fit_condition(inp)
        _merge_05(sim_data, cell_specs, out)

        print(f"  Stage 5 (fit_condition) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
            'condition_outputs': out.condition_outputs,
            'translation_supply_rate': out.translation_supply_rate,
        }


# ---------------------------------------------------------------------------
# Stage 6: Promoter Binding (COUPLED)
# ---------------------------------------------------------------------------

class PromoterBindingStep(Step):
    """Stage 6: Fit transcription factor binding probabilities."""

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return {
            'state': 'parca_state',
            'r_vector': 'overwrite',
            'r_columns': 'overwrite',
        }

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        inp = _extract_06(sim_data, cell_specs)
        out = compute_promoter_binding(inp)
        _merge_06(sim_data, cell_specs, out)

        print(f"  Stage 6 (promoter_binding) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
            'r_vector': out.r_vector,
            'r_columns': out.r_columns,
        }


# ---------------------------------------------------------------------------
# Stage 7: Adjust Promoters (COUPLED)
# ---------------------------------------------------------------------------

class AdjustPromotersStep(Step):
    """Stage 7: Adjust ligand concentrations and construct RNAP recruitment."""

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return {
            'state': 'parca_state',
            'basal_prob': 'overwrite',
            'delta_prob': 'overwrite',
        }

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        inp = _extract_07(sim_data, cell_specs)
        out = compute_adjust_promoters(inp)
        _merge_07(sim_data, cell_specs, out)

        print(f"  Stage 7 (adjust_promoters) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
            'basal_prob': out.basal_prob,
            'delta_prob': out.delta_prob,
        }


# ---------------------------------------------------------------------------
# Stage 8: Set Conditions — Extract / Pure / Merge
# ---------------------------------------------------------------------------

# Port schema shared by Extract (outputs) and SetConditionsStep (inputs)
_STAGE_08_INPUT_PORTS = {
    'conditions': 'overwrite',
    'is_mRNA': 'overwrite',
    'is_tRNA': 'overwrite',
    'is_rRNA': 'overwrite',
    'includes_ribosomal_protein': 'overwrite',
    'includes_RNAP': 'overwrite',
}

# Port schema shared by SetConditionsStep (outputs) and Merge (inputs)
_STAGE_08_OUTPUT_PORTS = {
    'rnaSynthProbFraction': 'overwrite',
    'rnapFractionActiveDict': 'overwrite',
    'rnaSynthProbRProtein': 'overwrite',
    'rnaSynthProbRnaPolymerase': 'overwrite',
    'rnaPolymeraseElongationRateDict': 'overwrite',
    'expectedDryMassIncreaseDict': 'overwrite',
    'ribosomeElongationRateDict': 'overwrite',
    'ribosomeFractionActiveDict': 'overwrite',
    'condition_outputs': 'overwrite',
}


class ExtractForStage8Step(Step):
    """Helper: extract fields from parca_state for the pure Stage 8 step."""

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return dict(_STAGE_08_INPUT_PORTS)

    def update(self, state):
        parca_state = state['state']
        inp = _extract_08(parca_state.sim_data, parca_state.cell_specs)
        return {
            'conditions': inp.conditions,
            'is_mRNA': inp.is_mRNA,
            'is_tRNA': inp.is_tRNA,
            'is_rRNA': inp.is_rRNA,
            'includes_ribosomal_protein': inp.includes_ribosomal_protein,
            'includes_RNAP': inp.includes_RNAP,
        }


class SetConditionsStep(Step):
    """Stage 8: PURE — every field is an explicit typed port.

    No parca_state in inputs or outputs.  Operates entirely on
    individually-named data ports extracted by ExtractForStage8Step.
    """

    config_schema = {
        'verbose': {'_type': 'integer', '_default': 1},
    }

    def inputs(self):
        return dict(_STAGE_08_INPUT_PORTS)

    def outputs(self):
        return dict(_STAGE_08_OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()
        inp = SetConditionsInput(
            conditions=state['conditions'],
            is_mRNA=state['is_mRNA'],
            is_tRNA=state['is_tRNA'],
            is_rRNA=state['is_rRNA'],
            includes_ribosomal_protein=state['includes_ribosomal_protein'],
            includes_RNAP=state['includes_RNAP'],
            verbose=self.config.get('verbose', 1),
        )
        out = compute_set_conditions(inp)
        print(f"  Stage 8 (set_conditions) completed in {time.time() - t0:.1f}s")
        return {
            'rnaSynthProbFraction': out.rnaSynthProbFraction,
            'rnapFractionActiveDict': out.rnapFractionActiveDict,
            'rnaSynthProbRProtein': out.rnaSynthProbRProtein,
            'rnaSynthProbRnaPolymerase': out.rnaSynthProbRnaPolymerase,
            'rnaPolymeraseElongationRateDict': out.rnaPolymeraseElongationRateDict,
            'expectedDryMassIncreaseDict': out.expectedDryMassIncreaseDict,
            'ribosomeElongationRateDict': out.ribosomeElongationRateDict,
            'ribosomeFractionActiveDict': out.ribosomeFractionActiveDict,
            'condition_outputs': out.condition_outputs,
        }


class MergeAfterStage8Step(Step):
    """Helper: merge Stage 8 outputs back into parca_state."""

    def inputs(self):
        return {
            'state': 'parca_state',
            **_STAGE_08_OUTPUT_PORTS,
        }

    def outputs(self):
        return {'state': 'parca_state'}

    def update(self, state):
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        out = SetConditionsOutput(
            rnaSynthProbFraction=state['rnaSynthProbFraction'],
            rnapFractionActiveDict=state['rnapFractionActiveDict'],
            rnaSynthProbRProtein=state['rnaSynthProbRProtein'],
            rnaSynthProbRnaPolymerase=state['rnaSynthProbRnaPolymerase'],
            rnaPolymeraseElongationRateDict=state['rnaPolymeraseElongationRateDict'],
            expectedDryMassIncreaseDict=state['expectedDryMassIncreaseDict'],
            ribosomeElongationRateDict=state['ribosomeElongationRateDict'],
            ribosomeFractionActiveDict=state['ribosomeFractionActiveDict'],
            condition_outputs=state['condition_outputs'],
        )
        _merge_08(sim_data, cell_specs, out)

        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
        }


# ---------------------------------------------------------------------------
# Stage 9: Final Adjustments (COUPLED)
# ---------------------------------------------------------------------------

class FinalAdjustmentsStep(Step):
    """Stage 9: Apply final expression adjustments, set amino acid supply
    constants, and configure ppGpp kinetics."""

    def inputs(self):
        return {'state': 'parca_state'}

    def outputs(self):
        return {'state': 'parca_state'}

    def update(self, state):
        t0 = time.time()
        parca_state = state['state']
        sim_data = parca_state.sim_data
        cell_specs = parca_state.cell_specs

        inp = _extract_09(sim_data, cell_specs)
        out = compute_final_adjustments(inp)
        _merge_09(sim_data, cell_specs, out)

        print(f"  Stage 9 (final_adjustments) completed in {time.time() - t0:.1f}s")
        return {
            'state': ParcaState(sim_data=sim_data, cell_specs=cell_specs),
        }


# ---------------------------------------------------------------------------
# Registry of all Step classes (for allocate_core discovery)
# ---------------------------------------------------------------------------

ALL_STEP_CLASSES = {
    'InitializeStep': InitializeStep,
    'ExtractForStage2Step': ExtractForStage2Step,
    'InputAdjustmentsStep': InputAdjustmentsStep,
    'MergeAfterStage2Step': MergeAfterStage2Step,
    'BasalSpecsStep': BasalSpecsStep,
    'TfConditionSpecsStep': TfConditionSpecsStep,
    'FitConditionStep': FitConditionStep,
    'PromoterBindingStep': PromoterBindingStep,
    'AdjustPromotersStep': AdjustPromotersStep,
    'ExtractForStage8Step': ExtractForStage8Step,
    'SetConditionsStep': SetConditionsStep,
    'MergeAfterStage8Step': MergeAfterStage8Step,
    'FinalAdjustmentsStep': FinalAdjustmentsStep,
}
