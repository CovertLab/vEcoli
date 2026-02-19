"""
ParCa pipeline stages as pure functions with explicit Input/Output dataclasses.

Pipeline Overview
=================
Stage 1 (initialize) is a constructor in fit_sim_data_1.py — not a module here.

Stages 2-9 each provide three functions:
    extract_input(sim_data, cell_specs, **kwargs) -> StageInput
    compute_*(inp: StageInput) -> StageOutput
    merge_output(sim_data, cell_specs, out: StageOutput)

Stage Registry
==============
Stage  Module                         Purity       Input/Output Types
-----  ------                         ------       ------------------
  2    stage_02_input_adjustments     PURE         InputAdjustmentsInput → InputAdjustmentsOutput
  3    stage_03_basal_specs           COUPLED      BasalSpecsInput (sim_data_ref) → BasalSpecsOutput
  4    stage_04_tf_condition_specs    COUPLED      TfConditionSpecsInput (sim_data_ref) → TfConditionSpecsOutput
  5    stage_05_fit_condition         READ-ONLY    FitConditionInput (sim_data_ref) → FitConditionOutput
  6    stage_06_promoter_binding      COUPLED      PromoterBindingInput (sim_data_ref, cell_specs_ref) → PromoterBindingOutput
  7    stage_07_adjust_promoters      COUPLED      AdjustPromotersInput (sim_data_ref, cell_specs_ref) → AdjustPromotersOutput
  8    stage_08_set_conditions        PURE         SetConditionsInput → SetConditionsOutput
  9    stage_09_final_adjustments     COUPLED      FinalAdjustmentsInput (sim_data_ref, cell_specs_ref) → FinalAdjustmentsOutput

Purity legend:
  PURE      — compute function has no sim_data/cell_specs access; fully testable with synthetic data
  READ-ONLY — compute reads sim_data via ref but does not mutate it
  COUPLED   — compute mutates sim_data via ref (future refactoring target)

Shared Utilities
================
  _math.py    — Pure math functions (no sim_data): distributions, loss rates, mass rescaling
  _fitting.py — sim_data-reading functions: expressionConverge, fitExpression, createBulkContainer
  _shared.py  — Backward-compatible re-export shim (imports from _math and _fitting)
  _types.py   — All Input/Output dataclasses for stages 2-9

External Dependencies
=====================
  parca_promoter_fitting.py — Matrix builders and CVXPY optimization for stages 6 and 7
"""
