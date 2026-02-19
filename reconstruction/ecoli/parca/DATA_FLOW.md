# ParCa Pipeline Data Flow Reference

This document maps what each stage reads from and writes to `sim_data` and
`cell_specs`.  Use it to answer: "If I change attribute X, which stages
are affected?"

## Pipeline Overview

```
  ┌──────────────────┐
  │  1. initialize   │  (fit_sim_data_1.py — constructs sim_data from raw_data)
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  2. input_adj.   │  PURE — adjusts translation efficiencies, expression, deg rates
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  3. basal_specs  │  COUPLED — fits basal expression, Km, dark ATP
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  4. tf_cond_spec │  COUPLED — fits per-TF condition expression (parallel)
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  5. fit_condition│  READ-ONLY — computes bulk containers + AA supply
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  6. promoter_bdg │  COUPLED — fits promoter binding probabilities
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  7. adjust_prom. │  COUPLED — fits ligand concentrations, RNAP recruitment
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  8. set_conds    │  PURE — rescales masses, sets per-condition dicts
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │  9. final_adj.   │  COUPLED — attenuation, ppGpp, metabolism constants
  └─────────────────-┘
```

## Per-Stage Data Flow

### Stage 2: input_adjustments (PURE)

**kwargs:** `debug`

| Direction | Attribute Path |
|-----------|----------------|
| READ  | `sim_data.process.translation.monomer_data["id"]` |
| READ  | `sim_data.process.translation.translation_efficiencies_by_monomer` |
| READ  | `sim_data.adjustments.translation_efficiencies_adjustments` |
| READ  | `sim_data.adjustments.balanced_translation_efficiencies` |
| READ  | `sim_data.process.transcription.rna_data["id"]` |
| READ  | `sim_data.process.transcription.rna_expression["basal"]` |
| READ  | `sim_data.adjustments.rna_expression_adjustments` |
| READ  | `sim_data.process.transcription.cistron_data["id"]` |
| READ  | `sim_data.process.transcription.cistron_id_to_rna_indexes()` |
| READ  | `sim_data.process.transcription.rna_data.struct_array["deg_rate"]` |
| READ  | `sim_data.process.transcription.cistron_data.struct_array["deg_rate"]` |
| READ  | `sim_data.adjustments.rna_deg_rates_adjustments` |
| READ  | `sim_data.process.translation.monomer_data.struct_array["deg_rate"]` |
| READ  | `sim_data.adjustments.protein_deg_rates_adjustments` |
| READ  | `sim_data.tf_to_active_inactive_conditions` |
| WRITE | `sim_data.process.translation.translation_efficiencies_by_monomer[:]` |
| WRITE | `sim_data.process.transcription.rna_expression["basal"][:]` |
| WRITE | `sim_data.process.transcription.rna_data.struct_array["deg_rate"][:]` |
| WRITE | `sim_data.process.transcription.cistron_data.struct_array["deg_rate"][:]` |
| WRITE | `sim_data.process.translation.monomer_data.struct_array["deg_rate"][:]` |
| WRITE | `sim_data.tf_to_active_inactive_conditions` (debug only — truncated) |

**cell_specs:** not used.

---

### Stage 3: basal_specs (COUPLED)

**kwargs:** `variable_elongation_transcription`, `variable_elongation_translation`,
`disable_ribosome_capacity_fitting`, `disable_rnapoly_capacity_fitting`, `cache_dir`

| Direction | Attribute Path |
|-----------|----------------|
| READ  | Full `sim_data` ref (passed to `expressionConverge`, Km optimization) |
| WRITE | `sim_data.mass.avg_cell_dry_mass_init` |
| WRITE | `sim_data.mass.avg_cell_dry_mass` |
| WRITE | `sim_data.mass.avg_cell_water_mass_init` |
| WRITE | `sim_data.mass.fitAvgSolubleTargetMolMass` |
| WRITE | `sim_data.process.transcription.rna_expression["basal"][:]` |
| WRITE | `sim_data.process.transcription.rna_synth_prob["basal"][:]` |
| WRITE | `sim_data.process.transcription.fit_cistron_expression["basal"]` |
| WRITE | `sim_data.process.transcription.rna_data["Km_endoRNase"]` |
| WRITE | `sim_data.process.transcription.mature_rna_data["Km_endoRNase"]` |
| WRITE | `sim_data.process.rna_decay.Km_first_order_decay` |
| WRITE | `sim_data.process.rna_decay.sensitivity_analysis_*` (alpha, kcat) |
| WRITE | `sim_data.process.rna_decay.stats_fit[*]` (LossKm, ResKm, etc.) |
| WRITE | `sim_data.constants.darkATP` |
| WRITE | `sim_data.process.transcription` (via `set_ppgpp_expression()`) |

| Direction | cell_specs Path |
|-----------|-----------------|
| WRITE | `cell_specs["basal"]["concDict"]` |
| WRITE | `cell_specs["basal"]["expression"]` |
| WRITE | `cell_specs["basal"]["synthProb"]` |
| WRITE | `cell_specs["basal"]["fit_cistron_expression"]` |
| WRITE | `cell_specs["basal"]["doubling_time"]` |
| WRITE | `cell_specs["basal"]["avgCellDryMassInit"]` |
| WRITE | `cell_specs["basal"]["fitAvgSolubleTargetMolMass"]` |
| WRITE | `cell_specs["basal"]["bulkContainer"]` |

---

### Stage 4: tf_condition_specs (COUPLED)

**kwargs:** `variable_elongation_transcription`, `variable_elongation_translation`,
`disable_ribosome_capacity_fitting`, `disable_rnapoly_capacity_fitting`, `cpus`

| Direction | Attribute Path |
|-----------|----------------|
| READ  | Full `sim_data` ref (passed to `expressionConverge`, fold-change logic) |
| WRITE | `sim_data.process.transcription.rna_expression[conditionKey]` (per TF + combined) |
| WRITE | `sim_data.process.transcription.rna_synth_prob[conditionKey]` |
| WRITE | `sim_data.process.transcription.cistron_expression[conditionKey]` |
| WRITE | `sim_data.process.transcription.fit_cistron_expression[conditionKey]` |

| Direction | cell_specs Path |
|-----------|-----------------|
| WRITE | `cell_specs[conditionKey]["concDict"]` |
| WRITE | `cell_specs[conditionKey]["expression"]` |
| WRITE | `cell_specs[conditionKey]["synthProb"]` |
| WRITE | `cell_specs[conditionKey]["fit_cistron_expression"]` |
| WRITE | `cell_specs[conditionKey]["doubling_time"]` |
| WRITE | `cell_specs[conditionKey]["avgCellDryMassInit"]` |
| WRITE | `cell_specs[conditionKey]["fitAvgSolubleTargetMolMass"]` |
| WRITE | `cell_specs[conditionKey]["bulkContainer"]` |
| WRITE | `cell_specs[conditionKey]["cistron_expression"]` (non-basal only) |

---

### Stage 5: fit_condition (READ-ONLY)

**kwargs:** `cpus`

| Direction | Attribute Path |
|-----------|----------------|
| READ  | Full `sim_data` ref (read-only in compute) |
| READ  | `sim_data.conditions[label]["nutrients"]` |
| WRITE | `sim_data.translation_supply_rate[nutrients]` |

| Direction | cell_specs Path |
|-----------|-----------------|
| READ  | `cell_specs[cond]["expression"]` |
| READ  | `cell_specs[cond]["concDict"]` |
| READ  | `cell_specs[cond]["avgCellDryMassInit"]` |
| READ  | `cell_specs[cond]["doubling_time"]` |
| WRITE | `cell_specs[cond]["bulkAverageContainer"]` |
| WRITE | `cell_specs[cond]["bulkDeviationContainer"]` |
| WRITE | `cell_specs[cond]["proteinMonomerAverageContainer"]` |
| WRITE | `cell_specs[cond]["proteinMonomerDeviationContainer"]` |
| WRITE | `cell_specs[cond]["translation_aa_supply"]` |

---

### Stage 6: promoter_binding (COUPLED)

| Direction | Attribute Path |
|-----------|----------------|
| READ  | Full `sim_data` ref |
| READ  | `sim_data.tf_to_active_inactive_conditions` |
| READ  | `sim_data.process.transcription_regulation.tf_to_tf_type` |
| READ  | `sim_data.process.transcription_regulation.active_to_bound` |
| READ  | `sim_data.tf_to_fold_change` |
| READ  | `sim_data.process.transcription.rna_data` |
| READ  | `sim_data.process.transcription_regulation.tf_ids` |
| READ  | `sim_data.condition_to_doubling_time` |
| READ  | `sim_data.process.replication.get_average_copy_number()` |
| READ  | `sim_data.process.equilibrium.get_rev_rate()`, `get_fwd_rate()`, etc. |
| READ  | `sim_data.process.two_component_system.active_to_inactive_tf` |
| READ  | `sim_data.relation.rna_id_to_regulating_tfs` |
| WRITE | `sim_data.pPromoterBound` |
| WRITE | `sim_data.process.transcription.rna_synth_prob[cond]` (all conditions) |

| Direction | cell_specs Path |
|-----------|-----------------|
| READ  | `cell_specs[cond]["avgCellDryMassInit"]` |
| READ  | `cell_specs[cond]["bulkAverageContainer"]` |
| WRITE | `cell_specs["basal"]["r_vector"]` |
| WRITE | `cell_specs["basal"]["r_columns"]` |

---

### Stage 7: adjust_promoters (COUPLED)

| Direction | Attribute Path |
|-----------|----------------|
| READ  | Full `sim_data` ref |
| READ  | `sim_data.tf_to_active_inactive_conditions` |
| READ  | `sim_data.process.transcription_regulation.tf_to_tf_type` |
| READ  | `sim_data.process.transcription_regulation.active_to_bound` |
| READ  | `sim_data.pPromoterBound` (from stage 6) |
| READ  | `sim_data.relation.rna_id_to_regulating_tfs` |
| WRITE | `sim_data.process.transcription_regulation.basal_prob` |
| WRITE | `sim_data.process.transcription_regulation.delta_prob` |
| WRITE | `sim_data.process.metabolism.concentration_updates.molecule_set_amounts[metabolite]` |
| WRITE | `sim_data.process.equilibrium` reverse rates (per 1CS TF) |

| Direction | cell_specs Path |
|-----------|-----------------|
| READ  | `cell_specs[cond]["avgCellDryMassInit"]` |
| READ  | `cell_specs[cond]["bulkAverageContainer"]` |
| READ  | `cell_specs["basal"]["r_vector"]` (from stage 6) |
| READ  | `cell_specs["basal"]["r_columns"]` (from stage 6) |

---

### Stage 8: set_conditions (PURE)

| Direction | Attribute Path |
|-----------|----------------|
| READ  | `sim_data.conditions[label]["nutrients"]`, `["perturbations"]` |
| READ  | `sim_data.condition_to_doubling_time[label]` |
| READ  | `sim_data.process.metabolism.concentration_updates.concentrations_based_on_nutrients()` |
| READ  | `sim_data.mass.getBiomassAsConcentrations()` |
| READ  | `sim_data.getter.get_masses()` |
| READ  | `sim_data.mass.get_component_masses()` |
| READ  | `sim_data.mass.avg_cell_to_initial_cell_conversion_factor` |
| READ  | `sim_data.growth_rate_parameters.*` (elongation rates, active fractions) |
| READ  | `sim_data.constants.cell_density`, `sim_data.constants.n_avogadro` |
| READ  | `sim_data.process.transcription.rna_synth_prob[label]` |
| READ  | `sim_data.process.transcription.rna_data` (is_mRNA, is_tRNA, is_rRNA, includes_ribosomal_protein, includes_RNAP) |
| WRITE | `sim_data.process.transcription.rnaSynthProbFraction` |
| WRITE | `sim_data.process.transcription.rnapFractionActiveDict` |
| WRITE | `sim_data.process.transcription.rnaSynthProbRProtein` |
| WRITE | `sim_data.process.transcription.rnaSynthProbRnaPolymerase` |
| WRITE | `sim_data.process.transcription.rnaPolymeraseElongationRateDict` |
| WRITE | `sim_data.expectedDryMassIncreaseDict` |
| WRITE | `sim_data.process.translation.ribosomeElongationRateDict` |
| WRITE | `sim_data.process.translation.ribosomeFractionActiveDict` |

| Direction | cell_specs Path |
|-----------|-----------------|
| READ  | `cell_specs[cond]["bulkContainer"]` |
| READ  | `cell_specs[cond]["avgCellDryMassInit"]` |
| WRITE | `cell_specs[cond]["avgCellDryMassInit"]` (updated/rescaled) |
| WRITE | `cell_specs[cond]["fitAvgSolublePoolMass"]` |
| WRITE | `cell_specs[cond]["bulkContainer"]` (updated) |

---

### Stage 9: final_adjustments (COUPLED)

| Direction | Attribute Path |
|-----------|----------------|
| READ  | Full `sim_data` ref |
| READ  | Full `cell_specs` ref |
| WRITE | `sim_data.process.transcription` (via `calculate_attenuation()`) |
| WRITE | `sim_data.process.transcription` (via `adjust_polymerizing_ppgpp_expression()`) |
| WRITE | `sim_data.process.transcription` (via `adjust_ppgpp_expression_for_tfs()`) |
| WRITE | `sim_data.process.metabolism` (via `set_phenomological_supply_constants()`) |
| WRITE | `sim_data.process.metabolism` (via `set_mechanistic_supply_constants()`) |
| WRITE | `sim_data.process.metabolism` (via `set_mechanistic_export_constants()`) |
| WRITE | `sim_data.process.metabolism` (via `set_mechanistic_uptake_constants()`) |
| WRITE | `sim_data.process.transcription` (via `set_ppgpp_kinetics_parameters()`) |

---

## Cross-Reference Index

"If I change **X**, which stages should I check?"

| Attribute | Read by | Written by |
|-----------|---------|------------|
| `sim_data.adjustments.*` | 2 | — (raw_data) |
| `sim_data.tf_to_active_inactive_conditions` | 2, 4, 6, 7 | 2 (debug) |
| `sim_data.process.translation.translation_efficiencies_by_monomer` | 2 | 2 |
| `sim_data.process.translation.monomer_data["deg_rate"]` | 2 | 2 |
| `sim_data.process.transcription.rna_expression["basal"]` | 2 | 2, 3 |
| `sim_data.process.transcription.rna_data["deg_rate"]` | 2 | 2 |
| `sim_data.process.transcription.rna_synth_prob[cond]` | 8 | 3, 4, 6 |
| `sim_data.process.transcription.fit_cistron_expression[cond]` | — | 3, 4 |
| `sim_data.process.transcription.rna_data["Km_endoRNase"]` | — | 3 |
| `sim_data.process.rna_decay.*` | — | 3 |
| `sim_data.constants.darkATP` | — | 3 |
| `sim_data.mass.avg_cell_dry_mass_init` | — | 3 |
| `sim_data.mass.fitAvgSolubleTargetMolMass` | — | 3 |
| `sim_data.pPromoterBound` | 7 | 6 |
| `sim_data.process.transcription_regulation.basal_prob` | — | 7 |
| `sim_data.process.transcription_regulation.delta_prob` | — | 7 |
| `sim_data.process.equilibrium` (reverse rates) | 6 | 7 |
| `sim_data.process.metabolism.concentration_updates.molecule_set_amounts` | — | 7 |
| `sim_data.process.transcription.rnaSynthProbFraction` | — | 8 |
| `sim_data.process.transcription.rnapFractionActiveDict` | — | 8 |
| `sim_data.expectedDryMassIncreaseDict` | — | 8 |
| `sim_data.process.translation.ribosomeElongationRateDict` | — | 8 |
| `sim_data.translation_supply_rate` | — | 5 |
| `cell_specs["basal"]` (expression, concDict, etc.) | 5, 6, 7 | 3 |
| `cell_specs[cond]` (TF conditions) | 5, 6, 7 | 4 |
| `cell_specs[cond]["bulkAverageContainer"]` | 6, 7 | 5 |
| `cell_specs[cond]["bulkContainer"]` | 8 | 3, 4, 8 |
| `cell_specs["basal"]["r_vector"]`, `["r_columns"]` | 7 | 6 |
| `cell_specs[cond]["fitAvgSolublePoolMass"]` | — | 8 |
