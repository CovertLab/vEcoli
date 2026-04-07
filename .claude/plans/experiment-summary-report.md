# Plan: Multi-Parca Analysis Report

## Status: IMPLEMENTED AND VALIDATED ✓

Script: `wholecell/io/multiparca_analysis.py`

End-to-end tested on `out/test_multi_parca_20260323-171234` (ref_0001 vs gbw_0001_v2).
Output matched known run result: parca_0 COMPLETED, parca_1 FAILED (CYS kinetics).

**Next step**: increase `generations` and `n_init_sims` in `test_multi_parca.json` once
the gbw_0001_v2 fitting issue is resolved, then re-run to validate multi-generation
and multi-seed behavior of the report and companion analyses.

---

## Goal

After a multi-parca workflow run, produce a concise report that answers:
- Which parca runs succeeded / failed, and with which datasets?
- How many simulations completed per dataset?
- What were the key simulation outcomes per dataset (generation time, growth rate, mass)?

Output: a summary CSV table + a set of comparison plots.

---

## Usage

```bash
uv run wholecell/io/multiparca_analysis.py --out_dir out/my_experiment_TIMESTAMP -o out/reports/
```

Outputs written to the specified `-o` directory:
- `parca_summary.csv`
- `parca_status.png`
- `cell_distributions.png`

---

## Data Sources

| Data | Location | Used for |
|---|---|---|
| Parca run status | Nextflow trace CSV (`trace--{expId}--{ts}.csv`, written to CWD) | Did parca succeed? |
| Per-parca dataset label | `out/{exp}/nextflow/parca_config_{i}.json` → `parca_options.rnaseq_basal_dataset_id` | Label rows by dataset |
| Variant → parca mapping | `out/{exp}/nextflow/workflow_config.json` | Which variant indices belong to which parca? |
| Per-cell stats | `out/{exp}/history/` Parquet | Gen time, protein mass |
| Cell mass / volume | `out/{exp}/analyses/variant={v}/lineage_seed=*/plots/higher_order_properties.tsv` | If cd1_higher_order_properties has run |

---

## Output Schema

### `parca_summary.csv` — one row per parca run

| column | source |
|---|---|
| `parca_id` | channel index |
| `dataset_id` | `parca_config_{i}.json` → `rnaseq_basal_dataset_id` |
| `parca_status` | trace CSV `status` for `runParca (k)` |
| `parca_error` | last Error/Exception line of `{workdir}/.command.err` for failed runs |
| `parca_duration_min` | trace CSV `duration` / 60000 |
| `n_sims_succeeded` | cells in doubling-time data for this parca's variant range |
| `mean_gen_time_hr` | mean doubling time across cells |
| `std_gen_time_hr` | std of same |
| `mean_protein_mass_fg` | mean time-averaged protein mass per cell |
| `mean_cell_mass_mg_per_1e9` | from higher_order_properties.tsv (if available) |
| `mean_cell_volume_um3` | from higher_order_properties.tsv (if available) |

### Plots

- **`parca_status.png`** — bar chart of parca durations colored by status
- **`cell_distributions.png`** — seaborn violin + strip, faceted by metric,
  hued by dataset_id. Metrics: doubling time, protein mass, plus cell mass and
  cell volume if `higher_order_properties.tsv` is available.

---

## Companion Analyses (enabled in `configs/test_multi_parca.json`)

These run through Nextflow and produce standalone HTML plots for mechanistic drill-down.
They are reviewed alongside the summary report, not assimilated into it.

| Script | Level | Output | Purpose |
|---|---|---|---|
| `cd1_higher_order_properties` | multiseed | `higher_order_properties.tsv` | Cell mass, volume, DNA/RNA fractions — **feeds into summary report** |
| `mass_fraction_summary` | single | `mass_fraction_summary.html` | Mass component breakdown per cell over time; generation drift visible with >1 gen |
| `ribosome_components` | multigeneration | `ribosome_components.html` | Ribosome subunit composition across generations |
| `ribosome_production` | multigeneration | `ribosome_production_report.html` | Ribosome synthesis rates |
| `ribosome_usage` | multigeneration | `ribosome_usage_report.html` | Ribosome occupancy / translation activity |

**Note**: `mass_fraction_summary` generation-drift diagnostic requires `generations > 1`
to be meaningful. All three ribosome plots are most useful for diagnosing protein
production differences between datasets.

---

## Known Limitations

- **`higher_order_properties.tsv` optional**: if `cd1_higher_order_properties` hasn't
  run, `mean_cell_mass` and `mean_cell_volume` are `None` in the CSV and cell mass/volume
  are absent from the distribution plot. Script prints a clear message.
- **Trace CSV location**: written to the repo root CWD, not the output dir. If not found,
  parca status appears as "unknown" rather than failing.
- **Parca failure = no sim data**: failed parca's variant range is absent from Parquet;
  handled gracefully (`n_sims_succeeded = 0`, cell stats `NaN`).
- **Single-parca compatibility**: `pickles_per_parca` maps all variants to `parca_id = 0`.
  Produces a single-row summary labeled with the one dataset_id (or "legacy").

---

## Future Work

- Increase `generations` and `n_init_sims` in `test_multi_parca.json` for richer
  distributions and generation-drift diagnostics (planned once gbw_0001_v2 issue resolved).
- Nextflow integration: wrap as an `analysisMultiVariant` step so it runs automatically
  at the end of a workflow. Currently standalone only.
