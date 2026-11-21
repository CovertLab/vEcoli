# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Vivarium *E. coli* (vEcoli) is a port of the Covert Lab's wcEcoli whole-cell model to the
[vivarium-core](https://github.com/vivarium-collective/vivarium-core) framework. Full docs:
https://covertlab.github.io/vEcoli/ (sources in `doc/`).

## Environment and commands

Python is pinned to 3.12.12 and managed with `uv`. Always run scripts through uv with the repo's
`.env` file (the README's `uvenv` alias is `uv run --env-file <repo>/.env --project <repo>`), never
with a bare `python`.

```bash
uv sync --frozen --extra dev          # install (CI uses USE_CYTHON=1 to build the .pyx extensions in wholecell/utils)
uv run pre-commit install             # ruff lint + format hook

uv run ruff check doc ecoli wholecell runscripts validation reconstruction
uv run ruff format
uv run mypy                           # file list comes from mypy.ini

uv run --env-file .env pytest                                  # full suite (also runs doctests)
uv run --env-file .env pytest path/to/test_file.py::test_name  # single test
uv run --env-file .env pytest -m "not slow"
```

- `pytest.ini` sets `python_files = *.py` and `--doctest-modules`, so any `test_*` function in any
  module under `ecoli`, `runscripts`, `wholecell`, `reconstruction` is collected. Markers: `slow`,
  `noci`, `master`.
- Many tests need ParCa output first. CI runs
  `uv run --env-file .env runscripts/parca.py --config configs/gha_pytest.json` before pytest.
- Ruff uses only the pre-0.16 default rule set (`E4`, `E7`, `E9`, `F`).

### Running the model

```bash
# End-to-end Nextflow workflow (ParCa -> sims -> analyses); output in out/<experiment_id>
uv run --env-file .env runscripts/workflow.py --config configs/test_installation.json

# Individual steps
uv run --env-file .env runscripts/parca.py --config configs/run_parca.json
uv run --env-file .env ecoli/experiments/ecoli_master_sim.py --config <config.json>
uv run --env-file .env runscripts/analysis.py --config <config.json>
```

`workflow.py` needs `nextflow` installed and renders the templates in `runscripts/nextflow/`.
For reproducibility checks, `runscripts/debug/compare_pickles.py` diffs two ParCa outputs and
`runscripts/debug/diff_simouts.py` diffs two simulation outputs (CI runs both).

### Fast iteration (use this by default)
- Don't rerun the ParCa unless you changed something under `reconstruction/` (including flat
  files). Process, listener and analysis changes do NOT need a new sim_data.
- To skip the ParCa, set top-level `sim_data_path` to an existing pickle, e.g.
  `out/test_installation/parca/kb/simData.cPickle` (from the test installation run).
- For quick checks, use `configs/dev_short.json` [TODO: create; inherits test_installation,
  sets `sim_data_path` as above and a short `max_duration`, e.g. 600]. Full-length runs only
  when asked.
- When reconstruction changes do require a ParCa, try `parca_options.debug_parca: true` first
  for a fast smoke test.

### Verifying changes
- Before modifying behavior, run a baseline with the same config and seed under a distinct
  `experiment_id` (e.g. `baseline_<feature>`).
- Refactors that should not change behavior: confirm with
  `uv run --env-file .env runscripts/debug/diff_simouts.py -o out <baseline_id> <new_id>`
  (must pass).
- Intended behavior changes: diff_simouts will fail by design. Instead, run the health-check
  analyses [TODO: list, e.g. mass_fraction_summary, ...] on both runs and report what changed,
  including anything unexpected.

## Architecture

### Pipeline
1. **Raw data** — TSV flat files in `reconstruction/ecoli/flat/`, loaded by
   `reconstruction/ecoli/knowledge_base_raw.py`.
2. **ParCa** (`reconstruction/ecoli/fit_sim_data_1.py`, driven by `runscripts/parca.py`) fits
   parameters and pickles a `SimulationDataEcoli` object (`reconstruction/ecoli/simulation_data.py`,
   sub-objects in `reconstruction/ecoli/dataclasses/`) to `<outdir>/kb/simData.cPickle`.
3. **Variants** — `runscripts/create_variants.py` imports `ecoli.variants.<name>` and calls its
   `apply_variant(sim_data, params) -> sim_data` to produce one modified sim_data per variant.
4. **Simulation** — `ecoli/experiments/ecoli_master_sim.py` (`SimConfig`, `EcoliSim`) builds the
   cell from config and runs it, emitting to Parquet via `ecoli/library/parquet_emitter.py`.
5. **Analysis** — `runscripts/analysis.py` imports `ecoli.analysis.<type>.<name>` and calls its
   `plot(params, conn, history_sql, config_sql, success_sql, sim_data_paths, validation_data_paths,
   outdir, variant_metadata, variant_names)`. `<type>` is the data scope (`single`,
   `multigeneration`, `multiseed`, `multivariant`, ...). Plots query Parquet output with DuckDB
   (helpers like `read_stacked_columns` in `parquet_emitter.py`); column names are flattened store
   paths such as `listeners__mass__dry_mass`.

### Configuration
All runs are configured by JSON in `configs/`. Every config is merged on top of
`configs/default.json`, and configs can chain via `"inherit_from"`. `default.json` defines the
ordered `processes` list, extra `topology` entries, and the `flow` dict that sets the execution
dependencies between Steps. Configs can also use `add_processes`, `exclude_processes`,
`swap_processes`, and `process_configs`. The analyses to run go under `analysis_options`, keyed by
analysis type then module name.

### Simulation internals
- Processes live in `ecoli/processes/` and are registered by name in
  `ecoli/processes/__init__.py` (`process_registry`). Topologies are registered in
  `topology_registry` (`ecoli/processes/registries.py`). A new process must be registered there to
  be usable from a config.
- Process parameters come from sim_data through `LoadSimData.get_config_by_name`
  (`ecoli/library/sim_data.py`), which maps each process name to a `get_*_config` method.
- **Partitioning**: most biological processes subclass `PartitionedProcess`
  (`ecoli/processes/partition.py`) and implement `calculate_request` and `evolve_state`. The
  composer (`ecoli/composites/ecoli_master.py`) wraps each in a `Requester` and an `Evolver` step,
  with `allocator.py` dividing limited bulk molecules among the requesters in between.
- **State**: bulk molecules are one structured numpy array (indexed with `bulk_name_to_idx`);
  unique molecules (RNAP, ribosomes, chromosome domains, ...) are `MetadataArray`s updated through
  `UniqueNumpyUpdater`. Schema helpers, updaters and division functions are in
  `ecoli/library/schema.py`.
- Listeners (`ecoli/processes/listeners/`) record per-timestep data under the `listeners` store,
  which is what analyses read.
- `wholecell/` holds utilities carried over from wcEcoli (units, polymerize, Cython extensions,
  fitting helpers).

## Scientific rules
- Every new or changed parameter needs a cited source in the relevant flat file. Never invent
  values or hand-tune parameters to make output match data.
- `validation/ecoli/flat/` is for evaluating the model, never for fitting it.
- If results disagree with expectations, say so plainly. Don't explain them away.

## Git and boundaries
- Work on a feature branch. Never push to master; never force-push.
- All commits must be signed (SSH signing is configured globally). If signing fails, stop and
  tell me; never commit unsigned or change git identity/signing config yourself.
- Before committing, run `uv run ruff check --fix` and `uv run ruff format`; if the pre-commit
  hook modifies files, re-stage and commit again.
- Ask before: editing `.env`, `uv.lock` or `configs/default.json`; deleting anything under
  `out/`; submitting Sherlock or cloud jobs.
- Before adding or changing a process, read `doc/processes.rst` and `doc/stores.rst`.
- Open an issue before non-trivial changes (from `doc/contributing.rst`).
