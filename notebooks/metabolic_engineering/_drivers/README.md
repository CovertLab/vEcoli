# `_drivers/` — moving FBA workflows out of notebooks

An early experiment in migrating standalone-FBA functionality out of
notebooks and into lightweight CLI scripts that operate on a **pickled
augmented model**. The pattern is intended to grow into a small toolkit
as the strain-design workstream produces more analyses that benefit
from CLI access (parameter sweeps, batch runs, composability).

## Pattern

```
notebook (one-time)              driver(s) (cheap, repeatable)
──────────────────────           ──────────────────────────────
load_sim, extract coefs           dill.load(state.pkl)
build augmented NetworkFlowModel  →  single-LP solve   →  CSV / JSON
pickle state                         parameter sweep      text output
                                     plot rendering
```

The notebook (`wcm_fba_biomass_prodenvs.ipynb`) stays canonical for
building the augmented model and narrating the science. The drivers
load the pickled state and run targeted analyses without re-paying
Jupyter startup or re-running upstream cells per parameter change.

## Pickled-state contract

Drivers expect a `dill`-loadable pickle (default path
`out/biomass_reaction/state.pkl`) containing the augmented
`NetworkFlowModel` plus the index/scaling metadata needed to interpret
its biomass column and run sweeps:

- `model_e` (or `model_t`): augmented `NetworkFlowModel`
- `biomass_idx`, `biomass_scale`: column index + WT scaling
- `kinetic_targets_arr`, `maintenance`: time-averaged listener arrays
- `metabolites`, `exchange_metabolites`: index lookups
- `glucose_exch_idx`, `glucose_bound_counts_s_default`

## Tools

- `solve_biomass.py` — single LP. Flags: `--kinetic-weight`,
  `--glucose-multiplier`, `--objective biomass | secrete:<exch>`,
  `--fix-biomass`, `--solver`, etc. Returns status, v_biomass_user,
  top mass-balance duals, near-cap fluxes.
- `envelope_kinweight_grid.py` — sweeps a (product × kinetic weight)
  grid. Writes long-format CSV + extremes JSON.
- `plot_kinweight_grid.py` — renders the family-of-curves HTML from
  the grid CSV.

## Forward-looking — design-algorithm fits

The same shape (pickled state + config flags → structured output)
applies cleanly to the next round of strain-design work:

- **FSEOF** — scan reactions whose flux scales with product secretion
  along a Mode B/C envelope; output a ranked enzyme-target CSV.
- **MOMA-KO** — quadratic-programming reaction knockouts vs. a WT
  reference; output a ranked KO-impact table.
- **FluxRETAP / OptKnock** — multi-product target deconvolution;
  output intervention candidates.

As that catalog grows, this directory becomes a small toolkit and the
notebook stays the place where the model gets built and stories get
told. Today's three scripts are the prototype.
