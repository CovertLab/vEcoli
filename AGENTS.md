# AGENTS: vEcoli - Whole-Cell E. coli Model in Vivarium

## Project Overview

This repository is Vivarium E. coli (vEcoli), a port of the Covert Lab's whole-cell E. coli model to the Vivarium
framework. The codebase simulates the full lifecycle of an E. coli cell by integrating modular biological processes —
transcription, translation, metabolism, replication, and division — each implemented as a composable Vivarium `Process`
or
`Step`. Simulation state is emitted as Parquet files for downstream analysis, and all configuration is managed through
JSON files under `configs/`.

The purpose of the project is to refine gene integration for the metabolism module within whole-cell E.coli model. In
2022, our lab has incorporated ~300 new metabolic genes, which encode enzymes catalyzing ~600 new metabolic reactions in
the model. The goal of this effort is to make a responsive metabolism module that has robust prediction of physiological
behavior when simulated under various conditions, including media, stress, or oxygen depletion etc. Out of the ~300 new
genes have been added to the model, only 3.5% of the new genes were used during basal simulation. There are four mains
reasons for this:

1. Condition-specific gene expression: many of the new genes may only be expressed under specific environmental
   conditions or stressors that were not present in the basal simulation. For exmaple, some genes are responsible for
   the utilization of other carbon sources, and hence would not sustain flux under glucose minimal media, which is the
   basal condition in the project
2. Dead-end metabolite: some of the new genes may be involved in reactions that produce or consume metabolites that are
   not connected to the rest of the metabolic network, leading to dead-end pathways that do not carry flux. We need to
   identify and address these dead-end metabolites via literature search and/or BLAST genes in close organisms to
   propose potential noval reactions to ensure that the new genes can be properly integrated into the metabolic network.
3. FBA produces scarce solution: the flux balance analysis (FBA) approach used in the metabolism module produce a sparse
   solution where only a subset of reactions carry flux, and many reactions are inactive. This can lead to
   underutilization of all genes (including new genes). Added flux diversity in my metabolism process `obj_div` to push
   for non-sparse solution. Requires new weight determination to the multi-objective problem. Want to find what
   combination of the five lambda weights produces a metabolic flux solution that (1) accurately reproduces central
   carbon metabolism fluxes from Toya et al. 2010 (quantified by toya_r_squared > 0.5), (2) keeps both homeostatic and
   kinetic objectives low, and (3) sits at the edge of the homeostatic–kinetic Pareto frontier so that a simulated
   enzyme knockdown propagates a measurable, biologically interpretable compensatory response in homeostatic flux?
4. Homeostatic need: many of the new genes are involved in creation of biomass precursors that are not currently
   considered as homeostatic need, hence, there is no incentive to produce flux through those reactions. Need to expand
   the homeostatic need to include more biomass precursors, which will incentivize flux through the new genes.

In regards to the metabolic process being used, the metabolic process at the center of Heena's current work is MetabolismReduxClassic (
ecoli/processes/metabolism_redux_classic.py), which solves a multi-objective Network Flow Model (convex optimization via
CVXPY/GLOP) at each simulation timestep. The objective is a weighted sum of five terms: homeostatic need (obj_hom),
kinetic flux matching (obj_kin), metabolic efficiency (obj_eff), secretion (obj_sec), and flux diversity (obj_div), each
scaled by a corresponding weight lambda_*. The central scientific question driving the work in notebooks/Heena
notebooks/Metabolism_New Genes/ is: what combination of the five lambda weights produces a metabolic flux solution
that (1) accurately reproduces central carbon metabolism fluxes from Toya et al. 2010 (quantified by toya_r_squared >
0.5), (2) keeps both homeostatic and kinetic objectives low, and (3) sits at the edge of the homeostatic–kinetic Pareto
frontier so that a simulated enzyme knockdown propagates a measurable, biologically interpretable compensatory response
in homeostatic flux?

## Style Conventions

- **TAB indentation** (4-space width display) — never use spaces for indentation
- snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants
- Scientific names like `Kcat`, `kM` are acceptable exceptions to casing rules
- Line length: soft 79, hard 99
- Import order: stdlib, then third-party, then local — alphabetical within each group
- Docstrings: triple double-quotes, imperative mood ("Calculate X", not "Calculates X")
- Line breaks go *before* binary operators (PEP 8)
- Always update docstrings, comments, and variable names for consistency when editing code


## Architecture

- **Variant system**: `ecoli/variants/` — functions that modify `sim_data` to create experimental conditions
- **Analysis scripts**: `ecoli/analysis/{single,multigen,cohort,variant}/`
- **Data Reconstruction**: `reconstruction/ecoli/*` for reconstructing raw files from `reconstruction/ecoli/flats/*` and `sim_data` creation
- **sim_data**: pickled `SimulationDataEcoli` object — the central data structure built by Parca, which is built using Data Reconstruction
- **Processes**: `ecoli/processes/` — each is a `Process` subclass
- **Listeners**: `models/ecoli/listeners/` — record simulation state each timestep
- **Output**: `out/` directory with experiment type related subdirectories
