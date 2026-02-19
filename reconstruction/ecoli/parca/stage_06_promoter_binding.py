"""
Stage 6: promoter_binding — Fit transcription factor binding probabilities
and their effects on RNA synthesis.

Three public functions:
    extract_input(sim_data, cell_specs, **kwargs) -> PromoterBindingInput
    compute_promoter_binding(inp) -> PromoterBindingOutput
    merge_output(sim_data, cell_specs, out)

This stage:
1. Calculates initial TF-promoter binding probabilities from bulk average
   counts of TFs and ligands for each condition
2. Uses convex optimization (CVXPY/ECOS) to fit parameters alpha and r
   such that computed RNA synthesis probabilities match measured values
3. Iteratively optimizes both the binding probabilities (P) and the
   recruitment parameters (r) until convergence

NOTE: compute_promoter_binding mutates sim_data_ref because
fitPromoterBoundProbability writes pPromoterBound and rna_synth_prob
to sim_data.  merge_output only writes cell_specs.
"""

from reconstruction.ecoli.parca._types import (
    PromoterBindingInput,
    PromoterBindingOutput,
)
from reconstruction.ecoli.parca_promoter_fitting import (
    fitPromoterBoundProbability,
)


# ============================================================================
# Extract / Merge
# ============================================================================


def extract_input(sim_data, cell_specs, **kwargs) -> PromoterBindingInput:
    """Pull sim_data and cell_specs references for promoter fitting."""
    return PromoterBindingInput(
        sim_data_ref=sim_data,
        cell_specs_ref=cell_specs,
    )


def merge_output(sim_data, cell_specs, out: PromoterBindingOutput):
    """Write computed results into cell_specs.

    sim_data mutations (pPromoterBound, rna_synth_prob) are already
    applied by compute_promoter_binding via sim_data_ref.
    """
    cell_specs["basal"]["r_vector"] = out.r_vector
    cell_specs["basal"]["r_columns"] = out.r_columns


# ============================================================================
# Compute
# ============================================================================


def compute_promoter_binding(inp: PromoterBindingInput) -> PromoterBindingOutput:
    """Run the full promoter_binding stage.

    This function mutates inp.sim_data_ref as a side effect because
    fitPromoterBoundProbability sets pPromoterBound and updates
    rna_synth_prob on sim_data.
    """
    sim_data = inp.sim_data_ref
    cell_specs = inp.cell_specs_ref

    print("Fitting promoter binding")

    r_vector, r_columns = fitPromoterBoundProbability(sim_data, cell_specs)

    return PromoterBindingOutput(
        r_vector=r_vector,
        r_columns=r_columns,
    )
