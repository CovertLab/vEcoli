"""
Stage 7: adjust_promoters — Adjust ligand concentrations and construct
RNAP recruitment parameters from fitted promoter binding probabilities.

Three public functions:
    extract_input(sim_data, cell_specs, **kwargs) -> AdjustPromotersInput
    compute_adjust_promoters(inp) -> AdjustPromotersOutput
    merge_output(sim_data, cell_specs, out)

This stage:
1. Adjusts ligand metabolite concentrations and Kd values for 1CS TFs
   to match fitted pPromoterBound probabilities (fitLigandConcentrations)
2. Constructs the basal_prob vector and delta_prob matrix from the
   fitted r parameters (calculateRnapRecruitment)

NOTE: compute_adjust_promoters mutates sim_data_ref because
fitLigandConcentrations writes molecule_set_amounts and equilibrium
reverse rates to sim_data.  merge_output writes basal_prob and
delta_prob to sim_data.
"""

from reconstruction.ecoli.parca._types import (
    AdjustPromotersInput,
    AdjustPromotersOutput,
)
from reconstruction.ecoli.parca_promoter_fitting import (
    fitLigandConcentrations,
    calculateRnapRecruitment,
)


# ============================================================================
# Extract / Merge
# ============================================================================


def extract_input(sim_data, cell_specs, **kwargs) -> AdjustPromotersInput:
    """Pull sim_data and cell_specs references for promoter adjustment."""
    return AdjustPromotersInput(
        sim_data_ref=sim_data,
        cell_specs_ref=cell_specs,
    )


def merge_output(sim_data, cell_specs, out: AdjustPromotersOutput):
    """Write computed results into sim_data.

    sim_data mutations from fitLigandConcentrations (molecule_set_amounts,
    equilibrium reverse rates) are already applied by compute via
    sim_data_ref.  This function writes basal_prob and delta_prob.
    """
    sim_data.process.transcription_regulation.basal_prob = out.basal_prob
    sim_data.process.transcription_regulation.delta_prob = out.delta_prob


# ============================================================================
# Compute
# ============================================================================


def compute_adjust_promoters(inp: AdjustPromotersInput) -> AdjustPromotersOutput:
    """Run the full adjust_promoters stage.

    This function mutates inp.sim_data_ref as a side effect because
    fitLigandConcentrations writes molecule_set_amounts and equilibrium
    reverse rates to sim_data process objects.
    """
    sim_data = inp.sim_data_ref
    cell_specs = inp.cell_specs_ref

    fitLigandConcentrations(sim_data, cell_specs)
    basal_prob, delta_prob = calculateRnapRecruitment(sim_data, cell_specs)

    return AdjustPromotersOutput(
        basal_prob=basal_prob,
        delta_prob=delta_prob,
    )
