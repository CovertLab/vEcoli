from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def get_new_gene_indices(sim_data):
    """
    Determines the indices of new gene mRNAs and proteins using the new
    gene flag in sim_data.

    Returns:
        new_gene_mRNA_indices: indices in rna_data table for new gene mRNAs
        new_monomer_indices: indices in monomer_data table for new monomers
    """
    mRNA_sim_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    new_gene_mRNA_ids = mRNA_sim_data[mRNA_sim_data['is_new_gene']]['id'].tolist()
    mRNA_monomer_id_dict = dict(
        zip(monomer_sim_data['cistron_id'], monomer_sim_data['id']))
    new_gene_monomer_ids = [
        mRNA_monomer_id_dict.get(mRNA_id) for mRNA_id in new_gene_mRNA_ids]
    if len(new_gene_mRNA_ids) == 0:
        raise Exception("This variant  is intended to be run on simulations "
            "where the new gene option was enabled, but no new gene mRNAs were "
            "found.")
    if len(new_gene_monomer_ids) == 0:
        raise Exception("This variant is intended to be run on simulations where"
            " the new gene option was enabled, but no new gene proteins "
            "were found.")
    assert len(new_gene_monomer_ids) == len(new_gene_mRNA_ids), \
        'number of new gene monomers and mRNAs should be equal'
    rna_data = sim_data.process.transcription.rna_data
    mRNA_idx_dict = {rna[:-3]: i for i, rna in enumerate(rna_data['id'])}
    new_gene_indices = [
        mRNA_idx_dict.get(mRNA_id) for mRNA_id in new_gene_mRNA_ids]
    monomer_idx_dict = {
        monomer: i for i, monomer in enumerate(monomer_sim_data['id'])}
    new_monomer_indices = [
        monomer_idx_dict.get(monomer_id) for monomer_id in new_gene_monomer_ids]

    return new_gene_indices, new_monomer_indices


def modify_new_gene_exp_trl(
    sim_data: "SimulationDataEcoli",
    expression: float,
    translation_efficiency: float
):
    """
    Sets expression and translation effiencies of new genes. Modifies::

        sim_data.process.transcription.rna_synth_prob
        sim_data.process.transcription.rna_expression
        sim_data.process.transcription.exp_free
        sim_data.process.transcription.exp_ppgpp
        sim_data.process.transcription.attenuation_basal_prob_adjustments
        sim_data.process.transcription_regulation.basal_prob
        sim_data.process.transcription_regulation.delta_prob
        sim_data.process.translation.translation_efficiencies_by_monomer

    Args:
        sim_data: Simulation data
        expression: Factor by which to adjust new gene expression levels
        translation_efficiency: Translation efficiency for new genes
    """
    # Determine ids and indices of new genes
    new_gene_indices, new_monomer_indices = get_new_gene_indices(sim_data)

    # Modify expression and translation efficiency for new genes
    for gene_idx, monomer_idx in zip(new_gene_indices, new_monomer_indices):
        sim_data.adjust_new_gene_final_expression([gene_idx], [expression])
        sim_data.process.translation.translation_efficiencies_by_monomer[
            monomer_idx] = translation_efficiency


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, list[Any]]
) -> "SimulationDataEcoli":
    """
    Modify sim_data so new gene expression and translation efficiency
    are modified for certain generation ranges.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Generation to induce new gene expression
                "induction_gen": Optional(int),
                "exp_trl_eff": {
                    # Factor by which to multiply new gene expression once induced
                    "exp": float,
                    # Translation efficiency for new gene once induced
                    "trl_eff": float,
                }
                # Generation to knock out new gene expression (> induction gen)
                "knockout_gen": Optional(int),
                # Environmental condition: "basal", "with_aa", "acetate",
                # "succinate", "no_oxygen"
                "condition": str,
            }

    Returns:
        Simulation data with the following attributes modified::

            sim_data.condition
            sim_data.external_state.current_timeline_id
            # Mapping of generation number to function that takes sim_data
            # as first argument and optional extra arguments contained in a
            # tuple that will be star expanded. Here, modify_new_gene_exp_trl
            # takes two extra arguments: expression and translation_efficiency.
            sim_data.internal_shift_dict

    """
    # Set media condition
    sim_data.condition = params["condition"]
    sim_data.external_state.current_timeline_id = params["condition"]
    sim_data.external_state.saved_timelines[params["condition"]] = [(
        0, sim_data.conditions[params["condition"]]["nutrients"])]

    # Initialize internal shift dictionary
    sim_data.internal_shift_dict = {}

    # Add the new gene induction to the internal_shift instructions
    if params["induction_gen"] != -1:
        sim_data.internal_shift_dict[params["induction_gen"]] = (
            modify_new_gene_exp_trl,
            (params["exp_trl_eff"]["exp"], params["exp_trl_eff"]["trl_eff"])
        )
    if params["knockout_gen"] != -1:
        assert params["knockout_gen"] > params["induction_gen"], (
            "New genes are knocked out by default, so induction should happen"
            " before knockout."
        )
        sim_data.internal_shift_dict[params["knockout_gen"]] = (
            modify_new_gene_exp_trl,
            (0, params["exp_trl_eff"]["trl_eff"])
        )

    return sim_data
