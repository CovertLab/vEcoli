"""
Extract gene metadata and reference basal abundances from a completed parca
simData.cPickle.

Outputs two TSVs to --output_dir:
  gene_metadata.tsv         — boolean model-role flags + adjustment info per gene
  ref_basal_abundances.tsv  — mRNA/monomer expression fractions and ss mean counts
                              at the reference (basal) condition

Usage:
    uv run runscripts/extract_gene_metadata.py \\
        --sim_data_path out/my_run/kb/simData.cPickle \\
        --output_dir out/gene_metadata/
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from reconstruction.ecoli.fit_sim_data_1 import (
    totalCountIdDistributionProtein,
    totalCountIdDistributionRNA,
)

FLAT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "reconstruction", "ecoli", "flat"
)
ADJ_DIR = os.path.join(FLAT_DIR, "adjustments")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim_data_path",
        required=True,
        help="Path to a completed parca simData.cPickle",
    )
    parser.add_argument(
        "--output_dir",
        default="out/gene_metadata",
        help="Directory to write output TSVs (default: out/gene_metadata/)",
    )
    parser.add_argument(
        "--essential_genes_path",
        default=None,
        help="Path to essential_genes.tsv (FrameID column = EcoCyc gene_id). "
        "If omitted, is_essential will be False for all genes.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading simData from {args.sim_data_path} ...")
    with open(args.sim_data_path, "rb") as f:
        sim_data = pickle.load(f)

    print("Building gene_metadata.tsv ...")
    gene_meta_df = build_gene_metadata(sim_data, args.essential_genes_path)
    meta_path = os.path.join(args.output_dir, "gene_metadata.tsv")
    gene_meta_df.to_csv(meta_path, sep="\t", index=False)
    print(f"  Written: {meta_path}  ({len(gene_meta_df)} genes)")
    _print_role_summary(gene_meta_df)

    print("Building ref_basal_abundances.tsv ...")
    abundances_df = build_ref_basal_abundances(sim_data)
    abund_path = os.path.join(args.output_dir, "ref_basal_abundances.tsv")
    abundances_df.to_csv(abund_path, sep="\t", index=False)
    print(f"  Written: {abund_path}  ({len(abundances_df)} genes)")


# ---------------------------------------------------------------------------
# gene_metadata.tsv
# ---------------------------------------------------------------------------


def build_gene_metadata(sim_data, essential_genes_path=None):
    transcription = sim_data.process.transcription
    cistron_data = transcription.cistron_data

    cistron_id_to_gene_id = dict(zip(cistron_data["id"], cistron_data["gene_id"]))
    in_model_set = set(cistron_data["gene_id"])

    monomer_id_to_cistron = {
        row["id"]: row["cistron_id"]
        for row in sim_data.process.translation.monomer_data
    }

    # Build symbol → gene_id lookup for TF resolution (tf_to_gene_id values are symbols)
    all_genes = _read_genes_tsv(os.path.join(FLAT_DIR, "genes.tsv"))
    symbol_to_gene_id = {symbol: gene_id for gene_id, symbol in all_genes}

    aa_pway_gene_ids = _aa_pway_enzyme_genes(
        sim_data, monomer_id_to_cistron, cistron_id_to_gene_id
    )
    tf_gene_ids = {
        symbol_to_gene_id[sym]
        for sym in sim_data.process.transcription_regulation.tf_to_gene_id.values()
        if sym in symbol_to_gene_id
    }
    ribosomal_gene_ids = _monomer_ids_to_gene_ids(
        sim_data.molecule_groups.ribosomal_proteins,
        monomer_id_to_cistron,
        cistron_id_to_gene_id,
    )
    rnap_gene_ids = _monomer_ids_to_gene_ids(
        sim_data.molecule_groups.RNAP_subunits,
        monomer_id_to_cistron,
        cistron_id_to_gene_id,
    )
    reaction_table_gene_ids = _reaction_table_gene_ids(
        sim_data, monomer_id_to_cistron, cistron_id_to_gene_id
    )

    essential_gene_ids = _read_essential_genes(essential_genes_path)

    expr_adj_by_gene = _read_adjustment_tsv(
        os.path.join(ADJ_DIR, "rna_expression_adjustments.tsv")
    )
    deg_adj_by_gene = _read_adjustment_tsv(
        os.path.join(ADJ_DIR, "rna_deg_rates_adjustments.tsv")
    )

    rows = []
    for gene_id, symbol in all_genes:
        in_reaction_table = gene_id in reaction_table_gene_ids
        is_tf = gene_id in tf_gene_ids
        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": symbol,
                "in_model": gene_id in in_model_set,
                "is_essential": gene_id in essential_gene_ids,
                "aa_pway_enzyme": gene_id in aa_pway_gene_ids,
                "is_tf": is_tf,
                "is_ribosomal_translation_mach": gene_id in ribosomal_gene_ids,
                "is_rnap_transcription_mach": gene_id in rnap_gene_ids,
                "in_reaction_table": in_reaction_table,
                "any_role": in_reaction_table or is_tf,
                "has_expression_adjustment": gene_id in expr_adj_by_gene,
                "expression_adjustment_factor": expr_adj_by_gene.get(gene_id, np.nan),
                "has_deg_rate_adjustment": gene_id in deg_adj_by_gene,
                "deg_rate_adjustment_factor": deg_adj_by_gene.get(gene_id, np.nan),
            }
        )

    return pd.DataFrame(rows)


def _aa_pway_enzyme_genes(sim_data, monomer_id_to_cistron, cistron_id_to_gene_id):
    complexation = sim_data.process.complexation
    gene_ids = set()
    for pathway in sim_data.process.metabolism.aa_synthesis_pathways.values():
        all_enzymes = list(pathway["enzymes"]) + list(pathway["reverse enzymes"])
        for enzyme_id in all_enzymes:
            try:
                subunit_ids = complexation.get_monomers(enzyme_id)["subunitIds"]
            except Exception:
                subunit_ids = [enzyme_id]
            for monomer_id in subunit_ids:
                cistron_id = monomer_id_to_cistron.get(monomer_id)
                if cistron_id:
                    gene_id = cistron_id_to_gene_id.get(cistron_id)
                    if gene_id:
                        gene_ids.add(gene_id)
    return gene_ids


def _print_role_summary(df):
    """Print counts for each role flag and the in_model × any_role cross-tab."""
    total = len(df)
    in_model = int(df["in_model"].sum())
    print(f"\nGene role summary ({total} total genes; {in_model} in_model):")
    role_cols = [
        "aa_pway_enzyme",
        "is_tf",
        "is_ribosomal_translation_mach",
        "is_rnap_transcription_mach",
        "in_reaction_table",
        "any_role",
    ]
    for col in role_cols:
        n_all = int(df[col].sum())
        n_in_model = int((df[col] & df["in_model"]).sum())
        print(f"  {col:32s} {n_all:5d}  ({n_in_model} also in_model)")

    # Cross-tab: in_model × any_role — the "expressed but passive" cell is the
    # count we actually care about for the research question.
    passive = int((df["in_model"] & ~df["any_role"]).sum())
    active_not_expressed = int((~df["in_model"] & df["any_role"]).sum())
    print(f"\n  in_model ∧ ¬any_role (expressed but passive): {passive}")
    print(f"  any_role ∧ ¬in_model (role but not transcribed): {active_not_expressed}")


def _reaction_table_gene_ids(sim_data, monomer_id_to_cistron, cistron_id_to_gene_id):
    """Return gene_ids whose monomer (directly, or as a subunit of a referenced
    complex) appears in any of the model's reaction tables.

    Sources unioned:
      - metabolism.catalyst_ids              (enzymes for metabolic reactions)
      - complexation.molecule_names          (subunits + complexes)
      - equilibrium.molecule_names           (ligand/receptor participants)
      - two_component_system.molecule_names  (sensor/response regulator participants)

    Non-protein participants (metabolites) are filtered out automatically: they
    don't appear in monomer_data, so monomer_id_to_cistron lookup returns None.
    TFs that don't appear in any of these tables are handled separately via the
    existing is_tf column; OR them in at the call site for `any_role`.
    """
    complexation = sim_data.process.complexation

    participant_ids = set()
    participant_ids.update(sim_data.process.metabolism.catalyst_ids)
    participant_ids.update(sim_data.process.complexation.molecule_names)
    participant_ids.update(sim_data.process.equilibrium.molecule_names)
    participant_ids.update(sim_data.process.two_component_system.molecule_names)
    # tRNA charging: synthetase enzymes
    participant_ids.update(sim_data.process.transcription.synthetase_names)
    # RNA decay: endo- and exo-ribonucleases
    participant_ids.update(sim_data.process.rna_decay.endoRNase_ids)
    participant_ids.update(sim_data.molecule_groups.exoRNases)
    # DNA replication: replisome subunits (DnaB, DnaG, ligase, etc.)
    participant_ids.update(sim_data.molecule_groups.replisome_trimer_subunits)
    participant_ids.update(sim_data.molecule_groups.replisome_monomer_subunits)

    gene_ids = set()
    for participant_id in participant_ids:
        try:
            subunit_ids = complexation.get_monomers(participant_id)["subunitIds"]
        except Exception:
            subunit_ids = [participant_id]
        for monomer_id in subunit_ids:
            cistron_id = monomer_id_to_cistron.get(monomer_id)
            if cistron_id:
                gene_id = cistron_id_to_gene_id.get(cistron_id)
                if gene_id:
                    gene_ids.add(gene_id)
    return gene_ids


def _monomer_ids_to_gene_ids(monomer_ids, monomer_id_to_cistron, cistron_id_to_gene_id):
    gene_ids = set()
    for mid in monomer_ids:
        cistron_id = monomer_id_to_cistron.get(mid)
        if cistron_id:
            gene_id = cistron_id_to_gene_id.get(cistron_id)
            if gene_id:
                gene_ids.add(gene_id)
    return gene_ids


def _read_adjustment_tsv(path):
    """Return {gene_id: factor} from an adjustment TSV (RNA IDs stripped to gene IDs)."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            name = parts[0].strip('"')
            if name == "name":
                continue
            # RNA IDs have the form "EG10581_RNA"; strip suffix to get gene_id
            gene_id = name.removesuffix("_RNA")
            try:
                factor = float(parts[1].strip('"'))
            except (IndexError, ValueError):
                continue
            result[gene_id] = factor
    return result


def _read_genes_tsv(path):
    """Return list of (gene_id, symbol) for all genes in genes.tsv."""
    df = pd.read_csv(path, sep="\t", comment="#", quotechar='"')
    return list(zip(df["id"], df["symbol"]))


def _read_essential_genes(path):
    """Return set of essential gene_ids from essential_genes.tsv (FrameID column)."""
    if path is None:
        return set()
    df = pd.read_csv(path, sep="\t", comment="#", quotechar='"')
    return set(df["FrameID"])


# ---------------------------------------------------------------------------
# ref_basal_abundances.tsv
# ---------------------------------------------------------------------------


def build_ref_basal_abundances(sim_data):
    transcription = sim_data.process.transcription
    cistron_data = transcription.cistron_data
    cistron_id_to_gene_id = dict(zip(cistron_data["id"], cistron_data["gene_id"]))

    basal_dt = sim_data.condition_to_doubling_time["basal"]
    basal_rna_expression = transcription.rna_expression["basal"]

    # --- mRNA fractions per cistron ---
    # cistron_tu_mapping_matrix: (n_cistrons × n_TUs); dot with TU expression
    # gives per-cistron expression summed across all TUs containing each cistron
    cistron_expression = transcription.cistron_tu_mapping_matrix.dot(
        basal_rna_expression
    )
    is_mRNA = cistron_data["is_mRNA"]

    # --- mRNA ss mean counts per TU, then map to cistron ---
    total_rna_count, rna_ids, rna_distribution = totalCountIdDistributionRNA(
        sim_data, basal_rna_expression.copy(), basal_dt
    )
    # counts per TU
    rna_counts = total_rna_count * rna_distribution
    # map TU counts → cistron counts via the same mapping matrix
    cistron_counts = transcription.cistron_tu_mapping_matrix.dot(rna_counts)

    # --- protein fractions + ss mean counts per monomer ---
    total_prot_count, prot_ids, prot_distribution = totalCountIdDistributionProtein(
        sim_data, basal_rna_expression.copy(), basal_dt
    )
    prot_counts = total_prot_count * prot_distribution

    monomer_data = sim_data.process.translation.monomer_data
    # cistron_index = {cid: i for i, cid in enumerate(cistron_data["id"])}

    # Build per-cistron protein aggregates (sum monomers sharing the same cistron)
    prot_frac_by_cistron = {}
    prot_count_by_cistron = {}
    for i, row in enumerate(monomer_data):
        cid = row["cistron_id"]
        prot_frac_by_cistron[cid] = prot_frac_by_cistron.get(cid, 0.0) + float(
            prot_distribution[i]
        )
        prot_count_by_cistron[cid] = prot_count_by_cistron.get(cid, 0.0) + float(
            prot_counts[i]
        )

    # --- Assemble per-gene output ---
    # Aggregate cistrons to genes (most genes have one cistron; polycistronics share)
    gene_rows = {}
    for i, cistron_id in enumerate(cistron_data["id"]):
        gene_id = cistron_id_to_gene_id[cistron_id]
        if gene_id not in gene_rows:
            gene_rows[gene_id] = {
                "mrna_expression_fraction": 0.0,
                "mrna_ss_mean_count": 0.0,
                "monomer_expression_fraction": 0.0,
                "monomer_ss_mean_count": 0.0,
            }
        if is_mRNA[i]:
            gene_rows[gene_id]["mrna_expression_fraction"] += float(
                cistron_expression[i]
            )
            gene_rows[gene_id]["mrna_ss_mean_count"] += float(cistron_counts[i])
        gene_rows[gene_id]["monomer_expression_fraction"] += prot_frac_by_cistron.get(
            cistron_id, 0.0
        )
        gene_rows[gene_id]["monomer_ss_mean_count"] += prot_count_by_cistron.get(
            cistron_id, 0.0
        )

    all_genes = _read_genes_tsv(os.path.join(FLAT_DIR, "genes.tsv"))
    rows = []
    for gene_id, symbol in all_genes:
        data = gene_rows.get(gene_id, {})
        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": symbol,
                "mrna_expression_fraction": data.get(
                    "mrna_expression_fraction", np.nan
                ),
                "mrna_ss_mean_count": data.get("mrna_ss_mean_count", np.nan),
                "monomer_expression_fraction": data.get(
                    "monomer_expression_fraction", np.nan
                ),
                "monomer_ss_mean_count": data.get("monomer_ss_mean_count", np.nan),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
