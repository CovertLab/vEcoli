"""
Analysis of ~307 new metabolic genes added to the E. coli whole-cell model.

Plot 1 — Proteome Cost vs. Flux Contribution (scatter per gene):
    For each new metabolic gene, compare its average protein mass (fg) against
    the average absolute flux (mmol/L/h) of reactions it catalyzes.  Points are
    colored by broad pathway category (Pathways parent column of the annotation
    CSV).  Top-cost / low-flux and top-flux outlier genes are annotated with
    their gene name.  One facet panel per simulation variant.
    Output: new_metabolic_genes_proteome_vs_flux.html

Plot 2 — Expression Burden (grouped bar chart):
    Four burden metrics—protein mass fraction, mRNA mass fraction, ribosome
    fraction, and RNAP fraction—are computed for the new gene set (averaged
    across all time steps and cells) and displayed as a grouped bar chart, one
    group of bars per variant.  Numeric values are also printed to stdout.
    Output: new_metabolic_genes_expression_burden.html
"""

from __future__ import annotations

import ast
import os
import pickle
import warnings
from typing import Any, TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
    from reconstruction.ecoli.fit_sim_data_1 import SimulationDataEcoli

alt.data_transformers.enable("vegafusion")

ANNOTATION_CSV = os.path.join(
    os.path.dirname(__file__),
    "../../../notebooks/Heena notebooks/Metabolism_New Genes"
    "/new_metabolic_gene_annotation.csv",
)
ANNOTATION_CSV = os.path.normpath(ANNOTATION_CSV)

LOG_EPS = 1e-10
S_PER_HR = 3600.0
PASTEL = px.colors.qualitative.Pastel + px.colors.qualitative.Pastel2


# ---------------------------------------------------------------------------
# Helpers — map monomer/mRNA IDs to listener column indices
# ---------------------------------------------------------------------------


def _get_mRNA_ids_from_monomer_ids(
    sim_data: "SimulationDataEcoli", target_monomer_ids: list[str]
) -> list[list[str]]:
    """Return a list (one entry per monomer) of mRNA TU ID lists."""
    monomer_ids = sim_data.process.translation.monomer_data["id"]
    cistron_ids = sim_data.process.translation.monomer_data["cistron_id"]
    monomer_to_cistron = dict(zip(monomer_ids, cistron_ids))
    RNA_ids = sim_data.process.transcription.rna_data["id"]
    result: list[list[str]] = []
    for mid in target_monomer_ids:
        cid = monomer_to_cistron.get(mid)
        if cid is None:
            result.append([])
            continue
        tu_idx = sim_data.process.transcription.cistron_id_to_rna_indexes(cid)
        result.append(RNA_ids[tu_idx].tolist())
    return result


def _mRNA_field_index_dict(
    conn: "DuckDBPyConnection", config_sql: str
) -> dict[str, int]:
    """Zero-based index map for ``listeners__rna_counts__mRNA_counts``."""
    return {
        rna: i
        for i, rna in enumerate(
            field_metadata(conn, config_sql, "listeners__rna_counts__mRNA_counts")
        )
    }


def _monomer_field_index_dict(
    conn: "DuckDBPyConnection", config_sql: str
) -> dict[str, int]:
    """Zero-based index map for ``listeners__monomer_counts``."""
    return {
        m: i
        for i, m in enumerate(
            field_metadata(conn, config_sql, "listeners__monomer_counts")
        )
    }


def _partial_mRNA_field_index_dict(
    conn: "DuckDBPyConnection", config_sql: str
) -> dict[str, int]:
    """Zero-based index map for ``listeners__rna_counts__partial_mRNA_counts``."""
    return {
        rna: i
        for i, rna in enumerate(
            field_metadata(
                conn, config_sql, "listeners__rna_counts__partial_mRNA_counts"
            )
        )
    }


def _parse_reactions(val: Any) -> list[str]:
    if isinstance(val, str) and val.startswith("["):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
) -> None:
    """
    Proteome-cost vs. flux contribution scatter and expression-burden bar chart
    for the ~307 new metabolic genes.
    """
    print("Loading simulation data...")
    with open_arbitrary_sim_data(sim_data_dict) as fh:
        sim_data: "SimulationDataEcoli" = pickle.load(fh)

    # ── Annotation CSV ─────────────────────────────────────────────────────
    annotation_df = pd.read_csv(ANNOTATION_CSV)
    gene_ids_set = set(annotation_df["Gene ID (EcoCyc)"].dropna().astype(str))
    annotation_df["Reactions_parsed"] = annotation_df["Reactions"].apply(
        _parse_reactions
    )

    # ── Map new genes → monomer metadata ───────────────────────────────────
    monomer_data = sim_data.process.translation.monomer_data
    all_monomer_gene_ids = np.array(monomer_data["gene_id"], dtype=str)
    all_monomer_ids = np.array(monomer_data["id"], dtype=str)
    all_monomer_mws = np.array(monomer_data["mw"])  # g/mol

    new_gene_mask = np.isin(all_monomer_gene_ids, list(gene_ids_set))
    new_monomer_ids = all_monomer_ids[new_gene_mask]
    new_monomer_mws = all_monomer_mws[new_gene_mask]
    new_monomer_gene_ids = all_monomer_gene_ids[new_gene_mask]

    if len(new_monomer_ids) == 0:
        warnings.warn("No monomers matched the annotation CSV gene IDs; aborting.")
        return

    print(f"  Matched {len(new_monomer_ids)} monomers for {len(gene_ids_set)} genes.")

    # Build gene_id → row(s) in annotation_df for name / pathway lookups
    gene_to_annotation: dict[str, pd.Series] = {}
    for _, row in annotation_df.iterrows():
        gid = str(row["Gene ID (EcoCyc)"])
        if gid not in gene_to_annotation:
            gene_to_annotation[gid] = row

    # ── DuckDB index lookups (1-indexed for list_select; 0-indexed elsewhere)
    monomer_idx_map = _monomer_field_index_dict(conn, config_sql)  # 0-based
    mRNA_idx_map = _mRNA_field_index_dict(conn, config_sql)  # 0-based
    partial_mRNA_idx_map = _partial_mRNA_field_index_dict(conn, config_sql)  # 0-based

    # Monomer positions in listener array
    new_monomer_listener_indices: list[int] = []
    for mid in new_monomer_ids:
        idx = monomer_idx_map.get(mid)
        if idx is None:
            warnings.warn(f"Monomer {mid} not found in listener metadata; skipping.")
        new_monomer_listener_indices.append(idx)  # may contain None

    valid_mask = np.array([i is not None for i in new_monomer_listener_indices])
    valid_monomer_ids = new_monomer_ids[valid_mask]
    valid_monomer_mws = new_monomer_mws[valid_mask]
    valid_monomer_gene_ids = new_monomer_gene_ids[valid_mask]
    valid_listener_indices = [
        i for i in new_monomer_listener_indices if i is not None
    ]  # list[int]

    # mRNA TU mappings (per valid monomer)
    all_mRNA_ids_per_monomer = _get_mRNA_ids_from_monomer_ids(
        sim_data, valid_monomer_ids.tolist()
    )
    # For mRNA counts listener (mRNA_counts field)
    mRNA_listener_indices: list[list[int]] = []
    for mRNA_ids in all_mRNA_ids_per_monomer:
        idxs = [mRNA_idx_map[r] for r in mRNA_ids if r in mRNA_idx_map]
        mRNA_listener_indices.append(idxs)

    # For partial_mRNA_counts listener (RNAP occupancy proxy)
    partial_mRNA_listener_indices: list[list[int]] = []
    for mRNA_ids in all_mRNA_ids_per_monomer:
        idxs = [partial_mRNA_idx_map[r] for r in mRNA_ids if r in partial_mRNA_idx_map]
        partial_mRNA_listener_indices.append(idxs)

    # mRNA molecular weights from rna_data
    rna_data = sim_data.process.transcription.rna_data
    all_rna_ids = np.array(rna_data["id"], dtype=str)
    rna_id_to_idx = {r: i for i, r in enumerate(all_rna_ids)}
    all_rna_mws = np.array(rna_data["mw"])  # g/mol

    # ── FBA reaction metadata ───────────────────────────────────────────────
    all_rxn_ids: list[str] = field_metadata(
        conn, config_sql, "listeners__fba_results__solution_fluxes"
    )
    rxn_id_to_idx = {r: i for i, r in enumerate(all_rxn_ids)}

    # Build per-monomer reaction index lists
    rxn_indices_per_monomer: list[list[int]] = []
    for gid in valid_monomer_gene_ids:
        ann_row = gene_to_annotation.get(gid)
        if ann_row is None:
            rxn_indices_per_monomer.append([])
            continue
        rxns = _parse_reactions(ann_row["Reactions"])
        idxs = [rxn_id_to_idx[r] for r in rxns if r in rxn_id_to_idx]
        rxn_indices_per_monomer.append(idxs)

    # Reverse map: mRNA listener index → rna_data index (for mw lookup)
    mrna_listener_idx_to_rna_data_idx: dict[int, int] = {}
    for rna_id, listener_idx in mRNA_idx_map.items():
        rna_data_idx = rna_id_to_idx.get(rna_id)
        if rna_data_idx is not None:
            mrna_listener_idx_to_rna_data_idx[listener_idx] = rna_data_idx

    # ── Load raw listener data ──────────────────────────────────────────────
    print("Loading listener data...")
    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            [
                "listeners__monomer_counts",
                "listeners__mass__protein_mass",
                "listeners__mass__mRna_mass",
                "listeners__rna_counts__mRNA_counts",
                "listeners__rna_counts__partial_mRNA_counts",
                "listeners__ribosome_data__n_ribosomes_per_transcript",
                "listeners__unique_molecule_counts__active_ribosome",
                "listeners__unique_molecule_counts__active_RNAP",
                "listeners__fba_results__solution_fluxes",
            ],
            order_results=True,
            conn=conn,
            remove_first=True,
        )
    )

    if raw.is_empty():
        print("new_metabolic_genes_analysis: no rows returned; skipping.")
        return

    n_rows = len(raw)
    print(f"  Loaded {n_rows} time-step rows.")

    # Convert list columns → numpy arrays
    monomer_counts_arr = ndlist_to_ndarray(
        raw["listeners__monomer_counts"]
    )  # (T, n_monomers)
    protein_mass_arr = raw["listeners__mass__protein_mass"].to_numpy()  # (T,)
    mrna_mass_arr = raw["listeners__mass__mRna_mass"].to_numpy()  # (T,)
    mrna_counts_arr = ndlist_to_ndarray(
        raw["listeners__rna_counts__mRNA_counts"]
    )  # (T, n_mRNAs)
    partial_mrna_arr = ndlist_to_ndarray(
        raw["listeners__rna_counts__partial_mRNA_counts"]
    )  # (T, n_mRNAs)
    ribosome_per_transcript_arr = ndlist_to_ndarray(
        raw["listeners__ribosome_data__n_ribosomes_per_transcript"]
    )  # (T, n_monomers)
    active_ribosome_arr = raw[
        "listeners__unique_molecule_counts__active_ribosome"
    ].to_numpy()  # (T,)
    active_rnap_arr = raw[
        "listeners__unique_molecule_counts__active_RNAP"
    ].to_numpy()  # (T,)
    solution_fluxes_arr = ndlist_to_ndarray(
        raw["listeners__fba_results__solution_fluxes"]
    )  # (T, n_rxns)
    variants_col = np.array(raw["variant"].to_list())

    # Convert fluxes to mmol/(L·h) using counts-to-molar if available, else
    # assume the field already has appropriate units (dimensionless ratio).
    # For proteome vs flux we only need relative differences so unit
    # consistency within a plot is sufficient; we label axes accordingly.
    # solution_fluxes are typically in mmol/(L·s) × counts_to_molar;
    # here we just convert seconds → hours.
    solution_fluxes_hr = solution_fluxes_arr * S_PER_HR  # (T, n_rxns)

    unique_variants: list[int] = sorted(raw["variant"].unique().to_list())

    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    def _make_label(v: int) -> str:
        lbl = _variant_label(v, per_variant_params)
        return " ".join(lbl) if isinstance(lbl, list) else str(lbl)

    variant_label_map = {v: _make_label(v) for v in unique_variants}

    # ── Avogadro / unit helpers ─────────────────────────────────────────────
    # monomer_mws are in g/mol → convert to fg/molecule:
    # fg/molecule = g/mol / n_avogadro * 1e15
    n_avogadro = float(sim_data.constants.n_avogadro.asNumber())
    g_per_mol_to_fg = 1e15 / n_avogadro  # fg per molecule for mw in g/mol

    # ── Plot 1: Proteome cost vs. flux contribution ─────────────────────────
    print("Computing proteome cost vs. flux contribution...")

    pathway_colors: dict[str, str] = {}
    scatter_rows: list[dict] = []

    for v in unique_variants:
        v_label = variant_label_map[v]
        v_mask = variants_col == v
        mon_counts_v = monomer_counts_arr[v_mask]  # (T_v, n_mon)
        fluxes_v = solution_fluxes_hr[v_mask]  # (T_v, n_rxns)

        for i, (mid, gid) in enumerate(zip(valid_monomer_ids, valid_monomer_gene_ids)):
            li = valid_listener_indices[i]
            mw_fg = valid_monomer_mws[i] * g_per_mol_to_fg

            avg_protein_mass_fg = float(np.mean(mon_counts_v[:, li]) * mw_fg)

            rxn_idxs = rxn_indices_per_monomer[i]
            if rxn_idxs:
                avg_flux = float(np.mean(np.sum(np.abs(fluxes_v[:, rxn_idxs]), axis=1)))
            else:
                avg_flux = 0.0

            ann_row = gene_to_annotation.get(gid)
            gene_name = str(ann_row["Gene name"]) if ann_row is not None else gid
            pathway_parent = (
                str(ann_row["Pathways parent"])
                if ann_row is not None
                and not pd.isna(ann_row.get("Pathways parent", float("nan")))
                else "Unknown"
            )

            if pathway_parent not in pathway_colors:
                ci = len(pathway_colors) % len(PASTEL)
                pathway_colors[pathway_parent] = PASTEL[ci]

            scatter_rows.append(
                {
                    "Gene": gene_name,
                    "Monomer ID": mid,
                    "Gene ID": gid,
                    "Avg Protein Mass (fg)": avg_protein_mass_fg,
                    "Avg Flux (mmol/L/h)": avg_flux,
                    "log_protein_mass": float(np.log10(avg_protein_mass_fg + LOG_EPS)),
                    "log_flux": float(np.log10(avg_flux + LOG_EPS)),
                    "Pathway": pathway_parent,
                    "Variant": v_label,
                    "n_reactions": len(rxn_idxs),
                }
            )

    scatter_df = pd.DataFrame(scatter_rows)

    # Annotate outliers: high cost + low flux, or high flux
    annotated_genes: set[str] = set()
    for v_label in scatter_df["Variant"].unique():
        sub = scatter_df[scatter_df["Variant"] == v_label]
        cost_thresh = sub["log_protein_mass"].quantile(0.95)
        flux_med = sub["log_flux"].median()
        top_flux_thresh = sub["log_flux"].quantile(0.95)
        high_cost_low_flux = sub[
            (sub["log_protein_mass"] >= cost_thresh) & (sub["log_flux"] <= flux_med)
        ]["Gene"].tolist()
        high_flux = sub[sub["log_flux"] >= top_flux_thresh]["Gene"].tolist()
        annotated_genes.update(high_cost_low_flux + high_flux)

    scatter_df["annotate"] = scatter_df["Gene"].isin(annotated_genes)
    scatter_df["annotation_label"] = scatter_df.apply(
        lambda r: r["Gene"] if r["annotate"] else "", axis=1
    )

    n_pathways = scatter_df["Pathway"].nunique()
    pathway_domain = scatter_df["Pathway"].unique().tolist()
    pathway_range = [PASTEL[i % len(PASTEL)] for i in range(n_pathways)]
    color_scale = alt.Scale(domain=pathway_domain, range=pathway_range)

    num_cols = min(len(unique_variants), 4)

    scatter_pts = (
        alt.Chart()
        .mark_circle(size=60, opacity=0.75)
        .encode(
            x=alt.X(
                "log_protein_mass:Q",
                title="log₁₀(Average Protein Mass (fg))",
            ),
            y=alt.Y(
                "log_flux:Q",
                title="log₁₀(Average Metabolic Flux (mmol/L/h))",
            ),
            color=alt.Color(
                "Pathway:N",
                scale=color_scale,
                legend=alt.Legend(title="Pathway (parent)"),
            ),
            tooltip=[
                alt.Tooltip("Gene:N", title="Gene"),
                alt.Tooltip(
                    "Avg Protein Mass (fg):Q", format=".3e", title="Protein mass (fg)"
                ),
                alt.Tooltip(
                    "Avg Flux (mmol/L/h):Q", format=".3e", title="Flux (mmol/L/h)"
                ),
                alt.Tooltip("Pathway:N"),
                alt.Tooltip("n_reactions:Q", title="# reactions"),
            ],
        )
    )

    text_layer = (
        alt.Chart()
        .mark_text(fontSize=8, dx=4, dy=-4, align="left")
        .transform_filter("datum.annotate")
        .encode(
            x=alt.X("log_protein_mass:Q"),
            y=alt.Y("log_flux:Q"),
            text=alt.Text("Gene:N"),
        )
    )

    scatter_faceted = (
        alt.layer(scatter_pts, text_layer, data=scatter_df)
        .properties(width=300, height=300)
        .facet(facet=alt.Facet("Variant:N", title="Variant"), columns=num_cols)
        .resolve_scale(x="independent", y="independent", color="shared")
        .properties(title="Proteome Cost vs. Flux Contribution for New Metabolic Genes")
    )

    out1 = os.path.join(outdir, "new_metabolic_genes_proteome_vs_flux.html")
    scatter_faceted.save(out1)
    print(f"  Saved scatter plot → {out1}")

    # ── Plot 2: Expression burden ───────────────────────────────────────────
    print("Computing expression burden metrics...")

    burden_rows: list[dict] = []

    for v in unique_variants:
        v_label = variant_label_map[v]
        v_mask = variants_col == v

        mon_counts_v = monomer_counts_arr[v_mask]
        prot_mass_v = protein_mass_arr[v_mask]
        mrna_counts_v = mrna_counts_arr[v_mask]
        mrna_mass_v = mrna_mass_arr[v_mask]
        partial_mrna_v = partial_mrna_arr[v_mask]
        rib_per_tx_v = ribosome_per_transcript_arr[v_mask]
        active_rib_v = active_ribosome_arr[v_mask]
        active_rnap_v = active_rnap_arr[v_mask]

        # 1. Protein mass fraction
        new_gene_prot_mass_ts = np.zeros(v_mask.sum())
        for i, li in enumerate(valid_listener_indices):
            mw_fg = valid_monomer_mws[i] * g_per_mol_to_fg
            new_gene_prot_mass_ts += mon_counts_v[:, li] * mw_fg

        with np.errstate(divide="ignore", invalid="ignore"):
            prot_frac_ts = np.where(
                prot_mass_v > 0, new_gene_prot_mass_ts / prot_mass_v, 0.0
            )
        mean_prot_frac = float(np.mean(prot_frac_ts))
        std_prot_frac = float(np.std(prot_frac_ts))

        # 2. mRNA mass fraction — mRNA counts × mRNA mw / total mRNA mass
        # For each monomer's TUs: sum counts × rna_mw across TUs
        new_gene_mrna_mass_ts = np.zeros(v_mask.sum())
        for i, mRNA_idxs in enumerate(mRNA_listener_indices):
            for ri in mRNA_idxs:
                rna_data_idx = mrna_listener_idx_to_rna_data_idx.get(ri)
                if rna_data_idx is None:
                    continue
                mw_fg = all_rna_mws[rna_data_idx] * g_per_mol_to_fg
                new_gene_mrna_mass_ts += mrna_counts_v[:, ri] * mw_fg

        with np.errstate(divide="ignore", invalid="ignore"):
            mrna_frac_ts = np.where(
                mrna_mass_v > 0, new_gene_mrna_mass_ts / mrna_mass_v, 0.0
            )
        mean_mrna_frac = float(np.mean(mrna_frac_ts))
        std_mrna_frac = float(np.std(mrna_frac_ts))

        # 3. Ribosome fraction
        new_gene_rib_ts = np.zeros(v_mask.sum())
        for i, li in enumerate(valid_listener_indices):
            new_gene_rib_ts += rib_per_tx_v[:, li]

        with np.errstate(divide="ignore", invalid="ignore"):
            rib_frac_ts = np.where(
                active_rib_v > 0, new_gene_rib_ts / active_rib_v, 0.0
            )
        mean_rib_frac = float(np.mean(rib_frac_ts))
        std_rib_frac = float(np.std(rib_frac_ts))

        # 4. RNAP fraction via partial_mRNA_counts
        new_gene_rnap_ts = np.zeros(v_mask.sum())
        for i, partial_idxs in enumerate(partial_mRNA_listener_indices):
            for ri in partial_idxs:
                new_gene_rnap_ts += partial_mrna_v[:, ri]

        with np.errstate(divide="ignore", invalid="ignore"):
            rnap_frac_ts = np.where(
                active_rnap_v > 0, new_gene_rnap_ts / active_rnap_v, 0.0
            )
        mean_rnap_frac = float(np.mean(rnap_frac_ts))
        std_rnap_frac = float(np.std(rnap_frac_ts))

        print(
            f"\nVariant: {v_label}"
            f"\n  Protein mass fraction : {mean_prot_frac:.4f} ± {std_prot_frac:.4f}"
            f"\n  mRNA mass fraction    : {mean_mrna_frac:.4f} ± {std_mrna_frac:.4f}"
            f"\n  Ribosome fraction     : {mean_rib_frac:.4f} ± {std_rib_frac:.4f}"
            f"\n  RNAP fraction         : {mean_rnap_frac:.4f} ± {std_rnap_frac:.4f}"
        )

        for metric, mean_val, std_val in [
            ("Protein Mass Fraction", mean_prot_frac, std_prot_frac),
            ("mRNA Mass Fraction", mean_mrna_frac, std_mrna_frac),
            ("Ribosome Fraction", mean_rib_frac, std_rib_frac),
            ("RNAP Fraction", mean_rnap_frac, std_rnap_frac),
        ]:
            burden_rows.append(
                {
                    "Metric": metric,
                    "Mean": mean_val * 100,
                    "Std": std_val * 100,
                    "Variant": v_label,
                }
            )

    burden_df = pd.DataFrame(burden_rows)

    n_variants = burden_df["Variant"].nunique()
    variant_domain = burden_df["Variant"].unique().tolist()
    variant_color_range = PASTEL[:n_variants]
    variant_color_scale = alt.Scale(domain=variant_domain, range=variant_color_range)

    metric_order = [
        "Protein Mass Fraction",
        "mRNA Mass Fraction",
        "Ribosome Fraction",
        "RNAP Fraction",
    ]

    bar_base = alt.Chart(burden_df).encode(
        x=alt.X(
            "Variant:N",
            title=None,
            axis=alt.Axis(labelAngle=-30),
        ),
        color=alt.Color("Variant:N", scale=variant_color_scale, legend=None),
        tooltip=[
            alt.Tooltip("Variant:N"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Mean:Q", format=".3f", title="Mean (%)"),
            alt.Tooltip("Std:Q", format=".3f", title="Std (%)"),
        ],
    )

    bars = bar_base.mark_bar(opacity=0.85).encode(
        y=alt.Y("Mean:Q", title="Fraction (%)"),
    )

    error_bars = bar_base.mark_errorbar().encode(
        y=alt.Y("Mean:Q"),
        yError=alt.YError("Std:Q"),
    )

    burden_chart = (
        alt.layer(bars, error_bars)
        .properties(width=120, height=250)
        .facet(
            facet=alt.Facet(
                "Metric:N",
                title="Expression Burden Metric",
                sort=metric_order,
            ),
            columns=4,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(title="Expression Burden of ~307 New Metabolic Genes")
    )

    out2 = os.path.join(outdir, "new_metabolic_genes_expression_burden.html")
    burden_chart.save(out2)
    print(f"\n  Saved burden chart → {out2}")
    print(f"Saved plots to {outdir}")
