"""
Strain health diagnostic dashboard.

Emits a single ``strain_dashboard.html`` plus a few CSVs summarising
mass / growth / resource-allocation / new-gene / essential-protein
metrics across all variants in an experiment. Designed to validate
strain-design variants (`ecoli.variants.strain_design` and
`ecoli.variants.native_translation_perturbation`).

Each diagnostic is a single per-cell summary number (mean / max / event
time), drawn as a boxplot grouped by variant. The essential-protein
panel shows the *distribution* of per-protein proteome-fraction ratios
against the control variant (one point per essential protein per variant).

Config usage::

    "analysis_options": {
        "multivariant": {
            "strain_dashboard": {
                "skip_first_n_gens": 0,
                "control_variant": 0
            }
        }
    }
"""

from __future__ import annotations

import json
import math
import os
import pickle
import warnings
from typing import Any, cast

import altair as alt
import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps import (
    get_indexes,
    get_mRNA_ids_from_monomer_ids,
)
from ecoli.library.parquet_emitter import field_metadata, open_arbitrary_sim_data


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _gene_class_cistron_ids(sim_data) -> dict[str, list[str]]:
    """
    Cistron-id lists for each gene class. Uses the boolean masks on
    ``cistron_data``.
    """
    cd = sim_data.process.transcription.cistron_data.struct_array
    out: dict[str, list[str]] = {}
    if "is_rRNA" in cd.dtype.names:
        out["rrna"] = cd[cd["is_rRNA"]]["id"].tolist()
    if "is_ribosomal_protein" in cd.dtype.names:
        out["ribosomal_protein"] = cd[cd["is_ribosomal_protein"]]["id"].tolist()
    if "is_RNAP" in cd.dtype.names:
        out["rnap_subunit"] = cd[cd["is_RNAP"]]["id"].tolist()
    if "is_new_gene" in cd.dtype.names:
        out["new_gene"] = cd[cd["is_new_gene"]]["id"].tolist()
    return out


def _gene_class_monomer_ids(sim_data, cistron_ids: list[str]) -> list[str]:
    monomer_data = sim_data.process.translation.monomer_data.struct_array
    cistron_to_monomer = dict(zip(monomer_data["cistron_id"], monomer_data["id"]))
    return [cistron_to_monomer[c] for c in cistron_ids if c in cistron_to_monomer]


def _resolve_perturbed_cistron_ids(variant_params: Any, sim_data) -> list[str]:
    if not isinstance(variant_params, dict):
        return []
    perturbations = variant_params.get("perturbations") or {}
    if not isinstance(perturbations, dict) or not perturbations:
        return []
    cd = sim_data.process.transcription.cistron_data.struct_array
    gene_id_to_cistron_id = dict(zip(cd["gene_id"], cd["id"]))
    return [
        gene_id_to_cistron_id[g] for g in perturbations if g in gene_id_to_cistron_id
    ]


def _load_essential_protein_monomer_indexes(
    validation_data_paths: list[str],
    conn: DuckDBPyConnection,
    config_sql: str,
) -> tuple[list[int], list[str]]:
    """Load essential proteins and return (indexes_1based, monomer_ids)."""
    if not validation_data_paths:
        return [], []
    try:
        with open(validation_data_paths[0], "rb") as f:
            vd = pickle.load(f)
        essential_proteins = list(vd.essential_genes.essential_proteins)
    except Exception as e:
        warnings.warn(f"strain_dashboard: could not load essential proteins: {e}")
        return [], []
    monomer_ids = field_metadata(conn, config_sql, "listeners__monomer_counts")
    idx_dict = {m: i + 1 for i, m in enumerate(monomer_ids)}
    matched = [(m, idx_dict[m]) for m in essential_proteins if m in idx_dict]
    if not matched:
        return [], []
    matched_ids = [m for m, _ in matched]
    matched_idx = [i for _, i in matched]
    return matched_idx, matched_ids


def _gen_filter_clause(skip_first_n_gens: int) -> str:
    if skip_first_n_gens <= 0:
        return ""
    return f" AND generation > {skip_first_n_gens}"


def _variant_label(variant_idx: int, variant_params: Any) -> str:
    if not isinstance(variant_params, dict):
        return f"variant {variant_idx} (baseline)"
    pert = variant_params.get("perturbations") or {}
    ng = variant_params.get("new_gene_shift") or {}
    bits = []
    if pert:
        bits.append(f"{len(pert)} perturbations")
    if ng:
        trl = (ng.get("exp_trl_eff") or {}).get("trl_eff")
        if trl is not None:
            bits.append(f"GFP trl_eff={trl}")
    return f"variant {variant_idx}" + (f" ({', '.join(bits)})" if bits else "")


def _cell_id(variant: int, lineage_seed: int, generation: int, agent_id: str) -> str:
    return f"Cell: {variant}_{lineage_seed}_{generation}_{agent_id}"


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------


def _cistron_to_monomer_id_map(sim_data) -> dict[str, str]:
    monomer = sim_data.process.translation.monomer_data.struct_array
    return dict(zip(monomer["cistron_id"], monomer["id"]))


def _tu_indexes_for_class(
    conn: DuckDBPyConnection,
    config_sql: str,
    sim_data,
    cistron_ids: list[str],
) -> list[int]:
    if not cistron_ids:
        return []
    cmap = _cistron_to_monomer_id_map(sim_data)
    monomer_ids = [cmap[c] for c in cistron_ids if c in cmap]
    if not monomer_ids:
        return []
    nested = cast(
        list[list[int | None]],
        get_indexes(
            conn,
            config_sql,
            "mRNA",
            get_mRNA_ids_from_monomer_ids(sim_data, monomer_ids),
        ),
    )
    return sorted({i for grp in nested for i in (grp or []) if i is not None})


def _monomer_indexes_for_class(
    conn: DuckDBPyConnection,
    config_sql: str,
    sim_data,
    cistron_ids: list[str],
) -> list[int]:
    if not cistron_ids:
        return []
    cmap = _cistron_to_monomer_id_map(sim_data)
    monomer_ids = [cmap[c] for c in cistron_ids if c in cmap]
    if not monomer_ids:
        return []
    idxs = cast(list[int | None], get_indexes(conn, config_sql, "monomer", monomer_ids))
    return [i for i in idxs if i is not None]


def _idx_list_sql(indexes: list[int]) -> str:
    return "[" + ",".join(str(i) for i in indexes) + "]"


def _per_cell_metrics(
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    sim_data,
    class_cistron_ids: dict[str, list[str]],
    perturbed_cistron_ids_per_variant: dict[int, list[str]],
    new_gene_monomer_ids: list[str],
    essential_idx: list[int],
    skip_clause: str,
) -> pl.DataFrame:
    """
    Compute one summary row per cell (variant, lineage_seed, generation,
    agent_id) with all scalar diagnostics. Heavy SQL but only one full pass.
    """
    # Build index sets that are variant-independent.
    rnap_idx = {
        cls: _tu_indexes_for_class(conn, config_sql, sim_data, ids)
        for cls, ids in class_cistron_ids.items()
        if cls != "rrna"
    }
    rnap_idx["rrna"] = []  # rRNA uses a separate listener entirely
    ribo_idx = {
        cls: _monomer_indexes_for_class(conn, config_sql, sim_data, ids)
        for cls, ids in class_cistron_ids.items()
    }
    ribo_idx["rrna"] = []  # rRNAs aren't translated
    new_gene_monomer_indexes = (
        [
            i
            for i in cast(
                list[int | None],
                get_indexes(conn, config_sql, "monomer", new_gene_monomer_ids),
            )
            if i is not None
        ]
        if new_gene_monomer_ids
        else []
    )

    # SQL: per-cell aggregations. Each variant has its own "perturbed" set,
    # so we compute that class via a CASE expression that switches index list
    # by variant. To keep SQL bounded, do per-variant SELECT then UNION ALL.
    parts: list[str] = []
    for variant, pert_cistron_ids in perturbed_cistron_ids_per_variant.items():
        pert_tu = _tu_indexes_for_class(conn, config_sql, sim_data, pert_cistron_ids)
        pert_mono = _monomer_indexes_for_class(
            conn, config_sql, sim_data, pert_cistron_ids
        )
        rnap_class_exprs = [
            (f"rnap_frac_{cls}", _rnap_class_numer_expr(cls, rnap_idx[cls]))
            for cls in ["rrna", "ribosomal_protein", "rnap_subunit", "new_gene"]
        ]
        rnap_class_exprs.append(
            ("rnap_frac_perturbed", _rnap_class_numer_expr("perturbed", pert_tu))
        )
        ribo_class_exprs = [
            (f"ribo_frac_{cls}", _ribo_class_numer_expr(ribo_idx[cls]))
            for cls in ["rrna", "ribosomal_protein", "rnap_subunit", "new_gene"]
        ]
        ribo_class_exprs.append(
            ("ribo_frac_perturbed", _ribo_class_numer_expr(pert_mono))
        )
        new_gene_peak_expr = (
            f"max(list_sum(list_select(listeners__monomer_counts, "
            f"{_idx_list_sql(new_gene_monomer_indexes)})))"
            if new_gene_monomer_indexes
            else "NULL"
        )
        ess_frac_expr = (
            f"avg(list_sum(list_select(listeners__monomer_counts, "
            f"{_idx_list_sql(essential_idx)}))::DOUBLE "
            f"/ NULLIF(list_sum(listeners__monomer_counts), 0))"
            if essential_idx
            else "NULL"
        )
        alloc_select_lines = ",\n            ".join(
            f"avg(({expr})::DOUBLE / NULLIF("
            f"listeners__unique_molecule_counts__active_{kind}, 0)) AS {alias}"
            for kind, (alias, expr) in (
                [("RNAP", e) for e in rnap_class_exprs]
                + [("ribosome", e) for e in ribo_class_exprs]
            )
        )
        parts.append(
            f"""
            SELECT
                {variant} AS variant,
                lineage_seed, generation, agent_id,
                max(listeners__mass__cell_mass) AS peak_cell_mass_fg,
                max(listeners__mass__protein_mass) AS peak_protein_mass_fg,
                avg(CASE
                    WHEN listeners__mass__instantaneous_growth_rate > 0
                    THEN LN(2) / listeners__mass__instantaneous_growth_rate / 60.0
                    ELSE NULL END) AS doubling_time_min,
                avg(listeners__ribosome_data__effective_elongation_rate)
                    AS mean_aa_per_s_per_ribo,
                sum(listeners__ribosome_data__actual_elongations)
                    AS total_aa_elongations,
                sum(listeners__rnap_data__actual_elongations)
                    AS total_nt_elongations,
                avg(listeners__unique_molecule_counts__active_ribosome)
                    AS mean_n_ribosome,
                avg(listeners__unique_molecule_counts__active_RNAP)
                    AS mean_n_rnap,
                {new_gene_peak_expr} AS new_gene_protein_peak,
                {ess_frac_expr} AS essential_proteome_fraction,
                {alloc_select_lines}
            FROM ({history_sql})
            WHERE variant = {variant} {skip_clause}
            GROUP BY lineage_seed, generation, agent_id
            """
        )
    sql = " UNION ALL ".join(parts)
    base = conn.sql(sql).pl()
    # Coerce Decimal columns produced by DuckDB avg() to Float64 so altair
    # can serialise them.
    base = base.with_columns(
        *[
            pl.col(c).cast(pl.Float64).alias(c)
            for c in base.columns
            if base.schema[c] not in (pl.Int64, pl.Int32, pl.Utf8)
        ]
    )

    # Add replication-initiation per cell.
    repl_init = _replication_initiation_per_cell(conn, history_sql, skip_clause)
    return base.join(
        repl_init,
        on=["variant", "lineage_seed", "generation", "agent_id"],
        how="left",
    ).sort(["variant", "lineage_seed", "generation", "agent_id"])


def _rnap_class_numer_expr(cls: str, tu_indexes: list[int]) -> str:
    """SQL fragment for the numerator of an RNAP allocation fraction."""
    if cls == "rrna":
        # rRNA RNAPs are in their own listener entirely; tu_indexes ignored.
        return "list_sum(listeners__rna_counts__partial_rRNA_counts)"
    if not tu_indexes:
        return "0"
    return (
        f"list_sum(list_select(listeners__rna_counts__partial_mRNA_counts, "
        f"{_idx_list_sql(tu_indexes)}))"
    )


def _ribo_class_numer_expr(monomer_indexes: list[int]) -> str:
    if not monomer_indexes:
        return "0"
    return (
        f"list_sum(list_select("
        f"listeners__ribosome_data__n_ribosomes_per_transcript, "
        f"{_idx_list_sql(monomer_indexes)}))"
    )


def _replication_initiation_per_cell(
    conn: DuckDBPyConnection, history_sql: str, skip_clause: str
) -> pl.DataFrame:
    sql = f"""
        WITH base AS (
            SELECT variant, lineage_seed, generation, agent_id, time,
                listeners__mass__cell_mass AS cell_mass_fg,
                listeners__replication_data__number_of_oric AS n_oric
            FROM ({history_sql})
            WHERE 1=1 {skip_clause}
        ),
        cell_bounds AS (
            SELECT variant, lineage_seed, generation, agent_id,
                MIN(time) AS time_birth,
                arg_min(n_oric, time) AS n_oric_at_birth
            FROM base
            GROUP BY variant, lineage_seed, generation, agent_id
        ),
        elevated AS (
            SELECT base.variant, base.lineage_seed, base.generation, base.agent_id,
                MIN(base.time) AS time_init,
                arg_min(base.cell_mass_fg, base.time) AS cell_mass_at_init
            FROM base
            JOIN cell_bounds USING (variant, lineage_seed, generation, agent_id)
            WHERE base.n_oric > cell_bounds.n_oric_at_birth
            GROUP BY base.variant, base.lineage_seed, base.generation, base.agent_id
        )
        SELECT cell_bounds.variant, cell_bounds.lineage_seed,
            cell_bounds.generation, cell_bounds.agent_id,
            (time_init - time_birth) / 60.0 AS time_to_init_min,
            cell_mass_at_init AS cell_mass_at_init_fg
        FROM cell_bounds
        LEFT JOIN elevated USING (variant, lineage_seed, generation, agent_id)
    """
    return conn.sql(sql).pl()


# ---------------------------------------------------------------------------
# Per-essential-protein ratios
# ---------------------------------------------------------------------------


def _essential_protein_ratios(
    conn: DuckDBPyConnection,
    history_sql: str,
    essential_idx: list[int],
    essential_ids: list[str],
    control_variant: int,
    skip_clause: str,
    cell_keys: list[tuple[int, int, int, str]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute, for each essential protein i and each variant V:

        frac_i^V = mean across cells of variant V of (mean over time of
                   count_i / total_count)
        ratio_i^V = frac_i^V / frac_i^{control}

    Returns:
        - ratios_df: row per protein, columns: monomer_id, frac_variant_<N>,
          ratio_variant_<N>.
        - summary_df: row per variant, columns: variant, n_proteins, mean,
          std, median, min, max.
    """
    if not essential_idx:
        return pl.DataFrame(), pl.DataFrame()

    idx_str = _idx_list_sql(essential_idx)
    sql = f"""
        WITH per_row AS (
            SELECT variant, lineage_seed, generation, agent_id, time,
                list_select(listeners__monomer_counts, {idx_str}) AS ess_counts,
                list_sum(listeners__monomer_counts) AS total_count
            FROM ({history_sql})
            WHERE 1=1 {skip_clause}
        ),
        per_row_unnested AS (
            SELECT variant, lineage_seed, generation, agent_id,
                unnest(ess_counts) AS count_i,
                generate_subscripts(ess_counts, 1) AS protein_idx,
                total_count
            FROM per_row
        ),
        per_cell AS (
            SELECT variant, lineage_seed, generation, agent_id, protein_idx,
                avg(count_i::DOUBLE / NULLIF(total_count, 0)) AS frac
            FROM per_row_unnested
            GROUP BY variant, lineage_seed, generation, agent_id, protein_idx
        )
        SELECT variant, lineage_seed, generation, agent_id, protein_idx, frac
        FROM per_cell
        ORDER BY variant, lineage_seed, generation, agent_id, protein_idx
    """
    long = conn.sql(sql).pl().with_columns(pl.col("frac").cast(pl.Float64))
    if long.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    # Per-variant mean fraction (averaged across cells of that variant).
    per_variant = long.group_by(["variant", "protein_idx"]).agg(pl.col("frac").mean())
    wide = per_variant.pivot(index="protein_idx", on="variant", values="frac").rename(
        {str(v): f"frac_variant_{v}" for v in per_variant["variant"].unique()}
    )
    # Attach monomer_id (protein_idx is 1-indexed → 0-based).
    wide = wide.with_columns(
        monomer_id=pl.Series(
            "monomer_id",
            [essential_ids[i - 1] for i in wide["protein_idx"].to_list()],
        )
    )
    # Per-cell proteome-fraction columns "Cell: <variant>_<lineage>_<gen>_<agent>".
    if cell_keys is None:
        cell_keys = []
    per_cell_pivot = pl.DataFrame()
    if cell_keys:
        # Wide per-cell frame: one column per cell.
        per_cell_long = long.with_columns(
            cell_label=pl.format(
                "Cell: {}_{}_{}_{}",
                pl.col("variant"),
                pl.col("lineage_seed"),
                pl.col("generation"),
                pl.col("agent_id"),
            )
        )
        per_cell_pivot = per_cell_long.pivot(
            index="protein_idx", on="cell_label", values="frac"
        )
    if not per_cell_pivot.is_empty():
        wide = wide.join(per_cell_pivot, on="protein_idx", how="left")
    control_col = f"frac_variant_{control_variant}"
    if control_col not in wide.columns:
        return wide, pl.DataFrame()
    ratio_cols = []
    for col in [c for c in wide.columns if c.startswith("frac_variant_")]:
        v = int(col.split("_")[-1])
        if v == control_variant:
            continue
        wide = wide.with_columns(
            (pl.col(col) / pl.col(control_col)).alias(f"ratio_variant_{v}")
        )
        ratio_cols.append(f"ratio_variant_{v}")
    # Final column order: monomer_id, per-variant fractions, per-variant ratios,
    # then per-cell columns.
    frac_cols = sorted(c for c in wide.columns if c.startswith("frac_variant_"))
    cell_cols = [c for c in wide.columns if c.startswith("Cell: ")]
    wide = wide.select(["monomer_id"] + frac_cols + ratio_cols + cell_cols)
    # Summary across proteins per non-control variant.
    summary_rows: list[dict[str, Any]] = []
    for col in ratio_cols:
        v = int(col.split("_")[-1])
        s = wide[col].drop_nulls()
        if s.is_empty():
            continue
        summary_rows.append(
            {
                "variant": v,
                "n_proteins": int(s.len()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
                "n_below_0_5": int((s < 0.5).sum()),
            }
        )
    summary = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    return wide, summary


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


# Diagnostic name → (column in per-cell frame, display title).
_DIAGNOSTIC_LABELS: list[tuple[str, str]] = [
    ("doubling_time_min", "Doubling time (min)"),
    ("peak_cell_mass_fg", "Peak cell mass (fg)"),
    ("peak_protein_mass_fg", "Peak protein mass (fg)"),
    ("mean_aa_per_s_per_ribo", "Effective elongation rate (AA/s/ribo)"),
    ("total_aa_elongations", "Total AA elongated (per cell)"),
    ("total_nt_elongations", "Total nt elongated (per cell)"),
    ("mean_n_ribosome", "Mean active ribosomes"),
    ("mean_n_rnap", "Mean active RNAPs"),
    ("time_to_init_min", "Time to replication initiation (min)"),
    ("cell_mass_at_init_fg", "Cell mass at initiation (fg)"),
    ("essential_proteome_fraction", "Essential proteome fraction"),
    ("new_gene_protein_peak", "Peak new-gene protein (sum)"),
    ("rnap_frac_rrna", "RNAP fraction on rRNA"),
    ("rnap_frac_ribosomal_protein", "RNAP fraction on ribosomal proteins"),
    ("rnap_frac_rnap_subunit", "RNAP fraction on RNAP subunits"),
    ("rnap_frac_new_gene", "RNAP fraction on new genes"),
    ("rnap_frac_perturbed", "RNAP fraction on perturbed genes"),
    ("ribo_frac_ribosomal_protein", "Ribosome fraction on ribosomal proteins"),
    ("ribo_frac_rnap_subunit", "Ribosome fraction on RNAP subunits"),
    ("ribo_frac_new_gene", "Ribosome fraction on new genes"),
    ("ribo_frac_perturbed", "Ribosome fraction on perturbed genes"),
]


def _per_cell_diagnostics_csv(per_cell: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot per-cell metrics into the user-specified row-per-diagnostic shape:
    diagnostic | mean | std | Cell:<id> | Cell:<id> | ...
    """
    if per_cell.is_empty():
        return pl.DataFrame()
    cell_cols: list[tuple[str, dict[str, Any]]] = []
    for row in per_cell.iter_rows(named=True):
        label = _cell_id(
            row["variant"], row["lineage_seed"], row["generation"], row["agent_id"]
        )
        cell_cols.append((label, row))
    rows: list[dict[str, Any]] = []
    for metric_col, metric_label in _DIAGNOSTIC_LABELS:
        if metric_col not in per_cell.columns:
            continue
        values = per_cell[metric_col].to_list()
        floats = [v for v in values if v is not None and not _is_nan(v)]
        row_out: dict[str, Any] = {
            "diagnostic": metric_label,
            "mean": (float(np.mean(floats)) if floats else None),
            "std": (float(np.std(floats, ddof=1)) if len(floats) > 1 else None),
        }
        for label, row in cell_cols:
            row_out[label] = row.get(metric_col)
        rows.append(row_out)
    return pl.DataFrame(rows)


def _per_variant_diagnostics_csv(per_cell: pl.DataFrame) -> pl.DataFrame:
    """
    Per-variant aggregate table: variant | n_cells | <metric>_mean |
    <metric>_std for each metric. Handy for spreadsheet diffing.
    """
    if per_cell.is_empty():
        return pl.DataFrame()
    aggs: list[pl.Expr] = [pl.len().alias("n_cells")]
    for metric_col, _label in _DIAGNOSTIC_LABELS:
        if metric_col not in per_cell.columns:
            continue
        aggs.append(pl.col(metric_col).mean().alias(f"{metric_col}_mean"))
        aggs.append(pl.col(metric_col).std().alias(f"{metric_col}_std"))
    return per_cell.group_by("variant", maintain_order=True).agg(*aggs).sort("variant")


def _strain_metadata_table(
    variant_metadata: dict[int, Any], variant_name: str
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_idx in sorted(variant_metadata.keys()):
        params = variant_metadata[variant_idx]
        row: dict[str, Any] = {
            "variant": variant_idx,
            "variant_name": variant_name,
        }
        if isinstance(params, dict):
            row["condition"] = params.get("condition") or ""
            pert = params.get("perturbations")
            row["perturbations_json"] = (
                json.dumps(pert) if isinstance(pert, dict) and pert else ""
            )
            ng = params.get("new_gene_shift") or {}
            ng_eff = ng.get("exp_trl_eff") or {}
            row["new_gene_exp"] = ng_eff.get("exp")
            row["new_gene_trl_eff"] = ng_eff.get("trl_eff")
            row["induction_gen"] = ng.get("induction_gen")
            row["knockout_gen"] = ng.get("knockout_gen")
        else:
            row["condition"] = ""
            row["perturbations_json"] = "baseline"
            row["new_gene_exp"] = None
            row["new_gene_trl_eff"] = None
            row["induction_gen"] = None
            row["knockout_gen"] = None
        rows.append(row)
    return pl.DataFrame(rows)


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _shorten_perturbations(json_str: str, max_chars: int = 90) -> str:
    if not json_str or json_str == "baseline":
        return json_str or "—"
    if len(json_str) <= max_chars:
        return json_str
    return json_str[: max_chars - 1] + "…"


def _metadata_table_chart(metadata_df: pl.DataFrame) -> alt.Chart:
    """A proper-looking strain metadata table with header row + per-variant rows."""
    columns = [
        ("Variant", "variant", 70),
        ("Condition", "condition", 100),
        ("Perturbations", "_perturbations_short", 500),
        ("GFP exp", "new_gene_exp", 90),
        ("GFP trl_eff", "new_gene_trl_eff", 90),
        ("Induction gen", "induction_gen", 90),
        ("Knockout gen", "knockout_gen", 90),
    ]
    total_w = sum(w for _, _, w in columns)
    starts: dict[str, float] = {}
    centers: dict[str, float] = {}
    accumulated = 0.0
    for label, _key, w in columns:
        starts[label] = accumulated
        centers[label] = accumulated + w / 2
        accumulated += w
    n = metadata_df.height
    row_h = 26
    header_h = 30
    height = header_h + n * row_h

    # Prepare cell text.
    text_rows: list[dict[str, Any]] = []
    rect_rows: list[dict[str, Any]] = []
    # Header backgrounds + text.
    for label, _key, w in columns:
        rect_rows.append(
            {
                "x": starts[label],
                "x2": starts[label] + w,
                "y": 0,
                "y2": header_h,
                "fill": "#dbe5f1",
            }
        )
        text_rows.append(
            {
                "x": centers[label],
                "y": header_h / 2,
                "text": label,
                "bold": True,
            }
        )
    # Data rows + alternating row backgrounds.
    for i, row in enumerate(
        metadata_df.with_columns(
            _perturbations_short=pl.col("perturbations_json").map_elements(
                _shorten_perturbations, return_dtype=pl.Utf8
            )
        ).iter_rows(named=True)
    ):
        y_top = header_h + i * row_h
        rect_rows.append(
            {
                "x": 0,
                "x2": total_w,
                "y": y_top,
                "y2": y_top + row_h,
                "fill": "#ffffff" if i % 2 == 0 else "#f4f6fa",
            }
        )
        for label, key, w in columns:
            val = row.get(key)
            if val is None or val == "":
                text = "—"
            elif isinstance(val, float):
                text = f"{val:g}"
            else:
                text = str(val)
            text_rows.append(
                {
                    "x": centers[label],
                    "y": y_top + row_h / 2,
                    "text": text,
                    "bold": False,
                }
            )

    rect_df = pl.DataFrame(rect_rows)
    text_df = pl.DataFrame(text_rows)
    header_df = text_df.filter(pl.col("bold"))
    body_df = text_df.filter(~pl.col("bold"))

    common_x = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, total_w]))
    common_y = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[height, 0]))

    rects = (
        alt.Chart(rect_df)
        .mark_rect()
        .encode(
            x=common_x,
            x2="x2:Q",
            y=common_y,
            y2="y2:Q",
            color=alt.Color("fill:N", scale=None, legend=None),
        )
    )
    header = (
        alt.Chart(header_df)
        .mark_text(
            align="center",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
            font="Helvetica",
        )
        .encode(x=common_x, y=common_y, text="text:N")
    )
    body = (
        alt.Chart(body_df)
        .mark_text(align="center", baseline="middle", fontSize=11, font="Helvetica")
        .encode(x=common_x, y=common_y, text="text:N")
    )
    return (rects + header + body).properties(
        title="Strain metadata", width=total_w, height=height
    )


def _boxplot(
    per_cell: pl.DataFrame, metric_col: str, title: str, y_title: str
) -> alt.Chart:
    if metric_col not in per_cell.columns:
        return _placeholder(title, "not computed")
    sub = per_cell.select(["variant", metric_col]).drop_nulls()
    if sub.is_empty():
        return _placeholder(title, "no data")
    box = (
        alt.Chart(sub)
        .mark_boxplot(size=30, outliers=True)
        .encode(
            x=alt.X("variant:N").title("Variant"),
            y=alt.Y(f"{metric_col}:Q").title(y_title),
        )
    )
    points = (
        alt.Chart(sub)
        .mark_circle(size=40, opacity=0.55, color="#555")
        .encode(
            x=alt.X("variant:N"),
            y=alt.Y(f"{metric_col}:Q"),
            xOffset="jitter:Q",
            tooltip=[
                alt.Tooltip("variant:N"),
                alt.Tooltip(f"{metric_col}:Q", format=".4g"),
            ],
        )
        .transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
    )
    return (box + points).properties(title=title, width=220, height=200)


def _placeholder(title: str, message: str) -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"msg": [message]}))
        .mark_text(size=14, color="#888")
        .encode(text="msg:N")
        .properties(title=title, width=220, height=140)
    )


def _essential_ratio_chart(
    ratios_long: pl.DataFrame, control_variant: int
) -> alt.Chart | None:
    if ratios_long.is_empty():
        return None
    box = (
        alt.Chart(ratios_long)
        .mark_boxplot(size=40, outliers=True)
        .encode(
            x=alt.X("variant:N").title("Variant"),
            y=alt.Y("ratio:Q")
            .title(f"Proteome-fraction ratio vs variant {control_variant}")
            .scale(zero=False),
        )
    )
    points = (
        alt.Chart(ratios_long)
        .mark_circle(size=22, opacity=0.4, color="#555")
        .encode(
            x=alt.X("variant:N"),
            y=alt.Y("ratio:Q"),
            xOffset="jitter:Q",
            tooltip=[
                alt.Tooltip("monomer_id:N"),
                alt.Tooltip("variant:N"),
                alt.Tooltip("ratio:Q", format=".3f"),
            ],
        )
        .transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
    )
    threshold = (
        alt.Chart(pl.DataFrame({"y": [0.5]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return (box + points + threshold).properties(
        title=(
            "Essential-protein proteome fraction "
            f"(per-protein ratio vs variant {control_variant}; red line = 0.5)"
        ),
        width=420,
        height=260,
    )


def _resource_allocation_chart_grid(per_cell: pl.DataFrame) -> alt.Chart:
    classes = [
        ("rrna", "rRNA"),
        ("ribosomal_protein", "ribosomal proteins"),
        ("rnap_subunit", "RNAP subunits"),
        ("new_gene", "new genes"),
        ("perturbed", "perturbed genes"),
    ]
    rnap_charts = [
        _boxplot(
            per_cell,
            f"rnap_frac_{cls}",
            f"RNAP on {label}",
            "mean fraction of active RNAP",
        )
        for cls, label in classes
    ]
    ribo_charts = [
        _boxplot(
            per_cell,
            f"ribo_frac_{cls}",
            f"Ribosomes on {label}",
            "mean fraction of active ribosomes",
        )
        for cls, label in classes
    ]
    return alt.vconcat(
        alt.hconcat(*rnap_charts).resolve_scale(y="independent"),
        alt.hconcat(*ribo_charts).resolve_scale(y="independent"),
    ).properties(title="Resource allocation (per cell)")


def _scalar_diagnostic_grid(per_cell: pl.DataFrame) -> alt.Chart:
    """Boxplots for non-allocation diagnostics."""
    items = [
        ("doubling_time_min", "Doubling time", "min"),
        ("peak_cell_mass_fg", "Peak cell mass", "fg"),
        ("peak_protein_mass_fg", "Peak protein mass", "fg"),
        ("mean_aa_per_s_per_ribo", "Eff. elongation rate", "AA/s/ribo"),
        ("total_aa_elongations", "Total AA elongated", "per cell"),
        ("total_nt_elongations", "Total nt elongated", "per cell"),
        ("mean_n_ribosome", "Mean active ribosomes", "count"),
        ("mean_n_rnap", "Mean active RNAPs", "count"),
        ("time_to_init_min", "Time to repl. initiation", "min"),
        ("cell_mass_at_init_fg", "Cell mass at initiation", "fg"),
        ("new_gene_protein_peak", "Peak new-gene protein", "count"),
        ("essential_proteome_fraction", "Essential proteome fraction", "fraction"),
    ]
    charts = [_boxplot(per_cell, col, title, ytitle) for col, title, ytitle in items]
    rows = []
    cols_per_row = 4
    for i in range(0, len(charts), cols_per_row):
        rows.append(alt.hconcat(*charts[i : i + cols_per_row]))
    return alt.vconcat(*rows).properties(title="Per-cell diagnostics")


# ---------------------------------------------------------------------------
# Main plot entry point
# ---------------------------------------------------------------------------


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    skip_first_n_gens = int(params.get("skip_first_n_gens") or 0)
    control_variant = int(params.get("control_variant") or 0)
    skip_clause = _gen_filter_clause(skip_first_n_gens)

    exp_id = next(iter(variant_metadata.keys()))
    vm = variant_metadata[exp_id]
    variant_name = variant_names.get(exp_id, "")

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    class_cistron_ids = _gene_class_cistron_ids(sim_data)
    new_gene_cistron_ids = class_cistron_ids.get("new_gene", [])
    new_gene_monomer_ids = _gene_class_monomer_ids(sim_data, new_gene_cistron_ids)

    perturbed_cistron_ids_per_variant: dict[int, list[str]] = {
        v: _resolve_perturbed_cistron_ids(p, sim_data) for v, p in vm.items()
    }

    essential_idx, essential_ids = _load_essential_protein_monomer_indexes(
        validation_data_paths, conn, config_sql
    )
    if not essential_idx:
        print("strain_dashboard: no essential-protein data; omitting that panel.")

    print("strain_dashboard: computing per-cell metrics…")
    per_cell = _per_cell_metrics(
        conn,
        history_sql,
        config_sql,
        sim_data,
        class_cistron_ids,
        perturbed_cistron_ids_per_variant,
        new_gene_monomer_ids,
        essential_idx,
        skip_clause,
    )

    print("strain_dashboard: computing per-protein essential-fraction ratios…")
    cell_keys = [
        (row["variant"], row["lineage_seed"], row["generation"], row["agent_id"])
        for row in per_cell.iter_rows(named=True)
    ]
    ratios_wide, ratio_summary = _essential_protein_ratios(
        conn,
        history_sql,
        essential_idx,
        essential_ids,
        control_variant,
        skip_clause,
        cell_keys=cell_keys,
    )

    metadata_df = _strain_metadata_table(vm, variant_name)
    diagnostics_csv = _per_cell_diagnostics_csv(per_cell)
    per_variant_csv = _per_variant_diagnostics_csv(per_cell)

    # Write CSVs early so failure in chart assembly still yields data.
    diagnostics_csv.write_csv(os.path.join(outdir, "per_cell_diagnostics.csv"))
    per_variant_csv.write_csv(os.path.join(outdir, "per_variant_diagnostics.csv"))
    metadata_df.write_csv(os.path.join(outdir, "strain_metadata.csv"))
    if not ratios_wide.is_empty():
        ratios_wide.write_csv(os.path.join(outdir, "essential_protein_ratios.csv"))
    if not ratio_summary.is_empty():
        ratio_summary.write_csv(
            os.path.join(outdir, "essential_protein_ratio_summary.csv")
        )
        print("strain_dashboard: essential-protein ratio summary:")
        for row in ratio_summary.iter_rows(named=True):
            print(
                f"  variant {row['variant']:>3}: "
                f"mean={row['mean']:.3f}  std={row['std']:.3f}  "
                f"median={row['median']:.3f}  n={row['n_proteins']}  "
                f"n<0.5={row['n_below_0_5']}"
            )

    # Build chart sections.
    sections: list[alt.Chart] = [_metadata_table_chart(metadata_df)]

    # Cross-variant essential-protein boxplot.
    if not ratios_wide.is_empty():
        ratio_cols = [c for c in ratios_wide.columns if c.startswith("ratio_variant_")]
        if ratio_cols:
            long = ratios_wide.unpivot(
                index=["monomer_id"],
                on=ratio_cols,
                variable_name="ratio_col",
                value_name="ratio",
            ).with_columns(
                variant=pl.col("ratio_col").str.split("_").list.last().cast(pl.Int64)
            )
            chart = _essential_ratio_chart(long, control_variant)
            if chart is not None:
                sections.append(chart)

    sections.append(_scalar_diagnostic_grid(per_cell))
    sections.append(_resource_allocation_chart_grid(per_cell))

    final = alt.vconcat(*sections).resolve_scale(x="independent", y="independent")
    output_path = os.path.join(outdir, "strain_dashboard.html")
    final.save(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"strain_dashboard: wrote {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_stub_sim_data():
    """Minimal sim_data-shaped stub for unit testing the pure helpers."""

    class Stub:
        pass

    cistron_dtype = [
        ("id", "U16"),
        ("gene_id", "U16"),
        ("is_rRNA", "?"),
        ("is_ribosomal_protein", "?"),
        ("is_RNAP", "?"),
        ("is_new_gene", "?"),
        ("is_mRNA", "?"),
    ]
    cistrons = np.array(
        [
            ("lacZ", "EG10001", False, False, False, False, True),
            ("rpoB", "EG10002", False, False, True, False, True),
            ("rpsA", "EG10003", False, True, False, False, True),
            ("rrsA", "EG10004", True, False, False, False, False),
            ("gfp", "NG_FAKE_1", False, False, False, True, True),
        ],
        dtype=cistron_dtype,
    )
    monomer_dtype = [("id", "U24"), ("cistron_id", "U16")]
    monomers = np.array(
        [
            ("lacZ[c]", "lacZ"),
            ("rpoB[c]", "rpoB"),
            ("rpsA[c]", "rpsA"),
            ("gfp[c]", "gfp"),
        ],
        dtype=monomer_dtype,
    )
    sd = Stub()
    sd.process = Stub()
    sd.process.transcription = Stub()
    sd.process.transcription.cistron_data = Stub()
    sd.process.transcription.cistron_data.struct_array = cistrons
    sd.process.translation = Stub()
    sd.process.translation.monomer_data = Stub()
    sd.process.translation.monomer_data.struct_array = monomers
    return sd


def test_gene_class_cistron_ids_partitions():
    sd = _build_stub_sim_data()
    classes = _gene_class_cistron_ids(sd)
    assert classes["rrna"] == ["rrsA"]
    assert classes["ribosomal_protein"] == ["rpsA"]
    assert classes["rnap_subunit"] == ["rpoB"]
    assert classes["new_gene"] == ["gfp"]


def test_gene_class_monomer_ids_drops_non_coding():
    sd = _build_stub_sim_data()
    assert _gene_class_monomer_ids(sd, ["rrsA"]) == []
    assert _gene_class_monomer_ids(sd, ["gfp"]) == ["gfp[c]"]


def test_resolve_perturbed_cistron_ids_baseline_string():
    sd = _build_stub_sim_data()
    assert _resolve_perturbed_cistron_ids("baseline", sd) == []
    assert _resolve_perturbed_cistron_ids({}, sd) == []


def test_resolve_perturbed_cistron_ids_happy_path():
    sd = _build_stub_sim_data()
    cistrons = _resolve_perturbed_cistron_ids(
        {"perturbations": {"EG10001": 0.0, "EG10003": 2.0, "EG_UNKNOWN": 5.0}}, sd
    )
    assert set(cistrons) == {"lacZ", "rpsA"}


def test_variant_label_baseline_and_perturbed():
    assert _variant_label(0, "baseline") == "variant 0 (baseline)"
    assert (
        _variant_label(3, {"perturbations": {"EG10001": 0.0, "EG10002": 2.0}})
        == "variant 3 (2 perturbations)"
    )


def test_gen_filter_clause_round_trip():
    assert _gen_filter_clause(0) == ""
    assert _gen_filter_clause(2) == " AND generation > 2"


def test_cell_id_format():
    assert _cell_id(0, 0, 1, "0") == "Cell: 0_0_1_0"
    assert _cell_id(1, 0, 2, "00") == "Cell: 1_0_2_00"


def test_strain_metadata_table_handles_baseline_and_dict():
    vm = {
        0: "baseline",
        1: {
            "condition": "basal",
            "perturbations": {"EG10001": 0.0},
            "new_gene_shift": {
                "induction_gen": 2,
                "exp_trl_eff": {"exp": 1e6, "trl_eff": 5.0},
            },
        },
    }
    df = _strain_metadata_table(vm, "strain_design")
    assert df.height == 2
    assert df["variant"].to_list() == [0, 1]
    assert df["perturbations_json"].to_list() == [
        "baseline",
        json.dumps({"EG10001": 0.0}),
    ]
    assert df["new_gene_trl_eff"].to_list() == [None, 5.0]


def test_shorten_perturbations_handles_long_and_short():
    assert _shorten_perturbations("") == "—"
    assert _shorten_perturbations("baseline") == "baseline"
    short = '{"EG1": 0.5}'
    assert _shorten_perturbations(short) == short
    long = "x" * 200
    assert _shorten_perturbations(long, max_chars=20).endswith("…")
    assert len(_shorten_perturbations(long, max_chars=20)) == 20


def test_per_cell_diagnostics_csv_shape():
    """Smoke test: pivot a fake per-cell frame into the diagnostic shape."""
    pc = pl.DataFrame(
        {
            "variant": [0, 0, 1, 1],
            "lineage_seed": [0, 0, 0, 0],
            "generation": [1, 2, 1, 2],
            "agent_id": ["0", "00", "0", "00"],
            "doubling_time_min": [50.0, 49.0, 45.0, 44.0],
            "peak_cell_mass_fg": [2300.0, 2310.0, 2400.0, 2410.0],
            "mean_n_ribosome": [19000, 19500, 20000, 20500],
        }
    )
    csv = _per_cell_diagnostics_csv(pc)
    cell_cols = [c for c in csv.columns if c.startswith("Cell: ")]
    assert len(cell_cols) == 4
    assert "Cell: 0_0_1_0" in csv.columns
    assert "Cell: 1_0_2_00" in csv.columns
    # Doubling time row should have mean ~47.0
    dt_row = csv.filter(pl.col("diagnostic") == "Doubling time (min)").row(0)
    mean_idx = csv.columns.index("mean")
    assert abs(dt_row[mean_idx] - 47.0) < 1e-6
