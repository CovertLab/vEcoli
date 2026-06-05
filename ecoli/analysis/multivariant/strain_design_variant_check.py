"""
Verify that strain-modification variants hit their targets.

For every native gene listed in any variant's ``perturbations`` dict, this
plot compares the configured fold-change against the observed protein
fold-change relative to a reference variant (default V0). For every new
gene (``is_new_gene`` in sim_data), it reports the configured
new-gene ``trl_eff`` alongside the observed mRNA and protein counts per
variant.

Output is a single self-contained HTML file with:

- A "native perturbations" table — one row per gene, with the reference
  (V0) protein count, expected ratio per variant, observed ratio per
  variant, and a ``hit`` flag (observed within a tolerance of expected).
- A "new genes" table — one row per (variant, new-gene) with trl_eff,
  mRNA, protein.
- A scatter plot of expected vs observed protein ratio (one point per
  gene per non-reference variant) with a y=x diagonal for at-a-glance
  verification.

Config usage::

    "analysis_options": {
        "multivariant": {
            "strain_design_variant_check": {
                "reference_variant": 0,
                "skip_first_n_gens": 1,
                "tolerance": 0.3,
                "include_new_genes": true
            }
        }
    }

``tolerance`` is the fractional deviation allowed before a variant is
flagged as missing its target. ``0.3`` means observed must lie within
30% of expected (relative). For knockouts (expected = 0), the check
becomes ``observed / V0 < tolerance``.
"""

from __future__ import annotations

import html as html_lib
import os
import pickle
from typing import Any

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
    read_stacked_columns,
)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _flatten_metadata(
    variant_metadata: dict[str, dict[int, Any]],
) -> dict[int, dict[str, Any]]:
    """Collapse the {exp_id: {variant: params}} nesting into {variant: params}.

    Variants are expected to share parameters across experiment IDs in the
    composed-variant runs we target. If a variant appears in multiple
    experiments with the same params, the first non-empty params dict wins.
    """
    flat: dict[int, dict[str, Any]] = {}
    for exp_variants in variant_metadata.values():
        for vid_raw, params in exp_variants.items():
            try:
                vid = int(vid_raw)
            except (TypeError, ValueError):
                continue
            if vid in flat:
                continue
            flat[vid] = params if isinstance(params, dict) else {}
    return flat


def _expected_ratio(params: dict[str, Any], gene_id: str) -> float:
    """Expected protein fold-change for a native gene under one variant.

    A missing entry means "no perturbation" (ratio = 1). Variant 0 in
    strain-design runs typically has no ``perturbations`` dict at all,
    which collapses to 1.0 here for every gene.
    """
    pert = params.get("perturbations") or {}
    if not isinstance(pert, dict):
        return 1.0
    if gene_id not in pert:
        return 1.0
    try:
        return float(pert[gene_id])
    except (TypeError, ValueError):
        return 1.0


def _classify_role(expected_ratios: list[float]) -> str:
    """Categorise a gene as KO / KD / OE / mixed based on its multipliers."""
    non_ref = [r for r in expected_ratios if r != 1.0]
    if not non_ref:
        return "—"
    has_zero = any(r == 0.0 for r in non_ref)
    has_down = any(0.0 < r < 1.0 for r in non_ref)
    has_up = any(r > 1.0 for r in non_ref)
    tags: list[str] = []
    if has_zero:
        tags.append("KO")
    if has_down:
        tags.append("KD")
    if has_up:
        tags.append("OE")
    return "/".join(tags) if tags else "—"


def _perturbed_genes(meta: dict[int, dict[str, Any]]) -> list[str]:
    """Union of all native genes perturbed in any non-reference variant."""
    seen: list[str] = []
    for params in meta.values():
        pert = params.get("perturbations") or {}
        if not isinstance(pert, dict):
            continue
        for g in pert:
            if g not in seen:
                seen.append(g)
    return seen


def _new_gene_trl_eff(params: dict[str, Any]) -> float | None:
    ng = params.get("new_gene_shift") or {}
    if not isinstance(ng, dict):
        return None
    ete = ng.get("exp_trl_eff") or {}
    if not isinstance(ete, dict):
        return None
    val = ete.get("trl_eff")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Gene-id resolution (mirrors gene_counts._resolve_gene_ids but keeps the
# reverse maps so we can report mRNA for new genes too)
# ---------------------------------------------------------------------------


def _resolve(
    sim_data, gene_ids: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (resolved_gene_ids, cistron_ids, monomer_ids, unresolved)."""
    cd = sim_data.process.transcription.cistron_data.struct_array
    md = sim_data.process.translation.monomer_data.struct_array
    g2c = dict(zip(cd["gene_id"], cd["id"]))
    c2m = dict(zip(md["cistron_id"], md["id"]))
    out_g: list[str] = []
    out_c: list[str] = []
    out_m: list[str] = []
    unresolved: list[str] = []
    for g in gene_ids:
        cid = g2c.get(g)
        if cid is None:
            unresolved.append(g)
            continue
        mid = c2m.get(cid)
        if mid is None:
            unresolved.append(g)
            continue
        out_g.append(g)
        out_c.append(cid)
        out_m.append(mid)
    return out_g, out_c, out_m, unresolved


# ---------------------------------------------------------------------------
# DuckDB aggregations
# ---------------------------------------------------------------------------


def _avg_counts_per_variant(
    conn: DuckDBPyConnection,
    history_sql: str,
    listener_col: str,
    skip_first_n_gens: int,
) -> pl.DataFrame:
    """Average a per-element list listener across cells, per variant.

    Returns one row per (variant, index) with column ``avg_count``. Indices
    are 1-based DuckDB array positions; callers subtract 1 to align with the
    Python lookups returned by :py:func:`field_metadata`.
    """
    subquery = read_stacked_columns(history_sql, [listener_col], order_results=False)
    return conn.sql(f"""
        WITH src AS (
            SELECT variant, generation, {listener_col} AS counts
            FROM ({subquery})
            WHERE generation >= {int(skip_first_n_gens)}
        ),
        unnested AS (
            SELECT variant,
                   unnest(counts) AS count,
                   generate_subscripts(counts, 1) AS idx
            FROM src
        )
        SELECT variant, idx, avg(count) AS avg_count
        FROM unnested
        GROUP BY variant, idx
        ORDER BY variant, idx
    """).pl()


# ---------------------------------------------------------------------------
# Table HTML
# ---------------------------------------------------------------------------


_TABLE_STYLE = (
    "border-collapse:collapse;font-family:monospace;font-size:12px;margin:8px 0 16px 0;"
)
_TH = (
    'style="border:1px solid #999;padding:4px 8px;background:#222;color:#eee;'
    'text-align:left;"'
)
_TD = 'style="border:1px solid #999;padding:4px 8px;"'


def _fmt_float(x: float | None, places: int = 2) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return ""
    return f"{x:.{places}f}"


def _fmt_int(x: float | None) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return f"{int(round(x)):,}"


def _native_table_html(
    gene_rows: list[dict[str, Any]],
    non_ref_variants: list[int],
) -> str:
    """Render the native-gene table. The expected/observed columns list only
    the non-reference variants — the reference is 1.0 by definition."""
    headers = [
        "<th " + _TH + ">gene</th>",
        "<th " + _TH + ">role</th>",
        "<th " + _TH + ">V<sub>ref</sub> prot</th>",
        f"<th {_TH}>Expected (V{', V'.join(str(v) for v in non_ref_variants)})</th>",
        f"<th {_TH}>Observed (V{', V'.join(str(v) for v in non_ref_variants)})</th>",
        "<th " + _TH + ">hit</th>",
    ]
    body: list[str] = []
    for r in gene_rows:
        exp_str = ", ".join(_fmt_float(e) for e in r["expected_non_ref"])
        obs_str = ", ".join(_fmt_float(o) for o in r["observed_non_ref"])
        hit_glyph = "&#10003;" if r["hit"] else "&#10007;"
        body.append(
            "<tr>"
            f"<td {_TD}>{html_lib.escape(r['gene'])}</td>"
            f"<td {_TD}>{html_lib.escape(r['role'])}</td>"
            f"<td {_TD}>{_fmt_int(r['ref_protein'])}</td>"
            f"<td {_TD}>{exp_str}</td>"
            f"<td {_TD}>{obs_str}</td>"
            f"<td {_TD}>{hit_glyph}</td>"
            "</tr>"
        )
    return (
        f'<table style="{_TABLE_STYLE}">'
        f"<thead><tr>{''.join(headers)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def _new_gene_table_html(rows: list[dict[str, Any]]) -> str:
    headers = [
        f"<th {_TH}>variant</th>",
        f"<th {_TH}>gene</th>",
        f"<th {_TH}>new-gene trl_eff</th>",
        f"<th {_TH}>mRNA</th>",
        f"<th {_TH}>Protein</th>",
    ]
    body: list[str] = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td {_TD}>V{r['variant']}</td>"
            f"<td {_TD}>{html_lib.escape(r['gene'])}</td>"
            f"<td {_TD}>{_fmt_float(r['trl_eff'], 1)}</td>"
            f"<td {_TD}>{_fmt_float(r['mRNA'], 1)}</td>"
            f"<td {_TD}>{_fmt_int(r['protein'])}</td>"
            "</tr>"
        )
    return (
        f'<table style="{_TABLE_STYLE}">'
        f"<thead><tr>{''.join(headers)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Scatter chart
# ---------------------------------------------------------------------------


def _scatter_chart(scatter_df: pl.DataFrame) -> alt.Chart:
    if scatter_df.is_empty():
        return (
            alt.Chart(pl.DataFrame({"msg": ["no perturbed variants to plot"]}))
            .mark_text(size=14, color="#888")
            .encode(text="msg:N")
            .properties(width=480, height=80)
        )
    # Use a symmetric log-ish scale: linear is fine if all values are small,
    # but KOs (expected=0) and OEs (expected=5+) together call for log scale.
    # Clip expected/observed to >=1e-3 so log axes don't blow up on KOs.
    plot_df = scatter_df.with_columns(
        expected_plot=pl.max_horizontal(pl.col("expected"), pl.lit(1e-3)),
        observed_plot=pl.max_horizontal(pl.col("observed"), pl.lit(1e-3)),
    )
    points = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=80, opacity=0.85)
        .encode(
            x=alt.X(
                "expected_plot:Q",
                title="Expected ratio (config)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "observed_plot:Q",
                title="Observed ratio (mean protein / Vref)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("gene:N", legend=alt.Legend(title="Gene")),
            shape=alt.Shape("variant:N", legend=alt.Legend(title="Variant")),
            tooltip=[
                alt.Tooltip("gene:N"),
                alt.Tooltip("variant:N"),
                alt.Tooltip("expected:Q", format=".3f"),
                alt.Tooltip("observed:Q", format=".3f"),
                alt.Tooltip("hit:N"),
            ],
        )
    )
    # y = x reference line over the union of plotted ranges.
    lo = float(min(plot_df["expected_plot"].min(), plot_df["observed_plot"].min()))
    hi = float(max(plot_df["expected_plot"].max(), plot_df["observed_plot"].max()))
    diag = (
        alt.Chart(pl.DataFrame({"v": [lo, hi]}))
        .mark_line(color="#888", strokeDash=[4, 4])
        .encode(x="v:Q", y="v:Q")
    )
    return (diag + points).properties(width=480, height=420).interactive()


# ---------------------------------------------------------------------------
# Main entry point
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
    reference_variant = int(params.get("reference_variant", 0))
    skip_first_n_gens = int(params.get("skip_first_n_gens", 1))
    tolerance = float(params.get("tolerance", 0.3))
    include_new_genes = bool(params.get("include_new_genes", True))

    meta = _flatten_metadata(variant_metadata)
    if not meta:
        print(
            "multivariant strain_design_variant_check: no variant metadata; skipping."
        )
        return
    variant_order = sorted(meta.keys())
    if reference_variant not in meta:
        print(
            f"multivariant strain_design_variant_check: reference_variant={reference_variant} "
            f"not present in variant metadata {variant_order}; skipping."
        )
        return

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    native_genes = _perturbed_genes(meta)
    new_gene_ids: list[str] = []
    if include_new_genes:
        cd = sim_data.process.transcription.cistron_data.struct_array
        new_gene_ids = cd[cd["is_new_gene"]]["gene_id"].tolist()

    all_genes = list(native_genes) + [g for g in new_gene_ids if g not in native_genes]
    if not all_genes:
        print(
            "multivariant strain_design_variant_check: no perturbed or new genes; skipping."
        )
        return

    resolved_genes, cistron_ids, monomer_ids, unresolved = _resolve(sim_data, all_genes)
    if unresolved:
        print(
            f"multivariant strain_design_variant_check: skipping unresolved genes {unresolved}"
        )
    if not resolved_genes:
        print("multivariant strain_design_variant_check: nothing resolvable; skipping.")
        return

    monomer_fields = field_metadata(conn, config_sql, "listeners__monomer_counts")
    cistron_fields = field_metadata(
        conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
    )
    monomer_to_idx = {m: i for i, m in enumerate(monomer_fields)}
    cistron_to_idx = {c: i for i, c in enumerate(cistron_fields)}

    avg_protein = _avg_counts_per_variant(
        conn, history_sql, "listeners__monomer_counts", skip_first_n_gens
    )
    avg_mrna = _avg_counts_per_variant(
        conn,
        history_sql,
        "listeners__rna_counts__mRNA_cistron_counts",
        skip_first_n_gens,
    )

    # Build {(variant, gene): avg_protein}; idx in DuckDB is 1-based.
    protein_lookup: dict[tuple[int, str], float] = {}
    for g, mid in zip(resolved_genes, monomer_ids):
        idx0 = monomer_to_idx.get(mid)
        if idx0 is None:
            continue
        rows = avg_protein.filter(pl.col("idx") == idx0 + 1)
        for variant, val in zip(rows["variant"].to_list(), rows["avg_count"].to_list()):
            protein_lookup[(int(variant), g)] = float(val)

    mrna_lookup: dict[tuple[int, str], float] = {}
    for g, cid in zip(resolved_genes, cistron_ids):
        idx0 = cistron_to_idx.get(cid)
        if idx0 is None:
            continue
        rows = avg_mrna.filter(pl.col("idx") == idx0 + 1)
        for variant, val in zip(rows["variant"].to_list(), rows["avg_count"].to_list()):
            mrna_lookup[(int(variant), g)] = float(val)

    # ---- Native rows + scatter data ------------------------------------------------
    non_ref_variants = [v for v in variant_order if v != reference_variant]
    native_rows: list[dict[str, Any]] = []
    scatter_records: list[dict[str, Any]] = []
    for g in (g for g in resolved_genes if g in native_genes):
        ref_prot = protein_lookup.get((reference_variant, g))
        expected_non_ref = [_expected_ratio(meta[v], g) for v in non_ref_variants]
        observed_non_ref: list[float | None] = []
        hit_flags: list[bool] = []
        for v, e in zip(non_ref_variants, expected_non_ref):
            obs_prot = protein_lookup.get((v, g))
            if ref_prot is None or obs_prot is None or ref_prot == 0:
                observed_non_ref.append(None)
                hit_flags.append(False)
                continue
            o = obs_prot / ref_prot
            observed_non_ref.append(o)
            if e == 0.0:
                hit_flags.append(o < tolerance)
            else:
                hit_flags.append(abs(o - e) / e <= tolerance)
            scatter_records.append(
                {
                    "gene": g,
                    "variant": v,
                    "expected": e,
                    "observed": o,
                    "hit": "yes" if hit_flags[-1] else "no",
                }
            )
        native_rows.append(
            {
                "gene": g,
                "role": _classify_role([1.0] + expected_non_ref),
                "ref_protein": ref_prot,
                "expected_non_ref": expected_non_ref,
                "observed_non_ref": observed_non_ref,
                "hit": all(hit_flags) if hit_flags else False,
            }
        )

    # ---- New-gene rows -------------------------------------------------------------
    new_gene_rows: list[dict[str, Any]] = []
    for g in (g for g in resolved_genes if g in new_gene_ids):
        for v in variant_order:
            new_gene_rows.append(
                {
                    "variant": v,
                    "gene": g,
                    "trl_eff": _new_gene_trl_eff(meta[v]),
                    "mRNA": mrna_lookup.get((v, g)),
                    "protein": protein_lookup.get((v, g)),
                }
            )

    # ---- Assemble HTML -------------------------------------------------------------
    parts: list[str] = []
    parts.append(
        "<html><head><meta charset='utf-8'>"
        "<title>Variant check</title></head>"
        "<body style='font-family:sans-serif;padding:18px;'>"
    )
    parts.append(
        f"<h2>Variant check &mdash; reference variant V{reference_variant}, "
        f"skip first {skip_first_n_gens} gen(s), tolerance &plusmn;{int(tolerance * 100)}%</h2>"
    )

    if native_rows:
        parts.append("<h3>Knockdowns / overexpressions / knockouts</h3>")
        parts.append(_native_table_html(native_rows, non_ref_variants))

    if scatter_records:
        chart_path = os.path.join(outdir, "strain_design_variant_check_scatter.html")
        scatter_df = pl.DataFrame(scatter_records)
        _scatter_chart(scatter_df).save(chart_path)
        # Embed as iframe to keep the page self-contained but avoid bundling
        # Vega into the table HTML.
        rel = os.path.basename(chart_path)
        parts.append("<h3>Expected vs observed protein ratio</h3>")
        parts.append(
            f"<iframe src='{rel}' style='width:560px;height:520px;border:0;'></iframe>"
        )

    if new_gene_rows:
        parts.append("<h3>New gene induction</h3>")
        parts.append(_new_gene_table_html(new_gene_rows))

    parts.append("</body></html>")

    output_path = os.path.join(outdir, "strain_design_variant_check.html")
    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(
        f"multivariant strain_design_variant_check: wrote {output_path} — "
        f"{len(native_rows)} native gene(s), {len(new_gene_rows)} new-gene row(s)"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_flatten_metadata_collapses_nested_dict():
    raw = {
        "exp_A": {
            0: {"perturbations": {"EG1": 0.5}},
            1: {"perturbations": {"EG1": 2.0}},
        },
    }
    flat = _flatten_metadata(raw)
    assert set(flat.keys()) == {0, 1}
    assert flat[1]["perturbations"]["EG1"] == 2.0


def test_expected_ratio_defaults_to_one_when_missing():
    assert _expected_ratio({}, "EG1") == 1.0
    assert _expected_ratio({"perturbations": {}}, "EG1") == 1.0
    assert _expected_ratio({"perturbations": {"EG1": 0.0}}, "EG1") == 0.0
    assert _expected_ratio({"perturbations": {"EG1": 3.5}}, "EG1") == 3.5


def test_classify_role_partitions():
    assert _classify_role([1.0, 1.0, 1.0]) == "—"
    assert _classify_role([1.0, 0.0, 0.0]) == "KO"
    assert _classify_role([1.0, 0.5, 0.3]) == "KD"
    assert _classify_role([1.0, 2.0, 5.0]) == "OE"
    assert _classify_role([1.0, 0.0, 0.5, 2.0]) == "KO/KD/OE"


def test_perturbed_genes_unions_across_variants():
    meta = {
        0: {},
        1: {"perturbations": {"EG1": 0.5, "EG2": 0.0}},
        2: {"perturbations": {"EG2": 0.3, "EG3": 2.0}},
    }
    assert _perturbed_genes(meta) == ["EG1", "EG2", "EG3"]


def test_new_gene_trl_eff_extraction():
    p = {"new_gene_shift": {"exp_trl_eff": {"trl_eff": 5.0, "exp": 1e6}}}
    assert _new_gene_trl_eff(p) == 5.0
    assert _new_gene_trl_eff({}) is None
    assert _new_gene_trl_eff({"new_gene_shift": {}}) is None


def test_resolve_partitions_known_and_unknown():
    from ecoli.analysis.multivariant.gene_counts import _build_stub_sim_data

    sd = _build_stub_sim_data()
    g, c, m, unresolved = _resolve(sd, ["EG10001", "EG10003", "EG10002", "EG_NOPE"])
    assert g == ["EG10001", "EG10003"]
    assert c == ["lacZ", "rpoB"]
    assert m == ["lacZ[c]", "rpoB[c]"]
    # Non-coding cistron (EG10002 -> rrsA, no monomer) and EG_NOPE both unresolved.
    assert set(unresolved) == {"EG10002", "EG_NOPE"}
