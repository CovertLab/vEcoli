"""
Multivariant dashboard summarising whether each new gene is bottlenecked
by RNA-polymerase or ribosome crowding under every variant.

For each new gene (``is_new_gene`` in sim_data), the dashboard reports
per-variant:

- the configured ``new_gene_shift.exp_trl_eff.exp`` and ``trl_eff``
- the fraction of (post-skip) timesteps where the TU was flagged
  ``tu_is_overcrowded`` (RNA-polymerase bottleneck)
- the fraction of timesteps where the ribosome target prob exceeded the
  actual prob for the new-gene monomer (ribosome bottleneck)
- the mean new-gene mRNA and protein counts

The goal is to visualise *diminishing returns*: as ``exp`` or ``trl_eff``
is ramped, protein output should increase roughly linearly until either
the TU saturates (RNAP overcrowded) or the monomer's translation prob
saturates (ribosome overcrowded). The dashboard makes that knee visible
both numerically (table) and visually (scatter of mean protein vs
``trl_eff`` coloured by ribosome overcrowding, and vs ``exp`` coloured by
RNAP overcrowding).

Config usage::

    "analysis_options": {
        "multivariant": {
            "new_gene_crowding_dashboard": {
                "skip_first_n_gens": 1,
                "rnap_warn_frac": 0.1,
                "rnap_alert_frac": 0.5,
                "ribo_warn_frac": 0.1,
                "ribo_alert_frac": 0.5
            }
        }
    }
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
)


# ---------------------------------------------------------------------------
# Metadata helpers (shared idioms with strain_design_variant_check)
# ---------------------------------------------------------------------------


def _flatten_metadata(
    variant_metadata: dict[str, dict[int, Any]],
) -> dict[int, dict[str, Any]]:
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


def _new_gene_exp_trl_eff(params: dict[str, Any]) -> tuple[float | None, float | None]:
    ng = params.get("new_gene_shift") or {}
    if not isinstance(ng, dict):
        return None, None
    ete = ng.get("exp_trl_eff") or {}
    if not isinstance(ete, dict):
        return None, None

    def _coerce(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return _coerce(ete.get("exp")), _coerce(ete.get("trl_eff"))


def _native_pert_summary(params: dict[str, Any]) -> str:
    pert = params.get("perturbations") or {}
    if not isinstance(pert, dict) or not pert:
        return ""
    return ", ".join(f"{g}={v:g}" for g, v in pert.items())


# ---------------------------------------------------------------------------
# DuckDB stats per variant
# ---------------------------------------------------------------------------


def _per_variant_stats(
    conn: DuckDBPyConnection,
    history_sql: str,
    tu_idx_1based: int,
    cistron_mrna_idx_1based: int,
    monomer_idx_1based: int,
    skip_first_n_gens: int,
) -> pl.DataFrame:
    """Compute per-variant crowding fractions and mean counts for one new gene.

    All indices are 1-based DuckDB array positions. Returns one row per
    variant with: ``variant``, ``rnap_oc_frac``, ``ribo_oc_frac``,
    ``mean_mrna``, ``mean_protein``, ``n_steps``.
    """
    query = f"""
    WITH src AS (
        SELECT
            variant,
            listeners__rna_synth_prob__tu_is_overcrowded[{tu_idx_1based}]
                AS rnap_oc,
            listeners__rna_synth_prob__target_rna_synth_prob[{tu_idx_1based}]
                AS rnap_target,
            listeners__rna_synth_prob__actual_rna_synth_prob[{tu_idx_1based}]
                AS rnap_actual,
            listeners__ribosome_data__target_prob_translation_per_transcript[{monomer_idx_1based}]
                AS ribo_target,
            listeners__ribosome_data__actual_prob_translation_per_transcript[{monomer_idx_1based}]
                AS ribo_actual,
            listeners__rna_counts__mRNA_cistron_counts[{cistron_mrna_idx_1based}]
                AS mrna_count,
            listeners__monomer_counts[{monomer_idx_1based}] AS protein_count
        FROM ({history_sql})
        WHERE generation >= {int(skip_first_n_gens)}
    )
    SELECT
        variant,
        avg(case when rnap_oc then 1.0 else 0.0 end) AS rnap_oc_frac,
        avg(case when ribo_target > ribo_actual then 1.0 else 0.0 end)
            AS ribo_oc_frac,
        avg(mrna_count) AS mean_mrna,
        avg(protein_count) AS mean_protein,
        avg(rnap_target) AS mean_rnap_target,
        avg(rnap_actual) AS mean_rnap_actual,
        avg(ribo_target) AS mean_ribo_target,
        avg(ribo_actual) AS mean_ribo_actual,
        count(*) AS n_steps
    FROM src
    GROUP BY variant
    ORDER BY variant
    """
    return conn.sql(query).pl()


# ---------------------------------------------------------------------------
# Verdict / colour helpers
# ---------------------------------------------------------------------------


def _verdict(
    rnap_oc: float,
    ribo_oc: float,
    rnap_warn: float,
    rnap_alert: float,
    ribo_warn: float,
    ribo_alert: float,
) -> tuple[str, str]:
    """Return (label, bg_colour) for the dashboard cell."""
    rnap_bad = rnap_oc >= rnap_alert
    ribo_bad = ribo_oc >= ribo_alert
    rnap_warn_hit = rnap_oc >= rnap_warn
    ribo_warn_hit = ribo_oc >= ribo_warn

    if rnap_bad and ribo_bad:
        return ("RNAP+ribosome saturated", "#7a0d0d")
    if rnap_bad:
        return ("RNAP saturated", "#a33")
    if ribo_bad:
        return ("ribosome saturated", "#a33")
    if rnap_warn_hit and ribo_warn_hit:
        return ("approaching limits", "#8a6d3b")
    if rnap_warn_hit:
        return ("RNAP nearing limit", "#8a6d3b")
    if ribo_warn_hit:
        return ("ribosome nearing limit", "#8a6d3b")
    return ("headroom", "#2b6e2b")


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{x * 100:.1f}%"


def _fmt_float(x: float | None, places: int = 1) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{x:.{places}f}"


def _fmt_int(x: float | None) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{int(round(x)):,}"


def _fmt_exp(x: float | None) -> str:
    """Compact format for ``exp`` values: small numbers as floats, large in
    scientific notation. ``exp_trl_eff.exp`` is usually ~1 or ~1e6."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    if abs(x) < 1000:
        return f"{x:.4g}"
    return f"{x:.2e}"


# ---------------------------------------------------------------------------
# HTML table
# ---------------------------------------------------------------------------


_TABLE_STYLE = (
    "border-collapse:collapse;font-family:monospace;font-size:12px;margin:8px 0 16px 0;"
)
_TH = (
    'style="border:1px solid #999;padding:4px 8px;background:#222;color:#eee;'
    'text-align:left;"'
)
_TD = 'style="border:1px solid #999;padding:4px 8px;"'


def _gene_table_html(rows: list[dict[str, Any]]) -> str:
    headers = [
        f"<th {_TH}>variant</th>",
        f"<th {_TH}>exp</th>",
        f"<th {_TH}>trl_eff</th>",
        f"<th {_TH}>native perturbations</th>",
        f"<th {_TH}>RNAP overcrowded</th>",
        f"<th {_TH}>ribosome overcrowded</th>",
        f"<th {_TH}>mean mRNA</th>",
        f"<th {_TH}>mean protein</th>",
        f"<th {_TH}>verdict</th>",
    ]
    body: list[str] = []
    for r in rows:
        verdict_label, bg = r["verdict"]
        verdict_cell = (
            f'<td style="border:1px solid #999;padding:4px 8px;'
            f'background:{bg};color:#fff;">{html_lib.escape(verdict_label)}</td>'
        )
        body.append(
            "<tr>"
            f"<td {_TD}>V{r['variant']}</td>"
            f"<td {_TD}>{_fmt_exp(r['exp'])}</td>"
            f"<td {_TD}>{_fmt_float(r['trl_eff'], 2)}</td>"
            f"<td {_TD}>{html_lib.escape(r['native'])}</td>"
            f"<td {_TD}>{_fmt_pct(r['rnap_oc'])}</td>"
            f"<td {_TD}>{_fmt_pct(r['ribo_oc'])}</td>"
            f"<td {_TD}>{_fmt_float(r['mean_mrna'])}</td>"
            f"<td {_TD}>{_fmt_int(r['mean_protein'])}</td>"
            f"{verdict_cell}"
            "</tr>"
        )
    return (
        f'<table style="{_TABLE_STYLE}">'
        f"<thead><tr>{''.join(headers)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Diminishing-returns scatter
# ---------------------------------------------------------------------------


def _scatter_chart(df: pl.DataFrame, gene_label: str) -> alt.Chart:
    """Two side-by-side panels:
       (left) mean_protein vs trl_eff, colour = ribosome OC frac
       (right) mean_mrna vs exp,       colour = rnap OC frac
    Only variants with finite metadata values render in each panel."""

    def _panel(
        x_col: str,
        y_col: str,
        oc_col: str,
        x_title: str,
        y_title: str,
        oc_title: str,
    ) -> alt.Chart:
        sub = df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
        if sub.is_empty():
            return (
                alt.Chart(pl.DataFrame({"msg": [f"no {x_col} data"]}))
                .mark_text(size=14, color="#888")
                .encode(text="msg:N")
                .properties(width=320, height=260, title=f"{gene_label} — {x_title}")
            )
        return (
            alt.Chart(sub)
            .mark_point(filled=True, size=140, opacity=0.9)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_title),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                color=alt.Color(
                    f"{oc_col}:Q",
                    scale=alt.Scale(scheme="reds", domain=[0, 1]),
                    legend=alt.Legend(title=oc_title, format=".0%"),
                ),
                tooltip=[
                    alt.Tooltip("variant:N"),
                    alt.Tooltip(f"{x_col}:Q", format=".3g"),
                    alt.Tooltip(f"{y_col}:Q", format=".3g"),
                    alt.Tooltip(f"{oc_col}:Q", format=".1%"),
                ],
            )
            .properties(width=320, height=260, title=f"{gene_label} — {x_title}")
        )

    return alt.hconcat(
        _panel(
            "trl_eff",
            "mean_protein",
            "ribo_oc",
            "protein vs trl_eff",
            "mean protein",
            "ribosome OC",
        ),
        _panel(
            "exp",
            "mean_mrna",
            "rnap_oc",
            "mRNA vs exp",
            "mean mRNA",
            "RNAP OC",
        ),
    )


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
    skip_first_n_gens = int(params.get("skip_first_n_gens", 1))
    rnap_warn = float(params.get("rnap_warn_frac", 0.1))
    rnap_alert = float(params.get("rnap_alert_frac", 0.5))
    ribo_warn = float(params.get("ribo_warn_frac", 0.1))
    ribo_alert = float(params.get("ribo_alert_frac", 0.5))

    meta = _flatten_metadata(variant_metadata)
    if not meta:
        print(
            "multivariant new_gene_crowding_dashboard: no variant metadata; skipping."
        )
        return

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    cd = sim_data.process.transcription.cistron_data.struct_array
    md = sim_data.process.translation.monomer_data.struct_array
    new_cistron_ids = cd[cd["is_new_gene"]]["id"].tolist()
    if not new_cistron_ids:
        print("multivariant new_gene_crowding_dashboard: no new genes; skipping.")
        return
    cistron_to_gene = dict(zip(cd["id"], cd["gene_id"]))
    cistron_to_monomer = dict(zip(md["cistron_id"], md["id"]))

    # Listener index dicts (1-based for DuckDB)
    rnap_tu_fields = field_metadata(
        conn, config_sql, "listeners__rna_synth_prob__target_rna_synth_prob"
    )
    monomer_fields = field_metadata(conn, config_sql, "listeners__monomer_counts")
    mrna_cistron_fields = field_metadata(
        conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
    )
    tu_to_idx = {tu: i + 1 for i, tu in enumerate(rnap_tu_fields)}
    monomer_to_idx = {m: i + 1 for i, m in enumerate(monomer_fields)}
    cistron_to_mrna_idx = {c: i + 1 for i, c in enumerate(mrna_cistron_fields)}

    variant_order = sorted(meta.keys())

    parts: list[str] = []
    parts.append(
        "<html><head><meta charset='utf-8'>"
        "<title>New-gene crowding dashboard</title></head>"
        "<body style='font-family:sans-serif;padding:18px;'>"
    )
    parts.append(
        "<h2>New-gene crowding dashboard</h2>"
        f"<p style='color:#555;'>Skipping first {skip_first_n_gens} generation(s). "
        f"RNAP warn/alert at {int(rnap_warn * 100)}%/{int(rnap_alert * 100)}%, "
        f"ribosome warn/alert at {int(ribo_warn * 100)}%/{int(ribo_alert * 100)}% of timesteps "
        "with target&gt;actual. Diminishing returns are visible when increasing exp or trl_eff "
        "across variants no longer increases mean mRNA / protein (right-most points stop "
        "rising and turn red).</p>"
    )

    saved_anything = False
    for cistron_id in new_cistron_ids:
        gene_id = cistron_to_gene.get(cistron_id, cistron_id)
        monomer_id = cistron_to_monomer.get(cistron_id)
        if monomer_id is None:
            print(
                f"multivariant new_gene_crowding_dashboard: cistron {cistron_id} has "
                "no monomer; skipping this gene."
            )
            continue
        # New-gene TUs are monocistronic — the TU id matches the cistron id
        # plus the compartment tag used in rna_data['id']. The listener field
        # metadata exposes those full TU ids.
        candidate_tu_ids = [t for t in rnap_tu_fields if t[:-3] == cistron_id]
        if not candidate_tu_ids:
            # Fallback: exact match (some new genes might not carry [c]).
            candidate_tu_ids = [t for t in rnap_tu_fields if t == cistron_id]
        if not candidate_tu_ids:
            print(
                f"multivariant new_gene_crowding_dashboard: no TU found for cistron "
                f"{cistron_id}; skipping."
            )
            continue
        tu_id = candidate_tu_ids[0]
        tu_idx = tu_to_idx.get(tu_id)
        mrna_idx = cistron_to_mrna_idx.get(cistron_id)
        monomer_idx = monomer_to_idx.get(monomer_id)
        if tu_idx is None or mrna_idx is None or monomer_idx is None:
            print(
                f"multivariant new_gene_crowding_dashboard: missing listener index "
                f"for {gene_id}; skipping."
            )
            continue

        stats = _per_variant_stats(
            conn,
            history_sql,
            tu_idx,
            mrna_idx,
            monomer_idx,
            skip_first_n_gens,
        )
        stats_dict = {int(row["variant"]): row for row in stats.iter_rows(named=True)}

        rows: list[dict[str, Any]] = []
        scatter_records: list[dict[str, Any]] = []
        for v in variant_order:
            params_v = meta[v]
            exp_v, trl_eff_v = _new_gene_exp_trl_eff(params_v)
            srow = stats_dict.get(v)
            rnap_oc = srow["rnap_oc_frac"] if srow else 0.0
            ribo_oc = srow["ribo_oc_frac"] if srow else 0.0
            mean_mrna = srow["mean_mrna"] if srow else None
            mean_protein = srow["mean_protein"] if srow else None
            verdict = _verdict(
                rnap_oc or 0.0,
                ribo_oc or 0.0,
                rnap_warn,
                rnap_alert,
                ribo_warn,
                ribo_alert,
            )
            rows.append(
                {
                    "variant": v,
                    "exp": exp_v,
                    "trl_eff": trl_eff_v,
                    "native": _native_pert_summary(params_v),
                    "rnap_oc": rnap_oc,
                    "ribo_oc": ribo_oc,
                    "mean_mrna": mean_mrna,
                    "mean_protein": mean_protein,
                    "verdict": verdict,
                }
            )
            scatter_records.append(
                {
                    "variant": v,
                    "exp": exp_v,
                    "trl_eff": trl_eff_v,
                    "rnap_oc": rnap_oc or 0.0,
                    "ribo_oc": ribo_oc or 0.0,
                    "mean_mrna": mean_mrna,
                    "mean_protein": mean_protein,
                }
            )

        parts.append(
            f"<h3>{html_lib.escape(gene_id)} ({html_lib.escape(cistron_id)})</h3>"
        )
        parts.append(_gene_table_html(rows))

        scatter_df = pl.DataFrame(scatter_records)
        chart_path = os.path.join(outdir, f"new_gene_crowding_{gene_id}_scatter.html")
        _scatter_chart(scatter_df, gene_id).save(chart_path)
        parts.append(
            f"<iframe src='{os.path.basename(chart_path)}' "
            "style='width:760px;height:330px;border:0;'></iframe>"
        )
        saved_anything = True

    parts.append("</body></html>")

    if not saved_anything:
        print("multivariant new_gene_crowding_dashboard: nothing to plot.")
        return

    output_path = os.path.join(outdir, "new_gene_crowding_dashboard.html")
    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"multivariant new_gene_crowding_dashboard: wrote {output_path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_new_gene_exp_trl_eff_extracts_both():
    p = {"new_gene_shift": {"exp_trl_eff": {"trl_eff": 5.0, "exp": 1e6}}}
    assert _new_gene_exp_trl_eff(p) == (1e6, 5.0)
    assert _new_gene_exp_trl_eff({}) == (None, None)
    assert _new_gene_exp_trl_eff({"new_gene_shift": {}}) == (None, None)


def test_verdict_priorities():
    # Both above alert -> combined label
    label, _ = _verdict(0.6, 0.6, 0.1, 0.5, 0.1, 0.5)
    assert "RNAP+ribosome" in label
    # Only RNAP above alert
    label, _ = _verdict(0.6, 0.0, 0.1, 0.5, 0.1, 0.5)
    assert label == "RNAP saturated"
    # Only ribosome above alert
    label, _ = _verdict(0.0, 0.6, 0.1, 0.5, 0.1, 0.5)
    assert label == "ribosome saturated"
    # Both below warn -> headroom
    label, _ = _verdict(0.0, 0.0, 0.1, 0.5, 0.1, 0.5)
    assert label == "headroom"
    # In between -> nearing limit
    label, _ = _verdict(0.2, 0.0, 0.1, 0.5, 0.1, 0.5)
    assert label == "RNAP nearing limit"


def test_flatten_metadata_collapses_nested_dict():
    raw = {"exp_A": {0: {"new_gene_shift": {}}, 1: {"new_gene_shift": {}}}}
    flat = _flatten_metadata(raw)
    assert set(flat.keys()) == {0, 1}


def test_native_pert_summary_formats_known_dict():
    s = _native_pert_summary({"perturbations": {"EG1": 0.5, "EG2": 2.0}})
    assert "EG1=0.5" in s and "EG2=2" in s
    assert _native_pert_summary({}) == ""
