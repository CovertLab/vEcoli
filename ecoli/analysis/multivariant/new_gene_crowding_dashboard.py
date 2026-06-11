"""
Multivariant dashboard summarising whether each new gene is bottlenecked
by RNA-polymerase or ribosome crowding under every variant.

For each new gene (``is_new_gene`` in sim_data), the dashboard emits two
tables — a **transcription** (RNA-polymerase) table and a **translation**
(ribosome) table — reporting per-variant:

- the new-gene knobs read per variant from ``sim_data.internal_shift_dict``
  (the scheduled ``modify_new_gene_exp_trl`` call, falling back to
  ``new_gene_shift.exp_trl_eff`` metadata): ``log10(exp)`` =
  ``log10(expression / baseline)`` (transcription) and ``trl_eff``
  (translation)
- the overcrowded fraction: timesteps flagged ``tu_is_overcrowded``
  (RNAP) / where ribosome target prob exceeded actual prob (ribosome)
- ``demand honored`` = mean actual ÷ mean target initiation prob, the
  *magnitude* the binary overcrowded flag throws away
- mean target / actual initiation prob and mean ``max_p`` (the
  per-promoter probability ceiling: ``rna_synth_prob.max_p`` for RNAP,
  ``ribosome_data.max_p_per_protein`` for ribosomes)
- ``target/max_p``, the per-gene overcrowding severity
- the mean new-gene mRNA / protein counts

The goal is to surface *diminishing returns*: as ``exp`` or ``trl_eff``
is ramped, output should increase roughly linearly until either the TU
saturates (RNAP overcrowded) or the monomer's translation prob saturates
(ribosome overcrowded). The tables make that knee visible — when ramping
a parameter no longer increases mean mRNA/protein and the crowding
fraction crosses the alert threshold, further ramping is wasted. Because
``max_p`` is a global RNAP/ribosome-supply signal, a less-contended
variant can reach higher output at lower crowding than a more-contended
one at the same knob value.

When a TU carries multiple new-gene cistrons (polycistronic cassette),
the RNAP overcrowded fraction is a TU-level signal shared across every
cistron in that TU; the ribosome overcrowded fraction and mean mRNA /
protein are per-cistron.

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
import math
import os
import pickle
from typing import Any

import polars as pl
from duckdb import DuckDBPyConnection
from fsspec import open as fsspec_open

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
)
from wholecell.utils import units


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


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _new_gene_exp_trl_eff(params: dict[str, Any]) -> tuple[float | None, float | None]:
    ng = params.get("new_gene_shift") or {}
    if not isinstance(ng, dict):
        return None, None
    ete = ng.get("exp_trl_eff") or {}
    if not isinstance(ete, dict):
        return None, None
    return _coerce_float(ete.get("exp")), _coerce_float(ete.get("trl_eff"))


def _new_gene_shift_from_sim_data(sim_data: Any) -> tuple[float | None, float | None]:
    """Read the new-gene ``(exp, trl_eff)`` straight from a variant's sim_data.

    The strain-design / new-gene-shift variants don't bake the shift into
    ``rna_expression`` at parca time — they schedule ``modify_new_gene_exp_trl``
    to run at the induction generation and stash the call in
    ``sim_data.internal_shift_dict`` as ``{gen: (func, (exp, trl_eff))}``. That
    dict *is* pickled, so it's the one place the per-variant knobs live in
    sim_data (``rna_expression`` for the new gene is still 0 in the pickle).

    ``exp`` is the multiplicative factor over ``new_gene_rna_expression_baseline``
    (final = baseline * exp), so it already equals expression/baseline. We take
    the earliest scheduled generation (induction); any later knockout entry sets
    exp=0 and is ignored. Returns ``(None, None)`` if no shift is scheduled.
    """
    shift = getattr(sim_data, "internal_shift_dict", None)
    if not isinstance(shift, dict) or not shift:
        return None, None
    best_gen: Any = None
    best: tuple[float | None, float | None] = (None, None)
    for gen, entry in shift.items():
        if not (isinstance(entry, tuple) and len(entry) == 2):
            continue
        func, args = entry
        if getattr(func, "__name__", "") != "modify_new_gene_exp_trl":
            continue
        if not (isinstance(args, tuple) and len(args) >= 2):
            continue
        if best_gen is None or gen < best_gen:
            best_gen = gen
            best = (_coerce_float(args[0]), _coerce_float(args[1]))
    return best


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
    variant with the overcrowded fractions, mean target/actual/max_p for
    both axes, mean mRNA/protein counts, the per-timestep initiation rates
    (``avg_rnap_inits_per_copy``, ``avg_ribo_inits_per_mrna``) used for
    inter-machine spacing, and ``n_steps``.
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
            listeners__rna_synth_prob__max_p AS rnap_max_p,
            listeners__rnap_data__rna_init_event[{tu_idx_1based}] AS rnap_init,
            listeners__rna_synth_prob__promoter_copy_number[{tu_idx_1based}]
                AS promoter_copies,
            listeners__ribosome_data__target_prob_translation_per_transcript[{monomer_idx_1based}]
                AS ribo_target,
            listeners__ribosome_data__actual_prob_translation_per_transcript[{monomer_idx_1based}]
                AS ribo_actual,
            listeners__ribosome_data__max_p_per_protein[{monomer_idx_1based}]
                AS ribo_max_p,
            listeners__ribosome_data__ribosome_init_event_per_monomer[{monomer_idx_1based}]
                AS ribo_init,
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
        avg(rnap_max_p) AS mean_rnap_max_p,
        avg(ribo_target) AS mean_ribo_target,
        avg(ribo_actual) AS mean_ribo_actual,
        avg(ribo_max_p) AS mean_ribo_max_p,
        avg(case when promoter_copies > 0
            then rnap_init::DOUBLE / promoter_copies else 0.0 end)
            AS avg_rnap_inits_per_copy,
        avg(case when mrna_count > 0
            then ribo_init::DOUBLE / mrna_count else 0.0 end)
            AS avg_ribo_inits_per_mrna,
        count(*) AS n_steps
    FROM src
    GROUP BY variant
    ORDER BY variant
    """
    return conn.sql(query).pl()


# ---------------------------------------------------------------------------
# Verdict / colour helpers
# ---------------------------------------------------------------------------


def _axis_verdict(
    oc: float,
    warn: float,
    alert: float,
    machine: str,
) -> tuple[str, str]:
    """Return (label, bg_colour) for a single bottleneck axis.

    ``machine`` is the human label for the limiting machine ("RNAP" or
    "ribosome"). The verdict is driven purely by that axis's overcrowded
    fraction vs. the warn/alert thresholds, so the transcription and
    translation tables each get their own independent verdict.
    """
    if oc >= alert:
        return (f"{machine} saturated", "#a33")
    if oc >= warn:
        return (f"{machine} nearing limit", "#8a6d3b")
    return ("headroom", "#2b6e2b")


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    """num / den, guarding against None, NaN, and zero/near-zero denominators."""
    if num is None or den is None:
        return None
    try:
        num = float(num)
        den = float(den)
    except (TypeError, ValueError):
        return None
    if den == 0 or den != den or num != num:
        return None
    return num / den


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{x * 100:.1f}%"


def _fmt_prob(x: float | None) -> str:
    """Format an initiation probability (typically tiny, e.g. 1e-4)."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    if x == 0:
        return "0"
    return f"{x:.3g}"


def _fmt_mult(x: float | None) -> str:
    """Format a ratio as a multiplier, e.g. 2.31x. >1 for target/max_p means
    the request overshoots the per-promoter ceiling."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{x:.2f}×"


def _spacing_cell(spacing_nt: float | None, footprint_nt: float | None) -> str:
    """Format an inter-machine spacing in nt. Highlights red when the spacing
    is below the physical footprint (machines can't fit that close together)."""
    if spacing_nt is None or (
        isinstance(spacing_nt, float) and spacing_nt != spacing_nt
    ):
        return f"<td {_TD}></td>"
    jammed = footprint_nt is not None and spacing_nt < footprint_nt
    text = f"{spacing_nt:,.0f}"
    if jammed:
        return (
            f'<td style="border:1px solid #999;padding:4px 8px;'
            f'background:#a33;color:#fff;">{text}</td>'
        )
    return f"<td {_TD}>{text}</td>"


def _fmt_float(x: float | None, places: int = 1) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{x:.{places}f}"


def _fmt_int(x: float | None) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return f"{int(round(x)):,}"


def _fmt_log_exp(exp: float | None) -> str:
    """Format ``exp`` as log10(exp). Since ``exp`` is the multiplicative factor
    over the new-gene baseline expression, this is log10(expression / baseline)
    — e.g. exp=1e6 -> 6.0. Non-positive/None -> blank."""
    if exp is None or (isinstance(exp, float) and exp != exp) or exp <= 0:
        return ""
    return f"{math.log10(exp):.2f}"


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


def _verdict_cell(verdict: tuple[str, str]) -> str:
    label, bg = verdict
    return (
        f'<td style="border:1px solid #999;padding:4px 8px;'
        f'background:{bg};color:#fff;">{html_lib.escape(label)}</td>'
    )


def _table_html(headers: list[str], body_rows: list[str]) -> str:
    head = "".join(f"<th {_TH}>{h}</th>" for h in headers)
    return (
        f'<table style="{_TABLE_STYLE}">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _transcription_table_html(
    rows: list[dict[str, Any]], rnap_footprint_nt: float | None
) -> str:
    """RNA-polymerase / transcription axis: how much of the requested synthesis
    probability got through, and how hard this TU is pushing against max_p."""
    fp = "" if rnap_footprint_nt is None else f" (footprint {rnap_footprint_nt:.0f})"
    headers = [
        "variant",
        "log₁₀(exp/base)",
        "native perturbations",
        "mean mRNA",
        "RNAP overcrowded",
        "demand honored",
        "mean target p",
        "mean actual p",
        "mean max_p",
        "target/max_p",
        f"RNAP spacing (nt){fp}",
        "verdict",
    ]
    body: list[str] = []
    for r in rows:
        honored = _safe_ratio(r["rnap_actual"], r["rnap_target"])
        target_over_maxp = _safe_ratio(r["rnap_target"], r["rnap_max_p"])
        body.append(
            "<tr>"
            f"<td {_TD}>V{r['variant']}</td>"
            f"<td {_TD}>{_fmt_log_exp(r['exp'])}</td>"
            f"<td {_TD}>{html_lib.escape(r['native'])}</td>"
            f"<td {_TD}>{_fmt_float(r['mean_mrna'])}</td>"
            f"<td {_TD}>{_fmt_pct(r['rnap_oc'])}</td>"
            f"<td {_TD}>{_fmt_pct(honored)}</td>"
            f"<td {_TD}>{_fmt_prob(r['rnap_target'])}</td>"
            f"<td {_TD}>{_fmt_prob(r['rnap_actual'])}</td>"
            f"<td {_TD}>{_fmt_prob(r['rnap_max_p'])}</td>"
            f"<td {_TD}>{_fmt_mult(target_over_maxp)}</td>"
            f"{_spacing_cell(r['rnap_spacing'], rnap_footprint_nt)}"
            f"{_verdict_cell(r['rnap_verdict'])}"
            "</tr>"
        )
    return _table_html(headers, body)


def _translation_table_html(
    rows: list[dict[str, Any]], ribo_footprint_nt: float | None
) -> str:
    """Ribosome / translation axis: per-monomer analogue of the transcription
    table, using the per-protein initiation cap (max_p_per_protein)."""
    fp = "" if ribo_footprint_nt is None else f" (footprint {ribo_footprint_nt:.0f})"
    headers = [
        "variant",
        "trl_eff",
        "native perturbations",
        "mean protein",
        "ribosome overcrowded",
        "demand honored",
        "mean target p",
        "mean actual p",
        "mean max_p",
        "target/max_p",
        f"ribosome spacing (nt){fp}",
        "verdict",
    ]
    body: list[str] = []
    for r in rows:
        honored = _safe_ratio(r["ribo_actual"], r["ribo_target"])
        target_over_maxp = _safe_ratio(r["ribo_target"], r["ribo_max_p"])
        body.append(
            "<tr>"
            f"<td {_TD}>V{r['variant']}</td>"
            f"<td {_TD}>{_fmt_float(r['trl_eff'], 2)}</td>"
            f"<td {_TD}>{html_lib.escape(r['native'])}</td>"
            f"<td {_TD}>{_fmt_int(r['mean_protein'])}</td>"
            f"<td {_TD}>{_fmt_pct(r['ribo_oc'])}</td>"
            f"<td {_TD}>{_fmt_pct(honored)}</td>"
            f"<td {_TD}>{_fmt_prob(r['ribo_target'])}</td>"
            f"<td {_TD}>{_fmt_prob(r['ribo_actual'])}</td>"
            f"<td {_TD}>{_fmt_prob(r['ribo_max_p'])}</td>"
            f"<td {_TD}>{_fmt_mult(target_over_maxp)}</td>"
            f"{_spacing_cell(r['ribo_spacing'], ribo_footprint_nt)}"
            f"{_verdict_cell(r['ribo_verdict'])}"
            "</tr>"
        )
    return _table_html(headers, body)


def _legend_html() -> str:
    """Bulleted description of every computed column, shown once at the top."""
    items = [
        (
            "variant",
            "Variant id (Vn). Rows are in variant-id order, not "
            "necessarily ascending exp/trl_eff.",
        ),
        (
            "log₁₀(exp/base) / trl_eff",
            "The new-gene knobs, read per variant from "
            "<code>sim_data.internal_shift_dict</code> (the scheduled "
            "<code>modify_new_gene_exp_trl</code> call), falling back to "
            "<code>new_gene_shift.exp_trl_eff</code> metadata. <code>exp</code> is the "
            "multiplicative factor over the new-gene baseline expression, so the "
            "transcription table shows <b>log₁₀(exp)</b> = log₁₀(expression / baseline) "
            "(e.g. exp=1e6 → 6.0). <code>trl_eff</code> (translation table) is shown "
            "linearly.",
        ),
        (
            "native perturbations",
            "Any native-gene fold-changes set for this "
            "variant (<code>perturbations</code>).",
        ),
        (
            "mean mRNA / mean protein",
            "Mean new-gene cistron mRNA count "
            "(transcription) / monomer count (translation), over post-skip timesteps.",
        ),
        (
            "RNAP overcrowded",
            "Fraction of timesteps the TU's promoter init prob "
            "was capped at <code>max_p</code> (<code>tu_is_overcrowded</code>). Binary "
            "per step: saturates at 100% and says nothing about magnitude.",
        ),
        (
            "ribosome overcrowded",
            "Fraction of timesteps the monomer's target "
            "translation prob exceeded its actual prob.",
        ),
        (
            "demand honored",
            "mean actual p ÷ mean target p — how much of the "
            "requested initiation actually got through. This is the magnitude the "
            "binary overcrowded flag throws away (100% overcrowded but 95% honored is "
            "very different from 100% overcrowded but 40% honored).",
        ),
        (
            "mean target p / mean actual p",
            "Mean requested vs. achieved initiation "
            "probability (per-TU for transcription, per-monomer for translation).",
        ),
        (
            "mean max_p",
            "Mean per-promoter probability ceiling — transcription uses "
            "<code>rna_synth_prob.max_p</code> (RNAP footprint ÷ RNAPs activated); "
            "translation uses <code>ribosome_data.max_p_per_protein</code>. Higher = "
            "more machine headroom that step, so a less-contended variant clips less "
            "even at the same exp/trl_eff.",
        ),
        (
            "target/max_p",
            "mean target p ÷ mean max_p. &gt;1× means this gene's "
            "request overshoots the per-promoter ceiling — the overcrowding severity "
            "for <i>this</i> gene specifically (vs. the global supply signal in max_p).",
        ),
        (
            "RNAP / ribosome spacing (nt)",
            "Average physical distance between "
            "successive machines on the template, à la wcEcoli "
            "<code>inter_rnap_distance</code> / <code>inter_ribosome_distance</code>: "
            "elongation rate ÷ initiation rate. Transcription uses "
            "<code>rna_init_event ÷ promoter_copy_number</code>; translation uses "
            "<code>ribosome_init_event_per_monomer ÷ mRNA_cistron_counts</code>. The "
            "header shows the footprint size; a cell turns red when spacing falls below "
            "it — the machines are packed tighter than physically fits, the hard "
            "crowding limit. (Initiation rate is per-timestep, i.e. assumes ~1&nbsp;s "
            "timesteps, matching <code>ribosome_spacing.py</code>.)",
        ),
        (
            "verdict",
            "Per-axis headroom / nearing-limit / saturated, from that "
            "axis's overcrowded fraction vs. the warn/alert thresholds.",
        ),
    ]
    lis = "".join(
        f"<li><b>{html_lib.escape(name)}</b> — {desc}</li>" for name, desc in items
    )
    return (
        "<details open style='margin:8px 0 16px 0;'>"
        "<summary style='cursor:pointer;font-weight:bold;'>Column reference</summary>"
        f"<ul style='color:#333;font-size:13px;line-height:1.5;'>{lis}</ul>"
        "</details>"
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

    # Elongation rates (nt/s) and physical footprints (nt) for inter-machine
    # spacing, à la wcEcoli inter_rnap_distance / inter_ribosome_distance.
    # Wrapped defensively: a missing condition just disables the spacing cells.
    rnap_elong_rate = ribo_elong_rate = None
    rnap_footprint_nt = ribo_footprint_nt = None
    try:
        nutrients = sim_data.conditions[sim_data.condition]["nutrients"]
        rnap_elong_rate = (
            sim_data.process.transcription.rnaPolymeraseElongationRateDict[
                nutrients
            ].asNumber(units.nt / units.s)
        )
        rnap_footprint_nt = (
            sim_data.process.transcription.active_rnap_footprint_size.asNumber(units.nt)
        )
        # ribosome rate is aa/s; *3 to convert codons -> nt (matches ribosome_spacing.py)
        ribo_elong_rate = (
            sim_data.process.translation.ribosomeElongationRateDict[nutrients].asNumber(
                units.aa / units.s
            )
            * 3
        )
        ribo_footprint_nt = (
            sim_data.process.translation.active_ribosome_footprint_size.asNumber(
                units.nt
            )
        )
    except (KeyError, AttributeError) as e:
        print(
            "multivariant new_gene_crowding_dashboard: could not resolve elongation "
            f"rates / footprints ({e!r}); inter-machine spacing columns disabled."
        )

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

    # Per-variant (exp, trl_eff), read from each variant's own sim_data
    # (internal_shift_dict). Falls back to variant metadata when a variant has
    # no scheduled shift (e.g. the control). Works regardless of how the
    # variant was configured, unlike reading new_gene_shift metadata alone.
    variant_paths: dict[int, Any] = {}
    for exp_variants in sim_data_dict.values():
        for vid_raw, path in exp_variants.items():
            try:
                vid = int(vid_raw)
            except (TypeError, ValueError):
                continue
            variant_paths.setdefault(vid, path)

    variant_exp_trl: dict[int, tuple[float | None, float | None]] = {}
    for v in variant_order:
        exp_v = trl_eff_v = None
        path = variant_paths.get(v)
        if path is not None:
            try:
                with fsspec_open(path, "rb") as fh:
                    sd_v = pickle.load(fh)
                exp_v, trl_eff_v = _new_gene_shift_from_sim_data(sd_v)
            except Exception as e:  # noqa: BLE001 - never let one variant break the table
                print(
                    "multivariant new_gene_crowding_dashboard: could not read shift "
                    f"from sim_data for variant {v} ({e!r}); using metadata."
                )
        if exp_v is None and trl_eff_v is None:
            exp_v, trl_eff_v = _new_gene_exp_trl_eff(meta[v])
        variant_exp_trl[v] = (exp_v, trl_eff_v)

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
        f"ribosome warn/alert at {int(ribo_warn * 100)}%/{int(ribo_alert * 100)}% of "
        "timesteps. Diminishing returns: when ramping exp or trl_eff across variants "
        "no longer increases mean mRNA / protein and the matching crowding fraction "
        "crosses the alert threshold, further ramping is wasted. Each new gene gets a "
        "<b>transcription</b> table (RNAP axis) and a <b>translation</b> table "
        "(ribosome axis).</p>"
    )
    parts.append(_legend_html())

    rna_data = sim_data.process.transcription.rna_data
    rna_ids = list(rna_data["id"])

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
        # Find every TU containing this cistron via the proper API
        # (handles polycistronic new-gene cassettes).
        try:
            tu_indices_in_rna_data = list(
                sim_data.process.transcription.cistron_id_to_rna_indexes(cistron_id)
            )
        except KeyError:
            tu_indices_in_rna_data = []
        candidate_tu_ids = [rna_ids[i] for i in tu_indices_in_rna_data]
        candidate_tu_ids = [t for t in candidate_tu_ids if t in tu_to_idx]
        if not candidate_tu_ids:
            print(
                f"multivariant new_gene_crowding_dashboard: no listener-indexed TU "
                f"found for cistron {cistron_id}; skipping."
            )
            continue
        # For a polycistronic cassette the cistron's RNAP-overcrowding signal
        # is shared with everything else on the same TU. Pick the first TU as
        # the canonical one (in practice new-gene cassettes are engineered
        # onto a single TU). If the cistron is on multiple TUs, note it.
        tu_id = candidate_tu_ids[0]
        tu_idx = tu_to_idx[tu_id]
        if len(candidate_tu_ids) > 1:
            print(
                f"multivariant new_gene_crowding_dashboard: cistron {cistron_id} sits "
                f"on {len(candidate_tu_ids)} TUs; using {tu_id} for RNAP overcrowding."
            )
        mrna_idx = cistron_to_mrna_idx.get(cistron_id)
        monomer_idx = monomer_to_idx.get(monomer_id)
        if mrna_idx is None or monomer_idx is None:
            print(
                f"multivariant new_gene_crowding_dashboard: missing mRNA/monomer "
                f"listener index for {gene_id}; skipping."
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
        for v in variant_order:
            params_v = meta[v]
            exp_v, trl_eff_v = variant_exp_trl[v]
            srow = stats_dict.get(v)
            rnap_oc = (srow["rnap_oc_frac"] if srow else 0.0) or 0.0
            ribo_oc = (srow["ribo_oc_frac"] if srow else 0.0) or 0.0
            # Inter-machine spacing (nt) = elongation rate / initiation rate.
            rnap_spacing = _safe_ratio(
                rnap_elong_rate, srow["avg_rnap_inits_per_copy"] if srow else None
            )
            ribo_spacing = _safe_ratio(
                ribo_elong_rate, srow["avg_ribo_inits_per_mrna"] if srow else None
            )
            rows.append(
                {
                    "variant": v,
                    "exp": exp_v,
                    "trl_eff": trl_eff_v,
                    "native": _native_pert_summary(params_v),
                    "rnap_oc": rnap_oc,
                    "ribo_oc": ribo_oc,
                    "mean_mrna": srow["mean_mrna"] if srow else None,
                    "mean_protein": srow["mean_protein"] if srow else None,
                    "rnap_target": srow["mean_rnap_target"] if srow else None,
                    "rnap_actual": srow["mean_rnap_actual"] if srow else None,
                    "rnap_max_p": srow["mean_rnap_max_p"] if srow else None,
                    "ribo_target": srow["mean_ribo_target"] if srow else None,
                    "ribo_actual": srow["mean_ribo_actual"] if srow else None,
                    "ribo_max_p": srow["mean_ribo_max_p"] if srow else None,
                    "rnap_spacing": rnap_spacing,
                    "ribo_spacing": ribo_spacing,
                    "rnap_verdict": _axis_verdict(
                        rnap_oc, rnap_warn, rnap_alert, "RNAP"
                    ),
                    "ribo_verdict": _axis_verdict(
                        ribo_oc, ribo_warn, ribo_alert, "ribosome"
                    ),
                }
            )

        tu_note = ""
        if len(candidate_tu_ids) > 1:
            tu_note = (
                f" <span style='color:#888;font-size:11px;'>"
                f"(TU {html_lib.escape(tu_id)} shared with "
                f"{len(candidate_tu_ids) - 1} other TU(s) containing this cistron)"
                "</span>"
            )
        parts.append(
            f"<h3>{html_lib.escape(gene_id)} ({html_lib.escape(cistron_id)})"
            f"{tu_note}</h3>"
        )
        parts.append("<h4 style='margin:8px 0 2px 0;'>Transcription (RNAP)</h4>")
        parts.append(_transcription_table_html(rows, rnap_footprint_nt))
        parts.append("<h4 style='margin:8px 0 2px 0;'>Translation (ribosome)</h4>")
        parts.append(_translation_table_html(rows, ribo_footprint_nt))
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


def test_new_gene_shift_from_sim_data_reads_induction():
    def modify_new_gene_exp_trl():  # name is what the helper matches on
        pass

    class _SD:
        # induction at gen 2 (exp 1e6), knockout at gen 4 (exp 0); take induction
        internal_shift_dict = {
            2: (modify_new_gene_exp_trl, (1e6, 5.0)),
            4: (modify_new_gene_exp_trl, (0, 5.0)),
        }

    assert _new_gene_shift_from_sim_data(_SD()) == (1e6, 5.0)
    # No shift dict -> (None, None) so the caller falls back to metadata
    assert _new_gene_shift_from_sim_data(object()) == (None, None)


def test_fmt_log_exp():
    assert _fmt_log_exp(1e6) == "6.00"
    assert _fmt_log_exp(1.0) == "0.00"
    assert _fmt_log_exp(0) == ""
    assert _fmt_log_exp(None) == ""


def test_axis_verdict_thresholds():
    # Above alert
    assert _axis_verdict(0.6, 0.1, 0.5, "RNAP")[0] == "RNAP saturated"
    assert _axis_verdict(0.6, 0.1, 0.5, "ribosome")[0] == "ribosome saturated"
    # Between warn and alert
    assert _axis_verdict(0.2, 0.1, 0.5, "RNAP")[0] == "RNAP nearing limit"
    # Below warn
    assert _axis_verdict(0.0, 0.1, 0.5, "ribosome")[0] == "headroom"


def test_safe_ratio_guards():
    assert _safe_ratio(1.0, 2.0) == 0.5
    assert _safe_ratio(1.0, 0.0) is None
    assert _safe_ratio(None, 2.0) is None
    assert _safe_ratio(float("nan"), 2.0) is None


def test_spacing_cell_flags_below_footprint():
    # Spacing below footprint -> red background
    jammed = _spacing_cell(10.0, 24.0)
    assert "#a33" in jammed and "10" in jammed
    # Spacing above footprint -> normal cell
    roomy = _spacing_cell(100.0, 24.0)
    assert "#a33" not in roomy and "100" in roomy
    # Missing spacing -> empty cell, no crash
    assert _spacing_cell(None, 24.0).endswith("></td>")
    # Missing footprint -> never flagged red
    assert "#a33" not in _spacing_cell(1.0, None)


def test_flatten_metadata_collapses_nested_dict():
    raw = {"exp_A": {0: {"new_gene_shift": {}}, 1: {"new_gene_shift": {}}}}
    flat = _flatten_metadata(raw)
    assert set(flat.keys()) == {0, 1}


def test_native_pert_summary_formats_known_dict():
    s = _native_pert_summary({"perturbations": {"EG1": 0.5, "EG2": 2.0}})
    assert "EG1=0.5" in s and "EG2=2" in s
    assert _native_pert_summary({}) == ""
