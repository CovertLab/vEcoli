"""
Custom Escher-style pathway flux map for the 5 pilot pathways curated in
``validation/ecoli/flat/core_pathway_reactions.tsv``, for multivariant
simulations.

Builds an Escher map JSON from scratch (Escher's documented map schema),
keyed on vEcoli's own EcoCyc-style reaction IDs (not BiGG), so no external
ID crosswalk is needed. Metabolite node labels are the full EcoCyc
stoichiometry IDs exactly as written in
``reconstruction/ecoli/flat/metabolic_reactions.tsv`` (e.g.
``OXALACETIC_ACID[CCO-CYTOSOL]``), read directly from that flat file rather
than the abbreviated ``[c]``-style IDs used elsewhere in sim_data.

Layout: each pathway is placed in its own horizontal row, ordered
left-to-right by the TSV's "order" column. Metabolites shared between
reactions (including across pathways, at the curated branch points) reuse
the same node, so the diagram connects naturally wherever stoichiometry
actually overlaps — this is a simple deterministic layout, not a
hand-tuned, textbook-quality diagram.

One Escher map HTML is saved per variant, colored by mean base reaction
flux (``listeners__fba_results__base_reaction_fluxes``, the same source
``pathway_flux_distribution.py`` uses).
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
import re
from typing import Any, TYPE_CHECKING

import escher
import polars as pl

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

PATHWAY_TSV = "validation/ecoli/flat/core_pathway_reactions.tsv"
METABOLIC_REACTIONS_TSV = "reconstruction/ecoli/flat/metabolic_reactions.tsv"

X_SPACING = 300.0
Y_SPACING = 400.0
MET_OFFSET_Y = 110.0
MET_FAN_X = 50.0

# Common energy/redox/group-carrier cofactors and small inorganic
# molecules, identified from the curated pathways' real stoichiometry.
# These get a fresh, reaction-local node every time they appear instead
# of sharing one global node (see get_met_node), since they participate
# in many unrelated reactions scattered across the map; sharing one node
# for them is what causes long lines criss-crossing the whole canvas.
# ACETYL-COA is deliberately excluded: it's the backbone carbon carrier
# at the TCA Cycle / Acetate Fermentation branch point, not a pure
# cofactor, and should stay shared like other backbone metabolites.
COFACTOR_IDS = frozenset(
    {
        "ATP[CCO-CYTOSOL]",
        "ADP[CCO-CYTOSOL]",
        "NAD[CCO-CYTOSOL]",
        "NADH[CCO-CYTOSOL]",
        "NADP[CCO-CYTOSOL]",
        "NADPH[CCO-CYTOSOL]",
        "CO-A[CCO-CYTOSOL]",
        "WATER[CCO-CYTOSOL]",
        "PROTON[CCO-CYTOSOL]",
        "Pi[CCO-CYTOSOL]",
        "CARBON-DIOXIDE[CCO-CYTOSOL]",
        "HCO3[CCO-CYTOSOL]",
        "FORMATE[CCO-CYTOSOL]",
        "UBIQUINONE-8[CCO-PM-BAC-NEG]",
        "CPD-9956[CCO-PM-BAC-NEG]",
    }
)


def _read_base_reaction_stoich(
    base_reaction_ids: set[str], reaction_id_to_base_reaction_id: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Raw (verbose-compartment EcoCyc) stoichiometry for the given base
    reaction IDs, read directly from metabolic_reactions.tsv.

    metabolic_reactions.tsv's "id" column uses fully-qualified variant IDs
    for reactions with multiple specific substrate/product forms (e.g.
    "SUCCINATE-DEHYDROGENASE-UBIQUINONE-RXN-SUC/UBIQUINONE-8//FUM/CPD-9956.31."),
    not the collapsed base ID used elsewhere (e.g. in base_reaction_fluxes).
    reaction_id_to_base_reaction_id only has entries for raw IDs that differ
    from their base ID (qualified-variant suffixes, or the "(reverse)"
    suffix used for the reverse direction of a reversible reaction); the
    forward/canonical raw ID is implicit and always equal to the base ID
    itself, so it's never a key there. Resolve base -> raw IDs by combining
    the base ID with any known variants, then merge stoichiometry across
    whichever of those raw IDs are actually present in the flat file.
    """
    base_to_raw: dict[str, list[str]] = {}
    for raw_id, base_id in reaction_id_to_base_reaction_id.items():
        if base_id in base_reaction_ids:
            base_to_raw.setdefault(base_id, []).append(raw_id)

    def raw_ids_for(base_id: str) -> list[str]:
        return [base_id, *base_to_raw.get(base_id, [])]

    all_raw_ids = {
        raw_id for base_id in base_reaction_ids for raw_id in raw_ids_for(base_id)
    }

    df = pl.read_csv(
        METABOLIC_REACTIONS_TSV,
        separator="\t",
        quote_char='"',
        columns=["id", "stoichiometry"],
    )
    df = df.filter(pl.col("id").is_in(all_raw_ids))
    raw_stoich = {
        row["id"]: json.loads(row["stoichiometry"]) for row in df.iter_rows(named=True)
    }

    base_stoich: dict[str, dict[str, int]] = {}
    for base_id in base_reaction_ids:
        merged: dict[str, int] = {}
        for raw_id in raw_ids_for(base_id):
            merged.update(raw_stoich.get(raw_id, {}))
        base_stoich[base_id] = merged
    return base_stoich


def _build_pathway_map(
    pathway_df: pl.DataFrame, stoich_by_rxn: dict[str, dict[str, int]]
) -> str:
    """Construct an Escher map JSON (as a string) for the curated pathways.

    Each pathway is its own horizontal row, ordered by the TSV's "order"
    column; backbone metabolites are reused across reactions wherever their
    (verbose EcoCyc) IDs match, so the pathway's carbon backbone connects
    naturally. Cofactors (COFACTOR_IDS) instead get a fresh node local to
    each reaction that uses them, rendered smaller via node_is_primary, to
    avoid long lines connecting back to one shared, distant node.
    """
    node_id_counter = itertools.count(1)
    nodes: dict[str, dict[str, Any]] = {}
    reactions: dict[str, dict[str, Any]] = {}
    met_node_id: dict[str, str] = {}

    def get_met_node(met_id: str, x: float, y: float) -> str:
        is_primary = met_id not in COFACTOR_IDS
        if is_primary and met_id in met_node_id:
            return met_node_id[met_id]
        nid = str(next(node_id_counter))
        nodes[nid] = {
            "node_type": "metabolite",
            "x": x,
            "y": y,
            "label_x": x + 10,
            "label_y": y - 10,
            "name": met_id,
            "bigg_id": met_id,
            "node_is_primary": is_primary,
        }
        if is_primary:
            met_node_id[met_id] = nid
        return nid

    pathways = sorted(pathway_df["pathway"].unique().to_list())
    for row_idx, pathway in enumerate(pathways):
        y = row_idx * Y_SPACING
        sub = pathway_df.filter(pl.col("pathway") == pathway).sort("order")
        for col_idx, row in enumerate(sub.iter_rows(named=True)):
            rxn_id = row["reaction_id"]
            x = col_idx * X_SPACING
            stoich = stoich_by_rxn[rxn_id]
            substrates = [m for m, c in stoich.items() if c < 0]
            products = [m for m, c in stoich.items() if c > 0]

            mid_id = str(next(node_id_counter))
            nodes[mid_id] = {"node_type": "midmarker", "x": x, "y": y}

            # Escher indexes beziers by segment ID alone across the whole
            # map (not per-reaction), so segment IDs must be drawn from the
            # same global counter as node IDs, not restarted at 1 per reaction.
            segments: dict[str, Any] = {}
            for i, met in enumerate(substrates):
                met_x = x - MET_FAN_X * (len(substrates) - 1) / 2 + i * MET_FAN_X
                mnid = get_met_node(met, met_x, y - MET_OFFSET_Y)
                segments[str(next(node_id_counter))] = {
                    "from_node_id": mnid,
                    "to_node_id": mid_id,
                    "b1": None,
                    "b2": None,
                }
            for i, met in enumerate(products):
                met_x = x - MET_FAN_X * (len(products) - 1) / 2 + i * MET_FAN_X
                mnid = get_met_node(met, met_x, y + MET_OFFSET_Y)
                segments[str(next(node_id_counter))] = {
                    "from_node_id": mid_id,
                    "to_node_id": mnid,
                    "b1": None,
                    "b2": None,
                }

            rxn_node_id = str(next(node_id_counter))
            reactions[rxn_node_id] = {
                "name": row["reaction_label"],
                "bigg_id": rxn_id,
                "reversibility": False,
                "label_x": x,
                "label_y": y - 20,
                "gene_reaction_rule": "",
                "genes": [],
                "metabolites": [
                    {"bigg_id": m, "coefficient": c} for m, c in stoich.items()
                ],
                "segments": segments,
            }

    all_x = [n["x"] for n in nodes.values()]
    all_y = [n["y"] for n in nodes.values()]
    canvas = {
        "x": min(all_x) - 150,
        "y": min(all_y) - 150,
        "width": max(all_x) - min(all_x) + 300,
        "height": max(all_y) - min(all_y) + 300,
    }
    header = {
        "map_name": "vEcoli core pathway map",
        "map_id": "vecoli_core_pathways",
        "map_description": (
            "Generated from validation/ecoli/flat/core_pathway_reactions.tsv"
        ),
        "homepage": "https://escher.github.io",
        "schema": "https://escher.github.io/escher/jsonschema/1-0-0#",
    }
    body = {"reactions": reactions, "nodes": nodes, "canvas": canvas, "text_labels": {}}
    return json.dumps([header, body])


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
    All options have default values (do not need to be explicitly provided).

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Config options look like this:

            .. code-block:: json

                {
                    // Whether base reaction fluxes are raw counts requiring
                    // conversion to molar via counts_to_molar (true for
                    // MetabolismReduxClassic).
                    "is_reduxclassic": true
                }
    """
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )
    is_reduxclassic = params.get("is_reduxclassic", True)

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    pathway_df = pl.read_csv(
        PATHWAY_TSV, separator="\t", comment_prefix="#", quote_char='"'
    )
    all_reaction_ids = sorted(set(pathway_df["reaction_id"].to_list()))
    stoich_by_rxn = _read_base_reaction_stoich(
        set(all_reaction_ids),
        sim_data.process.metabolism.reaction_id_to_base_reaction_id,
    )
    map_json = _build_pathway_map(pathway_df, stoich_by_rxn)

    base_reaction_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__base_reaction_fluxes"
    )
    reaction_idx = [base_reaction_ids.index(r) for r in all_reaction_ids]

    columns = [
        named_idx(
            "listeners__fba_results__base_reaction_fluxes",
            all_reaction_ids,
            [reaction_idx],
        ),
        "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
    ]
    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            columns,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
            remove_first=is_reduxclassic,
        )
    )
    if raw.is_empty():
        print("escher_pathway_map: no rows returned; skipping.")
        return

    if is_reduxclassic:
        raw = raw.with_columns(
            [
                (pl.col(r) * pl.col("counts_to_molar")).abs().alias(r)
                for r in all_reaction_ids
            ]
        )
    else:
        raw = raw.with_columns([pl.col(r).abs().alias(r) for r in all_reaction_ids])

    unique_variants = sorted(raw["variant"].unique().to_list())
    for variant_val in unique_variants:
        label_l = _variant_label(variant_val, per_variant_params)
        variant_label = " ".join(label_l) if isinstance(label_l, list) else label_l
        variant_pdf = raw.filter(pl.col("variant") == variant_val)
        reaction_data = {
            r: float(variant_pdf[r].to_numpy().mean()) for r in all_reaction_ids
        }

        builder = escher.Builder(map_json=map_json, reaction_data=reaction_data)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_label)
        out_path = os.path.join(outdir, f"escher_pathway_map_{safe_label}.html")
        builder.save_html(out_path)
        print(f"Saved escher pathway map ({variant_label}) to {out_path}")
