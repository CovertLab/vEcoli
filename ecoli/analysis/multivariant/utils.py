"""
File containing utility functions for multivariant analysis.
"""

from __future__ import annotations

import json
from typing import Any


def create_variant_label(
    variant_id: int,
    per_variant_params: dict[int, Any],
) -> str | list[str]:
    """Return a human-readable label for a variant.

    Uses the first key/value pair from the variant's parameter dict when
    available (e.g. ``fraction_kinetic_target = 0.5``), marks the baseline
    variant explicitly, and falls back to ``'Variant {id}'`` otherwise.
    """
    params = per_variant_params.get(variant_id, {})
    if params == "baseline":
        return f"Variant {variant_id}: Baseline"
    if not params:
        return f"Variant {variant_id}"
    variant_name = list(params.keys())
    if len(variant_name) == 1:
        return f"Variant {variant_id}: {params.get(variant_name[0])}"
    else:
        label = [f"Variant {variant_id}"]
        for key in variant_name:
            value = params[key]
            label.append(f"{key}={value}")
        return label


def compute_variant_grid(
    per_variant_params: dict[int, Any],
) -> tuple[int, int, list[int]]:
    """Determine a (rows, columns, ordered_variant_ids) grid layout for
    variants produced by a two-parameter cross product (e.g. objective
    weights x fraction_kinetic_target — see
    ``ecoli/variants/metabolism_redux_classic_combined.py`` and
    ``runscripts/create_variants.py``'s ``parse_variants()``/
    ``apply_and_save_variants()``).

    Each non-baseline variant's parameter dict is expected to have exactly 2
    top-level keys. Grouping by the first key's value gives "rows"; grouping
    by the second key's value gives "columns". Distinct values are ordered by
    first appearance when iterating variant ids in ascending numeric order,
    reconstructing the original sweep order (e.g. weight-combo list order,
    fraction_kinetic_target list order) without assuming the values are
    sortable (a dict-valued param, like a weight combo, isn't).

    Returns:
        rows: number of distinct first-parameter values, plus 1 if a baseline
            variant (``params == "baseline"``) is present.
        columns: number of distinct second-parameter values.
        ordered_variant_ids: variant ids in row-major grid order — baseline
            first (its own row) if present, then grouped by the first
            parameter's value (each group internally ordered by the second
            parameter's value).

    Falls back to ``(len(variants), 1, sorted(variant_ids))`` — a single
    column, matching prior behavior — if variant params aren't a clean 2-key
    cross product (e.g. single-parameter variant sweeps).
    """
    baseline_ids = [
        vid for vid, params in per_variant_params.items() if params == "baseline"
    ]
    other_ids = sorted(
        vid for vid, params in per_variant_params.items() if params != "baseline"
    )

    def _fallback() -> tuple[int, int, list[int]]:
        all_ids = sorted(per_variant_params.keys())
        n_col = 3
        return len(all_ids) / n_col, n_col, all_ids

    if not other_ids:
        return _fallback()

    keys_per_variant = [list(per_variant_params[vid].keys()) for vid in other_ids]
    if any(len(keys) != 2 for keys in keys_per_variant) or any(
        keys != keys_per_variant[0] for keys in keys_per_variant
    ):
        return _fallback()

    key1, key2 = keys_per_variant[0]

    def _canonical(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return repr(value)

    row_order: list[str] = []
    col_order: list[str] = []
    cell_by_keys: dict[tuple[str, str], int] = {}
    for vid in other_ids:
        params = per_variant_params[vid]
        row_key = _canonical(params[key1])
        col_key = _canonical(params[key2])
        if row_key not in row_order:
            row_order.append(row_key)
        if col_key not in col_order:
            col_order.append(col_key)
        cell_by_keys[(row_key, col_key)] = vid

    ordered_variant_ids = list(baseline_ids)
    for row_key in row_order:
        for col_key in col_order:
            vid = cell_by_keys.get((row_key, col_key))
            if vid is not None:
                ordered_variant_ids.append(vid)

    rows = len(row_order) + (1 if baseline_ids else 0)
    columns = len(col_order)
    return rows, columns, ordered_variant_ids
