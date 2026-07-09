"""
File containing utility functions for multivariant analysis.
"""

from __future__ import annotations

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
