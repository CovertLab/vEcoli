from pint import Quantity
from typing import Any, Optional
from vivarium.library.units import units as vivarium_units


def remove_units(
    quantity_tree: dict[str, Any],
    expected_units_tree: Optional[dict[str, Any]] = None,
    path: tuple[str, ...] = tuple(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a tree of Quantities into units and magnitudes.

    Args:
        quantity_tree: The nested dictionary representing the tree of
            Quantity objects to split. This tree may also contain
            non-Quantity leaf nodes, which will be treated as unitless
            values.
        expected_units_tree: A nested dictionary of the same shape as
            ``quantity_tree`` specifying expected units for various leaf
            nodes. If a leaf is present in ``quantity_tree`` but not
            ``expected_units_tree``, no assertion will be made as to
            whether that node has units. A Quantity with an associated
            expected unit will be converted to that unit before being
            split.
        path: Path to the root of ``quantity_tree``. Used mainly for
            recursion.

    Returns:
        Two trees (as nested dictionaries), one with the magnitudes and
        another with the units. The units are from after any conversions
        imposed by ``expected_units_tree``.

    Raises:
        ValueError: If a leaf node in ``quantity_tree`` unexpectedly has
            no units.
    """
    expected_units_tree = expected_units_tree or {}
    converted_state = {}
    saved_units = {}
    for key, value in quantity_tree.items():
        if isinstance(value, dict):
            value_no_units, new_saved_units = remove_units(
                value, expected_units_tree.get(key), path + (key,)
            )
            converted_state[key] = value_no_units
            saved_units[key] = new_saved_units
        elif isinstance(value, Quantity):
            expected_units = expected_units_tree.get(key)
            if expected_units:
                value = value.to(expected_units)
            saved_units[key] = value.units
            value_no_units = value.magnitude
            converted_state[key] = value_no_units
        else:
            if expected_units_tree.get(key):
                path = path + (key,)
                raise ValueError(f"Units missing at {path}")
            converted_state[key] = value

    return converted_state, saved_units


def add_units(magnitudes, units, strict=True):
    """Combine a tree of magnitudes with a tree of units.

    Intended to be used as the inverse of ``remove_units()``.
    """
    combined = magnitudes.copy()
    for key, sub_units in units.items():
        if key not in combined and not strict:
            continue
        sub_magnitudes = combined[key]
        if isinstance(sub_units, dict):
            combined[key] = add_units(sub_magnitudes, sub_units)
        else:
            combined[key] = sub_magnitudes * sub_units
    return combined


def test_add_remove_units():
    original = {
        "a": {
            "b": 1 * vivarium_units.m,
            "c": 2 * vivarium_units.m,
        },
        "d": 3,
    }
    expected_units = {
        "a": {
            "b": vivarium_units.mm,
        }
    }
    magnitudes, units = remove_units(original, expected_units)

    expected_magnitudes = {
        "a": {
            "b": 1000,
            "c": 2,
        },
        "d": 3,
    }
    expected_units = {
        "a": {
            "b": vivarium_units.mm,
            "c": vivarium_units.m,
        }
    }
    assert magnitudes == expected_magnitudes
    assert units == expected_units

    combined = add_units(magnitudes, units)

    expected_combined = {
        "a": {
            "b": 1000 * vivarium_units.mm,
            "c": 2 * vivarium_units.m,
        },
        "d": 3,
    }
    assert combined == expected_combined
