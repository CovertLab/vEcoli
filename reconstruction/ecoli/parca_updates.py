"""
Pure function infrastructure for the ParCa pipeline.

This module provides dataclasses and utilities for expressing ParCa stage
updates as pure functions that return update objects instead of mutating
sim_data and cell_specs in-place.

Benefits:
- Explicit data flow (each stage declares what it modifies)
- Better testability (pure functions can be tested in isolation)
- Cacheability (results can be memoized)
- Debuggability (updates can be inspected/logged)
- Gradual migration (one stage at a time)

Usage:
    from reconstruction.ecoli.parca_updates import (
        SimDataUpdate, ArrayUpdate, CellSpecsUpdate, StageResult,
        apply_sim_data_update, apply_cell_specs_update, apply_stage_result
    )

    def compute_input_adjustments(sim_data, cell_specs, **kwargs) -> StageResult:
        update = SimDataUpdate()
        update.attributes['tf_to_active_inactive_conditions'] = {...}
        update.arrays['process.translation.translation_efficiencies_by_monomer'] = ArrayUpdate(
            op='multiply', value=1.5, indices=[0, 1, 2]
        )
        return StageResult(sim_data_update=update, cell_specs_update=CellSpecsUpdate())

    # Apply the result
    sim_data, cell_specs = apply_stage_result(sim_data, cell_specs, result)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import copy

import numpy as np


@dataclass
class ArrayUpdate:
    """
    Describes an update to a numpy array without mutating the original.

    Attributes:
        op: The operation to perform:
            - 'set': Replace the entire array or specific indices with value
            - 'set_slice': Set a slice of the array (indices should be a slice or tuple)
            - 'multiply': Multiply the array or specific indices by value
            - 'add': Add value to the array or specific indices
            - 'divide': Divide the array or specific indices by value
            - 'normalize': Normalize the array (value is ignored, indices optional)
        value: The value to use in the operation
        indices: Optional indices/mask for partial updates. Can be:
            - None: Apply to entire array
            - int or list of ints: Specific indices
            - np.ndarray: Boolean mask or index array
            - slice: A slice object
            - tuple: For multi-dimensional indexing
        field: Optional field name for structured arrays (e.g., 'deg_rate')
    """
    op: str
    value: Any
    indices: Optional[Any] = None
    field: Optional[str] = None

    def __post_init__(self):
        valid_ops = {'set', 'set_slice', 'multiply', 'add', 'divide', 'normalize'}
        if self.op not in valid_ops:
            raise ValueError(f"Invalid op '{self.op}'. Must be one of {valid_ops}")


@dataclass
class SimDataUpdate:
    """
    Updates to sim_data using dot-notation paths.

    Paths use dot notation to specify nested attributes, e.g.:
    - 'tf_to_active_inactive_conditions' -> sim_data.tf_to_active_inactive_conditions
    - 'process.translation.translation_efficiencies_by_monomer' ->
        sim_data.process.translation.translation_efficiencies_by_monomer
    - 'mass.avg_cell_dry_mass_init' -> sim_data.mass.avg_cell_dry_mass_init

    Attributes:
        attributes: Dict mapping paths to new values (full replacement)
        arrays: Dict mapping paths to ArrayUpdate objects (partial updates)
        dicts: Dict mapping paths to dicts of updates to merge
        method_calls: List of (path, method_name, args, kwargs) for methods that need calling
    """
    attributes: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, ArrayUpdate] = field(default_factory=dict)
    dicts: Dict[str, Dict] = field(default_factory=dict)
    method_calls: List[Tuple[str, str, tuple, dict]] = field(default_factory=list)

    def merge(self, other: 'SimDataUpdate') -> 'SimDataUpdate':
        """Merge another SimDataUpdate into this one (other takes precedence)."""
        merged = SimDataUpdate(
            attributes={**self.attributes, **other.attributes},
            arrays={**self.arrays, **other.arrays},
            dicts={**self.dicts},
            method_calls=self.method_calls + other.method_calls,
        )
        # Merge dicts deeply
        for path, updates in other.dicts.items():
            if path in merged.dicts:
                merged.dicts[path] = {**merged.dicts[path], **updates}
            else:
                merged.dicts[path] = updates
        return merged


@dataclass
class CellSpecsUpdate:
    """
    Updates to the cell_specs dictionary.

    Attributes:
        conditions: Dict mapping condition keys to their updates.
            Each condition update is a dict of property -> value.
    """
    conditions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def merge(self, other: 'CellSpecsUpdate') -> 'CellSpecsUpdate':
        """Merge another CellSpecsUpdate into this one (other takes precedence)."""
        merged = CellSpecsUpdate(conditions={**self.conditions})
        for condition, updates in other.conditions.items():
            if condition in merged.conditions:
                merged.conditions[condition] = {**merged.conditions[condition], **updates}
            else:
                merged.conditions[condition] = updates
        return merged


@dataclass
class StageResult:
    """
    Result from a pure ParCa stage function.

    Contains both sim_data and cell_specs updates that should be applied
    after the stage completes.
    """
    sim_data_update: SimDataUpdate = field(default_factory=SimDataUpdate)
    cell_specs_update: CellSpecsUpdate = field(default_factory=CellSpecsUpdate)

    def merge(self, other: 'StageResult') -> 'StageResult':
        """Merge another StageResult into this one (other takes precedence)."""
        return StageResult(
            sim_data_update=self.sim_data_update.merge(other.sim_data_update),
            cell_specs_update=self.cell_specs_update.merge(other.cell_specs_update),
        )


def get_by_path(obj: Any, path: str) -> Any:
    """
    Get a nested attribute using dot-notation path.

    Args:
        obj: The object to traverse
        path: Dot-separated path (e.g., 'process.translation.monomer_data')

    Returns:
        The value at the specified path

    Raises:
        AttributeError: If any part of the path doesn't exist
    """
    if not path:
        return obj

    parts = path.split('.')
    current = obj
    for part in parts:
        current = getattr(current, part)
    return current


def set_by_path(obj: Any, path: str, value: Any) -> None:
    """
    Set a nested attribute using dot-notation path.

    Args:
        obj: The object to traverse and modify
        path: Dot-separated path (e.g., 'process.translation.monomer_data')
        value: The value to set

    Raises:
        AttributeError: If any part of the path (except last) doesn't exist
    """
    if not path:
        raise ValueError("Path cannot be empty")

    parts = path.split('.')
    if len(parts) == 1:
        setattr(obj, path, value)
    else:
        parent = get_by_path(obj, '.'.join(parts[:-1]))
        setattr(parent, parts[-1], value)


def apply_array_update(arr: np.ndarray, update: ArrayUpdate) -> np.ndarray:
    """
    Apply an ArrayUpdate to a numpy array, returning the modified array.

    This function modifies the array in-place for efficiency when copy=False
    is used in the caller.

    Args:
        arr: The numpy array to update
        update: The ArrayUpdate describing the modification

    Returns:
        The modified array (same reference as input)
    """
    # For structured array fields, use direct field assignment instead of slice assignment
    # This is necessary for UnitStructArray and similar wrappers
    is_field_update = update.field is not None

    if is_field_update:
        # For field updates on structured arrays, use direct field assignment
        if update.op == 'set' and update.indices is None:
            # Direct field assignment: arr['field'] = value
            arr[update.field] = update.value
            return arr

        # For other operations, get the target field
        if hasattr(arr, 'struct_array'):
            target = arr.struct_array[update.field]
        else:
            target = arr[update.field]
    else:
        target = arr

    # Apply the operation
    if update.op == 'set':
        if update.indices is None:
            target[:] = update.value
        else:
            target[update.indices] = update.value

    elif update.op == 'set_slice':
        if update.indices is None:
            target[:] = update.value
        else:
            target[update.indices] = update.value

    elif update.op == 'multiply':
        if update.indices is None:
            target *= update.value
        else:
            target[update.indices] *= update.value

    elif update.op == 'add':
        if update.indices is None:
            target += update.value
        else:
            target[update.indices] += update.value

    elif update.op == 'divide':
        if update.indices is None:
            target /= update.value
        else:
            target[update.indices] /= update.value

    elif update.op == 'normalize':
        if update.indices is None:
            total = target.sum()
            if total > 0:
                target /= total
        else:
            total = target[update.indices].sum()
            if total > 0:
                target[update.indices] /= total

    return arr


def extract_path(key: str) -> str:
    """
    Extract the actual path from a key that may have a unique identifier suffix.

    Keys can have the format 'path.to.attribute:unique_id' where the part after
    the colon is just for making the key unique in the dictionary. This function
    returns only the path part.

    Args:
        key: The dictionary key, possibly with a ':suffix' for uniqueness

    Returns:
        The actual path without any unique identifier suffix
    """
    if ':' in key:
        return key.split(':')[0]
    return key


def apply_sim_data_update(
    sim_data: Any,
    update: SimDataUpdate,
    copy_data: bool = False
) -> Any:
    """
    Apply a SimDataUpdate to sim_data.

    Args:
        sim_data: The SimulationDataEcoli object to update
        update: The SimDataUpdate describing the modifications
        copy_data: If True, deep copy sim_data before modifying. If False, modify in-place.

    Returns:
        The modified sim_data (same reference if copy_data=False, new object if copy_data=True)
    """
    if copy_data:
        import copy as copy_module
        sim_data = copy_module.deepcopy(sim_data)

    # Special case: replace the entire object (used by initialize)
    if '_replace_entire_object' in update.attributes:
        return update.attributes['_replace_entire_object']

    # Apply attribute replacements
    for key, value in update.attributes.items():
        path = extract_path(key)
        set_by_path(sim_data, path, value)

    # Apply array updates
    for key, array_update in update.arrays.items():
        path = extract_path(key)
        arr = get_by_path(sim_data, path)
        apply_array_update(arr, array_update)

    # Apply dict merges
    for key, dict_updates in update.dicts.items():
        path = extract_path(key)
        target_dict = get_by_path(sim_data, path)
        target_dict.update(dict_updates)

    # Apply method calls (for cases where we need to call a method)
    for path, method_name, args, kwargs in update.method_calls:
        obj = get_by_path(sim_data, path) if path else sim_data
        method = getattr(obj, method_name)
        method(*args, **kwargs)

    return sim_data


def apply_cell_specs_update(
    cell_specs: Dict,
    update: CellSpecsUpdate,
    copy_data: bool = False
) -> Dict:
    """
    Apply a CellSpecsUpdate to cell_specs.

    Args:
        cell_specs: The cell specifications dictionary to update
        update: The CellSpecsUpdate describing the modifications
        copy_data: If True, deep copy cell_specs before modifying. If False, modify in-place.

    Returns:
        The modified cell_specs (same reference if copy_data=False, new object if copy_data=True)
    """
    if copy_data:
        import copy as copy_module
        cell_specs = copy_module.deepcopy(cell_specs)

    for condition, condition_updates in update.conditions.items():
        if condition not in cell_specs:
            cell_specs[condition] = {}
        cell_specs[condition].update(condition_updates)

    return cell_specs


def apply_stage_result(
    sim_data: Any,
    cell_specs: Dict,
    result: StageResult,
    copy_data: bool = False
) -> Tuple[Any, Dict]:
    """
    Apply a StageResult to both sim_data and cell_specs.

    Args:
        sim_data: The SimulationDataEcoli object to update
        cell_specs: The cell specifications dictionary to update
        result: The StageResult containing both updates
        copy_data: If True, deep copy both objects before modifying

    Returns:
        Tuple of (modified sim_data, modified cell_specs)
    """
    sim_data = apply_sim_data_update(sim_data, result.sim_data_update, copy_data=copy_data)
    cell_specs = apply_cell_specs_update(cell_specs, result.cell_specs_update, copy_data=copy_data)
    return sim_data, cell_specs


def pure_stage_wrapper(pure_func):
    """
    Decorator to wrap a pure stage function for backward compatibility.

    The pure function should have signature:
        (sim_data, cell_specs, **kwargs) -> StageResult

    The wrapped function will have signature:
        (sim_data, cell_specs, **kwargs) -> (sim_data, cell_specs)

    This allows gradual migration: define the pure function, wrap it,
    and the wrapped version can be used as a drop-in replacement.

    Usage:
        def compute_input_adjustments(sim_data, cell_specs, **kwargs) -> StageResult:
            # Pure implementation
            return StageResult(...)

        @pure_stage_wrapper
        def input_adjustments_pure(sim_data, cell_specs, **kwargs):
            return compute_input_adjustments(sim_data, cell_specs, **kwargs)
    """
    def wrapper(sim_data, cell_specs, **kwargs):
        result = pure_func(sim_data, cell_specs, **kwargs)
        apply_stage_result(sim_data, cell_specs, result, copy_data=False)
        return sim_data, cell_specs

    wrapper.__name__ = pure_func.__name__
    wrapper.__doc__ = pure_func.__doc__
    return wrapper


# Helper functions for building common update patterns

def make_attribute_update(path: str, value: Any) -> SimDataUpdate:
    """Create a SimDataUpdate that sets a single attribute."""
    return SimDataUpdate(attributes={path: value})


def make_array_multiply_update(
    path: str,
    value: Any,
    indices: Optional[Any] = None,
    field: Optional[str] = None
) -> SimDataUpdate:
    """Create a SimDataUpdate that multiplies an array or array slice."""
    return SimDataUpdate(
        arrays={path: ArrayUpdate(op='multiply', value=value, indices=indices, field=field)}
    )


def make_array_set_update(
    path: str,
    value: Any,
    indices: Optional[Any] = None,
    field: Optional[str] = None
) -> SimDataUpdate:
    """Create a SimDataUpdate that sets an array or array slice."""
    return SimDataUpdate(
        arrays={path: ArrayUpdate(op='set', value=value, indices=indices, field=field)}
    )


def make_dict_update(path: str, updates: Dict) -> SimDataUpdate:
    """Create a SimDataUpdate that merges updates into a dict."""
    return SimDataUpdate(dicts={path: updates})


def collect_updates(*updates: SimDataUpdate) -> SimDataUpdate:
    """Merge multiple SimDataUpdate objects into one."""
    result = SimDataUpdate()
    for update in updates:
        result = result.merge(update)
    return result
