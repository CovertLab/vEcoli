"""
======================
E. coli Bigraph Types
======================

Custom bigraph-schema types for the E. coli whole-cell model.  These
allow ``core.infer()`` and ``translate_ports()`` to produce proper
typed schemas from the vivarium ``ports_schema()`` format.

Types provided:

- **Method** — serializable callables (updaters, dividers)
- **UnumUnits** — Unum unit objects
- **Quantity** — pint Quantity (value + units)
- **CSRMatrix** — scipy sparse matrices (stoichiometry)
- **UnitsArray** — wholecell UnitStructArray (structured arrays with units)
- **StepInstance** / **ProcessInstance** — vivarium process instance schemas

Also provides ``translate_ports()`` which converts vivarium
``ports_schema()`` dicts into bigraph-schema type trees.
"""

import os
import typing

import numpy as np
import pint
from plum import dispatch
from dataclasses import dataclass, field
from scipy.sparse._csr import csr_matrix
from unum import Unum

from bigraph_schema.schema import (
    Node,
    Float,
    Integer,
    Array,
    List,
    Tuple,
    Map,
    Overwrite,
    Wrap,
    Function,
    Quantity,
)
from bigraph_schema.methods import (
    infer,
    set_default,
    default,
    serialize,
    realize,
    render,
    wrap_default,
    resolve,
    reify_schema,
    validate,
    apply,
    reconcile,
    divide,
    bundle,
    BundleContext,
)
from bigraph_schema.methods.handle_parameters import align_parameters
from bigraph_schema import capture_object_state, restore_object_value
from bigraph_schema.methods.bundle import register_derived_function_resolver

from wholecell.utils.unit_struct_array import UnitStructArray


# Function type now lives in bigraph_schema. The infer dispatch below
# routes vEcoli-specific callables (bound methods → Method, exec'd
# lambdas → DerivedFunction) before falling back to Function/Object.


@dispatch
def infer(core, value: typing.Callable, path: tuple = ()):
    if hasattr(value, "__self__") and hasattr(value, "__func__"):
        # Bound method → Method type (references an instance)
        return set_default(Method(), value), []
    elif (
        hasattr(value, "__code__")
        and getattr(value.__code__, "co_filename", "") == "<string>"
    ):
        # Dynamic function created via exec (e.g. from build_ode) —
        # can't be imported by name, must be rebuilt from sympy source.
        return set_default(DerivedFunction(), value), []
    elif hasattr(value, "__name__") and hasattr(value, "__module__"):
        # Standalone function → Function type (in bigraph_schema)
        data = {
            "module": value.__module__,
            "instance": None,
            "attribute": value.__name__,
        }
        return set_default(Function(**data), value), []
    else:
        # Callable object (e.g. CubicSpline, functors) → Object type
        from bigraph_schema.schema import Object

        cls = type(value)
        class_path = f"{cls.__module__}.{cls.__name__}"
        schema = Object(_class=class_path)
        return set_default(schema, value), []


# ============================================================================
# UnumUnits type — Unum unit objects
# ============================================================================


def unum_dimension(value):
    dimension = {}
    for unit, scale in value._unit.items():
        entry = value._unitTable[unit]
        base_unit = {unit: scale}
        if entry[0]:
            dimension_unit = entry[0]._unit
            base_key = list(dimension_unit.keys())[0]
            base_unit = {base_key: scale}
        dimension.update(base_unit)
    return dimension


@dataclass(kw_only=True)
class UnumUnits(Node):
    """Parameterized Unum type: ``unum[<magnitude_type>,<unit_string>]``.

    Examples::

        unum                          # scalar float magnitude, no unit annotation
        unum[float,fg]                # scalar float, femtograms
        unum[array[16321|9,float],g/mol]  # 2D array magnitude

    The magnitude field carries the schema for the numeric payload
    (Float, Array, etc.). serialize/realize dispatch on it so arrays
    and scalars are handled without branches.
    """

    _schema_keys = Node._schema_keys | frozenset({"_units"})
    _dimension: typing.Dict = field(default_factory=dict)
    units: typing.Dict = field(default_factory=dict)
    magnitude: Node = field(default_factory=lambda: Float())
    _units: str = ""


@dispatch
def infer(core, value: Unum, path: tuple = ()):
    dimension = unum_dimension(value)
    magnitude_schema, _ = infer(core, value.asNumber(), path + (value.strUnit(),))
    schema = UnumUnits(
        _dimension=dimension, units=value._unit, magnitude=magnitude_schema
    )
    return set_default(schema, value), []


@dispatch
def default(schema: UnumUnits):
    if schema._default:
        return schema._default
    return Unum(schema.units, default(schema.magnitude))


@dispatch
def apply(schema: UnumUnits, state, update, path):
    """Unum quantities use overwrite semantics, but only for actual
    Unum updates — not for bare numeric defaults (0, 0.0)."""
    if update is None:
        return state, []
    if isinstance(update, (int, float)) and update == 0 and state is not None:
        return state, []
    return update, []


@dispatch
def serialize(schema: UnumUnits, state):
    if isinstance(state, dict):
        return state
    if state is None:
        return None
    return {
        "units": state._unit,
        "magnitude": serialize(schema.magnitude, state.asNumber()),
    }


# Pint registry for dimensionality lookups. Cached at module load so
# divide doesn't pay the (small but non-trivial) cost on every call.
_UREG = pint.UnitRegistry()


def _extensive_scale_factor(unit_dict):
    """Net exponent of size-scaling base dimensions in a unit.

    An extensive quantity (mass, amount, total volume) scales linearly
    with system size — a cell splits its mass in half, its molecule
    count in half, its volume in half. Intensive quantities
    (concentration, rate, time) do not scale.

    We detect by summing (mass_exp + substance_exp + length_exp/3)
    across the unit's pint dimensionality. Volume shows up as
    ``[length]^3`` in pint, so dividing length_exp by 3 gives
    "volume exponent".

    Returns a float: >0 extensive, 0 or <0 intensive. Returns 0 for
    empty/None unit dicts (dimensionless → intensive)."""
    if not unit_dict:
        return 0.0
    dim = {}
    for symbol, exp in unit_dict.items():
        try:
            unit_dim = _UREG.parse_units(symbol).dimensionality
        except Exception:
            continue
        for k, v in unit_dim.items():
            dim[k] = dim.get(k, 0.0) + float(v) * float(exp)
    mass = dim.get("[mass]", 0.0)
    substance = dim.get("[substance]", 0.0)
    length = dim.get("[length]", 0.0)
    return mass + substance + length / 3.0


@dispatch
def divide(schema: UnumUnits, state, context=None, path=(), rng=None):
    """Extensive quantities halve; intensive quantities share.

    A ``Unum`` carries its unit in ``state._unit``. See
    ``_extensive_scale_factor``: ``mol``, ``g``, ``L`` halve;
    ``mol/L``, ``s``, ``1/s`` share."""
    if state is None:
        return None, None
    if not isinstance(state, Unum):
        return state, state
    scale = _extensive_scale_factor(state._unit)
    if scale > 0:
        half = state / 2
        return half, half
    return state, state


@dispatch
def resolve(schema: UnumUnits, update: UnumUnits, path=()):
    return schema


@dispatch
def resolve(schema: UnumUnits, update: Float, path=()):
    """UnumUnits is more specific than Float — keep Unum."""
    return schema


@dispatch
def resolve(schema: Float, update: UnumUnits, path=()):
    """UnumUnits is more specific than Float — use Unum."""
    return update


@dispatch
def realize(core, schema: UnumUnits, encode, path=()):
    if isinstance(encode, Unum):
        return schema, encode, []
    if isinstance(encode, (int, float)):
        return schema, Unum(schema.units, encode), []
    if isinstance(encode, dict) and "units" in encode:
        # Dict form from serialize: {'units': {...}, 'magnitude': ...}
        _, magnitude, _ = realize(core, schema.magnitude, encode["magnitude"], path)
        return schema, Unum(encode["units"], magnitude), []
    return schema, encode, []


@dispatch
def render(schema: UnumUnits, defaults=False):
    mag_render = render(schema.magnitude)
    if schema._units:
        result = f"unum[{mag_render},{schema._units}]"
    elif mag_render != "float":
        result = f"unum[{mag_render}]"
    else:
        result = "unum"
    return wrap_default(schema, result) if defaults else result


@dispatch
def align_parameters(schema: UnumUnits, parameters):
    """Handle unum[magnitude_type,unit_string] or unum[unit_string].

    If a single parameter is a Node (schema type), it's the magnitude type.
    If it's a string, it's the unit annotation.
    Two parameters: first is magnitude, second is units.
    """
    if len(parameters) == 2:
        return {"magnitude": parameters[0], "_units": parameters[1]}
    if len(parameters) == 1:
        p = parameters[0]
        if isinstance(p, Node):
            return {"magnitude": p}
        return {"_units": p}
    return {}


@dispatch
def reify_schema(core, schema: UnumUnits, parameters):
    """Reify magnitude type and unit string from parameters."""
    if "magnitude" in parameters:
        mag_param = parameters["magnitude"]
        if isinstance(mag_param, str):
            schema.magnitude = core.access(mag_param)
        elif isinstance(mag_param, Node):
            schema.magnitude = mag_param
    if "_units" in parameters:
        units_param = parameters["_units"]
        if isinstance(units_param, str):
            schema._units = units_param
    return schema


# ============================================================================
# Quantity type — moved to bigraph_schema. Configure to use vivarium's
# shared pint registry so vEcoli Quantities created via vivarium and
# via the framework interoperate.
# ============================================================================

from vivarium.library.units import units as ureg
from bigraph_schema.units import set_quantity_registry

set_quantity_registry(ureg)


def units_dict(value):
    return {key: subvalue for key, subvalue in value.unit_items()}


@dispatch
def infer(core, value: pint.Quantity, path: tuple = ()):
    units = units_dict(value)
    magnitude, _ = infer(core, value.magnitude, path + ("magnitude",))
    schema = Quantity(units=units, magnitude=magnitude)
    return set_default(schema, value), []


@dispatch
def default(schema: Quantity):
    if schema._default:
        return schema._default
    return {"units": schema.units, "magnitude": default(schema.magnitude)}


@dispatch
def resolve(schema: Integer, update: Array, path=()):
    return update


@dispatch
def resolve(schema: Quantity, update: Quantity, path=()):
    if not schema.units or schema.units == update.units:
        return update
    return schema


@dispatch
def resolve(schema: Quantity, update: Integer, path=()):
    return schema


@dispatch
def resolve(schema: Tuple, update: List, path=()):
    return schema


@dispatch
def apply(schema: Quantity, state, update, path):
    """Quantities use overwrite semantics, but only for actual
    Quantity updates — not for bare numeric defaults (0, 0.0)
    which would erase a valid Quantity."""
    if update is None:
        return state, []
    if isinstance(update, (int, float)) and update == 0 and state is not None:
        return state, []
    return update, []


# ============================================================================
# CSRMatrix type — scipy sparse matrices
# ============================================================================


@dataclass(kw_only=True)
class CSRMatrix(Node):
    _shape: typing.Tuple[int] = field(default_factory=tuple)
    _data: np.dtype = field(default_factory=lambda: np.dtype("float64"))
    data: Array = field(default_factory=Array)
    indices: Array = field(default_factory=Array)
    pointers: Array = field(default_factory=Array)


@dispatch
def infer(core, value: csr_matrix, path: tuple = ()):
    data = {
        "_shape": value.shape,
        "_data": infer(core, value.dtype, path=path + ("_data",))[0],
        "data": infer(core, value.data, path=path + ("data",))[0],
        "indices": infer(core, value.indices, path=path + ("indices",))[0],
        "pointers": infer(core, value.indptr, path=path + ("pointers",))[0],
    }
    schema = CSRMatrix(**data)
    return set_default(schema, value), []


@dispatch
def serialize(schema: CSRMatrix, state):
    if isinstance(state, dict):
        return state
    if state is None:
        return None
    if isinstance(state, csr_matrix):
        return {
            "data": np.asarray(state.data).tolist(),
            "indices": np.asarray(state.indices).tolist(),
            "pointers": np.asarray(state.indptr).tolist(),
        }
    # Already a plain array or other form — convert to list
    if isinstance(state, np.ndarray):
        return state.tolist()
    return state


@dispatch
def realize(core, schema: CSRMatrix, encode, path=()):
    if isinstance(encode, csr_matrix):
        return schema, encode, []
    # Dense ndarray — the process stores it as dense, convert to CSR
    if isinstance(encode, np.ndarray):
        return (
            schema,
            csr_matrix(encode, shape=tuple(schema._shape) if schema._shape else None),
            [],
        )
    # List of lists — dense matrix from JSON
    if isinstance(encode, list):
        return schema, csr_matrix(np.array(encode)), []
    # Dict form {data, indices, pointers} — schema tells us the shape
    if isinstance(encode, dict) and "data" in encode:
        inner = tuple(
            realize(core, getattr(schema, key), encode[key], path + (key,))[1]
            for key in ["data", "indices", "pointers"]
        )
        return schema, csr_matrix(inner, shape=tuple(schema._shape)), []
    return schema, encode, []


@dispatch
def align_parameters(schema: CSRMatrix, parameters):
    """csr_matrix[rows|cols,dtype] — parse shape and dtype parameters."""
    result = {}
    # First parameter: shape string like '4538|3277' or a Tuple from parsing
    if len(parameters) >= 1:
        shape_param = parameters[0]
        if isinstance(shape_param, Tuple):
            result["_shape"] = tuple(int(v) for v in shape_param._values)
        elif isinstance(shape_param, (tuple, list)):
            result["_shape"] = tuple(int(v) for v in shape_param)
    if len(parameters) >= 2:
        result["_data"] = parameters[1]
    return result


@dispatch
def reify_schema(core, schema: CSRMatrix, parameters):
    if "_shape" in parameters:
        schema._shape = parameters["_shape"]
    if "_data" in parameters:
        data_param = parameters["_data"]
        from bigraph_schema.schema import schema_dtype

        if isinstance(data_param, str):
            resolved = core.access(data_param)
            schema._data = (
                schema_dtype(resolved)
                if hasattr(resolved, "_default")
                else np.dtype(data_param)
            )
        elif isinstance(data_param, Node):
            schema._data = schema_dtype(data_param)
        # Propagate the dtype to the data sub-array so realize produces
        # correct types for data values. scipy's csr_matrix uses int32
        # for indices and indptr (unless the matrix is >2B elements),
        # so declare that explicitly rather than int64.
        from bigraph_schema.schema import Array

        schema.data = Array(_shape=(), _data=schema._data)
        schema.indices = Array(_shape=(), _data=np.dtype("int32"))
        schema.pointers = Array(_shape=(), _data=np.dtype("int32"))
    # Legacy path: explicit key-value pairs like csr_matrix[data:array[float],...]
    for key, parameter in parameters.items():
        if key in ("_shape", "_data"):
            continue
        subkey = core.access(parameter)
        setattr(schema, key, subkey)
    return schema


@dispatch
def render(schema: CSRMatrix, defaults=False):
    data = {
        "_type": "csr_matrix",
        "_shape": schema._shape,
        "_data": render(schema._data),
        "data": render(schema.data),
        "indices": render(schema.indices),
        "pointers": render(schema.pointers),
    }
    return wrap_default(schema, data) if defaults else data


@dispatch
def validate(core, schema: CSRMatrix, state):
    return


# ============================================================================
# UnitsArray type — wholecell UnitStructArray
# ============================================================================


@dataclass(kw_only=True)
class UnitsArray(Node):
    struct: Array = field(default_factory=Array)
    units: Map = field(default_factory=lambda: Map(_value=UnumUnits()))


@dispatch
def infer(core, value: UnitStructArray, path: tuple = ()):
    data = {
        "struct": infer(core, value.struct_array, path=path + ("struct",))[0],
        "units": infer(core, value.units, path=path + ("units",))[0],
    }
    schema = UnitsArray(**data)
    return set_default(schema, value), []


@dispatch
def serialize(schema: UnitsArray, state):
    if isinstance(state, dict):
        return state
    # Use the dtype-preserving structured-array serializer so realize
    # can rebuild with the correct structured dtype instead of falling
    # back to ``schema.struct._data`` (which is ``float64`` by default
    # and can't absorb strings/bools/sub-arrays in the real rows).
    return {
        "struct": _serialize_structured_array(state.struct_array),
        "units": serialize(schema.units, state.units),
    }


@dispatch
def realize(core, schema: UnitsArray, encode, path=()):
    if isinstance(encode, UnitStructArray):
        return schema, encode, []
    if isinstance(encode, dict) and "struct" in encode:
        struct = _realize_structured_array(encode["struct"])
        _, units, _ = realize(core, schema.units, encode["units"], path + ("units",))
        return schema, UnitStructArray(struct, units), []
    return schema, encode, []


@dispatch
def render(schema: UnitsArray, defaults=False):
    result = "units_array"
    return wrap_default(schema, result) if defaults else result


# StepInstance/ProcessInstance/FunctionInstance and their infer
# dispatches were never exercised in the v2 build path — the composite
# document declares processes as ``StepLink``/``ProcessLink`` directly,
# never via ``infer()`` on a live vivarium instance. Removed.


# ============================================================================
# translate_ports — convert v1 ports_schema to bigraph schema
# ============================================================================


def translate_ports(core, ports, path=()):
    """Convert a vivarium ports_schema dict into a bigraph-schema type tree.

    Args:
        core: bigraph-schema Core instance (used for type inference).
        ports: Dict from ports_schema() with _default, _updater, etc.
        path: Current path for recursive descent.

    Returns:
        A bigraph-schema type tree (Node, Overwrite, dict of sub-schemas).
    """
    if isinstance(ports, dict):
        if not ports:
            return Node()

        if "_default" in ports:
            state = ports["_default"]
            if isinstance(state, tuple) and state == ():
                state = []
            # An empty collection (set()/{}/[]/None) carries no element-type
            # information, so core.infer() guesses a concrete Set/Map/List that
            # spuriously conflicts at a SHARED store with a hand-typed
            # (already-migrated) process's interface() — e.g. Set[Node] vs
            # Map[String,Float]. Emit the permissive Node (top type) instead;
            # it resolves to whatever concrete type the typed declaration uses.
            if state is None or (
                isinstance(state, (set, dict, list)) and len(state) == 0
            ):
                schema = Node()
            else:
                schema = core.infer(state)

            if "_updater" in ports and ports["_updater"] == "set":
                schema = Overwrite(_value=schema)

            schema._default = state
            return schema

        elif "_updater" in ports:
            schema = Node()
            if ports["_updater"] == "set":
                schema = Overwrite(_value=schema)
            return schema

        else:
            child_keys = [k for k in ports if not k.startswith("_")]
            # vivarium glob schema: {'*': subschema} declares an open-ended map
            # (any number of dynamically-keyed children of one type), e.g.
            # global_clock's next_update_time = {'*': {}}. Without this it would
            # become a store with a literal '*' child, so the process reads back
            # {'*': ...} instead of {key: value} and its arithmetic explodes.
            if child_keys == ["*"]:
                return {"_type": "map", "_value": translate_ports(core, ports["*"])}
            result = {}
            for key in child_keys:
                result[key] = translate_ports(core, ports[key])
            return result

    return Node()


# ============================================================================
# BulkArray — structured array with sparse count updates
# ============================================================================


@dataclass(kw_only=True)
class BulkArray(Array):
    """Structured numpy array where sparse [(index, delta)] updates
    target the 'count' field specifically."""

    pass


def _serialize_structured_array(state):
    """Serialize a structured numpy array to a JSON-safe dict."""
    if not isinstance(state, np.ndarray) or not state.dtype.names:
        return state
    rows = []
    for record in state:
        row = []
        for name in state.dtype.names:
            val = record[name]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif hasattr(val, "item"):
                val = val.item()
            row.append(val)
        rows.append(row)
    return {
        "__structured_array__": True,
        "dtype": str(state.dtype),
        "data": rows,
    }


def _realize_structured_array(state):
    """Realize a structured array from its serialized dict form."""
    import ast

    if isinstance(state, np.ndarray):
        return state
    if isinstance(state, dict) and state.get("__structured_array__"):
        dtype = np.dtype(ast.literal_eval(state["dtype"]))
        return np.array([tuple(r) for r in state["data"]], dtype=dtype)
    return state


@dispatch
def serialize(schema: BulkArray, state):
    return _serialize_structured_array(state)


@dispatch
def render(schema: BulkArray, defaults=False):
    result = "bulk_array"
    return wrap_default(schema, result) if defaults else result


@realize.dispatch
def realize(core, schema: BulkArray, state, path=()):
    if isinstance(state, np.ndarray):
        return schema, state, []
    state = _realize_structured_array(state)
    return schema, state, []


@dispatch
def apply(schema: BulkArray, state, update, path):
    """Apply sparse index updates to the ``count`` field of a bulk array.

    The hot path: ``update`` is a list of ``(index_array, delta_array)``
    tuples produced by partition's allocate/return cycle. Anything
    else falls through to the standard Array dispatch.
    """
    if isinstance(update, list):
        for idx, delta in update:
            state["count"][idx] += delta
        return state, []
    return apply(Array(_shape=schema._shape, _data=schema._data), state, update, path)


@dispatch
def divide(schema: BulkArray, state, context=None, path=(), rng=None):
    """Binomial split of bulk molecule counts. Delegates to v1's
    `divide_bulk` so v1 and v2 use bit-identical logic.

    v1's divide_bulk freezes the daughter arrays read-only as a
    defensive measure for vivarium's tree. v2 mutates state in place
    via apply(BulkArray) — we need them writeable.
    """
    if state is None:
        return None, None
    from ecoli.library.schema import divide_bulk

    a, b = divide_bulk(state)
    a = a.copy() if not a.flags.writeable else a
    b = b.copy() if not b.flags.writeable else b
    a.flags.writeable = True
    b.flags.writeable = True
    return a, b


# ============================================================================
# UniqueArray — structured array for unique molecules (set/add/delete ops)
# ============================================================================


@dataclass(kw_only=True)
class UniqueArray(Array):
    """Structured numpy array for unique molecule populations.

    Updates use dict format with 'set', 'add', and 'delete' keys:
      - set: list of {col: values} dicts — overwrite active rows
      - add: list of {col: values} dicts — activate inactive rows
      - delete: list of index arrays — deactivate rows

    The reconciler batches operations from multiple steps and applies
    them in order: set → add → delete.
    """

    pass


@dispatch
def serialize(schema: UniqueArray, state):
    return _serialize_structured_array(state)


@dispatch
def render(schema: UniqueArray, defaults=False):
    result = "unique_array"
    return wrap_default(schema, result) if defaults else result


@realize.dispatch
def realize(core, schema: UniqueArray, state, path=()):
    from ecoli.library.schema import MetadataArray

    if isinstance(state, MetadataArray):
        return schema, state, []
    if isinstance(state, np.ndarray):
        next_idx = int(state["unique_index"].max()) + 1 if len(state) > 0 else 0
        return schema, MetadataArray(state, next_idx), []
    state = _realize_structured_array(state)
    if isinstance(state, np.ndarray):
        next_idx = int(state["unique_index"].max()) + 1 if len(state) > 0 else 0
        state = MetadataArray(state, next_idx)
    return schema, state, []


def _get_free_indices(array, n_new):
    """Find inactive slots in a unique molecule array, extending if needed."""
    from ecoli.library.schema import get_free_indices

    return get_free_indices(array, n_new)


@dispatch
def reconcile(schema: UniqueArray, updates: list):
    """Batch set/add/delete operations across all updates."""
    sets = []
    adds = []
    deletes = []

    for update in updates:
        if update is None or not isinstance(update, dict):
            continue
        if "set" in update:
            val = update["set"]
            if isinstance(val, list):
                sets.extend(val)
            elif isinstance(val, dict):
                sets.append(val)
        if "add" in update:
            val = update["add"]
            if isinstance(val, list):
                adds.extend(val)
            elif isinstance(val, dict):
                adds.append(val)
        if "delete" in update:
            val = update["delete"]
            if isinstance(val, list):
                if len(val) > 0:
                    if isinstance(val[0], (list, np.ndarray)):
                        deletes.extend(val)
                    elif isinstance(val[0], (int, np.integer)):
                        deletes.append(val)
            elif isinstance(val, np.ndarray):
                deletes.append(val)

    result = {}
    if sets:
        result["set"] = sets
    if adds:
        result["add"] = adds
    if deletes:
        result["delete"] = deletes
    return result if result else None


def _as_op_list(value, leaf_check=None):
    """Normalize a unique-molecule operation value into a list.

    A single operation can be passed as either a dict (one op) or a
    list of dicts (batched ops). Reconcile produces lists; the
    reconcile fast-path passes single updates through unchanged, so
    apply must accept both forms. ``leaf_check`` (used for delete ops)
    decides whether a list is itself a single op or a batch — for
    deletes, a list of ints/np.integer IS the single op, while a list
    whose first element is a list/array is a batch.
    """
    if value is None:
        return ()
    if isinstance(value, dict):
        return (value,)
    if isinstance(value, list):
        if leaf_check is not None and len(value) > 0 and leaf_check(value[0]):
            return (value,)
        return tuple(value)
    if isinstance(value, np.ndarray):
        return (value,)
    return ()


def _is_index_leaf(x):
    return isinstance(x, (int, np.integer))


# Cache of dtype → 1-element zero record. ``UniqueArray.apply``'s
# delete branch was allocating a fresh ``np.zeros(1, dtype=result.dtype)``
# on every delete op; caching it once per dtype lets the deactivate
# write reuse the same buffer.
_ZERO_RECORD_CACHE: typing.Dict[typing.Any, np.ndarray] = {}


def _zero_record_for(dtype):
    cached = _ZERO_RECORD_CACHE.get(dtype)
    if cached is None:
        cached = np.zeros(1, dtype=dtype)
        _ZERO_RECORD_CACHE[dtype] = cached
    return cached


@dispatch
def apply(schema: UniqueArray, state, update, path):
    """Apply batched unique molecule operations: set → add → delete.

    Tolerates both reconciled (lists of ops) and raw (single op) forms
    so the top-level reconcile fast-path can pass single updates through.
    """
    if update is None or not isinstance(update, dict) or len(update) == 0:
        return state, []

    if not state.flags.owndata:
        result = state.copy()
    else:
        result = state
    result.flags.writeable = True

    # Pull the three op buckets up-front. Most calls only carry one
    # of {set, add, delete}; gating the work blocks below on the
    # explicit ``is None`` check skips the extra ``_as_op_list``
    # call + iterator setup for the missing branches.
    set_payload = update.get("set")
    add_payload = update.get("add")
    delete_payload = update.get("delete")

    # ``initially_active_idx`` MUST be captured before any ops run —
    # delete indices are relative to the pre-update active rows, not
    # post-add. Same for set's ``active_mask``, since add wires
    # newly-activated rows that set ops shouldn't touch.
    active_mask = None
    initially_active_idx = None
    if set_payload is not None or delete_payload is not None:
        active_mask = result["_entryState"].view(np.bool_)
    if delete_payload is not None:
        initially_active_idx = np.nonzero(active_mask)[0]

    # 1. Set operations: overwrite columns for active rows
    if set_payload is not None:
        for set_update in _as_op_list(set_payload):
            for col, col_values in set_update.items():
                result[col][active_mask] = col_values

    # 2. Add operations: activate inactive rows with new data
    if add_payload is not None:
        for add_update in _as_op_list(add_payload):
            n_new = len(next(iter(add_update.values())))
            result, free_indices = _get_free_indices(result, n_new)
            if "unique_index" not in add_update:
                result["unique_index"][free_indices] = (
                    np.arange(n_new) + result.metadata
                )
                result.metadata += n_new
            for col, col_values in add_update.items():
                result[col][free_indices] = col_values
            result["_entryState"][free_indices] = 1

    # 3. Delete operations: deactivate rows.
    if delete_payload is not None:
        zero_rec = _zero_record_for(result.dtype)
        for delete_indices in _as_op_list(delete_payload, leaf_check=_is_index_leaf):
            rows_to_delete = initially_active_idx[delete_indices]
            result[rows_to_delete] = zero_rec

    result.flags.writeable = False
    return result, []


# Map v1 divider name → divider function. v1's divider_registry is the
# source of truth at runtime, but importing the functions directly avoids
# a registry round-trip and lets divide() be self-contained.
def _get_unique_divider_fn(divider_name):
    from ecoli.library.schema import (
        divide_by_domain,
        divide_RNAs_by_domain,
        divide_ribosomes_by_RNA,
        empty_dict_divider,
        divide_set_none,
        divide_binomial,
    )

    return {
        "by_domain": divide_by_domain,
        "rna_by_domain": divide_RNAs_by_domain,
        "ribosome_by_RNA": divide_ribosomes_by_RNA,
        "empty_dict": empty_dict_divider,
        "set_none": divide_set_none,
        "binomial_ecoli": divide_binomial,
    }.get(divider_name)


def _resolve_topology_path(context, base_path, rel_path):
    """Resolve a vivarium-style relative path (with leading '..' segments)
    against a context dict, starting from base_path. Returns the value at
    the resolved path or None if any segment is missing.
    """
    # Walk base_path up for each '..' in rel_path
    abs_path = list(base_path)
    for seg in rel_path:
        if seg == "..":
            if abs_path:
                abs_path.pop()
        else:
            abs_path.append(seg)
    cur = context
    for seg in abs_path:
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


@dispatch
def divide(schema: UniqueArray, state, context=None, path=(), rng=None):
    """Divide a unique molecule structured array using v1's molecule-specific
    divider. The molecule name comes from the last segment of `path`
    (e.g. ('unique', 'full_chromosome') → 'full_chromosome'). Contextual
    dividers (by_domain, rna_by_domain, ribosome_by_RNA) need sibling
    unique molecule arrays, which we resolve from `context` using the
    relative paths declared in v1's UNIQUE_DIVIDERS topology.
    """
    if state is None:
        return None, None
    if not path:
        # No molecule identity — share by reference. Caller should
        # always pass the path so the right divider can be selected.
        return state, state

    from ecoli.library.schema import UNIQUE_DIVIDERS

    # The unique store uses SINGULAR molecule names (e.g. 'RNA',
    # 'active_RNAP', 'full_chromosome') but vEcoli's UNIQUE_DIVIDERS
    # dict mostly uses PLURAL names ('RNAs', 'active_RNAPs', etc.).
    # active_ribosome is the one exception that's singular in both.
    # Try several common plural forms.
    mol_name = path[-1]
    divider_info = UNIQUE_DIVIDERS.get(mol_name)
    if divider_info is None:
        # +s plural (most cases): RNA→RNAs, active_RNAP→active_RNAPs
        divider_info = UNIQUE_DIVIDERS.get(mol_name + "s")
    if divider_info is None:
        # +es plural (words ending in -x): DnaA_box→DnaA_boxes
        divider_info = UNIQUE_DIVIDERS.get(mol_name + "es")
    if divider_info is None:
        # No v1 divider registered for this molecule. Default: share.
        return state, state

    divider_fn = _get_unique_divider_fn(divider_info["divider"])
    if divider_fn is None:
        return state, state

    # Resolve topology fields against the parent context. Vivarium
    # `..` goes up from the FULL port path (not one segment above), so
    # pass `path` directly as the base.
    topology = divider_info.get("topology", {})
    divider_state_arg = {}
    if context is not None:
        for port_name, port_path in topology.items():
            resolved = _resolve_topology_path(context, path, port_path)
            if resolved is not None:
                divider_state_arg[port_name] = resolved

    return divider_fn(state, divider_state_arg)


# ============================================================================
# SimData references — resolve at realize time from a sim_data pickle
# ============================================================================

# Global sim_data instance, set once at document load time via
# set_sim_data(). Realize methods read from this.
_sim_data = None
_sim_data_path = None


def set_sim_data(sim_data, path=None):
    """Set the global sim_data instance for SimDataRef resolution."""
    global _sim_data, _sim_data_path
    _sim_data = sim_data
    _sim_data_path = path


def get_sim_data():
    """Get the global sim_data instance, loading from path if needed."""
    global _sim_data, _sim_data_path
    if _sim_data is None and _sim_data_path is not None:
        import pickle

        with open(_sim_data_path, "rb") as f:
            _sim_data = pickle.load(f)
    return _sim_data


def _resolve_dotted_path(obj, path_str):
    """Resolve a dotted attribute path with optional bracket indexing.

    Examples:
        'process.metabolism.stoichMatrix'
        'internal_state.bulk_molecules.bulk_data["id"]'
        'external_state.saved_media["minimal"]'
    """
    import re

    current = obj
    # Split on dots, but keep bracket expressions attached to their segment
    segments = re.split(r"\.(?![^[]*\])", path_str)
    for segment in segments:
        # Check for bracket indexing: 'attr["key"]' or 'attr[0]'
        bracket_match = re.match(r"([^[]+)\[(.+)\]$", segment)
        if bracket_match:
            attr_name = bracket_match.group(1)
            key_str = bracket_match.group(2).strip("\"'")
            # Resolve attribute first
            if isinstance(current, dict):
                current = current[attr_name]
            else:
                current = getattr(current, attr_name)
            # Then index
            if isinstance(current, dict):
                current = current[key_str]
            elif isinstance(current, np.ndarray) and current.dtype.names:
                current = current[key_str]
            else:
                try:
                    current = current[int(key_str)]
                except (ValueError, TypeError):
                    current = current[key_str]
        elif isinstance(current, dict):
            current = current[segment]
        elif hasattr(current, segment):
            current = getattr(current, segment)
        else:
            raise AttributeError(
                f"Cannot resolve '{segment}' in path '{path_str}' "
                f"on {type(current).__name__}"
            )
    return current


@dataclass(kw_only=True)
class SimDataRef(Wrap):
    """Reference to a value in sim_data, resolved at realize time.

    Parameterized: ``sim_data_ref[array[float]]`` carries the inner type
    so the schema knows what the resolved value will be.

    The state value is a dotted attribute path string, e.g.:
        'process.metabolism.stoichMatrix'
        'internal_state.bulk_molecules.bulk_data["id"]'

    At realize time, the path is resolved against the global sim_data
    instance and the actual value (ndarray, dict, etc.) is returned.
    """

    pass


@realize.dispatch
def realize(core, schema: SimDataRef, state, path=()):
    if isinstance(state, dict) and "path" in state:
        ref_path = state["path"]
    elif isinstance(state, str):
        ref_path = state
    else:
        return schema, state, []

    sim_data = get_sim_data()
    if sim_data is None:
        raise RuntimeError(
            f"SimDataRef at {path}: sim_data not loaded. "
            f"Call set_sim_data() before realize."
        )

    value = _resolve_dotted_path(sim_data, ref_path)
    return schema, value, []


@dataclass(kw_only=True)
class SimDataMethod(Node):
    """Reference to a callable method on sim_data, resolved at realize time.

    Similar to SimDataRef but the resolved value is expected to be a
    callable (bound method or function). No type parameter needed —
    the resolved value is always callable.
    """

    pass


@realize.dispatch
def realize(core, schema: SimDataMethod, state, path=()):
    if isinstance(state, dict) and "path" in state:
        ref_path = state["path"]
    elif isinstance(state, str):
        ref_path = state
    else:
        return schema, state, []

    sim_data = get_sim_data()
    if sim_data is None:
        raise RuntimeError(f"SimDataMethod at {path}: sim_data not loaded.")

    value = _resolve_dotted_path(sim_data, ref_path)
    if not callable(value):
        raise TypeError(
            f"SimDataMethod at {path}: resolved value is "
            f"{type(value).__name__}, not callable"
        )
    return schema, value, []


# ============================================================================
# SharedProcess / SharedProcessRef — moved to process_bigraph.types.process.
# Re-exported below so existing imports keep working.
# ============================================================================

from process_bigraph.types.process import (  # noqa: F401  (re-exported)
    SharedProcess,
    SharedProcessRef,
    _shared_processes,
    clear_shared_processes,
    _class_address_from_instance,
)


def _capture_internal_state(instance):
    """vEcoli's wrapper around the framework's ``capture_object_state``."""
    return capture_object_state(instance, debug=bool(os.environ.get("INTERNAL_DEBUG")))


def _restore_internal_value(value):
    return restore_object_value(value)


# ============================================================================
# Method — pointer to a method on a sim_data_objects instance
# ============================================================================

# Global registry of sim_data object instances, populated during realize
# of the sim_data_objects store.
_sim_data_object_instances: typing.Dict[str, typing.Any] = {}


@dataclass(kw_only=True)
class SimDataObjectStore(Node):
    """Store for sim_data object instances used by bound method references.

    On realize, each entry is realized as an Object type (reconstructing
    the Python instance from serialized __dict__), then registered in
    ``_sim_data_object_instances`` so Method can find them.
    """

    pass


@dispatch
def render(schema: SimDataObjectStore, defaults=False):
    return "sim_data_object_store"


# ============================================================================
# SympyMatrix type — language-agnostic serialization of sympy matrices
# ============================================================================


@dataclass(kw_only=True)
class SympyMatrix(Node):
    """A sympy Matrix serialized as element-wise srepr strings.

    Serialized form::

        {"rows": 39, "cols": 1,
         "elements": ["Add(Mul(Indexed(...), ...), ...)", ...]}

    On realize, reconstructs the sympy Matrix from the srepr strings.
    """

    pass


@dispatch
def render(schema: SympyMatrix, defaults=False):
    return "sympy_matrix"


@dispatch
def serialize(schema: SympyMatrix, state):
    if state is None:
        return None
    if isinstance(state, dict):
        return state
    import sympy as sp

    rows, cols = state.shape
    elements = [sp.srepr(state[i, j]) for i in range(rows) for j in range(cols)]
    return {"rows": rows, "cols": cols, "elements": elements}


@dispatch
def bundle(schema: SympyMatrix, state, context: typing.Optional[BundleContext] = None):
    return serialize(schema, state)


@realize.dispatch
def realize(core, schema: SympyMatrix, state, path=()):
    if state is None:
        return schema, None, []
    if not isinstance(state, dict):
        return schema, state, []
    import sympy

    ns = vars(sympy)
    rows = state["rows"]
    cols = state["cols"]
    elements = [eval(e, {"__builtins__": {}}, ns) for e in state["elements"]]
    matrix = sympy.Matrix(rows, cols, elements)
    return schema, matrix, []


import sympy as _sympy


@dispatch
def infer(core, value: _sympy.MatrixBase, path: tuple = ()):
    """Infer a sympy Matrix as a SympyMatrix type."""
    return set_default(SympyMatrix(), value), []


# ============================================================================
# DerivedFunction — a function derived from a sibling field
# ============================================================================


@dataclass(kw_only=True)
class DerivedFunction(Node):
    """A function that is derived from another field via a builder function.

    Serialized form::

        {"source_field": "symbolic_rates",
         "builder": "wholecell.utils.build_ode.rates",
         "index": 0}

    On realize (handled by Object realize), the builder is imported,
    called with the realized source field value, and the result (or
    element at index) replaces this placeholder.
    """

    _schema_keys = Node._schema_keys | frozenset(
        {"_source_field", "_builder", "_index"}
    )
    _source_field: str = ""
    _builder: str = ""
    _index: int = -1  # -1 means use full result, >=0 means index into tuple


@dispatch
def render(schema: DerivedFunction, defaults=False):
    return "derived_function"


@dispatch
def serialize(schema: DerivedFunction, state):
    """Serialize: store the derivation recipe, not the function."""
    if isinstance(state, dict):
        return state
    return {
        "source_field": schema._source_field,
        "builder": schema._builder,
        "index": schema._index,
    }


@dispatch
def bundle(
    schema: DerivedFunction, state, context: typing.Optional[BundleContext] = None
):
    return serialize(schema, state)


@realize.dispatch
def realize(core, schema: DerivedFunction, state, path=()):
    """Realize returns the recipe dict — Object realize handles the actual rebuild."""
    if callable(state):
        return schema, state, []
    return schema, state, []


@dispatch
def serialize(schema: SimDataObjectStore, state):
    """Serialize the store — each value as an Object."""
    if not isinstance(state, dict):
        return state
    from bigraph_schema.schema import Object

    result = {}
    for key, instance in state.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        result[key] = serialize(Object(), instance)
    return result


@dispatch
def bundle(
    schema: SimDataObjectStore, state, context: typing.Optional[BundleContext] = None
):
    """Bundle the store — each value as an Object."""
    if not isinstance(state, dict):
        return state
    from bigraph_schema.schema import Object

    result = {}
    for key, instance in state.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        result[key] = bundle(Object(), instance, context)
    return result


@realize.dispatch
def realize(core, schema: SimDataObjectStore, state, path=()):
    """Realize each entry as an Object and register in the global registry.

    Add-only semantics: does NOT call ``.clear()`` because the wrapped
    Composite-as-Process pattern relies on pre-registered entries
    surviving outer realize. (Previously cleared to avoid stale
    cross-test leakage, but in production each Composite owns its own
    process — cross-test leakage isn't a real risk and breaking the
    wrapped-Composite case isn't worth the defense.)
    """
    if not isinstance(state, dict):
        return schema, state, []
    from bigraph_schema.schema import Object

    result = {}
    for key, encoded in state.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        _, instance, _ = core.realize(Object(), encoded)
        result[key] = instance
        _sim_data_object_instances[key] = instance
    return schema, result, []


@dataclass(kw_only=True)
class SimDataObjectRef(Node):
    """Reference to a sim_data object instance by store key.

    Document form::

        {"_type": "sim_data_object_ref",
         "store_key": "external_state"}

    On realize(), looks up the instance in ``_sim_data_object_instances``.
    """

    pass


@dispatch
def render(schema: SimDataObjectRef, defaults=False):
    return "sim_data_object_ref"


@dispatch
def serialize(schema: SimDataObjectRef, state):
    if isinstance(state, dict):
        return state
    # Find the instance in the registry
    inst_id = id(state)
    for key, inst in _sim_data_object_instances.items():
        if id(inst) == inst_id:
            return {"store_key": key}
    return state


@dispatch
def bundle(
    schema: SimDataObjectRef, state, context: typing.Optional[BundleContext] = None
):
    return serialize(schema, state)


@realize.dispatch
def realize(core, schema: SimDataObjectRef, state, path=()):
    if isinstance(state, dict):
        store_key = state.get("store_key")
        if store_key:
            instance = _sim_data_object_instances.get(store_key)
            if instance is None:
                raise RuntimeError(
                    f"SimDataObjectRef at {path}: store_key '{store_key}' "
                    f"not found. Available: {sorted(_sim_data_object_instances.keys())}"
                )
            return schema, instance, []
    # Already an instance
    return schema, state, []


class Method(Node):
    """Reference to a bound method on a sim_data_objects instance.

    Document form::

        {"_type": "method",
         "instance_path": ["sim_data_objects", "transcription"],
         "attribute": "make_elongation_rates"}

    On realize(), looks up the instance in ``_sim_data_object_instances``
    and returns ``getattr(instance, attribute)`` — a bound method.
    """

    pass


@dispatch
def serialize(schema: Method, state):
    """Serialize: if it's a bound method, extract path + attribute."""
    if isinstance(state, dict):
        return state
    if callable(state) and hasattr(state, "__self__"):
        # Find the instance in the registry
        inst_id = id(state.__self__)
        for key, inst in _sim_data_object_instances.items():
            if id(inst) == inst_id:
                return {
                    "instance_path": ["sim_data_objects", key],
                    "attribute": state.__func__.__name__,
                }
        # Not found in registry — fall back to Function serialize
        return serialize(Function(), state)
    if callable(state):
        return serialize(Function(), state)
    return state


@dispatch
def bundle(schema: Method, state, context: typing.Optional[BundleContext] = None):
    return serialize(schema, state)


@dispatch
def render(schema: Method, defaults=False):
    return "method"


@realize.dispatch
def realize(core, schema: Method, state, path=()):
    """Resolve a bound method ref by looking up the instance and binding."""
    if callable(state):
        return schema, state, []
    if isinstance(state, dict):
        instance_path = state.get("instance_path", [])
        attribute = state.get("attribute")
        if len(instance_path) >= 2 and attribute:
            store_key = instance_path[1]
            instance = _sim_data_object_instances.get(store_key)
            if instance is None:
                raise RuntimeError(
                    f"Method at {path}: instance '{store_key}' "
                    f"not found in sim_data_objects. "
                    f"Available: {sorted(_sim_data_object_instances.keys())}"
                )
            method = getattr(instance, attribute)
            return schema, method, []
    return schema, state, []


# ============================================================================
# Lazy-attr hooks for sim_data dataclasses
# ============================================================================
#
# Some sim_data dataclasses lazily populate compiled-lambda attributes
# on first use (e.g. reconstruction/.../process/metabolism.py sets
# `_compiled_enzymes = None` in __init__, then compiles it on first
# `get_kinetic_constraints` call). When a Composite is reconstructed
# from a bundle, the sim_data object is rebuilt via __new__ + __dict__,
# which bypasses __init__. If the lazy attr wasn't serialized (lambdas
# can't be), the instance is missing the attribute entirely.
#
# We attach __post_realize__ to such classes here, at v2 module import
# time, without modifying the shared v1 dataclass. The hook runs after
# Object realize sets __dict__, and re-creates the missing attributes
# from their source fields.


def _install_sim_data_post_realize_hooks():
    try:
        from reconstruction.ecoli.dataclasses.process.metabolism import (
            Metabolism as _MetabolismDataclass,
        )
    except Exception:
        return

    def __post_realize__(self):
        # Mirror the None-init that __init__ does; get_kinetic_constraints
        # will compile on first call. Only set if the attribute is missing
        # (serialized state wins if it was present).
        if not hasattr(self, "_compiled_enzymes"):
            self._compiled_enzymes = None
        if not hasattr(self, "_compiled_saturation"):
            self._compiled_saturation = None

    _MetabolismDataclass.__post_realize__ = __post_realize__


_install_sim_data_post_realize_hooks()


# ============================================================================
# Derived function resolver — vEcoli-specific naming conventions
# ============================================================================
#
# sim_data dataclasses cache lazily-compiled ODE rate/derivative lambdas
# alongside their sympy source expressions. The bundle machinery in
# bigraph_schema needs to know how to reconstruct these lambdas from
# the source fields on load. Rather than baking vEcoli conventions
# into the framework, we register a resolver here.


def _ecoli_derived_function_resolver(key, obj_dict):
    """Map a DerivedFunction field name + parent __dict__ to its
    sympy source field and the wholecell builder function.

    Conventions (vEcoli / wholecell):
    - Field name containing 'jacobian' → source is 'symbolic_rates_jacobian'
    - Otherwise → source is 'symbolic_rates'
    - Builder is selected from the compiled function's argument signature:
        ('t', 'y', 'kf', 'kr') → wholecell.utils.build_ode.rates[_jacobian]
        ('y', 't')             → wholecell.utils.build_ode.derivatives[_jacobian]
    """
    if "jacobian" in key:
        source_field = "symbolic_rates_jacobian"
    else:
        source_field = "symbolic_rates"

    if source_field not in obj_dict:
        return None, None, None

    value = obj_dict[key]
    func = value[0] if isinstance(value, tuple) else value
    if not callable(func) or not hasattr(func, "__code__"):
        return None, None, None

    args = func.__code__.co_varnames[: func.__code__.co_argcount]
    if args == ("t", "y", "kf", "kr"):
        builder = (
            "wholecell.utils.build_ode.rates_jacobian"
            if "jacobian" in key
            else "wholecell.utils.build_ode.rates"
        )
    elif args == ("y", "t"):
        builder = (
            "wholecell.utils.build_ode.derivatives_jacobian"
            if "jacobian" in key
            else "wholecell.utils.build_ode.derivatives"
        )
    else:
        return None, None, None

    return source_field, builder, isinstance(value, tuple)


register_derived_function_resolver(_ecoli_derived_function_resolver)


# ============================================================================
# Type registry
# ============================================================================

ECOLI_TYPES = {
    # 'function', 'quantity' come from bigraph_schema.BASE_TYPES.
    # 'step', 'process', 'shared_process', 'shared_process_ref' come from
    # process_bigraph.types.process via package discovery.
    # vEcoli only registers vEcoli-specific types below.
    "unum": UnumUnits,
    "csr_matrix": CSRMatrix,
    "units_array": UnitsArray,
    "method": Method,
    "bulk_array": BulkArray,
    "unique_array": UniqueArray,
    "sim_data_ref": SimDataRef,
    "sim_data_method": SimDataMethod,
    "sympy_matrix": SympyMatrix,
    "derived_function": DerivedFunction,
    "sim_data_object_store": SimDataObjectStore,
    "sim_data_object_ref": SimDataObjectRef,
}


# ---------------------------------------------------------------------------
# Bundle overrides — write large data directly to Parquet
# ---------------------------------------------------------------------------


@dispatch
def bundle(schema: BulkArray, state, context: typing.Optional[BundleContext] = None):
    """Bundle a BulkArray: write the structured array directly to Parquet."""
    if not isinstance(state, np.ndarray) or not state.dtype.names:
        return state
    if context is not None and state.nbytes >= context.min_bytes:
        return context.save_array(state, "bulk")
    return _serialize_structured_array(state)


@dispatch
def bundle(schema: UniqueArray, state, context: typing.Optional[BundleContext] = None):
    """Bundle a UniqueArray: write the structured array directly to Parquet."""
    if not isinstance(state, np.ndarray):
        # MetadataArray wraps ndarray
        if hasattr(state, "base") and isinstance(
            getattr(state, "base", None), np.ndarray
        ):
            state = np.asarray(state)
        elif hasattr(state, "__array__"):
            state = np.asarray(state)
        else:
            return state
    if not state.dtype.names:
        return state
    if context is not None and state.nbytes >= context.min_bytes:
        return context.save_array(state, "unique")
    return _serialize_structured_array(state)


@dispatch
def bundle(schema: CSRMatrix, state, context: typing.Optional[BundleContext] = None):
    """Bundle a CSRMatrix: bundle the three component arrays (data,
    indices, pointers) through their schemas. Shape comes from the
    CSRMatrix schema, not from metadata markers."""
    if isinstance(state, dict):
        return state
    if state is None:
        return None
    # Dense ndarray with CSRMatrix schema — save directly as array
    if isinstance(state, np.ndarray):
        if context is not None and state.nbytes >= context.min_bytes:
            return context.save_array(state, "csr_matrix")
        return state.tolist()
    if isinstance(state, csr_matrix):
        # Bundle each CSR component through its schema (arrays go to parquet)
        return {
            "data": bundle(schema.data, state.data, context),
            "indices": bundle(schema.indices, state.indices, context),
            "pointers": bundle(schema.pointers, state.indptr, context),
        }
    return serialize(schema, state)


@dispatch
def bundle(schema: UnumUnits, state, context: typing.Optional[BundleContext] = None):
    """Bundle a Unum — same as serialize (scalars stay inline)."""
    return serialize(schema, state)


@dispatch
def bundle(schema: Quantity, state, context: typing.Optional[BundleContext] = None):
    """Bundle a Quantity — same as serialize (scalars stay inline)."""
    return serialize(schema, state)


@dispatch
def bundle(schema: UnitsArray, state, context: typing.Optional[BundleContext] = None):
    """Bundle a UnitStructArray — delegate to serialize.

    serialize produces {'struct': ..., 'units': {...}} where struct is
    the structured array (which bundle can write to parquet) and units
    is a dict mapping field names to Unum values.
    """
    if isinstance(state, UnitStructArray):
        # Sync schema.struct._data with the actual structured dtype so the
        # bundle(Array) strict-dtype check sees them as matching. A bare
        # `units_array` schema starts with a default float64 _data that
        # doesn't reflect the UnitStructArray's real named-field dtype.
        struct = state.struct_array
        if isinstance(struct, np.ndarray) and schema.struct._data != struct.dtype:
            schema.struct._data = struct.dtype
        struct_bundled = bundle(schema.struct, struct, context)
        units_serialized = serialize(schema.units, state.units)
        return {"struct": struct_bundled, "units": units_serialized}
    return serialize(schema, state)


@dispatch
def bundle(schema: Function, state, context: typing.Optional[BundleContext] = None):
    """Bundle a Function — same as serialize (small metadata dict)."""
    return serialize(schema, state)


def register_ecoli_types(core):
    """Type-provider for Ray actors. Registers the basic ECOLI_TYPES
    on the actor's freshly-allocated core. Wire up via:

        from process_bigraph.protocols.ray import register_type_provider
        register_type_provider(
            'ecoli.library.bigraph_types', 'register_ecoli_types')

    Then every actor (per-shard pool) calls this against its own core
    before instantiating the cell-Composite.
    """
    core.register_types(ECOLI_TYPES)


_LSD_CACHE: typing.Dict[str, typing.Any] = {}


def get_cached_load_sim_data(sim_data_path: str, sim_config: dict = None):
    """Return a cached ``LoadSimData`` for this path, building it if
    absent. ``EcoliCellComposite.initialize`` uses this so each
    daughter doesn't re-load the same sim_data pickle from disk."""
    lsd = _LSD_CACHE.get(sim_data_path)
    if lsd is None:
        from ecoli.library.sim_data import LoadSimData

        cfg = dict(sim_config or {})
        cfg.setdefault("sim_data_path", sim_data_path)
        cfg.setdefault("seed", 0)
        cfg.setdefault("agent_id", "0")
        lsd = LoadSimData(**cfg)
        _LSD_CACHE[sim_data_path] = lsd
    return lsd


def load_sim_data_provider(
    core,
    sim_data_path: str = None,
    sim_data_ref=None,
    sim_config: dict = None,
    aws_region: str = None,
):
    """Type-provider that pre-loads sim_data on the actor.

    Populates the module-global ``_sim_data_object_instances`` so
    ``SimDataObjectRef`` can resolve store_keys WITHOUT the actor having
    to receive sim_data via per-cell ``config['state']``. The daughter's
    CompositeDivision can therefore strip ``sim_data_objects`` from its
    shipped state.

    Two sim_data sources, in priority order:

    1. ``sim_data_ref`` (Ray ObjectRef path) — **CURRENTLY BROKEN
       for vEcoli**: arrays returned from plasma are READ-ONLY
       memoryviews, but vEcoli's Cython code (e.g.
       ``stochastic_arrow.Arrowhead.evolve`` invoked by
       complexation) requires writable buffers. Symptoms:
       ``ValueError: buffer source array is read-only``. To
       re-enable, every mutation of sim_data-derived arrays in the
       process tree would need a ``.copy()`` at the use site.
       Kept as a code path because the wiring is correct — only
       the consumer-side mutability needs auditing.
    2. ``sim_data_path`` (default for now): pickle.load from disk
       (or S3) via ``get_cached_load_sim_data``. One copy per actor
       (~500 MB each). Slower startup (S3 download per actor) but
       writable arrays.

    For S3 reads on GovCloud: Ray spawns actor processes that do NOT
    inherit driver env vars, so boto3/polars/fsspec default to
    us-east-1 and a HeadObject against a GovCloud bucket returns
    400 Bad Request. ``aws_region`` (default ``us-gov-west-1``) is
    set via ``os.environ.setdefault`` before any S3 op. Pass
    ``aws_region=''`` to skip the setdefault entirely if running
    against commercial AWS with credentials from a different chain.

    Wire up via:

        # Preferred: pre-load on driver, ray.put, share via plasma:
        sim_data_obj_ref = ray.put(loaded_sim_data)
        register_type_provider(
            'ecoli.library.bigraph_types',
            'load_sim_data_provider',
            kwargs={'sim_data_ref': sim_data_obj_ref,
                    'sim_data_path': '/path/for/cache_key.cPickle',
                    'sim_config': {...}})

        # Fallback: each actor loads from disk independently:
        register_type_provider(
            'ecoli.library.bigraph_types',
            'load_sim_data_provider',
            kwargs={'sim_data_path': '/path/to/simData.cPickle',
                    'sim_config': {...}})
    """
    import os as _os
    import sys as _sys

    # One-stop actor-process setup: AWS_REGION env, thread pins, and
    # s3fs CreateBucket monkey-patch. Same setup ``run_colony_ray.py``
    # does inline inside its ``@ray.remote`` body. Idempotent
    # (setdefault) so explicit env overrides survive.
    from ecoli.library.ray_actor_setup import setup_ray_actor_process

    setup_ray_actor_process(
        aws_region="us-gov-west-1" if aws_region is None else aws_region
    )
    # Resolve sim_data. Path (1) zero-copies via plasma; path (2)
    # falls through to the original disk-load behavior for back-compat.
    if sim_data_ref is not None:
        import ray

        _sys.stderr.write(
            f"[sim-data-provider] ray.get sim_data from plasma store "
            f"(ref={sim_data_ref})\n"
        )
        _sys.stderr.flush()
        sd = ray.get(sim_data_ref)
        # Wrap shared sim_data in a LoadSimData with the colony seed
        # so EcoliCellComposite's get_cached_load_sim_data picks it up.
        # ``sim_data=sd`` skips the disk-load path inside LoadSimData.
        from ecoli.library.sim_data import LoadSimData

        cfg = dict(sim_config or {})
        cfg.setdefault("sim_data_path", sim_data_path or "")
        cfg.setdefault("seed", 0)
        cfg.setdefault("agent_id", "0")
        cfg["sim_data"] = sd
        lsd = LoadSimData(**cfg)
        # Cache under BOTH a synthetic key and the actual path so
        # downstream get_cached_load_sim_data calls hit regardless of
        # which key they use.
        _LSD_CACHE["__ray_shared__"] = lsd
        if sim_data_path:
            _LSD_CACHE[sim_data_path] = lsd
    else:
        _sys.stderr.write(
            f"[sim-data-provider] loading sim_data on actor "
            f"from {sim_data_path} (AWS_REGION={_os.environ.get('AWS_REGION')})\n"
        )
        _sys.stderr.flush()
        lsd = get_cached_load_sim_data(sim_data_path, sim_config)
        sd = lsd.sim_data
    # Same mapping as build_ecoli_document populates into the
    # cell_state['sim_data_objects'] dict. Kept in sync manually —
    # adding a new key here requires mirroring it there. These are
    # NAMED REFERENCES into the single sim_data object above (sharing
    # underlying memory), not 12 independent copies.
    _paths = {
        "external_state": sd.external_state,
        "mass": sd.mass,
        "growth_rate_parameters": sd.growth_rate_parameters,
        "getter": sd.getter,
        "transcription": sd.process.transcription,
        "transcription_regulation": sd.process.transcription_regulation,
        "replication": sd.process.replication,
        "translation": sd.process.translation,
        "metabolism_data": sd.process.metabolism,
        "equilibrium_data": sd.process.equilibrium,
        "two_component_system": sd.process.two_component_system,
        "concentration_updates": sd.process.metabolism.concentration_updates,
    }
    for key, instance in _paths.items():
        if instance is not None:
            _sim_data_object_instances[key] = instance
    _sys.stderr.write(
        f"[sim-data-provider] populated {len(_sim_data_object_instances)} "
        f"sim_data attribute references on actor\n"
    )
    _sys.stderr.flush()
