"""
==================
E. coli Composite
==================

Builds the vEcoli whole-cell model as a process-bigraph composite
document from sim_data config.

The document is a self-contained dict that ``Composite(document)``
can load entirely through ``realize()``. No pre-built instances,
no manual state assembly — everything is declared and realize
handles instantiation, port wiring, and default filling.

Top-level entrypoint:
- ``build_ecoli_document(core, sim_config)`` — produces a composite
  document ready for ``Composite({'state': document}, core=core)``.
"""

import copy
from copy import deepcopy

import numpy as np

from bigraph_schema import (
    deep_merge,
    class_address as _class_address,
    make_arrays_writeable as _make_arrays_writeable,
    tuples_to_lists as _tuple_to_list,
)
from process_bigraph import wire_step_layers
from vivarium.core.engine import _StepGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_schema_defaults(target, schema):
    """Fill target dict with default values from a vivarium ports_schema.

    Walks the schema tree looking for ``_default`` entries and sets them
    in *target* only when the key is missing.  This ensures listeners
    that read their own prior state (e.g. ``ribosome_data`` reading
    ``rRNA_initiated_TU``) have valid values before the first seeding
    ``update()`` call.
    """
    for key, spec in schema.items():
        if key.startswith("_"):
            continue
        if isinstance(spec, dict):
            if "_default" in spec:
                target.setdefault(key, spec["_default"])
            else:
                sub = target.setdefault(key, {})
                if isinstance(sub, dict):
                    _fill_schema_defaults(sub, spec)


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_ecoli_document(core, sim_config, load_sim_data=None, flat=False):
    """Build a complete composite document from sim_config.

    The document contains:
    - Initial cell state (bulk, unique, environment) from sim_data
    - Process/step declarations (address + config + wires)
    - Step flow wiring (layer tokens and triggers)
    - Per-process runtime state (next_update_time, request, allocate)

    Args:
        load_sim_data: Optional pre-built ``LoadSimData`` to reuse.
            When ``None`` (default), a fresh one is constructed from
            ``sim_config`` (the v1 / per-gen / standard path). The
            composite_lineage driver passes a pre-built one whose
            ``sim_data`` pickle is shared across multiple per-gen
            wrappers, so the seed_library precompute and the gen-0
            build don't double-load the pickle.
        flat: When ``False`` (default), returns ``{'agents': {<agent_id>:
            cell_state}}`` — the wrapped form the greenfield colony
            driver and lineage runner expect. CompositeDivision's
            ``agents`` output wires to ``('..', '..', 'agents')``
            (lifts to the outer agents map).

            When ``True``, returns ``cell_state`` directly (no agents
            wrap) AND sets CompositeDivision's ``agents`` output wire
            to ``['agents']`` (a sibling slot inside the cell's own
            state that's empty at construction and gets created
            dynamically when divide fires). This is the shape needed
            for the cell-as-Composite-as-Process pattern (each cell
            wrapped as ``ray:Composite`` or ``local:Composite``) where
            the bridge propagates the inner ``agents`` slot upward to
            the outer agents map. Mirrors the ``grow_divide_agent``
            structure in ``process-bigraph/processes/growth_division.py``.

    Returns:
        The composite document. Shape depends on ``flat``.
        Load with ``Composite({'state': document}, core=core)``.
    """
    from ecoli.library.sim_data import LoadSimData, RAND_MAX

    if load_sim_data is None:
        load_sim_data = LoadSimData(**sim_config)
    agent_id = sim_config.get("agent_id", "0")
    time_step = sim_config.get("time_step", 1.0)

    # 1. Resolve process configs from sim_data
    configs, classes, partitioned, partitioned_configs = _resolve_process_configs(
        load_sim_data, sim_config
    )

    # 2. Build topology (port → wire path mapping)
    topology = _build_topology(sim_config, partitioned, configs, flat=flat)

    # 3. Build flow graph (step execution order)
    flow, configs, classes = _build_flow(
        sim_config,
        load_sim_data,
        configs,
        classes,
        partitioned,
        partitioned_configs,
        time_step,
    )

    # 3a. Optional per-cell CellParquetEmitter step. Added here (alongside
    # all other steps) so its schema merges cleanly with the cell tree
    # in the same realize pass — post-injection corrupts sibling steps'
    # precompile_link. See memory: wire-projection-requires-nested-dicts.
    # Activated by setting ``sim_config['parquet_emitter']`` with the
    # required output / metadata config. The TOPOLOGY for this step is
    # registered in ecoli/processes/cell_parquet_emitter.py.
    parquet_cfg = sim_config.get("parquet_emitter")
    if parquet_cfg:
        from ecoli.processes.cell_parquet_emitter import (
            CellParquetEmitter,
            TOPOLOGY as _PARQUET_TOPOLOGY,
        )

        # Stamp the cell's canonical agent_id into the parquet config
        # so daughters write to their OWN hive partition. Without this,
        # the upstream parquet_cfg keeps the mother's agent_id and every
        # daughter overwrites the mother's parquet batches at the same
        # generation=N/agent_id=<mother>/ path.
        parquet_cfg = deepcopy(parquet_cfg)
        parquet_cfg["agent_id"] = str(
            sim_config.get("agent_id", parquet_cfg.get("agent_id", "0"))
        )
        classes["parquet_emitter"] = CellParquetEmitter
        configs["parquet_emitter"] = parquet_cfg
        topology["parquet_emitter"] = deepcopy(_PARQUET_TOPOLOGY)
        # Flow: run after all other steps. Put it after division so
        # divided daughters' first tick also emits a row.
        if "division" in flow:
            flow["parquet_emitter"] = [("division",)]
        else:
            # No division in this run; depend on last unique_update.
            uu_names = sorted([k for k in flow if k.startswith("unique_update_")])
            if uu_names:
                flow["parquet_emitter"] = [(uu_names[-1],)]

    # 3b. Extract each edge's interface (inputs/outputs) via a temporary
    # instance BEFORE configs are rewritten into serializable refs.
    # Bound-method instances must still be callable for __init__.
    # Also keep the temp instance for the edge-type classification below.
    interfaces = {}
    temp_instances = {}
    from ecoli.library.bigraph_types import translate_ports

    for name, cls in classes.items():
        cfg = configs[name]
        inst = None
        try:
            # Try with no core first (vivarium / BigraphStep path).
            # Plain process-bigraph Steps require a core; retry with
            # one if the first call raised "must provide a core".
            try:
                inst = cls(cfg)
            except Exception as _err:
                if "must provide a core" in str(_err):
                    inst = cls(cfg, core=core)
                else:
                    raise
            interfaces[name] = inst.interface()
            temp_instances[name] = inst
        except Exception as _err:
            # Un-migrated process (no typed interface()): synthesize inputs/
            # outputs from its vivarium ports_schema() via the v1->v2 converter
            # (translate_ports). vivarium ports are bidirectional, so over-
            # declare every port as both input and output (safe). Without this,
            # the process is wired to nothing and dies on the first tick with a
            # KeyError on its un-allocated state path.
            ports = None
            if inst is not None and hasattr(inst, "ports_schema"):
                try:
                    ports = inst.ports_schema()
                except Exception:
                    ports = None
            if ports is not None:
                typed = translate_ports(core, ports)
                interfaces[name] = {"inputs": typed, "outputs": typed}
                temp_instances[name] = inst
            else:
                import traceback as _tb

                print(
                    f"[build_ecoli] interface() AND ports_schema() unavailable "
                    f"for {name}: {type(_err).__name__}: {_err}",
                    flush=True,
                )
                _tb.print_exc()
                interfaces[name] = {"inputs": {}, "outputs": {}}
                temp_instances[name] = None

    # 4. Get initial cell state from sim_data
    cell_state = _get_initial_state(load_sim_data, sim_config)
    if os.environ.get("VECOLI_DEBUG_DIVIDE"):
        import sys as _sys

        try:
            bulk = cell_state.get("bulk") if isinstance(cell_state, dict) else None
            if (
                bulk is not None
                and hasattr(bulk, "dtype")
                and bulk.dtype.names
                and "count" in bulk.dtype.names
            ):
                print(
                    f"[divide-debug] build_ecoli_document agent_id={sim_config.get('agent_id')} "
                    f"cell_state[bulk].count.sum={int(bulk['count'].sum())} (after _get_initial_state)",
                    file=_sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(
                f"[divide-debug] build_ecoli_document err: {e}",
                file=_sys.stderr,
                flush=True,
            )
    _make_arrays_writeable(cell_state)

    # 5. Add infrastructure topologies (allocator, unique_update)
    allocator_topology = {
        "request": ("request",),
        "allocate": ("allocate",),
        "bulk": ("bulk",),
    }
    for name in classes:
        if name.startswith("allocator_"):
            topology[name] = allocator_topology.copy()
        elif name.startswith("unique_update_"):
            # UniqueUpdate topology comes from its config's unique_topo
            unique_topology = configs[name].get("unique_topo", {})
            topology[name] = {k: v for k, v in unique_topology.items()}

    # 5b. Build sim_data_objects store FIRST — bound method refs in
    # SharedProcess and step configs need these instances at realize time.
    sd = load_sim_data.sim_data
    sim_data_objects = {}
    # Map from instance id to store key for deduplication
    _instance_to_key = {}
    _sim_data_paths = {
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
        # Nested objects that are also referenced directly in configs
        "concentration_updates": sd.process.metabolism.concentration_updates,
    }
    for key, instance in _sim_data_paths.items():
        if instance is not None:
            sim_data_objects[key] = instance
            _instance_to_key[id(instance)] = key
    sim_data_objects["_type"] = "sim_data_object_store"
    # When the receiving actor will pre-load sim_data via type-provider
    # (Ray cell-as-Composite path), don't ship sim_data_objects through
    # cell_state — keeps the per-cell shipping payload small and avoids
    # the cost of pickling bound-method references. The actor still
    # needs ``_instance_to_key`` populated for the config rewrite below
    # so the SimDataObjectRef strings get the right store_key.
    if not sim_config.get("skip_sim_data_objects_in_state"):
        cell_state["sim_data_objects"] = sim_data_objects

    # Now rewrite configs: replace bound methods and sim_data object
    # instances with references to the sim_data_objects store.
    def _rewrite_refs(config):
        if not isinstance(config, dict):
            return
        for key, val in list(config.items()):
            if callable(val) and hasattr(val, "__self__") and hasattr(val, "__func__"):
                inst_id = id(val.__self__)
                if inst_id in _instance_to_key:
                    config[key] = {
                        "_type": "method",
                        "instance_path": [
                            "sim_data_objects",
                            _instance_to_key[inst_id],
                        ],
                        "attribute": val.__func__.__name__,
                    }
            elif id(val) in _instance_to_key:
                config[key] = {
                    "_type": "sim_data_object_ref",
                    "store_key": _instance_to_key[id(val)],
                }

    for name, config in configs.items():
        _rewrite_refs(config)
    for name, config in partitioned_configs.items():
        _rewrite_refs(config)

    # 5c. Declare SharedProcess entries in the process store AFTER
    # sim_data_objects so realize() has bound method instances available.
    for proc_name in partitioned:
        proc_class = sim_config["processes"][proc_name]
        proc_config = partitioned_configs.get(proc_name, {})
        cell_state.setdefault("process", {})[proc_name] = {
            "_type": "shared_process",
            "address": _class_address(proc_class),
            "config": proc_config,
            "interval": 1.0,
        }

    # 6. Build process declarations and add to cell state
    for name, cls in classes.items():
        config = configs[name]
        edge_wires = topology.get(name, {})
        wires = _tuple_to_list(edge_wires) or {}

        # Use the interface we extracted before _rewrite_refs mutated
        # configs into serializable method refs.
        interface = interfaces.get(name, {"inputs": {}, "outputs": {}})

        input_ports = set(interface.get("inputs", {}).keys())
        output_ports = set(interface.get("outputs", {}).keys())
        for port_name in input_ports | output_ports:
            if port_name not in wires:
                wires[port_name] = [port_name]

        output_wires = {k: v for k, v in wires.items() if k in output_ports}

        # Determine edge type: Process (continuous-time) vs Step (event-driven).
        # BigraphProcess is the vEcoli bridge; ProcessBigraphProcess is the
        # underlying process-bigraph base class — also accepted in case a
        # process is registered without the bridge.
        from ecoli.library.bigraph_bridge import BigraphProcess
        from process_bigraph import Process as ProcessBigraphProcess

        # vivarium.Process covers un-migrated processes wired via the
        # translate_ports fallback (they are plain vivarium classes, not
        # BigraphProcess subclasses). vivarium.Step subclasses Process but
        # carries a 'triggers' attr, so the step-exclusion below still routes
        # derivers/listeners to 'step'. Without including vivarium.Process,
        # every un-migrated time-driven process (elongation, replication,
        # global_clock, …) is mislabeled a step → no 'interval' → the engine
        # KeyErrors running it as a process.
        from vivarium.core.process import Process as VivariumProcess

        instance = temp_instances.get(name)
        if (
            instance is not None
            and isinstance(
                instance, (BigraphProcess, ProcessBigraphProcess, VivariumProcess)
            )
            and not hasattr(instance, "triggers")
        ):
            edge_type = "process"
        else:
            edge_type = "step"

        # For the document, replace process instances in config with
        # string IDs (SharedProcessRef resolves them at realize time).
        from ecoli.processes.partition import is_partitioned_process

        doc_config = dict(config) if config else {}
        if is_partitioned_process(doc_config.get("process")):
            doc_config["process"] = doc_config["process"].name

        decl = {
            "_type": edge_type,
            "address": _class_address(cls),
            "config": doc_config,
            "_inputs": interface.get("inputs", {}),
            "_outputs": interface.get("outputs", {}),
            "inputs": copy.deepcopy(wires),
            "outputs": copy.deepcopy(output_wires),
        }
        if edge_type == "process":
            decl["interval"] = 1.0
        else:
            decl["priority"] = 1.0

        cell_state[name] = decl

    # global_time default is declared on global_clock.outputs() as
    # 'float{0.0}' for the framework's auto-init at realize time, but
    # the listener-seeding loop below builds views directly from
    # cell_state and bypasses the framework, so we still need the
    # cell_state seed here.
    cell_state.setdefault("global_time", 0.0)
    # timestep is config-derived (no producer process); keep setdefault.
    cell_state.setdefault("timestep", int(time_step))
    # listeners.mass.* defaults are declared in
    # ecoli/processes/listeners/mass_listener.py outputs() so the
    # framework auto-creates them on first read.

    # Listener seeding has been removed: the framework is responsible
    # for running derivers/listeners on init via ``run_steps_on_init``
    # or the first ``run()`` cycle. v1's ``prime_listeners`` equivalent
    # belongs in the framework, not in build_ecoli_document.

    # 9. Initialize per-process runtime state (except process store,
    # which was already declared in step 5b as SharedProcess entries).
    # `allocate.<proc>.bulk` is a full-size int64 array at runtime; declare
    # that explicitly so bundle() externalizes it to Parquet.
    import numpy as _np

    bulk_store = cell_state.get("bulk")
    if isinstance(bulk_store, _np.ndarray):
        n_bulk = len(bulk_store)
    elif isinstance(bulk_store, dict):
        n_bulk = len(bulk_store.get("id", []))
    else:
        n_bulk = 0
    for proc_name in partitioned:
        cell_state.setdefault("next_update_time", {}).setdefault(
            proc_name, float(time_step)
        )
        cell_state.setdefault("request", {}).setdefault(proc_name, {"bulk": []})
        cell_state.setdefault("allocate", {}).setdefault(
            proc_name, {"bulk": _np.zeros(n_bulk, dtype=_np.int64)}
        )

    # Seed allocator_rng deterministically from the Allocator config.
    # Without this, ``realize(NPRandom, None)`` falls back to
    # ``RandomState()`` (system entropy), making gen 1 non-reproducible
    # and producing a non-deterministic saved RNG state at division.
    # The Allocator schema's ``seed`` is ``lineage_seed[integer]``; the
    # config dict here holds the BASE value, so we replicate the same
    # ``(base + lineage) % RAND_MAX`` derivation the framework will
    # apply when constructing the Allocator process — keeping the
    # cell-level store in sync with the process's internal RandomState.
    from bigraph_schema.methods.derive import get_derivation_context

    allocator_seed = next(
        (
            cfg.get("seed")
            for nm, cfg in configs.items()
            if nm.startswith("allocator_") and isinstance(cfg, dict)
        ),
        None,
    )
    if allocator_seed is not None:
        derivation_context = get_derivation_context()
        if derivation_context is not None:
            allocator_seed = (
                int(allocator_seed) + int(derivation_context.lineage_seed)
            ) % RAND_MAX
        cell_state["allocator_rng"] = _np.random.RandomState(seed=int(allocator_seed))

    # 9b. For non-partitioned Steps whose topology references a
    # next_update_time store (e.g. Metabolism), initialize that store so
    # the perform_update() gate in process-bigraph's run_steps correctly
    # skips them at global_time=0.
    for proc_name, ports in topology.items():
        if proc_name in partitioned:
            continue
        nut_wire = ports.get("next_update_time") if isinstance(ports, dict) else None
        if isinstance(nut_wire, (list, tuple)) and len(nut_wire) >= 2:
            parent, key = nut_wire[0], nut_wire[1]
            cell_state.setdefault(parent, {}).setdefault(key, float(time_step))

    # 9. Wire step layers (flow tokens + triggers)
    if flow:
        step_layers = wire_step_layers(cell_state, flow)

        # 9a. Assign execution-order priorities so the engine's
        # ``determine_steps`` cycle-fallback yields TOPOLOGICAL order.
        #
        # The flow-token dependency graph is acyclic on the tokens alone,
        # but at runtime the full step network is CYCLIC: processes and
        # steps share the ``bulk`` / ``unique`` / ``listeners`` stores, so
        # build_step_network adds data-path back-edges among them. With a
        # cyclic graph, ``determine_steps`` can never find a step whose
        # inputs are all fulfilled and falls back to running the single
        # highest-``priority`` remaining step (composite.py:819-826).
        #
        # We were leaving every step at a flat ``priority = 1.0``, so that
        # fallback picked an essentially ARBITRARY step each cycle —
        # scrambling the requester→allocator→evolver ordering. The
        # polypeptide_initiation evolver then read its ``allocate`` store
        # BEFORE allocator_2 processed the fresh request, was allocated 0
        # ribosomal subunits every tick, made no new ribosomes, and
        # translation collapsed to zero active ribosomes by ~tick 89.
        #
        # Mirror v2ecoli's ``inject_flow_dependencies``: give each edge a
        # priority that strictly DECREASES with its execution-layer index,
        # so the cycle-fallback always picks the earliest-scheduled step
        # and the partition cascade runs in order every tick.
        ordered = [name for lvl in sorted(step_layers) for name in step_layers[lvl]]
        total = len(ordered)
        for idx, name in enumerate(ordered):
            edge = cell_state.get(name)
            if isinstance(edge, dict) and ("priority" in edge or "interval" in edge):
                edge["priority"] = float(total - idx)

    if flat:
        # Flat mode: caller is wrapping the cell as a Composite-as-
        # Process and supplies its own agents-map nesting at the
        # outer level. Return cell_state directly.
        result = cell_state
    else:
        result = {"agents": {agent_id: cell_state}}

    # Cell-as-Composite mode: add the dedicated ``divide_emit`` slot
    # where CompositeDivision routes its _add/_remove sentinels.
    # Typed loosely as ``map[node]`` so divide events of any daughter
    # shape land without schema fight. The cell-Composite's bridge
    # output wires here, so only divide events (not every inner
    # sub-process update) cross to the outer — see _build_topology
    # for the matching division_agents_wire routing.
    if sim_config.get("cell_as_composite_mode"):
        result["divide_emit"] = {"_type": "map[node]", "_value": {}}

    return result


# Keep backward compat alias
build_composite_native = build_ecoli_document


def collect_output_metadata_from_composite(composite):
    """Walk a realized Composite's state to collect ``output_metadata``
    from each process/step's ports_schema. Mirrors
    :py:meth:`EcoliSim.output_metadata` for the cell-as-Composite
    pipeline — needed so the colony's parquet ``configuration`` table
    has the ``output_metadata__listeners__...`` columns that
    multiseed / multigeneration analyses query for cistron IDs, gene
    names, and reaction lists.

    Each migrated process still implements Vivarium-style
    ``ports_schema()`` and may embed per-listener metadata under
    ``_properties.metadata`` (see ``ecoli.library.schema.listener_schema``).
    Walk the realized cell state, call ``ports_schema()`` on each
    instance, extract those annotations via ``extract_metadata``, and
    remap from port names to wire paths via each process's ``inputs``
    declaration so the resulting columns line up with the listener
    arrays in the history parquet.

    Pass the result into ``CellParquetEmitter`` via
    ``sim_config['parquet_emitter']['output_metadata']`` so it lands
    in the first ``configuration`` emit alongside experiment_id, etc.
    """
    from vivarium.library.topology import inverse_topology
    from vivarium.library.dict_utils import deep_merge_check
    from ecoli.experiments.ecoli_master_sim import extract_metadata

    output_metadata: dict = {}

    def _collect(node, path):
        """Yields (port_name_or_None, ports_schema, wires) for each
        process instance found by walking ``node``. Stops descending at
        any process/step boundary (don't recurse INTO an inner
        Composite's encapsulated steps from outside)."""
        results = []
        if not isinstance(node, dict):
            return results
        instance = node.get("instance")
        if instance is not None and hasattr(instance, "ports_schema"):
            try:
                ports = instance.ports_schema()
            except Exception:
                return results
            wires = node.get("inputs", {}) or {}
            proc_name = path[-1] if path else None
            results.append((proc_name, ports, wires))
            # Don't descend into process internals — opaque boundary
            # matches ``find_instances`` behavior in process_bigraph.
            return results
        for key, child in node.items():
            if isinstance(child, dict):
                results.extend(_collect(child, path + (key,)))
        return results

    for proc_name, ports, wires in _collect(composite.state, ()):
        extracted = extract_metadata(ports)
        if not extracted:
            continue
        # Remap port names to wire paths so listener metadata ends up
        # at the same path as the listener data in history. ``wires``
        # is the process decl's ``inputs`` dict (port → wire path).
        if wires:
            try:
                extracted = inverse_topology((), extracted, wires)
            except Exception:
                # Some processes have wires that aren't simple tuples
                # (e.g. ``__`` references) — fall back to leaving the
                # extracted dict keyed by port name. Analyses that need
                # the wire-path version will skip these annotations
                # cleanly via field_metadata lookups.
                pass
        try:
            output_metadata = deep_merge_check(
                output_metadata, extracted, check_equality=True
            )
        except Exception:
            output_metadata = {**extracted, **output_metadata}
    return output_metadata


def reseed_loaded_bundle(document, sim_data_path, cli_seed, agent_id="0"):
    """Recompute per-process seeds from sim_data with the current
    generation's ``cli_seed`` and overwrite them in a freshly-loaded
    bundle ``document``. Also resets the cell-level ``allocator_rng``
    and strips saved per-process ``rng_state`` so daughter processes
    start from freshly-seeded RandomStates instead of replaying
    mother's advanced state.

    Mirrors v1's per-generation reset where each daughter is
    constructed from ``LoadSimData(seed=cli_seed)``: per-process seed
    = ``crc32(_seedFromName_input, cli_seed) & 0xFFFFFFFF``.

    Mutates ``document`` in place. Call BEFORE constructing the
    Composite so realize sees the up-to-date config.
    """
    from ecoli.library.sim_data import LoadSimData

    sd = LoadSimData(sim_data_path=sim_data_path, seed=int(cli_seed))

    cell = document["state"]["agents"].get(agent_id)
    if cell is None:
        cell = document["state"]["agents"][next(iter(document["state"]["agents"]))]
    if not isinstance(cell, dict):
        return

    def _strip_partition_suffix(n):
        # Partition wraps ``ecoli-X`` as ``ecoli-X_requester`` and
        # ``ecoli-X_evolver``. Strip both. Map allocator_N to
        # the singular ``allocator`` sim_data key.
        if n.endswith("_requester"):
            return n[: -len("_requester")]
        if n.endswith("_evolver"):
            return n[: -len("_evolver")]
        if n.startswith("allocator_"):
            return "allocator"
        return n

    def _refresh(scope):
        for name, decl in scope.items():
            if not isinstance(decl, dict):
                continue
            cfg = decl.get("config")
            if not isinstance(cfg, dict) or "seed" not in cfg:
                continue
            try:
                fresh_cfg = sd.get_config_by_name(_strip_partition_suffix(name))
            except (KeyError, Exception):
                continue
            if isinstance(fresh_cfg, dict) and "seed" in fresh_cfg:
                cfg["seed"] = int(fresh_cfg["seed"])
            # Drop saved RandomState — daughter starts fresh from the
            # newly-derived seed (matches v1's daughter construction).
            decl.pop("rng_state", None)

    _refresh(cell)
    proc_block = cell.get("process")
    if isinstance(proc_block, dict):
        _refresh(proc_block)


def _reseed_allocator_rng(state, sim_data_path, cli_seed, agent_id="0"):
    """Reset ``allocator_rng`` to a freshly-seeded RandomState matching
    the cli_seed-derived seed for this generation.

    The bundle stores mother's advanced RandomState; a daughter that
    replayed it would diverge from v1's per-generation re-seeding.
    Mirrors v1's ``RandomState(seed=_seedFromName('BulkMolecules',
    cli_seed))``.

    Accepts either an ``{'agents': {...}}`` document or a bare cell
    dict. Mutates ``state`` in place.
    """
    import numpy as _np
    from ecoli.library.sim_data import LoadSimData

    if isinstance(state, dict) and "agents" in state:
        cell = state["agents"].get(agent_id)
        if cell is None:
            cell = state["agents"][next(iter(state["agents"]))]
    else:
        cell = state
    if not isinstance(cell, dict):
        return
    sd = LoadSimData(sim_data_path=sim_data_path, seed=int(cli_seed))
    seed = sd.get_allocator_config()["seed"]
    cell["allocator_rng"] = _np.random.RandomState(seed=int(seed))


# ---------------------------------------------------------------------------
# Run-to-division loop and daughter save (v1-style, fsspec, cloud-aware)
# ---------------------------------------------------------------------------
#
# The v1 workflow handed daughter state across generations as a single JSON
# file written via fsspec — works equally for local paths and s3:// / gs://.
# v2 originally used save_bundle (parquet + document.json), which is local-
# only: os.makedirs/os.path.join/open silently create local dirs literally
# named ``s3:`` instead of writing to S3, so daughter handoff broke on
# Atlantis. These helpers restore v1's pattern for daughter handoff while
# keeping the bundle infrastructure available for other uses (checkpoints).

# v2 has the same cell state v1 has, plus extra infrastructure that's
# either unserializable or recreated on the next-gen build:
#   - top-level edge declarations (process/step decls with address/instance)
#   - sim_data_objects (bound-method instance refs)
# These get stripped here. Everything else v1 ships, v2 ships too, so
# daughters inherit mother's listeners / boundary / process_state /
# allocate / request etc. — same as v1, same bit-parity story.


def _strip_v2_edges(d):
    """Remove top-level edge declarations and sim_data_objects refs.

    v1 has no equivalent because vivarium stores processes in a separate
    registry; v2 keeps them as state nodes (with ``address`` + ``instance``
    after realize). Strip them so the daughter JSON contains only data,
    matching v1's ``state.get_value(condition=not_a_process)`` output.
    """
    edge_keys = [
        k
        for k, v in d.items()
        if isinstance(v, dict)
        and (
            "address" in v
            or "instance" in v
            or v.get("_type") in ("process", "step", "shared_process", "shared_step")
        )
    ]
    for k in edge_keys:
        del d[k]
    d.pop("sim_data_objects", None)
    d.pop("step_flow", None)


def _v2_daughter_payload(agent_state):
    """Prepare a v2 cell state for in-memory daughter handoff (and
    JSON serialization for nextflow).

    **Mirrors v1's ``prepare_save_state`` exactly.** v1 ships
    everything from ``state.get_value(condition=not_a_process)``
    minus two unserializable keys (``process``, ``allocator_rng``)
    plus dtype metadata. v2 does the same, with v2-specific edge
    stripping (vivarium stores processes in a separate registry;
    v2 keeps them as state nodes with ``address`` + ``instance``
    after realize, so we strip those).

    Don't add per-key resets here — that's whack-a-mole. If a
    listener / process_state / accumulator diverges from v1, the
    fix belongs at the schema level (divide_reset on the type) or
    in the relevant process's state semantics, not as an ad-hoc
    drop in this function. The job here is "ship the same thing
    v1 ships".
    """
    _strip_v2_edges(agent_state)
    agent_state.pop("process", None)
    agent_state.pop("allocator_rng", None)
    if "bulk" in agent_state and hasattr(agent_state["bulk"], "dtype"):
        agent_state["bulk_dtypes"] = str(agent_state["bulk"].dtype)
    if "unique" in agent_state and isinstance(agent_state["unique"], dict):
        agent_state["unique_dtypes"] = {}
        for name, mols in list(agent_state["unique"].items()):
            agent_state["unique"][name] = np.asarray(mols)
            agent_state["unique_dtypes"][name] = str(mols.dtype)


def save_v2_daughters(state, daughter_outdir):
    """Write per-daughter JSON files (v1 single-JSON pattern, fsspec).

    Mirrors v1's ``EcoliSim.update_experiment`` daughter-save: one
    ``daughter_state_{i}.json`` per daughter under ``daughter_outdir``,
    plus ``daughter_state_{i}_uri.txt`` and ``division_time.sh`` in the
    current working directory for Nextflow to read.

    Args:
        state: Composite's full state dict (must contain ``agents`` key
            with exactly 2 daughter cells).
        daughter_outdir: Output directory or cloud URI prefix
            (``s3://``, ``gs://``, or local path).
    """
    from ecoli.library.logging_tools import write_json
    from wholecell.utils.filepath import cloud_path_join

    agents = state.get("agents", {})
    if len(agents) != 2:
        print(
            f"  WARNING: expected 2 daughters post-divide, got {len(agents)}",
            flush=True,
        )

    # Top-level non-agent state — strip v2 edges, keep everything else
    # (matches v1's `non_agent_state = {k:v for k,v in state.items()
    # if k != 'agents'}` after vivarium's not_a_process filter).
    non_agent_state = {k: deepcopy(v) for k, v in state.items() if k != "agents"}
    _strip_v2_edges(non_agent_state)

    for i, (agent_id, agent_state) in enumerate(sorted(agents.items())):
        # Deep-copy so the live composite state isn't mutated by the
        # in-place pruning in _v2_daughter_payload (the run loop may
        # continue after save in some test harnesses).
        agent_copy = deepcopy(agent_state)
        _v2_daughter_payload(agent_copy)

        daughter_path = cloud_path_join(
            daughter_outdir.rstrip("/"), f"daughter_state_{i}.json"
        )
        write_json(daughter_path, {**non_agent_state, "agents": {agent_id: agent_copy}})
        with open(f"daughter_state_{i}_uri.txt", "w") as f:
            f.write(daughter_path)

    division_time = float(state.get("global_time", 0.0))
    with open("division_time.sh", "w") as f:
        f.write(f"export division_time={division_time}")
    print(f"  wrote {len(agents)} daughter JSON(s) to {daughter_outdir}", flush=True)


def run_to_division(
    composite, max_duration, daughter_outdir=None, on_tick=None, poll_s=1.0
):
    """Tick the composite until first division or ``max_duration``.

    This is the v2 equivalent of v1's ``update_experiment`` loop:
    advance in short steps, poll for division (agent count grew), and on
    division write per-daughter JSONs via fsspec.

    Args:
        composite: process_bigraph.Composite instance to drive.
        max_duration: Max simulated seconds to run before stopping.
        daughter_outdir: If set, save daughters here on division.
        on_tick: Optional callback ``fn(composite)`` invoked after each
            successful tick (used by ecoli_master_sim to emit history
            rows to the parquet emitter).
        poll_s: Tick granularity in seconds (1.0 matches v1's per-tick
            emit cadence).

    Returns:
        ``(divided: bool, current_time: float)``.
    """
    from ecoli.processes.cell_division import DivisionDetected

    # Caller may have already set this; flip it on so the framework
    # halts the post-divide step cascade, leaving daughters' derived
    # state in mother's pre-divide state (matches v1's
    # DivisionDetected halt).
    composite._halt_after_structural = True

    pre_agent_count = len(composite.state.get("agents", {}))
    current_t = float(composite.state.get("global_time", 0.0))
    end_t = current_t + float(max_duration)
    divided = False

    while float(composite.state.get("global_time", 0.0)) < end_t:
        remaining = end_t - float(composite.state.get("global_time", 0.0))
        step = min(poll_s, remaining)
        try:
            composite.run(step)
        except DivisionDetected:
            divided = True
            break
        if on_tick is not None:
            on_tick(composite)
        if len(composite.state.get("agents", {})) > pre_agent_count:
            divided = True
            break

    if divided and daughter_outdir:
        save_v2_daughters(composite.state, daughter_outdir)

    return divided, float(composite.state.get("global_time", 0.0))


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _resolve_process_configs(load_sim_data, config):
    """Resolve process configs from sim_data without instantiation.

    Returns (configs, classes, partitioned_names) where:
    - configs: {step_name: config_dict}
    - classes: {step_name: class}
    - partitioned_names: [process_name, ...] for PartitionedProcesses
    """
    from ecoli.processes.partition import (
        is_partitioned_process,
        Requester,
        Evolver,
    )

    time_step = config["time_step"]
    process_configs = {}
    for name, cfg in config["process_configs"].items():
        if cfg == "sim_data":
            process_configs[name] = load_sim_data.get_config_by_name(name, time_step)
        elif cfg == "default":
            process_configs[name] = None
        elif isinstance(cfg, dict):
            try:
                default = load_sim_data.get_config_by_name(name, time_step)
            except KeyError:
                default = config["processes"][name].defaults
            process_configs[name] = deepcopy(default)
            process_configs[name] = deep_merge(process_configs[name], cfg)
            # Per-generation seed derivation
            # ((default + config["seed"]) % RAND_MAX) is now handled at
            # realize time by the framework's LineageSeed type — the
            # stored 'seed' field is the BASE value, and the active
            # DerivationContext.lineage_seed is combined just before
            # the process constructor sees it.

    configs = {}
    classes = {}
    partitioned = []

    partitioned_configs = {}  # original configs for SharedProcess declarations

    for process_name, process_class in config["processes"].items():
        if is_partitioned_process(process_class):
            parallel = process_configs[process_name].pop("_parallel", False)
            # Save the config for the SharedProcess declaration — share the
            # reference so bound method instances match the sim_data
            # instances in the sim_data_objects store.
            partitioned_configs[process_name] = process_configs[process_name]
            # Instantiate the PartitionedProcess (needed for Requester/Evolver config)
            process_instance = process_class(process_configs[process_name])
            req_config = {
                "time_step": time_step,
                "process": process_instance,
                "_parallel": parallel,
            }
            evo_config = {
                "time_step": time_step,
                "process": process_instance,
                "_parallel": parallel,
            }
            configs[f"{process_name}_requester"] = req_config
            configs[f"{process_name}_evolver"] = evo_config
            classes[f"{process_name}_requester"] = Requester
            classes[f"{process_name}_evolver"] = Evolver
            partitioned.append(process_name)
        else:
            configs[process_name] = process_configs.get(process_name)
            classes[process_name] = process_class

    return configs, classes, partitioned, partitioned_configs


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def _build_topology(config, partitioned, configs, flat=False):
    """Build port→wire topology from config."""
    topology = {}
    for process_id, ports in config["topology"].items():
        if process_id in partitioned:
            topology[f"{process_id}_requester"] = deepcopy(ports)
            topology[f"{process_id}_evolver"] = deepcopy(ports)
            topology[f"{process_id}_requester"]["request"] = ("request", process_id)
            topology[f"{process_id}_evolver"]["allocate"] = ("allocate", process_id)
            topology[f"{process_id}_requester"]["next_update_time"] = (
                "next_update_time",
                process_id,
            )
            topology[f"{process_id}_evolver"]["next_update_time"] = (
                "next_update_time",
                process_id,
            )
            topology[f"{process_id}_requester"]["process"] = ("process", process_id)
            topology[f"{process_id}_evolver"]["process"] = ("process", process_id)
            topology[f"{process_id}_requester"]["global_time"] = ("global_time",)
            topology[f"{process_id}_evolver"]["global_time"] = ("global_time",)
        else:
            topology[process_id] = deepcopy(ports)

    if config.get("divide"):
        if config.get("d_period"):
            topology["mark_d_period"] = {
                "full_chromosome": tuple(config["chromosome_path"]),
                "global_time": ("global_time",),
                "divide": ("divide",),
            }
        # Division ``agents`` wire depends on whether the cell tree is
        # wrapped in ``{agents: {<id>: cell_state}}`` (default) or flat
        # (cell_state directly at composite root). In the wrapped case
        # CompositeDivision is at ``agents/<id>/division`` so going
        # two levels up reaches the ``agents`` map. In flat mode
        # CompositeDivision is at ``<root>/division`` so the agents
        # slot is a sibling — wire is just ``('agents',)``. The slot
        # is empty at construction and gets created dynamically when
        # the divide sentinel is applied (mirrors how the Divide step
        # in ``process-bigraph/processes/growth_division.py`` emits to
        # an empty inner ``environment`` slot that the bridge then
        # propagates to the outer environment map).
        #
        # CELL-AS-COMPOSITE BRIDGE-EMIT MODE: when configured,
        # CompositeDivision's agents output routes to a DEDICATED
        # ``divide_emit`` slot (a sibling of ``agents`` in wrapped
        # mode, or at root in flat mode). The cell-Composite's bridge
        # output then wires to ``divide_emit`` instead of ``agents``,
        # so only divide events cross the bridge to the outer. Without
        # this, every inner sub-process update has shape
        # ``{'agents': {<id>: {<field>: <delta>}}}`` and the bridge
        # wire to ``['agents']`` would propagate ALL of them, drowning
        # the outer's apply_updates in O(N_subprocesses) reconciles
        # per tick (measured: 14.7s/30s wall in profile).
        if config.get("cell_as_composite_mode"):
            division_agents_wire = (
                ("divide_emit",) if flat else ("..", "..", "divide_emit")
            )
        else:
            division_agents_wire = ("agents",) if flat else ("..", "..", "agents")
        topology["division"] = {
            "division_variable": tuple(config["division_variable"]),
            "full_chromosome": tuple(config["chromosome_path"]),
            "agents": division_agents_wire,
            "media_id": ("environment", "media_id"),
            "division_threshold": ("division_threshold",),
        }
        # NOTE: cell_as_composite_mode previously added a mother_state
        # input wire here. Removed — CompositeDivision now reads
        # mother state directly from the cell-Composite instance via
        # the module-level ``_CELL_COMPOSITE_INSTANCE`` cache.
        # The wire approach corrupted numpy struct arrays (view walk
        # converted to dicts) and caused resolve_merges conflicts at
        # init even when using the precise cell tree schema as the
        # port type.

    return topology


# ---------------------------------------------------------------------------
# Flow graph
# ---------------------------------------------------------------------------


def _build_flow(
    config, load_sim_data, configs, classes, partitioned, partitioned_configs, time_step
):
    """Build step execution flow and add infrastructure steps."""
    from ecoli.processes.allocator import Allocator
    from ecoli.processes.unique_update import UniqueUpdate
    from ecoli.processes.cell_division import MarkDPeriod

    step_graph = _StepGraph()
    step_classes = dict(classes)  # will be extended with infra steps

    for process in config["processes"]:
        deps = config["flow"].get(process, [])
        tuplified_deps = []
        for dep_path in deps:
            if dep_path[-1] in partitioned:
                tuplified_deps.append(
                    tuple(dep_path[:-1]) + (f"{dep_path[-1]}_evolver",)
                )
            else:
                tuplified_deps.append(tuple(dep_path))
        if process in partitioned:
            step_graph.add((f"{process}_requester",), tuplified_deps)
            step_graph.add((f"{process}_evolver",), [(f"{process}_requester",)])
        elif process in classes:
            step_graph.add((process,), tuplified_deps)

    layers = step_graph.get_execution_layers()
    flow = {}
    allocator_counter = 1
    unique_update_counter = 1

    for layer_steps in layers:
        requesters = False
        for step_path in layer_steps:
            if "evolver" in step_path[-1]:
                flow[step_path[-1]] = [(f"allocator_{allocator_counter - 1}",)]
            elif unique_update_counter > 1:
                flow[step_path[-1]] = [(f"unique_update_{unique_update_counter - 1}",)]
                if "requester" in step_path[-1]:
                    requesters = True
            else:
                flow[step_path[-1]] = []
        if requesters:
            flow[f"allocator_{allocator_counter}"] = layer_steps
            allocator_counter += 1
        else:
            flow[f"unique_update_{unique_update_counter}"] = [step_path]
            unique_update_counter += 1

    # Allocator configs and classes
    allocator_config = load_sim_data.get_allocator_config(
        time_step, process_names=partitioned
    )
    allocator_topology = {
        "request": ("request",),
        "allocate": ("allocate",),
        "bulk": ("bulk",),
    }
    for i in range(1, allocator_counter):
        name = f"allocator_{i}"
        configs[name] = allocator_config
        classes[name] = Allocator

    # UniqueUpdate configs and classes
    unique_mols = (
        load_sim_data.sim_data.internal_state
    ).unique_molecule.unique_molecule_definitions.keys()
    unique_topology = {
        unique_mol + "s": ("unique", unique_mol)
        for unique_mol in unique_mols
        if unique_mol not in ["active_ribosome", "DnaA_box"]
    }
    unique_topology["active_ribosome"] = ("unique", "active_ribosome")
    unique_topology["DnaA_boxes"] = ("unique", "DnaA_box")
    unique_params = {
        "unique_topo": unique_topology,  # key matches UniqueUpdate's config
        "emit_unique": config["emit_unique"],
    }
    for i in range(1, unique_update_counter):
        name = f"unique_update_{i}"
        configs[name] = unique_params
        classes[name] = UniqueUpdate

    # Division steps
    if config.get("divide"):
        from ecoli.processes.cell_division import (
            CompositeDivision,
            daughter_phylogeny_id,
        )

        # Discover per-process seed paths so CompositeDivision can
        # emit per-daughter seed overrides at divide time. Two homes
        # for seeded process configs:
        #   1. Partitioned processes live in the cell's ``process``
        #      store as SharedProcess entries; their config['seed']
        #      lives at ``process/<name>/config/seed``.
        #   2. Non-partitioned steps live at the cell root; their
        #      config['seed'] is at ``<name>/config/seed``.
        # Anything with ``seed`` in its resolved config is a candidate.
        seed_paths = []
        for proc_name in partitioned:
            shared_cfg = partitioned_configs.get(proc_name, {})
            if isinstance(shared_cfg, dict) and "seed" in shared_cfg:
                seed_paths.append(["process", proc_name, "config", "seed"])
        for step_name, step_cfg in configs.items():
            if step_name == "division":
                continue  # division reseeds itself via its own override
            if step_name in partitioned:
                continue  # handled above as SharedProcess
            if not isinstance(step_cfg, dict):
                continue
            if "seed" in step_cfg:
                seed_paths.append([step_name, "config", "seed"])

        # v2 uses CompositeDivision (skips the v1 Composer roundtrip —
        # the framework handles daughter state reconstruction via
        # type-driven _divide_state and Link instantiation).
        division_config = {
            "division_threshold": config["division_threshold"],
            "agent_id": config["agent_id"],
            "dry_mass_inc_dict": load_sim_data.sim_data.expectedDryMassIncreaseDict,
            # Base seed is 0; the framework's LineageSeed type adds the
            # active DerivationContext.lineage_seed at realize time so the
            # constructor sees the per-generation value (matches v1's
            # division.seed = config["seed"] without pre-deriving here).
            "seed": 0,
            "daughter_ids_function": daughter_phylogeny_id,
            # Single-lineage mode: when True, CompositeDivision emits
            # only daughter 0 in the _divide sentinel. Currently
            # always False — both ``engine=composite`` (per-gen
            # Nextflow) and ``engine=composite_lineage`` (managed
            # loop in EcoliSim._run_composite_lineage) want both
            # daughters in state at division and let the driver
            # extract daughter 0. Reserved for a future tree-mode
            # honest-composite design where division stays in-place.
            "single_daughters": False,
            # Paths inside the cell tree for per-process seed reseeding
            # at divide. See CompositeDivision.config_schema['seed_paths']
            # for the format. Empty list disables (correlated daughters).
            "seed_paths": seed_paths,
        }
        # Cell-as-Composite mode: caller can pass a daughter wrap
        # template + cell schema in sim_config under the
        # ``division_wrap_template`` / ``division_cell_schema`` keys.
        # CompositeDivision uses them to build fully-wrapped daughter
        # Composite-Process decls and emit _add/_remove instead of the
        # legacy _divide sentinel. The matching topology change is in
        # _build_topology (adds a ``mother_state`` wire when this is
        # configured).
        if config.get("cell_as_composite_mode"):
            # cell_as_composite_mode is just a boolean flag — the actual
            # daughter wrap template and cell schema come from the
            # cell_division module-level cache (set via
            # ``set_daughter_wrap_template`` / ``set_cell_tree_schema``)
            # because storing them in config would trigger the type
            # system to walk them as state (schema → nonsense inferred
            # types; wrap_template → recursive realize-as-process).
            division_config["cell_as_composite_mode"] = True
            # daughter_address inherited by every daughter and flows
            # transparently across generations.
            division_config["daughter_address"] = config.get(
                "daughter_address", "local:Composite"
            )
        configs["division"] = division_config
        classes["division"] = CompositeDivision

        if config.get("d_period"):
            configs["mark_d_period"] = {}
            classes["mark_d_period"] = MarkDPeriod
            flow["mark_d_period"] = [(f"unique_update_{unique_update_counter - 1}",)]
            # Extra UniqueUpdate after MarkDPeriod
            uu_name = f"unique_update_{unique_update_counter}"
            configs[uu_name] = unique_params
            classes[uu_name] = UniqueUpdate
            flow[uu_name] = [("mark_d_period",)]
            flow["division"] = [(uu_name,)]
        else:
            flow["division"] = [(f"unique_update_{unique_update_counter - 1}",)]

    return flow, configs, classes


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def _get_initial_state(load_sim_data, config):
    """Get initial cell state from sim_data or a daughter handoff JSON.

    Same pattern as v1's update_experiment → get_state_from_file path:
    if ``initial_state_file`` is set, load it (fsspec-aware via
    ``load_states``) and use its agent's cell_state as-is.
    ``build_ecoli_document`` then layers fresh process / allocator /
    sim_data_objects / etc. on top, matching v1's per-gen reset for
    those infrastructure pieces while preserving mother's data state.
    """
    from ecoli.library.json_state import get_state_from_file
    from wholecell.utils.filepath import is_cloud_uri

    full_state = config.get("initial_state", None)
    if not full_state:
        initial_state_file = config.get("initial_state_file", None)
        if not initial_state_file:
            full_state = load_sim_data.generate_initial_state()
        else:
            if is_cloud_uri(initial_state_file) or initial_state_file.startswith("/"):
                state_path = initial_state_file
            else:
                state_path = f"data/{initial_state_file}.json"
            full_state = get_state_from_file(path=state_path)

    if "agents" in full_state:
        cell_state = full_state["agents"][config.get("agent_id", "0")]
    else:
        cell_state = full_state
    return _apply_overrides(cell_state, config)


def _apply_overrides(cell_state, config):
    """Apply ``initial_state_overrides`` files to a cell_state in place."""
    from ecoli.library.json_state import get_state_from_file

    overrides = config.get("initial_state_overrides", [])
    if overrides:
        bulk_map = {
            bulk_id: row_id for row_id, bulk_id in enumerate(cell_state["bulk"]["id"])
        }
    for override_file in overrides:
        override = get_state_from_file(path=f"data/{override_file}.json")
        bulk_overrides = override.pop("bulk", {})
        cell_state["bulk"].flags.writeable = True
        for molecule, count in bulk_overrides.items():
            cell_state["bulk"]["count"][bulk_map[molecule]] = count
        cell_state["bulk"].flags.writeable = False
        deep_merge(cell_state, override)

    return cell_state


# ---------------------------------------------------------------------------
# Whole-cell process wrapper
# ---------------------------------------------------------------------------
#
# EcoliProcess wraps the entire vEcoli composite as a single process so it
# can be embedded in a larger simulation. The class subclasses Composite
# and uses the standard process-bigraph bridge mechanism: external inputs
# are projected into internal stores, the inner sim runs for the requested
# interval, and bridge outputs are read back out at the end.
#
# Boundary (what an outer simulation sees):
#   inputs:  external (mM concentrations), media_id (str), global_time (s)
#   outputs: exchange (counts), cell_mass (fg), dry_mass (fg), volume (L),
#            agent_id (str)
#
# Inside, the cell state is nested under ``agents.<agent_id>`` so the v2
# division machinery (CompositeDivision adds new agents under the same key)
# continues to work without changes. For multi-cell outer sims, instantiate
# multiple EcoliProcess objects with distinct ``agent_id`` values.


_DEFAULT_CONFIG_PATH = "configs/default.json"


def _load_default_sim_config():
    """Load the vEcoli default sim config (configs/default.json).

    Cached because every EcoliProcess instance reads the same defaults.
    """
    import json
    import os

    if not hasattr(_load_default_sim_config, "_cache"):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        path = os.path.join(repo_root, _DEFAULT_CONFIG_PATH)
        with open(path, "r") as f:
            _load_default_sim_config._cache = json.load(f)
    return copy.deepcopy(_load_default_sim_config._cache)


def _resolve_sim_data(sim_data_path, parca_options):
    """Ensure a sim_data pickle exists; run parca if necessary.

    Returns the path to a usable simData.cPickle.

    Resolution order:
    1. If ``sim_data_path`` is set and the file exists, return it as-is.
    2. Else if ``parca_options`` is provided, run parca with those options
       and return the path to the resulting simData.cPickle (under
       ``parca_options['outdir']/kb/``).
    3. Else raise — caller must supply one or the other.
    """
    import os
    from wholecell.utils import constants
    from wholecell.utils.filepath import is_cloud_uri

    if sim_data_path:
        if is_cloud_uri(sim_data_path) or os.path.exists(sim_data_path):
            return sim_data_path

    if not parca_options:
        raise ValueError(
            "EcoliProcess: either 'sim_data_path' must point at an "
            "existing simData.cPickle, or 'parca_options' must be set "
            "so parca can be run."
        )

    # Run parca via the runscript helper. It writes simData.cPickle under
    # ``<outdir>/kb/`` and returns a content hash. We don't use the hash
    # here — first cut trusts the user's outdir as the cache key.
    from runscripts.parca import run_parca

    outdir = parca_options.get("outdir", "out")
    if not is_cloud_uri(outdir):
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
    parca_options = dict(parca_options)
    parca_options["outdir"] = outdir
    parca_options.setdefault(
        "cache_dir",
        os.path.join(outdir, "cache")
        if not is_cloud_uri(outdir)
        else os.path.join(os.getcwd(), "parca_cache"),
    )
    if not is_cloud_uri(parca_options["cache_dir"]):
        os.makedirs(parca_options["cache_dir"], exist_ok=True)

    resolved_path = os.path.join(outdir, "kb", constants.SERIALIZED_SIM_DATA_FILENAME)
    if os.path.exists(resolved_path):
        # Outdir already has a parca output — reuse it.
        return resolved_path

    print(f"[EcoliProcess] Running parca → {resolved_path}", flush=True)
    run_parca(parca_options)
    return resolved_path


def _build_inner_sim_config(user_config, sim_data_path):
    """Merge user config onto configs/default.json and resolve registries.

    Produces the dict consumed by ``build_ecoli_document``. Equivalent to
    what ``EcoliSim._run_composite`` does just before calling
    ``build_composite_native``: process names → classes, default topology,
    process_configs sourced from sim_data, etc.
    """
    sim_config = _load_default_sim_config()

    # Merge user-provided overrides on top. ``deep_merge`` mutates the
    # first arg, so we work on the defaults copy.
    for key, value in user_config.items():
        if value is None:
            continue
        if (
            key in sim_config
            and isinstance(sim_config[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(sim_config[key], value)
        else:
            sim_config[key] = value

    sim_config["sim_data_path"] = sim_data_path

    # Resolve process names → classes, topology overrides, process configs.
    # Reuse EcoliSim's helpers so registry lookup logic stays in one place.
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim(sim_config)
    sim.processes = sim._retrieve_processes(
        sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes
    )
    sim.topology = sim._retrieve_topology(
        sim.topology, sim.processes, sim.swap_processes, sim.log_updates
    )
    sim.process_configs = sim._retrieve_process_configs(
        sim.process_configs, sim.processes
    )
    return sim.config


def _ecoli_bridge(agent_id):
    """Bridge wires connecting external ports to internal cell stores.

    The cell state lives under ``agents.<agent_id>`` inside this Composite.
    """
    return {
        "inputs": {
            "external": ["agents", agent_id, "boundary", "external"],
            "media_id": ["agents", agent_id, "environment", "media_id"],
            "global_time": ["global_time"],
        },
        "outputs": {
            "exchange": ["agents", agent_id, "environment", "exchange"],
            "cell_mass": ["agents", agent_id, "listeners", "mass", "cell_mass"],
            "dry_mass": ["agents", agent_id, "listeners", "mass", "dry_mass"],
            "volume": ["agents", agent_id, "listeners", "mass", "volume"],
        },
    }


def _ecoli_interface():
    """Declared input/output schema for the wrapped cell.

    These are the ports an outer Composite wires up. Types match the
    internal stores they bridge to (see ExchangeData and Metabolism
    schemas in ecoli/processes/{environment/exchange_data,metabolism}.py).
    """
    return {
        "inputs": {
            "external": "map[quantity[millimolar]]",
            "media_id": "string",
            "global_time": "float",
        },
        "outputs": {
            "exchange": "map[integer]",
            "cell_mass": "float[fg]",
            "dry_mass": "float[fg]",
            "volume": "float[L]",
        },
    }


# Late import — Composite lives in process-bigraph and doesn't itself
# trigger any vEcoli-specific machinery, so it's safe to import at module
# top, but kept local to keep the module's import-time footprint similar
# to before this class was added.
from process_bigraph import Composite as _Composite


class EcoliProcess(_Composite):
    """Whole vEcoli model wrapped as a single process.

    Use this to embed E. coli in a larger simulation: instantiate it with
    a ``sim_data_path`` (or ``parca_options`` to run parca on demand) and
    wire its declared input/output ports to your outer composite's stores.

    Example::

        from ecoli.composites.ecoli_composite import EcoliProcess
        from process_bigraph import Composite
        from bigraph_schema import Core, BASE_TYPES

        core = Core(BASE_TYPES)
        # ... register process_bigraph + ECOLI_TYPES ...

        ecoli = EcoliProcess(
            {'sim_data_path': 'out/kb/simData.cPickle',
             'agent_id': '0', 'seed': 0},
            core=core)
        ecoli.run(60.0)  # advance 60 simulated seconds

    For embedding in a parent Composite, reference by class address
    (``local:!ecoli.composites.ecoli_composite.EcoliProcess``) and supply
    the same config_schema fields under the child's ``config`` key.
    """

    config_schema = {
        # --- sim_data sourcing (one of these is required at initialize) ---
        "sim_data_path": "maybe[string]",
        "parca_options": "maybe[tree[node]]",
        # --- cell identity ---
        "agent_id": "string",
        "seed": "integer",
        "time_step": "float",
        "initial_global_time": "float",
        # --- media / condition ---
        "media_id": "string",
        "fixed_media": "string",
        "condition": "string",
        "mar_regulon": "boolean",
        "amp_lysis": "boolean",
        # --- bulk overrides on the loaded sim config ---
        # Anything in here is deep-merged onto configs/default.json before
        # build_ecoli_document is called. Use this to add antibiotics
        # processes, custom topology, etc. (mirrors the JSON-config path).
        "sim_config": "tree[node]",
        # --- initial state options ---
        "initial_state": "tree[node]",
        "initial_state_file": "maybe[string]",
        "initial_state_overrides": "list[string]",
        # --- division (handled internally if true) ---
        "divide": "boolean",
        # --- pass-throughs to Composite ---
        "parallel_steps": "boolean",
        "parallel_workers": "maybe[integer]",
        "global_time_precision": "maybe[float]",
        # --- filled by ``initialize`` before delegating to Composite ---
        "state": "tree[node]",
        "schema": "schema",
        "interface": {"inputs": "schema", "outputs": "schema"},
        "bridge": {"inputs": "wires", "outputs": "wires"},
        "run_steps_on_init": "boolean",
    }

    def initialize(self, config=None):
        """Resolve sim_data, build the inner cell, configure the bridge."""
        # Ensure ECOLI_TYPES is on self.core. Required because the Ray
        # protocol's shadow constructs its template instance with a
        # fresh ``allocate_core()`` (no domain types registered), and
        # build_ecoli_document references vEcoli types like
        # ``sim_data_object_store`` that aren't in BASE_TYPES.
        # Idempotent for cores that already have the types.
        from ecoli.library.bigraph_types import ECOLI_TYPES

        try:
            self.core.register_types(ECOLI_TYPES)
        except Exception:
            pass  # already registered

        cfg = self._config

        # Collapse the user-facing keys into a sim_config dict that
        # build_ecoli_document understands.
        #
        # Filter out the empty-string / empty-dict defaults the
        # framework fills in for unset string/tree[node] fields. Those
        # are "not set" semantically; copying them would clobber the
        # real values from configs/default.json (e.g. condition='' would
        # overwrite condition='basal' and crash LoadSimData with an
        # unhelpful "process X is not known" because sim_data.condition_to_
        # doubling_time[''] raises KeyError, caught and re-raised by
        # get_config_by_name).
        user_config = dict(cfg.get("sim_config") or {})
        for key in (
            "agent_id",
            "seed",
            "time_step",
            "initial_global_time",
            "fixed_media",
            "condition",
            "mar_regulon",
            "amp_lysis",
            "initial_state",
            "initial_state_file",
            "initial_state_overrides",
            "divide",
        ):
            value = cfg.get(key)
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            if isinstance(value, dict) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            user_config[key] = value

        sim_data_path = _resolve_sim_data(
            cfg.get("sim_data_path"), cfg.get("parca_options")
        )

        agent_id = str(cfg.get("agent_id") or "0")

        # Build (or load) the inner state and stuff it into self._config so
        # Composite.initialize picks it up.
        initial_state_file = cfg.get("initial_state_file")
        if initial_state_file and os.path.isdir(initial_state_file):
            # Bundle reload path. Composite.load_bundle handles document
            # parsing; we replicate a minimal subset here so realize() runs
            # against the bundle's saved state.
            from process_bigraph.bundle import load_bundle

            document = load_bundle(initial_state_file, as_numpy=True)
            loaded_state = document.get("state", {})
            # Match v1: re-seed allocator_rng at each generation start.
            _reseed_allocator_rng(loaded_state, agent_id)
            cfg["state"] = loaded_state
            cfg.setdefault("schema", document.get("schema", {}))
        else:
            inner_sim_config = _build_inner_sim_config(user_config, sim_data_path)
            cfg["state"] = build_ecoli_document(self.core, inner_sim_config)

        cfg["bridge"] = _ecoli_bridge(agent_id)
        cfg["interface"] = _ecoli_interface()

        super().initialize(config)


# Late import: ``os`` is used by the bundle-reload branch above.
import os
