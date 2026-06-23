"""
================
Bigraph Bridge
================

Bridge base classes that inherit from both vivarium-core and
process-bigraph, providing a unified interface for vEcoli processes.
The "Bigraph" prefix marks these as the bridge: a vivarium-style API
running on the process-bigraph runtime. Plain vivarium processes still
work in the v1 engine; ``BigraphStep`` / ``BigraphProcess`` work in both.

Each vEcoli process defines:

- ``config_schema`` — typed configuration (replaces ``defaults``)
- ``inputs()`` — typed input port schema
- ``outputs()`` — typed output port schema
- ``update(state, interval)`` — primary computation method

The vivarium interface (``defaults``, ``ports_schema``,
``next_update``) is derived automatically for backward compatibility.

In v2 (composite engine), config is accessed via ``self.config``.
In v1 (vivarium engine), config is accessed via ``self.parameters``.
The bridge ensures both are available regardless of which path was used.
"""

from bigraph_schema.methods import default as schema_default
from vivarium.core.process import Step as VivariumStep, Process as VivariumProcess
from process_bigraph import (
    Step as ProcessBigraphStep,
    Process as ProcessBigraphProcess,
)


def normalize_vivarium_update(update):
    """Translate vivarium updater-dicts into plain bigraph updates.

    Vivarium lets a process return ``{'_value': v, '_updater': name}`` (and
    related schema keys like ``_reduce``) at a leaf to override the store's
    default updater. The process-bigraph engine has no notion of this
    convention, so such a dict would be applied LITERALLY — leaking the
    schema keys into the state (e.g. ``environment.media_id`` becomes the
    dict ``{'_value': 'minimal', '_updater': 'set'}`` instead of the string
    ``'minimal'``, which then breaks any consumer that treats it as a
    scalar). We recursively replace each such leaf-dict with its ``_value``.

    bigraph leaf ``apply`` REPLACES strings/booleans and ACCUMULATES numbers,
    which matches vivarium ``set`` (string ports) and ``accumulate`` (numeric
    ports) semantics. Numeric ``set`` is approximated as accumulate; this is
    acceptable for the run-clean milestone (numerical fidelity is a later
    concern). The walk only rebuilds dicts that actually contain an updater-
    dict, so updates with no such wrappers pass through untouched.
    """
    if not isinstance(update, dict):
        return update
    if '_value' in update:
        return normalize_vivarium_update(update['_value'])
    rebuilt = None
    for key, value in update.items():
        new_value = normalize_vivarium_update(value)
        if new_value is not value:
            if rebuilt is None:
                rebuilt = dict(update)
            rebuilt[key] = new_value
    return rebuilt if rebuilt is not None else update


class BigraphStep(VivariumStep, ProcessBigraphStep):
    """Base class for vEcoli steps.

    Subclasses implement ``update(state, interval)`` as the primary
    computation.  ``next_update(timestep, states)`` is provided for
    vivarium Engine compatibility and delegates to ``update()``.
    """

    # Base config fields inherited by all vEcoli steps/processes.
    # Vivarium injects 'timestep' automatically; declaring it here
    # ensures it survives realize/fill.
    config_schema = {
        'timestep': 'float{1.0}',
    }
    _output_ports = None
    _input_only_ports = None

    @classmethod
    def _defaults_from_schema(cls):
        """Derive vivarium-style defaults dict from config_schema.

        Handles both dict specs ``{'_type': 'float', '_default': 0.5}``
        and inline specs ``'float{0.5}'``.
        """
        if not cls.config_schema:
            return {}
        result = {}
        for key, spec in cls.config_schema.items():
            if isinstance(spec, dict) and '_default' in spec:
                result[key] = spec['_default']
            elif isinstance(spec, str) and '{' in spec:
                type_str, _, default_str = spec.partition('{')
                default_str = default_str.rstrip('}')
                type_str = type_str.strip()
                if type_str == 'float':
                    result[key] = float(default_str)
                elif type_str == 'integer':
                    result[key] = int(default_str)
                elif type_str == 'boolean':
                    result[key] = default_str.lower() in ('true', '1', 'yes')
                elif type_str == 'string':
                    result[key] = default_str
                else:
                    result[key] = default_str
        return result

    def __init__(self, config=None, core=None):
        # realize_link calls edge_class(config, core).
        # vivarium calls Class(parameters_dict).
        # Both pass config as the first positional arg.

        if self.config_schema and not self.__class__.__dict__.get('defaults'):
            self.__class__.defaults = self._defaults_from_schema()

        # vivarium init — sets self.parameters (merges with defaults)
        VivariumStep.__init__(self, parameters=config)

        # process-bigraph init — sets self.config via Edge.__init__
        if core is not None:
            ProcessBigraphStep.__init__(self, config=config or {}, core=core)
        else:
            # Ensure self.config is available even without core.
            # Use self.parameters (post-merge with defaults) so
            # self.config and self.parameters are always equivalent.
            self._config = self.parameters

    def perform_update(self, state):
        """Gate for step execution in v2 composite engine.

        Returns True if the step should run, False to skip.
        Default: always run. Override in subclasses to implement
        variable timestepping or conditional execution.

        Named differently from v1's update_condition to avoid
        collision — both methods coexist on dual-inheriting classes.
        """
        return True

    def invoke(self, state, interval=None):
        """Check perform_update before running.

        In v1 vivarium, the engine checked update_condition before
        calling next_update. In v2, invoke is called unconditionally
        by the step network — so we gate here via perform_update.
        """
        from process_bigraph.composite import SyncUpdate
        if not self.perform_update(state):
            return SyncUpdate({})
        update = self.update(state)
        return SyncUpdate(update)

    def next_update(self, timestep, states):
        """vivarium Engine entry point — delegates to update()."""
        return self.update(states, timestep)

    def update(self, state, interval=None):
        """process-bigraph entry point — subclasses override this.

        If a subclass overrides ``next_update`` (vivarium-style) but not
        ``update``, delegate to the subclass's ``next_update`` so the
        composite engine path also picks up the logic. We skip the
        delegation when ``next_update`` is BigraphStep's own version (which
        just calls ``update``) to avoid infinite recursion.
        """
        cls = type(self)
        _delegation_bases = (BigraphStep, BigraphProcess)
        for klass in cls.__mro__:
            if 'next_update' in klass.__dict__:
                if klass not in _delegation_bases:
                    return normalize_vivarium_update(
                        klass.next_update(self, interval or 0, state))
                break
        return {}


class BigraphProcess(VivariumProcess, ProcessBigraphProcess):
    """Base class for vEcoli processes (time-driven, with interval).

    Same bridge pattern as BigraphStep but for temporal processes.
    """

    config_schema = {
        'timestep': 'float{1.0}',
    }
    _output_ports = None

    _defaults_from_schema = BigraphStep._defaults_from_schema

    def __init__(self, config=None, core=None):
        if self.config_schema and not self.__class__.__dict__.get('defaults'):
            self.__class__.defaults = self._defaults_from_schema()

        VivariumProcess.__init__(self, parameters=config)

        if core is not None:
            ProcessBigraphProcess.__init__(self, config=config or {}, core=core)
        else:
            self._config = self.parameters

    def next_update(self, timestep, states):
        return self.update(states, timestep)

    def update(self, state, interval):
        """Same delegation as BigraphStep.update."""
        cls = type(self)
        _delegation_bases = (BigraphStep, BigraphProcess)
        for klass in cls.__mro__:
            if 'next_update' in klass.__dict__:
                if klass not in _delegation_bases:
                    return normalize_vivarium_update(
                        klass.next_update(self, interval or 0, state))
                break
        return {}

    def calculate_timestep(self, interval_or_state, state=None):
        """Bridge both signatures:
        vivarium: calculate_timestep(states)
        process-bigraph: calculate_timestep(interval, state)
        """
        if state is None:
            return self.parameters.get('timestep', 1.0)
        return interval_or_state
