from typing import Any, Callable, TYPE_CHECKING

from ecoli.variants.variant_test import variant_test

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli

VARIANT_REGISTRY: dict[
    str, Callable[["SimulationDataEcoli", dict[str, Any]], "SimulationDataEcoli"]
] = {"variant_test": variant_test}
"""
Make sure to import and add new variants to this dictionary.
"""
