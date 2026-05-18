"""
Wrapper variant that composes
:py:mod:`~ecoli.variants.native_translation_perturbation` and
:py:mod:`~ecoli.variants.new_gene_internal_shift`. Lets a single config
control: an environmental condition, native-gene translation perturbations,
and new-gene (e.g. GFP) induction/knockout shifts at specific generations.
"""

from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np

from ecoli.variants.condition import apply_variant as condition_variant
from ecoli.variants.native_translation_perturbation import (
    apply_variant as native_perturbation_variant,
)
from ecoli.variants.new_gene_internal_shift import (
    apply_variant as new_gene_shift_variant,
)

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Compose native-gene translation perturbations with optional new-gene
    internal shifts and an environmental condition.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Optional environmental condition (passed to condition variant).
                "condition": str,
                # Optional native-gene translation perturbations.
                # See native_translation_perturbation.apply_variant.
                "perturbations": {"EG10527": 0.0, ...},
                # Optional new-gene internal shift parameters. Forwarded to
                # new_gene_internal_shift.apply_variant. The "condition" key
                # is injected so the underlying variant has what it needs.
                "new_gene_shift": {
                    "induction_gen": int,
                    "exp_trl_eff": {"exp": float, "trl_eff": float},
                    "knockout_gen": int (optional),
                }
            }

    Returns:
        Simulation data with the union of attributes modified by the
        delegated variants.
    """
    if "condition" in params:
        sim_data = condition_variant(sim_data, {"condition": params["condition"]})

    if "perturbations" in params:
        sim_data = native_perturbation_variant(
            sim_data, {"perturbations": params["perturbations"]}
        )

    if "new_gene_shift" in params:
        new_gene_params = dict(params["new_gene_shift"])
        # new_gene_internal_shift requires a condition; inherit if not given.
        if "condition" not in new_gene_params and "condition" in params:
            new_gene_params["condition"] = params["condition"]
        sim_data = new_gene_shift_variant(sim_data, new_gene_params)

    return sim_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_stub_with_new_gene():
    """
    Build a sim_data-shaped stub with both native genes and one new gene,
    enough to exercise both delegated variants. Avoids importing
    new_gene_internal_shift code paths that touch real
    ``adjust_new_gene_final_expression`` since that requires far more
    sim_data state. The test focuses on the wrapper's dispatch logic.
    """
    from ecoli.variants.native_translation_perturbation import _build_stub_sim_data

    sim_data = _build_stub_sim_data()
    return sim_data


def test_strain_design_native_only():
    sim_data = _build_stub_with_new_gene()
    apply_variant(
        sim_data,
        {"perturbations": {"EG_FAKE_001": 0.0, "EG_FAKE_003": 2.0}},
    )
    trl = sim_data.process.translation.translation_efficiencies_by_monomer
    assert trl[0] == 0.0
    assert trl[2] == 2.0 * 2.0


def test_strain_design_empty_params_is_noop():
    sim_data = _build_stub_with_new_gene()
    baseline = sim_data.process.translation.translation_efficiencies_by_monomer.copy()
    apply_variant(sim_data, {})
    assert np.array_equal(
        sim_data.process.translation.translation_efficiencies_by_monomer, baseline
    )


def test_strain_design_order_independence_of_native_perturbations():
    """
    Two sequences of native-only edits yielding the same final dict should
    produce the same translation_efficiencies_by_monomer, regardless of
    iteration order or split into multiple calls. Confirms the
    absolute-write semantics documented in the docstring.
    """
    final_dict = {"EG_FAKE_001": 0.0, "EG_FAKE_002": 0.5, "EG_FAKE_003": 3.0}

    sim_a = _build_stub_with_new_gene()
    apply_variant(sim_a, {"perturbations": final_dict})

    sim_b = _build_stub_with_new_gene()
    apply_variant(sim_b, {"perturbations": {"EG_FAKE_003": 3.0}})
    apply_variant(sim_b, {"perturbations": {"EG_FAKE_002": 0.5}})
    apply_variant(sim_b, {"perturbations": {"EG_FAKE_001": 0.0}})

    assert np.array_equal(
        sim_a.process.translation.translation_efficiencies_by_monomer,
        sim_b.process.translation.translation_efficiencies_by_monomer,
    )


def _stub_delegates(sd_module):
    """
    Replace condition_variant and new_gene_shift_variant on the strain_design
    module with capture stubs. Returns (captured_dict, restore_callable).
    """
    captured = {}

    def fake_condition(sim_data_in, params_in):
        captured.setdefault("condition_calls", []).append(deepcopy(params_in))
        sim_data_in.condition = params_in["condition"]
        return sim_data_in

    def fake_new_gene(sim_data_in, params_in):
        captured.setdefault("new_gene_calls", []).append(deepcopy(params_in))
        sim_data_in.internal_shift_dict = {"stub": True}
        return sim_data_in

    original_condition = sd_module.condition_variant
    original_new_gene = sd_module.new_gene_shift_variant
    sd_module.condition_variant = fake_condition
    sd_module.new_gene_shift_variant = fake_new_gene

    def restore():
        sd_module.condition_variant = original_condition
        sd_module.new_gene_shift_variant = original_new_gene

    return captured, restore


def test_strain_design_new_gene_branch_dispatches():
    """
    The wrapper should pass new_gene_shift params (plus injected condition)
    to new_gene_internal_shift, and still apply native perturbations.
    """
    import ecoli.variants.strain_design as sd

    sim_data = _build_stub_with_new_gene()
    captured, restore = _stub_delegates(sd)
    try:
        apply_variant(
            sim_data,
            {
                "condition": "basal",
                "perturbations": {"EG_FAKE_001": 0.0},
                "new_gene_shift": {
                    "induction_gen": 2,
                    "exp_trl_eff": {"exp": 1e6, "trl_eff": 1.0},
                },
            },
        )
    finally:
        restore()

    # Native branch ran for real:
    assert sim_data.process.translation.translation_efficiencies_by_monomer[0] == 0.0
    # Condition branch was invoked once:
    assert captured["condition_calls"] == [{"condition": "basal"}]
    # New-gene branch received the merged params with inherited condition:
    assert len(captured["new_gene_calls"]) == 1
    new_gene_call = captured["new_gene_calls"][0]
    assert new_gene_call["induction_gen"] == 2
    assert new_gene_call["condition"] == "basal"
    # Internal-shift bookkeeping from the (stubbed) delegate is present:
    assert sim_data.internal_shift_dict == {"stub": True}


def test_strain_design_does_not_clobber_explicit_new_gene_condition():
    import ecoli.variants.strain_design as sd

    sim_data = _build_stub_with_new_gene()
    captured, restore = _stub_delegates(sd)
    try:
        apply_variant(
            sim_data,
            {
                "condition": "basal",
                "new_gene_shift": {
                    "condition": "with_aa",
                    "induction_gen": 1,
                    "exp_trl_eff": {"exp": 1.0, "trl_eff": 1.0},
                },
            },
        )
    finally:
        restore()

    assert captured["new_gene_calls"][0]["condition"] == "with_aa"
