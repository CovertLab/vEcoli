"""
Variant module that scales the translation efficiency of arbitrary native
genes. Supports knockouts (multiplier = 0), knockdowns (0 < m < 1), and
overexpressions (m > 1). Edits happen at the translation level only:
``sim_data.process.translation.translation_efficiencies_by_monomer``.

Caveats users should be aware of:

- Edits are translation-only. mRNAs continue to be transcribed and consume
  RNAP capacity even when their translation efficiency is set to zero.
- Polycistronic transcripts (on operons) share a single mRNA; scaling one
  monomer's efficiency does not affect other monomers on the same operon.
- Genes whose cistron has no associated monomer (non-coding RNAs) cannot be
  perturbed at the translation level and are rejected with a clear error.
"""

from typing import Any, TYPE_CHECKING, cast
import warnings

import numpy as np

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def resolve_native_gene_indices(
    sim_data: "SimulationDataEcoli", ecocyc_ids: list[str]
) -> list[int]:
    """
    Resolve a list of EcoCyc gene IDs (e.g. ``EG10527``) to their
    corresponding indices in
    ``sim_data.process.translation.monomer_data``.

    The join is: ``cistron_data["gene_id"] -> cistron_data["id"] ->
    monomer_data["cistron_id"] -> monomer_data`` index.

    Args:
        sim_data: Simulation data
        ecocyc_ids: List of EcoCyc gene IDs to resolve

    Returns:
        List of monomer indices, in the same order as ``ecocyc_ids``.

    Raises:
        ValueError: If any EcoCyc IDs are unknown, or if any resolve to a
            cistron that has no associated monomer (non-coding RNA).
    """
    cistron_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_data = sim_data.process.translation.monomer_data.struct_array

    gene_id_to_cistron_id = dict(zip(cistron_data["gene_id"], cistron_data["id"]))
    cistron_id_to_monomer_idx = {
        cistron_id: i for i, cistron_id in enumerate(monomer_data["cistron_id"])
    }

    unknown_ids: list[str] = []
    non_coding_ids: list[str] = []
    indices: list[int] = []
    for ecocyc_id in ecocyc_ids:
        cistron_id = gene_id_to_cistron_id.get(ecocyc_id)
        if cistron_id is None:
            unknown_ids.append(ecocyc_id)
            continue
        monomer_idx = cistron_id_to_monomer_idx.get(cistron_id)
        if monomer_idx is None:
            non_coding_ids.append(ecocyc_id)
            continue
        indices.append(int(monomer_idx))

    errors: list[str] = []
    if unknown_ids:
        errors.append(
            f"Unknown EcoCyc gene IDs (not found in cistron_data['gene_id']): "
            f"{unknown_ids}"
        )
    if non_coding_ids:
        errors.append(
            f"EcoCyc gene IDs resolved to cistrons with no associated monomer "
            f"(likely non-coding RNAs, cannot be perturbed at the translation "
            f"level): {non_coding_ids}"
        )
    if errors:
        raise ValueError("; ".join(errors))

    return indices


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Scale translation efficiencies of native genes by user-specified factors.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                # Mapping of EcoCyc gene ID -> multiplier.
                # multiplier == 0    -> knockout
                # 0 < multiplier < 1 -> knockdown
                # multiplier > 1     -> overexpression
                "perturbations": {"EG10527": 0.0, "EG11015": 2.5, ...},
            }

    Returns:
        Simulation data with the following attribute modified::

            sim_data.process.translation.translation_efficiencies_by_monomer
    """
    perturbations = params.get("perturbations")
    if perturbations is None:
        return sim_data
    if not isinstance(perturbations, dict):
        raise TypeError(
            f"'perturbations' must be a dict of {{EcoCyc_id: multiplier}}, got "
            f"{type(perturbations).__name__}"
        )
    if not perturbations:
        return sim_data

    bad: list[tuple[str, Any]] = []
    for gene_id, multiplier in perturbations.items():
        if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
            bad.append((gene_id, multiplier))
            continue
        if multiplier < 0:
            bad.append((gene_id, multiplier))
    if bad:
        raise ValueError(
            f"Multipliers must be non-negative numbers. Offending entries: {bad}"
        )

    no_ops = [g for g, m in perturbations.items() if float(m) == 1.0]
    if no_ops:
        warnings.warn(
            f"Multiplier of 1.0 has no effect; these entries are no-ops: {no_ops}",
            stacklevel=2,
        )

    ecocyc_ids = list(perturbations.keys())
    monomer_indices = resolve_native_gene_indices(sim_data, ecocyc_ids)

    translation = sim_data.process.translation
    trl_eff = translation.translation_efficiencies_by_monomer
    # Snapshot original efficiencies on first application so repeated calls
    # multiply against the unperturbed baseline rather than compounding.
    if not hasattr(translation, "_native_perturbation_baseline"):
        translation._native_perturbation_baseline = cast(np.ndarray, trl_eff).copy()
    baseline = translation._native_perturbation_baseline
    for ecocyc_id, monomer_idx in zip(ecocyc_ids, monomer_indices):
        trl_eff[monomer_idx] = baseline[monomer_idx] * float(perturbations[ecocyc_id])

    return sim_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_stub_sim_data():
    """
    Build a minimal sim_data-shaped stub for unit testing. Populates only the
    fields touched by this module: cistron_data, monomer_data, and
    translation_efficiencies_by_monomer. Includes one non-coding cistron
    (no corresponding monomer) to exercise that error path.
    """

    class Stub:
        pass

    cistron_dtype = [("id", "U16"), ("gene_id", "U16")]
    cistrons = np.array(
        [
            ("lacZ", "EG_FAKE_001"),
            ("recA", "EG_FAKE_002"),
            ("rpoB", "EG_FAKE_003"),
            ("araD", "EG_FAKE_004"),
            ("rrnA", "EG_FAKE_NC1"),  # non-coding
        ],
        dtype=cistron_dtype,
    )

    monomer_dtype = [("id", "U24"), ("cistron_id", "U16")]
    monomers = np.array(
        [
            ("lacZ[c]", "lacZ"),
            ("recA[c]", "recA"),
            ("rpoB[c]", "rpoB"),
            ("araD[c]", "araD"),
        ],
        dtype=monomer_dtype,
    )

    class _Holder:
        pass

    sim_data = Stub()
    sim_data.process = Stub()
    sim_data.process.transcription = Stub()
    sim_data.process.translation = Stub()
    sim_data.process.transcription.cistron_data = _Holder()
    sim_data.process.transcription.cistron_data.struct_array = cistrons
    sim_data.process.translation.monomer_data = _Holder()
    sim_data.process.translation.monomer_data.struct_array = monomers
    sim_data.process.translation.translation_efficiencies_by_monomer = np.array(
        [0.5, 1.0, 2.0, 4.0], dtype=np.float64
    )
    return sim_data


def test_resolve_native_gene_indices_happy_path():
    sim_data = _build_stub_sim_data()
    indices = resolve_native_gene_indices(
        sim_data, ["EG_FAKE_003", "EG_FAKE_001", "EG_FAKE_004"]
    )
    assert indices == [2, 0, 3]


def test_resolve_native_gene_indices_unknown_id_raises():
    sim_data = _build_stub_sim_data()
    try:
        resolve_native_gene_indices(sim_data, ["EG_FAKE_001", "EG_NOPE_999"])
    except ValueError as e:
        assert "EG_NOPE_999" in str(e)
        assert "Unknown" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_native_gene_indices_non_coding_raises():
    sim_data = _build_stub_sim_data()
    try:
        resolve_native_gene_indices(sim_data, ["EG_FAKE_NC1"])
    except ValueError as e:
        assert "EG_FAKE_NC1" in str(e)
        assert "non-coding" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_resolve_native_gene_indices_reports_all_bad_ids_at_once():
    sim_data = _build_stub_sim_data()
    try:
        resolve_native_gene_indices(sim_data, ["EG_NOPE_1", "EG_FAKE_NC1", "EG_NOPE_2"])
    except ValueError as e:
        msg = str(e)
        assert "EG_NOPE_1" in msg
        assert "EG_NOPE_2" in msg
        assert "EG_FAKE_NC1" in msg
    else:
        raise AssertionError("expected ValueError")


def test_knockout_sets_zero():
    sim_data = _build_stub_sim_data()
    apply_variant(sim_data, {"perturbations": {"EG_FAKE_002": 0.0}})
    trl = sim_data.process.translation.translation_efficiencies_by_monomer
    assert trl[1] == 0.0
    # untouched
    assert trl[0] == 0.5 and trl[2] == 2.0 and trl[3] == 4.0


def test_knockdown_scales():
    sim_data = _build_stub_sim_data()
    apply_variant(sim_data, {"perturbations": {"EG_FAKE_001": 0.3}})
    trl = sim_data.process.translation.translation_efficiencies_by_monomer
    assert abs(trl[0] - 0.5 * 0.3) < 1e-12
    assert trl[1] == 1.0 and trl[2] == 2.0 and trl[3] == 4.0


def test_overexpression_scales():
    sim_data = _build_stub_sim_data()
    apply_variant(sim_data, {"perturbations": {"EG_FAKE_003": 5.0}})
    trl = sim_data.process.translation.translation_efficiencies_by_monomer
    assert trl[2] == 2.0 * 5.0
    assert trl[0] == 0.5 and trl[1] == 1.0 and trl[3] == 4.0


def test_combined_perturbations():
    sim_data = _build_stub_sim_data()
    apply_variant(
        sim_data,
        {
            "perturbations": {
                "EG_FAKE_001": 0.0,  # knockout
                "EG_FAKE_002": 0.2,  # knockdown
                "EG_FAKE_003": 3.0,  # overexpression
            }
        },
    )
    trl = sim_data.process.translation.translation_efficiencies_by_monomer
    assert trl[0] == 0.0
    assert abs(trl[1] - 1.0 * 0.2) < 1e-12
    assert trl[2] == 2.0 * 3.0
    # untouched
    assert trl[3] == 4.0


def test_negative_multiplier_raises():
    sim_data = _build_stub_sim_data()
    try:
        apply_variant(sim_data, {"perturbations": {"EG_FAKE_001": -1.0}})
    except ValueError as e:
        assert "non-negative" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_perturbations_must_be_dict():
    sim_data = _build_stub_sim_data()
    try:
        apply_variant(sim_data, {"perturbations": [("EG_FAKE_001", 0.5)]})
    except TypeError as e:
        assert "dict" in str(e)
    else:
        raise AssertionError("expected TypeError")


def test_idempotent_on_repeated_apply():
    sim_data_a = _build_stub_sim_data()
    sim_data_b = _build_stub_sim_data()
    params = {"perturbations": {"EG_FAKE_001": 0.3, "EG_FAKE_003": 5.0}}
    apply_variant(sim_data_a, params)
    apply_variant(sim_data_b, params)
    apply_variant(sim_data_b, params)
    assert np.array_equal(
        sim_data_a.process.translation.translation_efficiencies_by_monomer,
        sim_data_b.process.translation.translation_efficiencies_by_monomer,
    )


import pytest  # noqa: E402


@pytest.mark.slow
@pytest.mark.noci
def test_integration_with_real_sim_data():
    """
    Optional integration test: if a real ``simData.cPickle`` is available,
    apply a perturbation against a known real EcoCyc gene and verify the
    corresponding monomer slot moved.

    Skipped when no pickle path is available.
    """
    import os
    import pickle

    sim_data_path = os.environ.get("VECOLI_SIM_DATA_PATH", "out/kb/simData.cPickle")
    if not os.path.exists(sim_data_path):
        pytest.skip(
            f"real simData.cPickle not found at {sim_data_path}; "
            "set VECOLI_SIM_DATA_PATH to enable"
        )

    with open(sim_data_path, "rb") as f:
        sim_data = pickle.load(f)

    # Pick the first real coding cistron we can find for a clean test.
    cistron_data = sim_data.process.transcription.cistron_data.struct_array
    monomer_data = sim_data.process.translation.monomer_data.struct_array
    cistron_id_set = set(monomer_data["cistron_id"])
    ecocyc_id = None
    for row in cistron_data:
        if row["id"] in cistron_id_set:
            ecocyc_id = row["gene_id"]
            break
    assert ecocyc_id is not None, "expected at least one coding cistron"

    [monomer_idx] = resolve_native_gene_indices(sim_data, [ecocyc_id])
    baseline = float(
        sim_data.process.translation.translation_efficiencies_by_monomer[monomer_idx]
    )
    apply_variant(sim_data, {"perturbations": {ecocyc_id: 0.25}})
    after = float(
        sim_data.process.translation.translation_efficiencies_by_monomer[monomer_idx]
    )
    assert abs(after - baseline * 0.25) < 1e-9


def test_empty_perturbations_is_noop():
    sim_data = _build_stub_sim_data()
    baseline = sim_data.process.translation.translation_efficiencies_by_monomer.copy()
    apply_variant(sim_data, {"perturbations": {}})
    assert np.array_equal(
        sim_data.process.translation.translation_efficiencies_by_monomer, baseline
    )
    apply_variant(sim_data, {})
    assert np.array_equal(
        sim_data.process.translation.translation_efficiencies_by_monomer, baseline
    )
