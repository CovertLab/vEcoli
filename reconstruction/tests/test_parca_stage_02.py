"""
Regression tests for Stage 2 (input_adjustments) of the ParCa pipeline.

Uses cached intermediates at reconstruction/sim_data/intermediates/ to verify
that the pure-function implementation produces identical results to the legacy
in-place mutations.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_02.py -v
"""

import os
import pickle

import numpy as np
import pytest

INTERMEDIATES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "sim_data",
    "intermediates",
)

# Skip all tests if intermediates are not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(INTERMEDIATES_DIR, "sim_data_initialize.cPickle")),
    reason="Cached intermediates not found — run the full ParCa pipeline first",
)


@pytest.fixture(scope="module")
def sim_data_before():
    with open(
        os.path.join(INTERMEDIATES_DIR, "sim_data_initialize.cPickle"), "rb"
    ) as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def cell_specs_before():
    with open(
        os.path.join(INTERMEDIATES_DIR, "cell_specs_initialize.cPickle"), "rb"
    ) as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def sim_data_after():
    with open(
        os.path.join(INTERMEDIATES_DIR, "sim_data_input_adjustments.cPickle"), "rb"
    ) as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def pure_output(sim_data_before, cell_specs_before):
    """Run the pure compute pipeline and return the output."""
    from reconstruction.ecoli.parca.stage_02_input_adjustments import (
        extract_input,
        compute_input_adjustments,
    )

    inp = extract_input(sim_data_before, cell_specs_before)
    return compute_input_adjustments(inp)


class TestInputAdjustmentsPure:
    """Test that the pure compute function matches the legacy output."""

    def test_translation_efficiencies(self, pure_output, sim_data_after):
        np.testing.assert_allclose(
            pure_output.translation_efficiencies,
            sim_data_after.process.translation.translation_efficiencies_by_monomer,
            rtol=1e-12,
            err_msg="Translation efficiencies do not match legacy output",
        )

    def test_basal_rna_expression(self, pure_output, sim_data_after):
        np.testing.assert_allclose(
            pure_output.basal_rna_expression,
            sim_data_after.process.transcription.rna_expression["basal"],
            rtol=1e-12,
            err_msg="Basal RNA expression does not match legacy output",
        )

    def test_rna_deg_rates(self, pure_output, sim_data_after):
        np.testing.assert_allclose(
            pure_output.rna_deg_rates,
            sim_data_after.process.transcription.rna_data.struct_array["deg_rate"],
            rtol=1e-12,
            err_msg="RNA degradation rates do not match legacy output",
        )

    def test_cistron_deg_rates(self, pure_output, sim_data_after):
        np.testing.assert_allclose(
            pure_output.cistron_deg_rates,
            sim_data_after.process.transcription.cistron_data.struct_array["deg_rate"],
            rtol=1e-12,
            err_msg="Cistron degradation rates do not match legacy output",
        )

    def test_protein_deg_rates(self, pure_output, sim_data_after):
        np.testing.assert_allclose(
            pure_output.protein_deg_rates,
            sim_data_after.process.translation.monomer_data.struct_array["deg_rate"],
            rtol=1e-12,
            err_msg="Protein degradation rates do not match legacy output",
        )

    def test_tf_conditions_not_filtered(self, pure_output, sim_data_after):
        """In non-debug mode, tf_to_active_inactive_conditions should not be modified."""
        assert pure_output.tf_to_active_inactive_conditions is None


class TestInputAdjustmentsRoundtrip:
    """Test that extract -> compute -> merge produces identical sim_data."""

    def test_full_roundtrip(self, sim_data_before, cell_specs_before, sim_data_after):
        """The full extract/compute/merge cycle must match the legacy output."""
        import copy

        from reconstruction.ecoli.parca.stage_02_input_adjustments import (
            extract_input,
            compute_input_adjustments,
            merge_output,
        )

        # Deep copy so we don't pollute the fixture
        sd = copy.deepcopy(sim_data_before)
        cs = copy.deepcopy(cell_specs_before)

        inp = extract_input(sd, cs)
        out = compute_input_adjustments(inp)
        merge_output(sd, cs, out)

        # Compare all output fields
        np.testing.assert_allclose(
            sd.process.translation.translation_efficiencies_by_monomer,
            sim_data_after.process.translation.translation_efficiencies_by_monomer,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            sd.process.transcription.rna_expression["basal"],
            sim_data_after.process.transcription.rna_expression["basal"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            sd.process.transcription.rna_data.struct_array["deg_rate"],
            sim_data_after.process.transcription.rna_data.struct_array["deg_rate"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            sd.process.transcription.cistron_data.struct_array["deg_rate"],
            sim_data_after.process.transcription.cistron_data.struct_array["deg_rate"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            sd.process.translation.monomer_data.struct_array["deg_rate"],
            sim_data_after.process.translation.monomer_data.struct_array["deg_rate"],
            rtol=1e-12,
        )


class TestSubFunctions:
    """Unit tests for individual pure sub-functions."""

    def test_adjust_translation_efficiencies(self):
        from reconstruction.ecoli.parca.stage_02_input_adjustments import (
            adjust_translation_efficiencies,
        )

        ids = np.array(["A[c]", "B[c]", "C[c]"])
        eff = np.array([1.0, 2.0, 3.0])
        adj = {"A[c]": 2.0, "C[c]": 0.5}
        result = adjust_translation_efficiencies(ids, eff, adj)
        np.testing.assert_array_equal(result, [2.0, 2.0, 1.5])
        # Original should be unchanged
        np.testing.assert_array_equal(eff, [1.0, 2.0, 3.0])

    def test_balance_translation_efficiencies(self):
        from reconstruction.ecoli.parca.stage_02_input_adjustments import (
            balance_translation_efficiencies,
        )

        ids = np.array(["A[c]", "B[c]", "C[c]", "D[c]"])
        eff = np.array([1.0, 3.0, 5.0, 7.0])
        groups = [["A", "C"]]  # mean of A(1.0) and C(5.0) = 3.0
        result = balance_translation_efficiencies(ids, eff, groups)
        np.testing.assert_array_equal(result, [3.0, 3.0, 3.0, 7.0])

    def test_adjust_protein_deg_rates(self):
        from reconstruction.ecoli.parca.stage_02_input_adjustments import (
            adjust_protein_deg_rates,
        )

        ids = np.array(["X[c]", "Y[c]"])
        rates = np.array([0.1, 0.2])
        adj = {"Y[c]": 3.0}
        result = adjust_protein_deg_rates(ids, rates, adj)
        np.testing.assert_allclose(result, [0.1, 0.6])
