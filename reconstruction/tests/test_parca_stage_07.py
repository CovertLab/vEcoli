"""
Tests for Stage 7 (adjust_promoters) of the ParCa pipeline.

Unit tests for imports and data types, smoke tests using cached
intermediates from stage 6, and regression tests comparing outputs
against cached intermediates from stage 7.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_07.py -v
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


class TestImports:
    """Verify all stage 7 modules can be imported."""

    def test_import_stage_module(self):
        from reconstruction.ecoli.parca.stage_07_adjust_promoters import (
            extract_input,
            compute_adjust_promoters,
            merge_output,
        )

        assert callable(extract_input)
        assert callable(compute_adjust_promoters)
        assert callable(merge_output)

    def test_import_promoter_fitting_functions(self):
        from reconstruction.ecoli.parca_promoter_fitting import (
            fitLigandConcentrations,
            calculateRnapRecruitment,
        )

        assert callable(fitLigandConcentrations)
        assert callable(calculateRnapRecruitment)


class TestDataTypes:
    """Verify dataclass instantiation works."""

    def test_input_creation(self):
        from reconstruction.ecoli.parca._types import AdjustPromotersInput

        inp = AdjustPromotersInput(
            sim_data_ref=None,
            cell_specs_ref={},
        )
        assert inp.sim_data_ref is None
        assert inp.cell_specs_ref == {}

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import AdjustPromotersOutput

        out = AdjustPromotersOutput(
            basal_prob=np.array([0.1, 0.2, 0.3]),
            delta_prob={
                "deltaI": np.array([0, 1]),
                "deltaJ": np.array([0, 0]),
                "deltaV": np.array([0.5, -0.3]),
                "shape": (3, 1),
            },
        )
        assert len(out.basal_prob) == 3
        assert "deltaI" in out.delta_prob


class TestAdjustPromotersSmokeTest:
    """Smoke test using promoter_binding intermediates."""

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_promoter_binding.cPickle")
        if not os.path.exists(path):
            pytest.skip("promoter_binding intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_promoter_binding.cPickle")
        if not os.path.exists(path):
            pytest.skip("promoter_binding intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_extract_input(self, sim_data_before, cell_specs_before):
        """Verify extract_input runs without error on real data."""
        from reconstruction.ecoli.parca.stage_07_adjust_promoters import (
            extract_input,
        )

        inp = extract_input(sim_data_before, cell_specs_before)

        assert inp.sim_data_ref is sim_data_before
        assert inp.cell_specs_ref is cell_specs_before

    def test_cell_specs_has_r_values(self, cell_specs_before):
        """Verify cell_specs has r_vector and r_columns from stage 6."""
        assert "r_vector" in cell_specs_before["basal"]
        assert "r_columns" in cell_specs_before["basal"]

    def test_sim_data_has_pPromoterBound(self, sim_data_before):
        """Verify sim_data has pPromoterBound from stage 6."""
        assert hasattr(sim_data_before, "pPromoterBound")
        assert isinstance(sim_data_before.pPromoterBound, dict)


class TestAdjustPromotersRegression:
    """Regression tests against cached intermediates.

    These tests verify that the after-state has the expected structure
    and properties.  The full compute is NOT re-run here because
    fitLigandConcentrations accesses deep sim_data process objects.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_promoter_binding.cPickle")
        if not os.path.exists(path):
            pytest.skip("promoter_binding intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def sim_data_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_adjust_promoters.cPickle")
        if not os.path.exists(path):
            pytest.skip("adjust_promoters intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_adjust_promoters.cPickle")
        if not os.path.exists(path):
            pytest.skip("adjust_promoters intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_basal_prob_set(self, sim_data_after):
        """Verify basal_prob is set on transcription_regulation."""
        reg = sim_data_after.process.transcription_regulation
        assert hasattr(reg, "basal_prob")
        assert isinstance(reg.basal_prob, np.ndarray)
        assert len(reg.basal_prob) > 0

    def test_basal_prob_non_negative(self, sim_data_after):
        """Verify basal_prob values are non-negative."""
        basal_prob = sim_data_after.process.transcription_regulation.basal_prob
        assert np.all(basal_prob >= 0), "Negative basal_prob values found"

    def test_delta_prob_set(self, sim_data_after):
        """Verify delta_prob is set on transcription_regulation."""
        reg = sim_data_after.process.transcription_regulation
        assert hasattr(reg, "delta_prob")
        dp = reg.delta_prob
        assert "deltaI" in dp
        assert "deltaJ" in dp
        assert "deltaV" in dp
        assert "shape" in dp

    def test_delta_prob_structure(self, sim_data_after):
        """Verify delta_prob arrays have consistent lengths."""
        dp = sim_data_after.process.transcription_regulation.delta_prob
        assert len(dp["deltaI"]) == len(dp["deltaJ"])
        assert len(dp["deltaI"]) == len(dp["deltaV"])
        assert len(dp["deltaI"]) > 0

    def test_delta_prob_shape(self, sim_data_after):
        """Verify delta_prob shape matches expected dimensions."""
        reg = sim_data_after.process.transcription_regulation
        dp = reg.delta_prob
        n_tus = len(sim_data_after.process.transcription.rna_data["id"])
        n_tfs = len(reg.tf_ids)
        assert dp["shape"] == (n_tus, n_tfs)

    def test_basal_prob_length(self, sim_data_after):
        """Verify basal_prob length matches number of TUs."""
        n_tus = len(sim_data_after.process.transcription.rna_data["id"])
        basal_prob = sim_data_after.process.transcription_regulation.basal_prob
        assert len(basal_prob) == n_tus

    def test_merge_output_roundtrip(self, sim_data_after):
        """Verify merge_output correctly writes basal_prob and delta_prob."""
        from reconstruction.ecoli.parca.stage_07_adjust_promoters import merge_output
        from reconstruction.ecoli.parca._types import AdjustPromotersOutput

        reg = sim_data_after.process.transcription_regulation
        expected_basal_prob = reg.basal_prob.copy()
        expected_delta_prob = {
            "deltaI": reg.delta_prob["deltaI"].copy(),
            "deltaJ": reg.delta_prob["deltaJ"].copy(),
            "deltaV": reg.delta_prob["deltaV"].copy(),
            "shape": reg.delta_prob["shape"],
        }

        out = AdjustPromotersOutput(
            basal_prob=expected_basal_prob,
            delta_prob=expected_delta_prob,
        )

        # Create a minimal mock to test merge_output
        class MockReg:
            basal_prob = None
            delta_prob = None

        class MockProcess:
            transcription_regulation = MockReg()

        class MockSimData:
            process = MockProcess()

        mock_sd = MockSimData()
        merge_output(mock_sd, {}, out)

        np.testing.assert_array_equal(
            mock_sd.process.transcription_regulation.basal_prob,
            expected_basal_prob,
        )
        np.testing.assert_array_equal(
            mock_sd.process.transcription_regulation.delta_prob["deltaI"],
            expected_delta_prob["deltaI"],
        )
        np.testing.assert_array_equal(
            mock_sd.process.transcription_regulation.delta_prob["deltaV"],
            expected_delta_prob["deltaV"],
        )

    def test_equilibrium_rates_differ(self, sim_data_before, sim_data_after):
        """Verify that fitLigandConcentrations modified some equilibrium rates."""
        # At least some reverse rates should have changed
        eq_before = sim_data_before.process.equilibrium
        eq_after = sim_data_after.process.equilibrium

        # Compare the reverse rates arrays
        rev_before = eq_before.rates_rev
        rev_after = eq_after.rates_rev
        assert not np.allclose(rev_before, rev_after, rtol=1e-12), (
            "No equilibrium reverse rates changed during fitLigandConcentrations"
        )
