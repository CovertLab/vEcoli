"""
Tests for Stage 6 (promoter_binding) of the ParCa pipeline.

Unit tests for imports and data types, smoke tests using cached
intermediates from stage 5, and regression tests comparing outputs
against cached intermediates from stage 6.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_06.py -v
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
    """Verify all stage 6 modules can be imported."""

    def test_import_stage_module(self):
        from reconstruction.ecoli.parca.stage_06_promoter_binding import (
            extract_input,
            compute_promoter_binding,
            merge_output,
        )

        assert callable(extract_input)
        assert callable(compute_promoter_binding)
        assert callable(merge_output)

    def test_import_promoter_fitting_functions(self):
        from reconstruction.ecoli.parca_promoter_fitting import (
            fitPromoterBoundProbability,
            calculatePromoterBoundProbability,
            calculateRnapRecruitment,
            fitLigandConcentrations,
        )

        assert callable(fitPromoterBoundProbability)
        assert callable(calculatePromoterBoundProbability)
        assert callable(calculateRnapRecruitment)
        assert callable(fitLigandConcentrations)

    def test_import_matrix_builders(self):
        from reconstruction.ecoli.parca_promoter_fitting import (
            _build_vector_k,
            _build_matrix_G,
            _build_matrix_Z,
            _build_matrix_T,
            _build_matrix_H,
            _build_matrix_pdiff,
        )

        assert callable(_build_vector_k)
        assert callable(_build_matrix_G)
        assert callable(_build_matrix_Z)
        assert callable(_build_matrix_T)
        assert callable(_build_matrix_H)
        assert callable(_build_matrix_pdiff)


class TestDataTypes:
    """Verify dataclass instantiation works."""

    def test_input_creation(self):
        from reconstruction.ecoli.parca._types import PromoterBindingInput

        inp = PromoterBindingInput(
            sim_data_ref=None,
            cell_specs_ref={},
        )
        assert inp.sim_data_ref is None
        assert inp.cell_specs_ref == {}

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import PromoterBindingOutput

        out = PromoterBindingOutput(
            r_vector=np.array([0.1, 0.2, 0.3]),
            r_columns={"col_a": 0, "col_b": 1, "col_c": 2},
        )
        assert len(out.r_vector) == 3
        assert "col_a" in out.r_columns


class TestPromoterBindingSmokeTest:
    """Smoke test using fit_condition intermediates.

    Verifies that extract_input can create a valid input dataclass from
    real sim_data and cell_specs.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_fit_condition.cPickle")
        if not os.path.exists(path):
            pytest.skip("fit_condition intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_fit_condition.cPickle")
        if not os.path.exists(path):
            pytest.skip("fit_condition intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_extract_input(self, sim_data_before, cell_specs_before):
        """Verify extract_input runs without error on real data."""
        from reconstruction.ecoli.parca.stage_06_promoter_binding import (
            extract_input,
        )

        inp = extract_input(sim_data_before, cell_specs_before)

        assert inp.sim_data_ref is sim_data_before
        assert inp.cell_specs_ref is cell_specs_before

    def test_cell_specs_has_bulk_average(self, cell_specs_before):
        """Verify cell_specs has bulkAverageContainer needed for promoter fitting."""
        for cond_key, spec in cell_specs_before.items():
            assert "bulkAverageContainer" in spec, (
                f"Missing bulkAverageContainer for condition '{cond_key}'"
            )
            assert "avgCellDryMassInit" in spec, (
                f"Missing avgCellDryMassInit for condition '{cond_key}'"
            )


class TestPromoterBindingRegression:
    """Regression tests against cached intermediates.

    These tests verify that the after-state has the expected structure
    and properties.  The full compute is NOT re-run here because the
    CVXPY optimization takes significant time.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_fit_condition.cPickle")
        if not os.path.exists(path):
            pytest.skip("fit_condition intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def sim_data_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_promoter_binding.cPickle")
        if not os.path.exists(path):
            pytest.skip("promoter_binding intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_promoter_binding.cPickle")
        if not os.path.exists(path):
            pytest.skip("promoter_binding intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_pPromoterBound_set(self, sim_data_after):
        """Verify pPromoterBound is set on sim_data."""
        assert hasattr(sim_data_after, "pPromoterBound")
        pPB = sim_data_after.pPromoterBound
        assert isinstance(pPB, dict)
        assert len(pPB) > 0

    def test_pPromoterBound_has_conditions(self, sim_data_after, cell_specs_after):
        """Verify pPromoterBound has entries for all cell_specs conditions."""
        pPB = sim_data_after.pPromoterBound
        for cond_key in cell_specs_after:
            assert cond_key in pPB, (
                f"Missing condition '{cond_key}' in pPromoterBound"
            )

    def test_pPromoterBound_values_in_range(self, sim_data_after):
        """Verify all promoter bound probabilities are in [0, 1]."""
        pPB = sim_data_after.pPromoterBound
        for cond_key, tf_probs in pPB.items():
            for tf, prob in tf_probs.items():
                assert 0.0 <= prob <= 1.0, (
                    f"Out of range probability {prob} for TF={tf} in {cond_key}"
                )

    def test_r_vector_in_cell_specs(self, cell_specs_after):
        """Verify r_vector is set in cell_specs['basal']."""
        assert "r_vector" in cell_specs_after["basal"]
        r = cell_specs_after["basal"]["r_vector"]
        assert isinstance(r, np.ndarray)
        assert len(r) > 0

    def test_r_columns_in_cell_specs(self, cell_specs_after):
        """Verify r_columns is set in cell_specs['basal']."""
        assert "r_columns" in cell_specs_after["basal"]
        r_cols = cell_specs_after["basal"]["r_columns"]
        assert isinstance(r_cols, dict)
        assert len(r_cols) > 0

    def test_r_vector_matches_r_columns_length(self, cell_specs_after):
        """Verify r_vector length matches number of r_columns."""
        r = cell_specs_after["basal"]["r_vector"]
        r_cols = cell_specs_after["basal"]["r_columns"]
        assert len(r) == len(r_cols), (
            f"r_vector length {len(r)} != r_columns length {len(r_cols)}"
        )

    def test_synth_prob_normalized(self, sim_data_after):
        """Verify RNA synthesis probabilities are normalized after fitting."""
        for condition, synth_prob in sim_data_after.process.transcription.rna_synth_prob.items():
            assert np.allclose(np.sum(synth_prob), 1.0, atol=1e-10), (
                f"rna_synth_prob not normalized for condition '{condition}': "
                f"sum={np.sum(synth_prob)}"
            )

    def test_synth_prob_non_negative(self, sim_data_after):
        """Verify RNA synthesis probabilities are non-negative."""
        for condition, synth_prob in sim_data_after.process.transcription.rna_synth_prob.items():
            assert np.all(synth_prob >= 0), (
                f"Negative rna_synth_prob for condition '{condition}'"
            )

    def test_synth_prob_changed(self, sim_data_before, sim_data_after):
        """Verify that promoter fitting actually changed some synth probs."""
        # At least one condition's synth_prob should differ
        any_changed = False
        for condition in sim_data_after.process.transcription.rna_synth_prob:
            if condition in sim_data_before.process.transcription.rna_synth_prob:
                before = sim_data_before.process.transcription.rna_synth_prob[condition]
                after = sim_data_after.process.transcription.rna_synth_prob[condition]
                if not np.allclose(before, after, rtol=1e-12):
                    any_changed = True
                    break
        assert any_changed, "No synth_prob values changed during promoter fitting"

    def test_merge_output_roundtrip(self, sim_data_before, cell_specs_after):
        """Verify merge_output produces the expected cell_specs structure."""
        from reconstruction.ecoli.parca.stage_06_promoter_binding import merge_output
        from reconstruction.ecoli.parca._types import PromoterBindingOutput

        r_vector = cell_specs_after["basal"]["r_vector"]
        r_columns = cell_specs_after["basal"]["r_columns"]

        out = PromoterBindingOutput(
            r_vector=r_vector,
            r_columns=r_columns,
        )

        # Create a test cell_specs with basal but without r_vector/r_columns
        test_cell_specs = {"basal": {k: v for k, v in cell_specs_after["basal"].items()
                                     if k not in ("r_vector", "r_columns")}}
        merge_output(sim_data_before, test_cell_specs, out)

        assert "r_vector" in test_cell_specs["basal"]
        assert "r_columns" in test_cell_specs["basal"]
        np.testing.assert_array_equal(
            test_cell_specs["basal"]["r_vector"], r_vector
        )
        assert test_cell_specs["basal"]["r_columns"] == r_columns

    def test_r_columns_have_alpha_entries(self, cell_specs_after):
        """Verify r_columns has both TF and alpha entries."""
        r_cols = cell_specs_after["basal"]["r_columns"]
        alpha_cols = [k for k in r_cols if k.endswith("__alpha")]
        tf_cols = [k for k in r_cols if not k.endswith("__alpha")]
        assert len(alpha_cols) > 0, "No alpha columns found in r_columns"
        assert len(tf_cols) > 0, "No TF columns found in r_columns"
