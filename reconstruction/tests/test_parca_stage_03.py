"""
Tests for Stage 3 (basal_specs) of the ParCa pipeline.

Unit tests for imports and data types, smoke tests using cached
intermediates from stage 2, and regression tests comparing outputs
against cached intermediates from stage 3.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_03.py -v
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
    """Verify all stage 3 modules can be imported."""

    def test_import_stage_module(self):
        from reconstruction.ecoli.parca.stage_03_basal_specs import (
            extract_input,
            compute_basal_specs,
            merge_output,
        )

        assert callable(extract_input)
        assert callable(compute_basal_specs)
        assert callable(merge_output)

    def test_import_sub_functions(self):
        from reconstruction.ecoli.parca.stage_03_basal_specs import (
            setKmCooperativeEndoRNonLinearRNAdecay,
            fitMaintenanceCosts,
        )

        assert callable(setKmCooperativeEndoRNonLinearRNAdecay)
        assert callable(fitMaintenanceCosts)

    def test_import_shared_functions(self):
        from reconstruction.ecoli.parca._shared import (
            expressionConverge,
            setInitialRnaExpression,
            createBulkContainer,
            fitExpression,
            setRNAPCountsConstrainedByPhysiology,
            setRibosomeCountsConstrainedByPhysiology,
            totalCountIdDistributionRNA,
            totalCountIdDistributionProtein,
            rescaleMassForSolubleMetabolites,
        )

        assert callable(expressionConverge)
        assert callable(setInitialRnaExpression)
        assert callable(createBulkContainer)
        assert callable(fitExpression)
        assert callable(setRNAPCountsConstrainedByPhysiology)
        assert callable(setRibosomeCountsConstrainedByPhysiology)
        assert callable(totalCountIdDistributionRNA)
        assert callable(totalCountIdDistributionProtein)
        assert callable(rescaleMassForSolubleMetabolites)


class TestDataTypes:
    """Verify dataclass instantiation works."""

    def test_input_creation(self):
        from reconstruction.ecoli.parca._types import BasalSpecsInput

        inp = BasalSpecsInput(
            variable_elongation_transcription=True,
            variable_elongation_translation=False,
            disable_ribosome_capacity_fitting=False,
            disable_rnapoly_capacity_fitting=False,
            cache_dir="/tmp",
            sim_data_ref=None,
        )
        assert inp.variable_elongation_transcription is True
        assert inp.cache_dir == "/tmp"

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import BasalSpecsOutput

        out = BasalSpecsOutput(
            conc_dict={"A[c]": 1.0},
            expression=np.array([0.5, 0.5]),
            synth_prob=np.array([0.5, 0.5]),
            fit_cistron_expression=np.array([0.5, 0.5]),
            doubling_time=1.0,
            avg_cell_dry_mass_init=1.0,
            fit_avg_soluble_target_mol_mass=1.0,
            bulk_container=np.zeros(1),
        )
        assert "A[c]" in out.conc_dict
        assert len(out.expression) == 2


class TestBasalSpecsSmokeTest:
    """Smoke test using input_adjustments intermediates.

    Verifies that extract_input can create a valid input dataclass from
    real sim_data.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_input_adjustments.cPickle")
        if not os.path.exists(path):
            pytest.skip("input_adjustments intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_extract_input(self, sim_data_before):
        """Verify extract_input runs without error on real data."""
        from reconstruction.ecoli.parca.stage_03_basal_specs import extract_input

        inp = extract_input(sim_data_before, {}, cache_dir="/tmp")

        assert inp.variable_elongation_transcription is True
        assert inp.variable_elongation_translation is False
        assert inp.disable_ribosome_capacity_fitting is False
        assert inp.disable_rnapoly_capacity_fitting is False
        assert inp.cache_dir == "/tmp"
        assert inp.sim_data_ref is sim_data_before

    def test_extract_input_with_kwargs(self, sim_data_before):
        """Verify kwargs are captured."""
        from reconstruction.ecoli.parca.stage_03_basal_specs import extract_input

        inp = extract_input(
            sim_data_before,
            {},
            disable_ribosome_capacity_fitting=True,
            variable_elongation_translation=True,
            cache_dir="/custom",
        )
        assert inp.disable_ribosome_capacity_fitting is True
        assert inp.variable_elongation_translation is True
        assert inp.cache_dir == "/custom"


class TestBasalSpecsRegression:
    """Regression tests against cached intermediates.

    These tests verify that the after-state has the expected structure
    and that extract_input works on the after-state.  The full compute
    is NOT re-run here because expressionConverge and Km fitting take
    ~10 minutes.
    """

    @pytest.fixture(scope="class")
    def sim_data_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_basal_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("basal_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_basal_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("basal_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_cell_specs_has_basal(self, cell_specs_after):
        """Verify cell_specs has the 'basal' key with expected fields."""
        assert "basal" in cell_specs_after
        basal = cell_specs_after["basal"]
        expected_keys = [
            "concDict",
            "expression",
            "synthProb",
            "fit_cistron_expression",
            "doubling_time",
            "avgCellDryMassInit",
            "fitAvgSolubleTargetMolMass",
            "bulkContainer",
        ]
        for key in expected_keys:
            assert key in basal, f"Missing key: {key}"

    def test_expression_normalized(self, cell_specs_after):
        """Verify expression sums to 1."""
        expression = cell_specs_after["basal"]["expression"]
        assert np.allclose(np.sum(expression), 1.0, atol=1e-10)

    def test_synth_prob_normalized(self, cell_specs_after):
        """Verify synthesis probabilities sum to 1."""
        synth_prob = cell_specs_after["basal"]["synthProb"]
        assert np.allclose(np.sum(synth_prob), 1.0, atol=1e-10)

    def test_sim_data_mass_values(self, sim_data_after):
        """Verify sim_data mass values are set."""
        assert sim_data_after.mass.avg_cell_dry_mass_init is not None
        assert sim_data_after.mass.avg_cell_dry_mass is not None
        assert sim_data_after.mass.avg_cell_water_mass_init is not None
        assert sim_data_after.mass.fitAvgSolubleTargetMolMass is not None

    def test_km_endoRNase_set(self, sim_data_after):
        """Verify Km_endoRNase values are set on rna_data and mature_rna_data."""
        km_rna = sim_data_after.process.transcription.rna_data["Km_endoRNase"]
        km_mature = sim_data_after.process.transcription.mature_rna_data["Km_endoRNase"]
        assert len(km_rna) > 0
        assert len(km_mature) > 0
        # Km values should be positive
        assert np.all(km_rna.asNumber() > 0)
        assert np.all(km_mature.asNumber() > 0)

    def test_dark_atp_set(self, sim_data_after):
        """Verify darkATP is set and positive."""
        assert hasattr(sim_data_after.constants, "darkATP")
        assert sim_data_after.constants.darkATP.asNumber() > 0

    def test_ppgpp_expression_set(self, sim_data_after):
        """Verify ppGpp expression attributes are set."""
        transcription = sim_data_after.process.transcription
        assert hasattr(transcription, "exp_ppgpp")
        assert hasattr(transcription, "exp_free")
        assert len(transcription.exp_ppgpp) > 0
        assert len(transcription.exp_free) > 0

    def test_basal_expression_matches_sim_data(self, sim_data_after, cell_specs_after):
        """Verify cell_specs expression matches sim_data basal expression."""
        cell_specs_expr = cell_specs_after["basal"]["expression"]
        sim_data_expr = sim_data_after.process.transcription.rna_expression["basal"]
        np.testing.assert_allclose(cell_specs_expr, sim_data_expr, rtol=1e-10)

    def test_bulk_container_structure(self, cell_specs_after):
        """Verify bulk container has expected structure."""
        bc = cell_specs_after["basal"]["bulkContainer"]
        assert "id" in bc.dtype.names
        assert "count" in bc.dtype.names
        assert len(bc) > 0
        # Counts should be non-negative
        assert np.all(bc["count"] >= 0)
