"""
Tests for Stage 5 (fit_condition) of the ParCa pipeline.

Unit tests for pure sub-functions and structural smoke tests that verify
the extract/compute/merge pipeline can process real data.

Regression tests against cached intermediates will be enabled once
intermediates for stages 4-5 are generated.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_05.py -v
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
    """Verify all stage 5 modules can be imported."""

    def test_import_stage_module(self):
        from reconstruction.ecoli.parca.stage_05_fit_condition import (
            extract_input,
            compute_fit_condition,
            merge_output,
        )

        assert callable(extract_input)
        assert callable(compute_fit_condition)
        assert callable(merge_output)

    def test_import_sub_functions(self):
        from reconstruction.ecoli.parca.stage_05_fit_condition import (
            totalCountIdDistributionRNA,
            totalCountIdDistributionProtein,
            calculateBulkDistributions,
            calculateTranslationSupply,
        )

        assert callable(totalCountIdDistributionRNA)
        assert callable(totalCountIdDistributionProtein)
        assert callable(calculateBulkDistributions)
        assert callable(calculateTranslationSupply)


class TestDataTypes:
    """Verify dataclass instantiation works."""

    def test_condition_input_creation(self):
        from reconstruction.ecoli.parca._types import FitConditionConditionInput

        inp = FitConditionConditionInput(
            condition_label="basal",
            nutrients="minimal",
            expression=np.array([0.5, 0.5]),
            conc_dict={"A[c]": 1.0},
            avg_cell_dry_mass_init=1.0,
            doubling_time=1.0,
        )
        assert inp.condition_label == "basal"

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import (
            FitConditionConditionOutput,
            FitConditionOutput,
        )

        out = FitConditionOutput(
            condition_outputs=[
                FitConditionConditionOutput(
                    condition_label="basal",
                    bulk_average_container=np.zeros(1),
                    bulk_deviation_container=np.zeros(1),
                    protein_monomer_average_container=np.zeros(1),
                    protein_monomer_deviation_container=np.zeros(1),
                    translation_aa_supply=np.zeros(21),
                )
            ],
            translation_supply_rate={"minimal": np.zeros(21)},
        )
        assert len(out.condition_outputs) == 1
        assert "minimal" in out.translation_supply_rate


class TestFitConditionSmokeTest:
    """Smoke test using tf_condition_specs intermediates.

    This verifies that extract_input can successfully read the data
    structures. Note: a full compute test requires running through
    the stochastic simulation which is expensive (~minutes per condition).
    """

    @pytest.fixture(scope="class")
    def sim_data_tf(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_tf_condition_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("tf_condition_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_tf(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_tf_condition_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("tf_condition_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_extract_input(self, sim_data_tf, cell_specs_tf):
        """Verify extract_input runs without error on real data."""
        from reconstruction.ecoli.parca.stage_05_fit_condition import extract_input

        inp = extract_input(sim_data_tf, cell_specs_tf)

        assert len(inp.conditions) == len(cell_specs_tf)
        assert inp.sim_data_ref is sim_data_tf
        assert inp.cpus == 1

        # Verify conditions are sorted
        labels = [c.condition_label for c in inp.conditions]
        assert labels == sorted(labels)

        # Verify per-condition data
        for cond in inp.conditions:
            assert cond.expression is not None
            assert cond.conc_dict is not None
            assert cond.doubling_time is not None

    def test_extract_with_cpus_kwarg(self, sim_data_tf, cell_specs_tf):
        """Verify cpus kwarg is captured."""
        from reconstruction.ecoli.parca.stage_05_fit_condition import extract_input

        inp = extract_input(sim_data_tf, cell_specs_tf, cpus=4)
        assert inp.cpus == 4


class TestFitConditionRegression:
    """Regression tests against cached intermediates.

    These tests verify the data flow (extract/merge) against saved
    intermediates.  The full compute is NOT re-run here because
    calculateBulkDistributions is a ~2 hour stochastic simulation.
    """

    @pytest.fixture(scope="class")
    def sim_data_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_fit_condition.cPickle")
        if not os.path.exists(path):
            pytest.skip("fit_condition intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_fit_condition.cPickle")
        if not os.path.exists(path):
            pytest.skip("fit_condition intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_after_state_has_expected_fields(self, sim_data_after, cell_specs_after):
        """Verify the cached after-state has all fields that merge_output sets."""
        # sim_data should have translation_supply_rate populated
        assert len(sim_data_after.translation_supply_rate) > 0

        # cell_specs should have bulk distribution fields for all conditions
        for label, spec in cell_specs_after.items():
            assert "bulkAverageContainer" in spec, f"Missing bulkAverageContainer in {label}"
            assert "bulkDeviationContainer" in spec, f"Missing bulkDeviationContainer in {label}"
            assert "proteinMonomerAverageContainer" in spec, f"Missing proteinMonomerAverageContainer in {label}"
            assert "proteinMonomerDeviationContainer" in spec, f"Missing proteinMonomerDeviationContainer in {label}"
            assert "translation_aa_supply" in spec, f"Missing translation_aa_supply in {label}"

    def test_extract_input_on_after_state(self, sim_data_after, cell_specs_after):
        """Verify extract_input works on the after-state (round-trip capability)."""
        from reconstruction.ecoli.parca.stage_05_fit_condition import extract_input

        inp = extract_input(sim_data_after, cell_specs_after)
        assert len(inp.conditions) == len(cell_specs_after)

        # Verify conditions have expected fields from cell_specs
        for cond in inp.conditions:
            assert cond.expression is not None
            assert cond.conc_dict is not None

    def test_translation_supply_rate_structure(self, sim_data_after):
        """Verify the translation_supply_rate dict has the expected shape."""
        for nutrients, supply in sim_data_after.translation_supply_rate.items():
            assert isinstance(nutrients, str)
            # Should have 21 amino acids
            assert len(supply) == 21
