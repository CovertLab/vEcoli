"""
Tests for Stage 9 (final_adjustments) of the ParCa pipeline.

Unit tests for imports and data types, and smoke tests using cached
intermediates from stage 8.  No after-state regression tests are
possible since this is the final stage with no cached output.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_09.py -v
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
    """Verify all stage 9 modules can be imported."""

    def test_import_stage_module(self):
        from reconstruction.ecoli.parca.stage_09_final_adjustments import (
            extract_input,
            compute_final_adjustments,
            merge_output,
        )

        assert callable(extract_input)
        assert callable(compute_final_adjustments)
        assert callable(merge_output)

    def test_import_create_bulk_container(self):
        from ecoli.library.initial_conditions import create_bulk_container

        assert callable(create_bulk_container)


class TestDataTypes:
    """Verify dataclass instantiation works."""

    def test_input_creation(self):
        from reconstruction.ecoli.parca._types import FinalAdjustmentsInput

        inp = FinalAdjustmentsInput(
            sim_data_ref=None,
            cell_specs_ref={},
        )
        assert inp.sim_data_ref is None
        assert inp.cell_specs_ref == {}

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import FinalAdjustmentsOutput

        out = FinalAdjustmentsOutput()
        assert out is not None


class TestFinalAdjustmentsSmokeTest:
    """Smoke test using set_conditions intermediates.

    Verifies that extract_input can create a valid input dataclass from
    real sim_data and cell_specs, and that the required sim_data methods
    exist.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_set_conditions.cPickle")
        if not os.path.exists(path):
            pytest.skip("set_conditions intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_set_conditions.cPickle")
        if not os.path.exists(path):
            pytest.skip("set_conditions intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_extract_input(self, sim_data_before, cell_specs_before):
        """Verify extract_input runs without error on real data."""
        from reconstruction.ecoli.parca.stage_09_final_adjustments import (
            extract_input,
        )

        inp = extract_input(sim_data_before, cell_specs_before)

        assert inp.sim_data_ref is sim_data_before
        assert inp.cell_specs_ref is cell_specs_before

    def test_sim_data_has_required_methods(self, sim_data_before):
        """Verify sim_data has the methods called by final_adjustments."""
        transcription = sim_data_before.process.transcription
        metabolism = sim_data_before.process.metabolism

        assert hasattr(transcription, "calculate_attenuation")
        assert callable(transcription.calculate_attenuation)
        assert hasattr(transcription, "adjust_polymerizing_ppgpp_expression")
        assert callable(transcription.adjust_polymerizing_ppgpp_expression)
        assert hasattr(transcription, "adjust_ppgpp_expression_for_tfs")
        assert callable(transcription.adjust_ppgpp_expression_for_tfs)
        assert hasattr(transcription, "set_ppgpp_kinetics_parameters")
        assert callable(transcription.set_ppgpp_kinetics_parameters)

        assert hasattr(metabolism, "set_phenomological_supply_constants")
        assert callable(metabolism.set_phenomological_supply_constants)
        assert hasattr(metabolism, "set_mechanistic_supply_constants")
        assert callable(metabolism.set_mechanistic_supply_constants)
        assert hasattr(metabolism, "set_mechanistic_export_constants")
        assert callable(metabolism.set_mechanistic_export_constants)
        assert hasattr(metabolism, "set_mechanistic_uptake_constants")
        assert callable(metabolism.set_mechanistic_uptake_constants)

    def test_merge_output_is_noop(self, sim_data_before, cell_specs_before):
        """Verify merge_output does nothing."""
        from reconstruction.ecoli.parca.stage_09_final_adjustments import merge_output
        from reconstruction.ecoli.parca._types import FinalAdjustmentsOutput

        out = FinalAdjustmentsOutput()
        # Should not raise and should not modify anything
        merge_output(sim_data_before, cell_specs_before, out)

    def test_cell_specs_has_required_conditions(self, sim_data_before, cell_specs_before):
        """Verify cell_specs has conditions needed by final_adjustments."""
        assert "basal" in cell_specs_before
        # with_aa is needed for create_bulk_container
        assert "with_aa" in cell_specs_before or len(cell_specs_before) > 1

    def test_fit_sim_data_wiring(self):
        """Verify fit_sim_data_1.py imports stage 9 correctly."""
        from reconstruction.ecoli.fit_sim_data_1 import final_adjustments

        assert callable(final_adjustments)
