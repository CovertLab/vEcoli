"""
Tests for Stage 4 (tf_condition_specs) of the ParCa pipeline.

Unit tests for imports and data types, smoke tests using cached
intermediates from stage 3, and regression tests comparing outputs
against cached intermediates from stage 4.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_04.py -v
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
    """Verify all stage 4 modules can be imported."""

    def test_import_stage_module(self):
        from reconstruction.ecoli.parca.stage_04_tf_condition_specs import (
            extract_input,
            compute_tf_condition_specs,
            merge_output,
        )

        assert callable(extract_input)
        assert callable(compute_tf_condition_specs)
        assert callable(merge_output)

    def test_import_sub_functions(self):
        from reconstruction.ecoli.parca.stage_04_tf_condition_specs import (
            buildTfConditionCellSpecifications,
            buildCombinedConditionCellSpecifications,
        )

        assert callable(buildTfConditionCellSpecifications)
        assert callable(buildCombinedConditionCellSpecifications)

    def test_import_shared_functions(self):
        from reconstruction.ecoli.parca._shared import (
            apply_updates,
            expressionConverge,
            expressionFromConditionAndFoldChange,
        )

        assert callable(apply_updates)
        assert callable(expressionConverge)
        assert callable(expressionFromConditionAndFoldChange)


class TestDataTypes:
    """Verify dataclass instantiation works."""

    def test_input_creation(self):
        from reconstruction.ecoli.parca._types import TfConditionSpecsInput

        inp = TfConditionSpecsInput(
            variable_elongation_transcription=True,
            variable_elongation_translation=False,
            disable_ribosome_capacity_fitting=False,
            disable_rnapoly_capacity_fitting=False,
            cpus=1,
            sim_data_ref=None,
        )
        assert inp.variable_elongation_transcription is True
        assert inp.cpus == 1

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import (
            TfConditionSpecsOutput,
            TfConditionSpecsConditionOutput,
        )

        cond_out = TfConditionSpecsConditionOutput(
            condition_label="test__active",
            conc_dict={"A[c]": 1.0},
            expression=np.array([0.5, 0.5]),
            synth_prob=np.array([0.5, 0.5]),
            cistron_expression=np.array([0.3, 0.7]),
            fit_cistron_expression=np.array([0.4, 0.6]),
            doubling_time=1.0,
            avg_cell_dry_mass_init=1.0,
            fit_avg_soluble_target_mol_mass=1.0,
            bulk_container=np.zeros(1),
        )
        out = TfConditionSpecsOutput(condition_outputs=[cond_out])
        assert len(out.condition_outputs) == 1
        assert out.condition_outputs[0].condition_label == "test__active"


class TestTfConditionSpecsSmokeTest:
    """Smoke test using basal_specs intermediates.

    Verifies that extract_input can create a valid input dataclass from
    real sim_data.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_basal_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("basal_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_basal_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("basal_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_extract_input(self, sim_data_before, cell_specs_before):
        """Verify extract_input runs without error on real data."""
        from reconstruction.ecoli.parca.stage_04_tf_condition_specs import (
            extract_input,
        )

        inp = extract_input(sim_data_before, cell_specs_before, cpus=2)

        assert inp.variable_elongation_transcription is True
        assert inp.variable_elongation_translation is False
        assert inp.disable_ribosome_capacity_fitting is False
        assert inp.disable_rnapoly_capacity_fitting is False
        assert inp.cpus == 2
        assert inp.sim_data_ref is sim_data_before

    def test_extract_input_with_kwargs(self, sim_data_before, cell_specs_before):
        """Verify kwargs are captured."""
        from reconstruction.ecoli.parca.stage_04_tf_condition_specs import (
            extract_input,
        )

        inp = extract_input(
            sim_data_before,
            cell_specs_before,
            disable_ribosome_capacity_fitting=True,
            variable_elongation_translation=True,
            cpus=4,
        )
        assert inp.disable_ribosome_capacity_fitting is True
        assert inp.variable_elongation_translation is True
        assert inp.cpus == 4


class TestTfConditionSpecsRegression:
    """Regression tests against cached intermediates.

    These tests verify that the after-state has the expected structure
    and properties.  The full compute is NOT re-run here because
    expressionConverge for all TF conditions takes a long time.
    """

    @pytest.fixture(scope="class")
    def sim_data_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_tf_condition_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("tf_condition_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_tf_condition_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("tf_condition_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_basal_specs.cPickle")
        if not os.path.exists(path):
            pytest.skip("basal_specs intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_cell_specs_has_tf_conditions(self, sim_data_after, cell_specs_after):
        """Verify cell_specs has TF condition entries (active/inactive)."""
        tf_conditions = [
            k for k in cell_specs_after if "__active" in k or "__inactive" in k
        ]
        assert len(tf_conditions) > 0, "No TF conditions found in cell_specs"

    def test_cell_specs_has_combined_conditions(self, sim_data_after, cell_specs_after):
        """Verify cell_specs has combined condition entries (with_aa, etc.)."""
        combined_conditions = [
            k
            for k in cell_specs_after
            if k != "basal" and "__active" not in k and "__inactive" not in k
        ]
        assert (
            len(combined_conditions) > 0
        ), "No combined conditions found in cell_specs"

    def test_condition_has_expected_keys(self, cell_specs_after):
        """Verify each non-basal condition has the expected keys."""
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
        for cond_key, spec in cell_specs_after.items():
            if cond_key == "basal":
                continue
            for key in expected_keys:
                assert key in spec, f"Missing key '{key}' in condition '{cond_key}'"

    def test_expression_normalized(self, cell_specs_after):
        """Verify expression arrays sum to 1 for all conditions."""
        for cond_key, spec in cell_specs_after.items():
            expression = spec["expression"]
            assert np.allclose(np.sum(expression), 1.0, atol=1e-10), (
                f"Expression not normalized for condition '{cond_key}': "
                f"sum={np.sum(expression)}"
            )

    def test_synth_prob_normalized(self, cell_specs_after):
        """Verify synthesis probabilities sum to 1 for all conditions."""
        for cond_key, spec in cell_specs_after.items():
            synth_prob = spec["synthProb"]
            assert np.allclose(np.sum(synth_prob), 1.0, atol=1e-10), (
                f"SynthProb not normalized for condition '{cond_key}': "
                f"sum={np.sum(synth_prob)}"
            )

    def test_cistron_expression_present_for_tf_conditions(self, cell_specs_after):
        """Verify cistron_expression is set for non-basal conditions."""
        for cond_key, spec in cell_specs_after.items():
            if cond_key == "basal":
                continue
            assert "cistron_expression" in spec, (
                f"Missing cistron_expression for condition '{cond_key}'"
            )
            assert spec["cistron_expression"] is not None

    def test_sim_data_expression_dicts_updated(self, sim_data_after, cell_specs_after):
        """Verify sim_data expression dicts match cell_specs for all conditions."""
        transcription = sim_data_after.process.transcription
        for cond_key, spec in cell_specs_after.items():
            if cond_key == "basal":
                continue
            np.testing.assert_allclose(
                transcription.rna_expression[cond_key],
                spec["expression"],
                rtol=1e-10,
                err_msg=f"rna_expression mismatch for {cond_key}",
            )
            np.testing.assert_allclose(
                transcription.rna_synth_prob[cond_key],
                spec["synthProb"],
                rtol=1e-10,
                err_msg=f"rna_synth_prob mismatch for {cond_key}",
            )

    def test_new_conditions_added_to_sim_data(self, sim_data_before, sim_data_after):
        """Verify new TF conditions were added to sim_data expression dicts."""
        before_keys = set(sim_data_before.process.transcription.rna_expression.keys())
        after_keys = set(sim_data_after.process.transcription.rna_expression.keys())
        new_keys = after_keys - before_keys
        assert len(new_keys) > 0, "No new conditions added to sim_data expression dicts"

    def test_bulk_container_structure(self, cell_specs_after):
        """Verify bulk container has expected structure for all conditions."""
        for cond_key, spec in cell_specs_after.items():
            bc = spec["bulkContainer"]
            assert "id" in bc.dtype.names, (
                f"bulkContainer missing 'id' field for {cond_key}"
            )
            assert "count" in bc.dtype.names, (
                f"bulkContainer missing 'count' field for {cond_key}"
            )
            assert len(bc) > 0
            assert np.all(bc["count"] >= 0), (
                f"Negative counts in bulkContainer for {cond_key}"
            )

    def test_merge_output_roundtrip(self, sim_data_before, cell_specs_after):
        """Verify merge_output produces the expected cell_specs structure."""
        from reconstruction.ecoli.parca.stage_04_tf_condition_specs import merge_output
        from reconstruction.ecoli.parca._types import (
            TfConditionSpecsOutput,
            TfConditionSpecsConditionOutput,
        )

        # Build output from cached cell_specs_after
        condition_outputs = []
        for cond_key, spec in sorted(cell_specs_after.items()):
            if cond_key == "basal":
                continue
            condition_outputs.append(
                TfConditionSpecsConditionOutput(
                    condition_label=cond_key,
                    conc_dict=spec["concDict"],
                    expression=spec["expression"],
                    synth_prob=spec["synthProb"],
                    cistron_expression=spec.get("cistron_expression"),
                    fit_cistron_expression=spec["fit_cistron_expression"],
                    doubling_time=spec["doubling_time"],
                    avg_cell_dry_mass_init=spec["avgCellDryMassInit"],
                    fit_avg_soluble_target_mol_mass=spec["fitAvgSolubleTargetMolMass"],
                    bulk_container=spec["bulkContainer"],
                )
            )

        out = TfConditionSpecsOutput(condition_outputs=condition_outputs)

        # Merge into fresh cell_specs (with basal)
        test_cell_specs = {"basal": cell_specs_after["basal"]}
        merge_output(sim_data_before, test_cell_specs, out)

        # Verify all conditions are present
        for cond_key in cell_specs_after:
            assert cond_key in test_cell_specs, f"Missing condition: {cond_key}"
            if cond_key == "basal":
                continue
            for field in ["concDict", "expression", "synthProb", "doubling_time"]:
                assert field in test_cell_specs[cond_key], (
                    f"Missing field '{field}' in condition '{cond_key}'"
                )
