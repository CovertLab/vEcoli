"""
Tests for Stage 8 (set_conditions) of the ParCa pipeline.

Unit tests for pure sub-functions plus a structural smoke test that
verifies the extract/compute/merge pipeline can process real data.

Regression tests against cached intermediates will be enabled once
intermediates for stages 5-8 are generated.

Run:
    python -m pytest reconstruction/tests/test_parca_stage_08.py -v
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


class TestComputeSynthProbFractions:
    """Unit tests for the compute_synth_prob_fractions sub-function."""

    def test_basic(self):
        from reconstruction.ecoli.parca.stage_08_set_conditions import (
            compute_synth_prob_fractions,
        )

        rna_synth_prob = np.array([0.1, 0.2, 0.3, 0.05, 0.15, 0.2])
        is_mRNA = np.array([True, True, False, False, False, False])
        is_tRNA = np.array([False, False, True, True, False, False])
        is_rRNA = np.array([False, False, False, False, True, True])

        result = compute_synth_prob_fractions(
            rna_synth_prob, is_mRNA, is_tRNA, is_rRNA
        )

        assert result == {"mRna": pytest.approx(0.3), "tRna": pytest.approx(0.35), "rRna": pytest.approx(0.35)}

    def test_empty_category(self):
        from reconstruction.ecoli.parca.stage_08_set_conditions import (
            compute_synth_prob_fractions,
        )

        rna_synth_prob = np.array([0.5, 0.5])
        is_mRNA = np.array([True, True])
        is_tRNA = np.array([False, False])
        is_rRNA = np.array([False, False])

        result = compute_synth_prob_fractions(
            rna_synth_prob, is_mRNA, is_tRNA, is_rRNA
        )
        assert result["mRna"] == pytest.approx(1.0)
        assert result["tRna"] == pytest.approx(0.0)
        assert result["rRna"] == pytest.approx(0.0)


class TestRescaleMassForSolubleMetabolites:
    """Unit tests for the pure rescale_mass function in _shared.py."""

    def test_import(self):
        """Verify the function can be imported."""
        from reconstruction.ecoli.parca._shared import (
            rescale_mass_for_soluble_metabolites,
        )

        assert callable(rescale_mass_for_soluble_metabolites)


class TestSetConditionsDataTypes:
    """Verify dataclass instantiation works."""

    def test_condition_input_creation(self):
        from reconstruction.ecoli.parca._types import SetConditionsConditionInput

        inp = SetConditionsConditionInput(
            condition_label="basal",
            nutrients="minimal",
            has_perturbations=False,
            doubling_time=1.0,
            target_molecule_ids=["A[c]", "WATER[c]"],
            target_molecule_concentrations=None,
            molecular_weights=None,
            non_small_molecule_initial_cell_mass=1.0,
            avg_cell_to_initial_cell_conversion_factor=0.5,
            cell_density=1.0,
            n_avogadro=6e23,
            bulk_container=np.zeros(1),
            avg_cell_dry_mass_init_old=1.0,
            rna_synth_prob=np.array([0.5, 0.5]),
            fraction_active_rnap=0.2,
            rnap_elongation_rate=50.0,
            ribosome_elongation_rate=20.0,
            fraction_active_ribosome=0.8,
        )
        assert inp.condition_label == "basal"

    def test_output_creation(self):
        from reconstruction.ecoli.parca._types import (
            SetConditionsConditionOutput,
            SetConditionsOutput,
        )

        out = SetConditionsOutput(
            rnaSynthProbFraction={"minimal": {"mRna": 0.3, "tRna": 0.3, "rRna": 0.4}},
            rnapFractionActiveDict={"minimal": 0.2},
            rnaSynthProbRProtein={"minimal": np.array([0.1])},
            rnaSynthProbRnaPolymerase={"minimal": np.array([0.05])},
            rnaPolymeraseElongationRateDict={"minimal": 50.0},
            expectedDryMassIncreaseDict={"minimal": 1.0},
            ribosomeElongationRateDict={"minimal": 20.0},
            ribosomeFractionActiveDict={"minimal": 0.8},
            condition_outputs=[
                SetConditionsConditionOutput(
                    condition_label="basal",
                    avg_cell_dry_mass_init=1.0,
                    fit_avg_soluble_pool_mass=0.1,
                    bulk_container=np.zeros(1),
                )
            ],
        )
        assert len(out.condition_outputs) == 1


class TestSetConditionsSmokeTest:
    """Smoke test using tf_condition_specs intermediates.

    This verifies that extract_input can successfully read the data
    structures from sim_data and cell_specs. It doesn't test the full
    pipeline because stages 5-7 intermediates are not available.
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
        from reconstruction.ecoli.parca.stage_08_set_conditions import extract_input

        inp = extract_input(sim_data_tf, cell_specs_tf)

        assert len(inp.conditions) == len(cell_specs_tf)
        assert inp.is_mRNA.dtype == bool
        assert inp.is_tRNA.dtype == bool
        assert inp.is_rRNA.dtype == bool

        # Verify conditions are sorted
        labels = [c.condition_label for c in inp.conditions]
        assert labels == sorted(labels)

        # Every condition should have target molecules including WATER[c]
        for cond in inp.conditions:
            assert "WATER[c]" in cond.target_molecule_ids

    def test_full_pipeline(self, sim_data_tf, cell_specs_tf):
        """Verify extract -> compute -> merge runs without error on real data.

        Note: This uses tf_condition_specs data which hasn't been through
        stages 5-7, so the rna_synth_prob values may not exactly match
        what stage 8 would normally see. But the pipeline should still
        execute correctly.
        """
        import copy

        from reconstruction.ecoli.parca.stage_08_set_conditions import (
            extract_input,
            compute_set_conditions,
            merge_output,
        )

        sd = copy.deepcopy(sim_data_tf)
        cs = copy.deepcopy(cell_specs_tf)

        inp = extract_input(sd, cs)
        out = compute_set_conditions(inp)
        merge_output(sd, cs, out)

        # Verify dicts were populated
        assert len(sd.process.transcription.rnaSynthProbFraction) > 0
        assert len(sd.expectedDryMassIncreaseDict) > 0
        assert len(sd.process.translation.ribosomeElongationRateDict) > 0

        # Verify cell_specs were updated
        for cond_out in out.condition_outputs:
            spec = cs[cond_out.condition_label]
            assert "fitAvgSolublePoolMass" in spec
            assert spec["avgCellDryMassInit"] is not None


class TestSetConditionsRegression:
    """Regression tests against cached intermediates.

    These tests require intermediates from stage 7 (adjust_promoters)
    and stage 8 (set_conditions). They will be skipped if not available.
    """

    @pytest.fixture(scope="class")
    def sim_data_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_adjust_promoters.cPickle")
        if not os.path.exists(path):
            pytest.skip("adjust_promoters intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_before(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_adjust_promoters.cPickle")
        if not os.path.exists(path):
            pytest.skip("adjust_promoters intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def sim_data_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "sim_data_set_conditions.cPickle")
        if not os.path.exists(path):
            pytest.skip("set_conditions intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def cell_specs_after(self):
        path = os.path.join(INTERMEDIATES_DIR, "cell_specs_set_conditions.cPickle")
        if not os.path.exists(path):
            pytest.skip("set_conditions intermediates not available")
        with open(path, "rb") as f:
            return pickle.load(f)

    def test_full_roundtrip(
        self, sim_data_before, cell_specs_before, sim_data_after, cell_specs_after
    ):
        """The full extract/compute/merge cycle must match the legacy output."""
        import copy

        from reconstruction.ecoli.parca.stage_08_set_conditions import (
            extract_input,
            compute_set_conditions,
            merge_output,
        )

        sd = copy.deepcopy(sim_data_before)
        cs = copy.deepcopy(cell_specs_before)

        inp = extract_input(sd, cs)
        out = compute_set_conditions(inp)
        merge_output(sd, cs, out)

        # Compare sim_data dicts
        for nutrients in sim_data_after.process.transcription.rnaSynthProbFraction:
            for key in ("mRna", "tRna", "rRna"):
                assert sd.process.transcription.rnaSynthProbFraction[nutrients][key] == pytest.approx(
                    sim_data_after.process.transcription.rnaSynthProbFraction[nutrients][key]
                )

        for nutrients in sim_data_after.expectedDryMassIncreaseDict:
            assert nutrients in sd.expectedDryMassIncreaseDict

        # Compare cell_specs
        for label in cell_specs_after:
            np.testing.assert_allclose(
                cs[label]["avgCellDryMassInit"].asNumber(),
                cell_specs_after[label]["avgCellDryMassInit"].asNumber(),
                rtol=1e-12,
            )
