"""
Tests for Nextflow workflow generation in workflow.py

These tests verify the correctness of:
- Channel grouping templates (MULTIDAUGHTER_CHANNEL, MULTIGENERATION_CHANNEL, etc.)
- Analysis batching logic with group_size for cache invalidation
- Full workflow generation via --build-only
- Nextflow stub execution to validate generated workflow files

Note: The mock createVariants stub always creates 3 variants:
  - variant_1, variant_2, baseline
This is used to verify exact group_size calculations in stub tests.
"""

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from runscripts.workflow import (
    generate_lineage,
    generate_code,
)

# Constants for test calculations
# The mock createVariants stub always produces 3 variants
MOCK_NUM_VARIANTS = 3


class TestGenerateLineage:
    """Test the generate_lineage function output."""

    def test_single_generation_no_analysis(self):
        """Test minimal lineage with no analyses."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=1,
            generations=1,
            single_daughters=True,
            analysis_config={},
        )

        # Should have simGen0 import
        assert any("simGen0" in imp for imp in sim_imports)
        # Should set simCh from metadata
        workflow_str = "\n".join(sim_workflow)
        assert "simCh" in workflow_str

    def test_multiseed_analysis_with_group_size(self):
        """Test that multiseed analysis includes group_size in channel."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=2,
            single_daughters=True,
            analysis_config={"multiseed": {"test_analysis": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Should include multiseed channel with size()
        assert "multiSeedCh" in workflow_str
        assert "it[0].size()" in workflow_str
        # Should have analysisMultiSeed import
        assert any("analysisMultiSeed" in imp for imp in sim_imports)

    def test_multigeneration_analysis_with_group_size(self):
        """Test that multigeneration analysis includes group_size in channel."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=3,
            single_daughters=True,
            analysis_config={"multigeneration": {"test_analysis": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Should include multigeneration channel with size()
        assert "multiGenerationCh" in workflow_str
        assert "it[0].size()" in workflow_str
        # Should have analysisMultiGeneration import
        assert any("analysisMultiGeneration" in imp for imp in sim_imports)

    def test_multivariant_analysis_with_group_size_sum(self):
        """Test that multivariant analysis sums group sizes."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=2,
            single_daughters=True,
            analysis_config={"multivariant": {"test_analysis": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Should include multivariant channel with sum()
        assert "multiVariantCh" in workflow_str
        assert "it[4].sum()" in workflow_str
        # Should have analysisMultiVariant import
        assert any("analysisMultiVariant" in imp for imp in sim_imports)

    def test_multidaughter_only_with_full_tree(self):
        """Test that multidaughter requires single_daughters=False."""
        # With single_daughters=True, multidaughter should not be generated
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=2,
            single_daughters=True,
            analysis_config={"multidaughter": {"test_analysis": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Should NOT include multidaughter channel
        assert "multiDaughterCh" not in workflow_str

    def test_multidaughter_analysis_with_full_tree(self):
        """Test multidaughter analysis when simulating both daughters."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=2,
            single_daughters=False,
            analysis_config={"multidaughter": {"test_analysis": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Should include multidaughter channel with size()
        assert "multiDaughterCh" in workflow_str
        assert "it[1].size()" in workflow_str
        # Should have generationSize mapping
        assert "generationSize = [1: 1, 2: 2]" in workflow_str
        # Should have analysisMultiDaughter import
        assert any("analysisMultiDaughter" in imp for imp in sim_imports)

    def test_multiple_analyses_per_type(self):
        """Test that multiple analyses of same type create proper channels."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=1,
            generations=1,
            single_daughters=True,
            analysis_config={
                "multiseed": {
                    "analysis_one": {},
                    "analysis_two": {},
                    "analysis_three": {},
                }
            },
        )

        workflow_str = "\n".join(sim_workflow)

        # Should have channel with all analysis names
        assert '"analysis_one"' in workflow_str
        assert '"analysis_two"' in workflow_str
        assert '"analysis_three"' in workflow_str

    def test_sims_per_seed_calculation_single_daughters(self):
        """Test correct sims_per_seed for single_daughters=True."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=3,
            single_daughters=True,
            analysis_config={"multigeneration": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # With single_daughters=True, sims_per_seed = generations = 3
        # So groupTuple size should be 3
        assert "size: 3" in workflow_str

    def test_sims_per_seed_calculation_full_tree(self):
        """Test correct sims_per_seed for single_daughters=False."""
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=3,
            single_daughters=False,
            analysis_config={"multigeneration": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # With single_daughters=False, sims_per_seed = 2^generations - 1 = 7
        # So groupTuple size should be 7
        assert "size: 7" in workflow_str


class TestGenerateCode:
    """Test the generate_code function."""

    @patch("runscripts.workflow.compute_file_hash", return_value="fakehash123")
    def test_with_sim_data_path_skips_parca(self, mock_hash):
        """Test that providing sim_data_path skips ParCa execution."""
        config = {
            "sim_data_path": "/path/to/simData.cPickle",
            "generations": 1,
            "n_init_sims": 1,
            "single_daughters": True,
            "analysis_options": {},
        }

        run_parca, sim_imports, sim_workflow = generate_code(config)

        # Should use file() to copy existing kb
        assert "file('/path/to')" in run_parca
        # Should NOT run runParca process
        assert "runParca(params.config)" not in run_parca
        # Should include computed kb_hash in parca_out channel
        assert "fakehash123" in run_parca
        mock_hash.assert_called_once_with("/path/to/simData.cPickle")

    def test_without_sim_data_path_runs_parca(self):
        """Test that without sim_data_path, ParCa is executed."""
        config = {
            "sim_data_path": None,
            "generations": 1,
            "n_init_sims": 1,
            "single_daughters": True,
            "analysis_options": {},
        }

        run_parca, sim_imports, sim_workflow = generate_code(config)

        # Should run runParca process
        assert "runParca(params.config)" in run_parca


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for test configs."""
    return tmp_path


def _check_nextflow_available():
    """Check if Nextflow is available on PATH."""
    try:
        result = subprocess.run(
            ["nextflow", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@pytest.mark.slow
@pytest.mark.skipif(
    not _check_nextflow_available(), reason="Nextflow not available on PATH"
)
class TestNextflowStubExecution:
    """
    Integration tests that generate workflow files and run Nextflow in stub mode.

    These tests require Nextflow to be installed and available on PATH.
    """

    @staticmethod
    def _build_workflow(config_path):
        """Run workflow.py with --build-only and assert success."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "runscripts.workflow",
                "--config",
                str(config_path),
                "--build-only",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Build failed: {result.stderr}"

    @staticmethod
    def _run_stub(build_dir):
        """Run Nextflow in stub mode and return the CompletedProcess."""
        return subprocess.run(
            [
                "nextflow",
                "run",
                str(build_dir / "main.nf"),
                "-stub",
                "-c",
                str(build_dir / "nextflow.config"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=build_dir,
        )

    def test_build_only_creates_required_files(self, temp_config_dir):
        """Test that --build-only creates main.nf, nextflow.config, and workflow_config.json."""
        exp_id = f"test_build_only_{uuid.uuid4().hex[:8]}"
        config = {
            "experiment_id": exp_id,
            "suffix_time": False,
            "analysis_options": {
                "multiseed": {"test_analysis": {}},
            },
            "emitter_arg": {
                "out_dir": str(temp_config_dir / "out"),
            },
            "sim_data_path": None,
            "generations": 1,
        }
        config_path = temp_config_dir / "test_build.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Run workflow.py with --build-only
        self._build_workflow(config_path)

        # Check that files were created
        repo_dir = Path(__file__).parent.parent
        build_dir = repo_dir / "nextflow_temp" / exp_id
        try:
            assert (build_dir / "main.nf").exists()
            assert (build_dir / "nextflow.config").exists()
            assert (build_dir / "workflow_config.json").exists()

            # Check main.nf contents include multiseed with group_size
            main_nf_content = (build_dir / "main.nf").read_text()
            assert "multiSeedCh" in main_nf_content
            assert "it[0].size()" in main_nf_content
        finally:
            # Cleanup
            if build_dir.exists():
                shutil.rmtree(build_dir)
            out_dir = temp_config_dir / "out" / exp_id
            if out_dir.exists():
                shutil.rmtree(out_dir)

    def test_stub_multiseed_outputs_group_size(self, temp_config_dir):
        """Test that stub execution outputs correct group_size for multiseed analysis.

        With 2 generations, 2 init_sims, single_daughters=True:
        - sims_per_seed = generations = 2
        - sims per variant = sims_per_seed * n_init_sims = 2 * 2 = 4
        - Each variant should have group_size=4
        - 3 variants means 3 multiseed analysis jobs
        """
        generations = 2
        n_init_sims = 2
        single_daughters = True
        sims_per_seed = generations if single_daughters else (2**generations - 1)
        expected_group_size = sims_per_seed * n_init_sims  # 4

        exp_id = f"test_stub_multiseed_{uuid.uuid4().hex[:8]}"
        config = {
            "experiment_id": exp_id,
            "suffix_time": False,
            "analysis_options": {
                "multiseed": {"mass_fraction_summary": {}},
            },
            "emitter_arg": {
                "out_dir": str(temp_config_dir / "out"),
            },
            "fail_at_max_duration": True,
            "sim_data_path": None,
            "generations": generations,
            "n_init_sims": n_init_sims,
            "single_daughters": single_daughters,
        }
        config_path = temp_config_dir / "test_multiseed.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        repo_dir = Path(__file__).parent.parent
        build_dir = repo_dir / "nextflow_temp" / exp_id
        out_dir = temp_config_dir / "out" / exp_id

        try:
            # Build workflow
            self._build_workflow(config_path)

            # Run stub
            self._run_stub(build_dir)

            # Find and check stub output files
            work_dir = build_dir / "work"
            assert work_dir.exists(), "Nextflow work directory not created"

            test_files = list(work_dir.rglob("test.txt"))
            multiseed_outputs = []
            for test_file in test_files:
                content = test_file.read_text()
                if "Multiseed" in content:
                    multiseed_outputs.append(content)

            # Should have exactly 3 multiseed outputs (one per variant)
            assert len(multiseed_outputs) == MOCK_NUM_VARIANTS, (
                f"Expected {MOCK_NUM_VARIANTS} Multiseed outputs, "
                f"got {len(multiseed_outputs)}"
            )

            # Each should have the correct group_size
            for content in multiseed_outputs:
                assert f"group_size={expected_group_size}" in content, (
                    f"Expected group_size={expected_group_size} in output:\n{content}"
                )

        finally:
            # Cleanup
            if build_dir.exists():
                shutil.rmtree(build_dir)
            if out_dir.exists():
                shutil.rmtree(out_dir)

    def test_stub_all_multi_analyses(self, temp_config_dir):
        """Test stub execution with all multi-* analysis types and verify exact group_sizes.

        With 2 generations, 2 init_sims, single_daughters=False (full tree):
        - sims_per_seed = 2^2 - 1 = 3 (cells per lineage seed)
        - 3 variants from mock createVariants

        Expected group sizes:
        - multiseed: sims_per_seed * n_init_sims = 3 * 2 = 6 per variant
        - multigeneration: sims_per_seed = 3 per (variant, seed) pair
        - multivariant: 3 variants * 6 sims per variant = 18 total
        - multidaughter: varies by generation (gen1=1, gen2=2 per seed)
        """
        generations = 2
        n_init_sims = 2
        single_daughters = False
        sims_per_seed = 2**generations - 1  # 3
        expected_multiseed_size = sims_per_seed * n_init_sims  # 6
        expected_multigen_size = sims_per_seed  # 3
        expected_multivariant_size = MOCK_NUM_VARIANTS * expected_multiseed_size  # 18

        exp_id = f"test_stub_all_multi_{uuid.uuid4().hex[:8]}"
        config = {
            "experiment_id": exp_id,
            "suffix_time": False,
            "analysis_options": {
                "multidaughter": {"test_analysis": {}},
                "multigeneration": {"test_analysis": {}},
                "multiseed": {"test_analysis": {}},
                "multivariant": {"test_analysis": {}},
            },
            "emitter_arg": {
                "out_dir": str(temp_config_dir / "out"),
            },
            "fail_at_max_duration": True,
            "sim_data_path": None,
            "generations": generations,
            "n_init_sims": n_init_sims,
            "single_daughters": single_daughters,
        }
        config_path = temp_config_dir / "test_all_multi.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        repo_dir = Path(__file__).parent.parent
        build_dir = repo_dir / "nextflow_temp" / exp_id
        out_dir = temp_config_dir / "out" / exp_id

        try:
            # Build workflow
            self._build_workflow(config_path)

            # Check main.nf includes correct groupTuple size parameters
            main_nf_content = (build_dir / "main.nf").read_text()

            # Verify all multi-* channels are present
            assert "multiDaughterCh" in main_nf_content
            assert "multiGenerationCh" in main_nf_content
            assert "multiSeedCh" in main_nf_content
            assert "multiVariantCh" in main_nf_content

            # Verify exact size parameters in groupTuple calls
            # multiseed/multivariant first grouping: by [2, 3] with expected size
            assert (
                f"groupTuple(by: [2, 3], size: {expected_multiseed_size}"
                in main_nf_content
            )
            # multigeneration: by [2, 3, 4] with sims_per_seed
            assert (
                f"groupTuple(by: [2, 3, 4], size: {expected_multigen_size}"
                in main_nf_content
            )
            # multidaughter uses dynamic generationSize
            assert "generationSize = [1: 1, 2: 2]" in main_nf_content

            # Verify group_size calculations are present
            assert "it[1].size()" in main_nf_content  # multidaughter
            assert "it[0].size()" in main_nf_content  # multigeneration, multiseed
            assert "it[4].sum()" in main_nf_content  # multivariant

            # Run stub (may fail on some analyses due to path handling, but
            # the important thing is that the workflow parses correctly)
            self._run_stub(build_dir)

            # Collect and verify stub outputs
            work_dir = build_dir / "work"
            if work_dir.exists():
                test_files = list(work_dir.rglob("test.txt"))

                # Categorize outputs by analysis type
                multiseed_outputs = []
                multigen_outputs = []
                multidaughter_outputs = []
                multivariant_outputs = []

                for test_file in test_files:
                    content = test_file.read_text()
                    if "Multiseed" in content:
                        multiseed_outputs.append(content)
                    elif "Multigeneration" in content:
                        multigen_outputs.append(content)
                    elif "Multicell" in content:  # multidaughter uses "Multicell"
                        multidaughter_outputs.append(content)
                    elif "Multivariant" in content:
                        multivariant_outputs.append(content)

                # Verify number of outputs and group_size values
                assert len(multiseed_outputs) == MOCK_NUM_VARIANTS, (
                    f"Expected {MOCK_NUM_VARIANTS} Multiseed outputs, "
                    f"got {len(multiseed_outputs)}"
                )
                for content in multiseed_outputs:
                    assert f"group_size={expected_multiseed_size}" in content, (
                        f"Multiseed expected group_size={expected_multiseed_size}, got: {content}"
                    )
                assert len(multigen_outputs) == MOCK_NUM_VARIANTS * n_init_sims, (
                    f"Expected {MOCK_NUM_VARIANTS * n_init_sims} Multigen outputs, "
                    f"got {len(multigen_outputs)}"
                )
                for content in multigen_outputs:
                    assert f"group_size={expected_multigen_size}" in content, (
                        f"Multigen expected group_size={expected_multigen_size}, got: {content}"
                    )
                assert len(multivariant_outputs) == 1, (
                    f"Expected 1 Multivariant output, got {len(multivariant_outputs)}"
                )
                for content in multivariant_outputs:
                    assert f"group_size={expected_multivariant_size}" in content, (
                        f"Multivariant expected group_size={expected_multivariant_size}, got: {content}"
                    )
                assert (
                    len(multidaughter_outputs)
                    == MOCK_NUM_VARIANTS * n_init_sims * generations
                ), (
                    f"Expected {MOCK_NUM_VARIANTS * n_init_sims * generations} Multidaughter outputs, "
                    f"got {len(multidaughter_outputs)}"
                )
                # Verify multidaughter outputs have valid group_sizes (1 or 2 for 2 generations)
                for content in multidaughter_outputs:
                    assert "group_size=1" in content or "group_size=2" in content, (
                        f"Multidaughter expected group_size=1 or 2, got: {content}"
                    )

        finally:
            # Cleanup
            if build_dir.exists():
                shutil.rmtree(build_dir)
            if out_dir.exists():
                shutil.rmtree(out_dir)

    @pytest.mark.parametrize("different_seeds_per_variant", [False, True])
    def test_stub_different_seeds_per_variant(
        self, different_seeds_per_variant, temp_config_dir
    ):
        """With different_seeds_per_variant set to False,
        every variant runs with the same seed range. When set
        to True, each variant gets a distinct, non-overlapping seed range.

        The simGen0 stub creates daughter-state directories named
        seed=<lineage_seed>, so we can read the filesystem to check which
        (variant, seed) pairs were actually executed.
        """
        lineage_seed = 10
        n_init_sims = 2

        exp_id = f"test_correlated_seeds_{uuid.uuid4().hex[:8]}"
        config = {
            "experiment_id": exp_id,
            "suffix_time": False,
            "analysis_options": {},
            "emitter_arg": {"out_dir": str(temp_config_dir / "out")},
            "sim_data_path": None,
            "generations": 1,
            "n_init_sims": n_init_sims,
            "single_daughters": True,
            "lineage_seed": lineage_seed,
            "different_seeds_per_variant": different_seeds_per_variant,
        }
        config_path = temp_config_dir / "test_correlated.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        repo_dir = Path(__file__).parent.parent
        build_dir = repo_dir / "nextflow_temp" / exp_id
        out_dir = temp_config_dir / "out" / exp_id

        try:
            self._build_workflow(config_path)

            result = self._run_stub(build_dir)
            assert result.returncode == 0, (
                f"Stub run failed:\n{result.stdout}\n{result.stderr}"
            )

            daughter_states_dir = out_dir / "daughter_states"
            assert daughter_states_dir.exists(), "daughter_states directory not created"

            if not different_seeds_per_variant:
                # All 3 mock variants should have the same shared seed range
                expected_seeds = {
                    f"seed={lineage_seed + i}" for i in range(n_init_sims)
                }
                for variant in ["0", "1", "2"]:
                    variant_dir = daughter_states_dir / f"variant={variant}"
                    assert variant_dir.exists(), f"variant={variant} dir not found"
                    actual_seeds = {p.name for p in variant_dir.iterdir() if p.is_dir()}
                    assert actual_seeds == expected_seeds, (
                        f"variant={variant}: expected {expected_seeds}, got {actual_seeds}"
                    )
            else:
                # The mock createVariants stub emits variant names "0", "1", "2".
                # With decorrelation, variant i gets seeds:
                #   [lineage_seed + i*n_init_sims, ..., lineage_seed + (i+1)*n_init_sims - 1]
                all_seed_sets = []
                for variant_idx, variant in enumerate(["0", "1", "2"]):
                    variant_dir = daughter_states_dir / f"variant={variant}"
                    assert variant_dir.exists(), f"variant={variant} dir not found"
                    actual_seeds = {p.name for p in variant_dir.iterdir() if p.is_dir()}

                    expected_seeds = {
                        f"seed={lineage_seed + variant_idx * n_init_sims + i}"
                        for i in range(n_init_sims)
                    }
                    assert actual_seeds == expected_seeds, (
                        f"variant={variant}: expected {expected_seeds}, got {actual_seeds}"
                    )
                    all_seed_sets.append(actual_seeds)

                # Verify all seed ranges are disjoint across variants
                for i in range(len(all_seed_sets)):
                    for j in range(i + 1, len(all_seed_sets)):
                        assert all_seed_sets[i].isdisjoint(all_seed_sets[j]), (
                            f"Variants {i} and {j} share seeds: "
                            f"{all_seed_sets[i] & all_seed_sets[j]}"
                        )

        finally:
            if build_dir.exists():
                shutil.rmtree(build_dir)
            if out_dir.exists():
                shutil.rmtree(out_dir)

    def test_stub_sim_data_path_skips_parca(self, temp_config_dir):
        """Test that supplying sim_data_path skips ParCa and sims run correctly.

        Verifies with 2 generations and 2 seeds (n_init_sims=2, single_daughters=True):
        - main.nf uses Channel.value(...) for parca_out instead of running runParca
        - Stub succeeds (exit code 0)
        - daughter_states has expected structure:
            3 variants × 2 seeds × 2 generations
        - The kb directory was copied to the output parca/kb location
        """
        generations = 2
        n_init_sims = 2
        lineage_seed = 0

        # Create a dummy kb directory with a simData.cPickle file so
        # compute_file_hash and file().copyTo() both work correctly
        kb_dir = temp_config_dir / "kb"
        kb_dir.mkdir()
        sim_data_file = kb_dir / "simData.cPickle"
        sim_data_file.write_bytes(b"Mock sim_data for testing")

        exp_id = f"test_sim_data_path_{uuid.uuid4().hex[:8]}"
        config = {
            "experiment_id": exp_id,
            "suffix_time": False,
            "analysis_options": {
                "single": True,
            },
            "emitter_arg": {"out_dir": str(temp_config_dir / "out")},
            "sim_data_path": str(sim_data_file),
            "generations": generations,
            "n_init_sims": n_init_sims,
            "single_daughters": True,
            "lineage_seed": lineage_seed,
        }
        config_path = temp_config_dir / "test_sim_data_path.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        repo_dir = Path(__file__).parent.parent
        build_dir = repo_dir / "nextflow_temp" / exp_id
        out_dir = temp_config_dir / "out" / exp_id

        try:
            self._build_workflow(config_path)

            # Verify main.nf skips runParca and uses Channel.value for parca_out
            main_nf_content = (build_dir / "main.nf").read_text()
            assert "Channel.value(" in main_nf_content, (
                "Expected Channel.value(...) for parca_out when sim_data_path is set"
            )
            assert "runParca(params.config)" not in main_nf_content, (
                "runParca should not appear in main.nf when sim_data_path is set"
            )

            result = self._run_stub(build_dir)
            assert result.returncode == 0, (
                f"Stub run failed:\n{result.stdout}\n{result.stderr}"
            )

            # Verify kb was copied to parca output location
            parca_kb_dir = out_dir / "parca" / "kb"
            assert parca_kb_dir.exists(), (
                "parca/kb directory should exist after file().copyTo()"
            )
            assert (parca_kb_dir / "simData.cPickle").exists(), (
                "simData.cPickle should be present in the copied parca/kb directory"
            )

            # Verify daughter_states directory structure:
            # MOCK_NUM_VARIANTS variants × n_init_sims seeds × generations generations
            daughter_states_dir = out_dir / "daughter_states"
            assert daughter_states_dir.exists(), "daughter_states directory not created"

            expected_seeds = {f"seed={lineage_seed + i}" for i in range(n_init_sims)}
            expected_generations = {f"generation={g + 1}" for g in range(generations)}

            for variant in ["0", "1", "2"]:
                variant_dir = daughter_states_dir / f"variant={variant}"
                assert variant_dir.exists(), f"variant={variant} dir not found"

                actual_seeds = {p.name for p in variant_dir.iterdir() if p.is_dir()}
                assert actual_seeds == expected_seeds, (
                    f"variant={variant}: expected seeds {expected_seeds}, "
                    f"got {actual_seeds}"
                )

                for seed_dir in variant_dir.iterdir():
                    actual_generations = {
                        p.name for p in seed_dir.iterdir() if p.is_dir()
                    }
                    assert actual_generations == expected_generations, (
                        f"variant={variant}/{seed_dir.name}: "
                        f"expected generations {expected_generations}, "
                        f"got {actual_generations}"
                    )

            # Verify analysisSingle ran once per simulation:
            # MOCK_NUM_VARIANTS variants × n_init_sims seeds × generations = 12 runs
            expected_single_count = MOCK_NUM_VARIANTS * n_init_sims * generations
            analyses_dir = out_dir / "analyses"
            assert analyses_dir.exists(), "analyses directory not created"
            single_outputs = list(analyses_dir.rglob("test.txt"))
            assert len(single_outputs) == expected_single_count, (
                f"Expected {expected_single_count} analysisSingle outputs "
                f"(MOCK_NUM_VARIANTS={MOCK_NUM_VARIANTS} × n_init_sims={n_init_sims} "
                f"× generations={generations}), got {len(single_outputs)}"
            )

        finally:
            if build_dir.exists():
                shutil.rmtree(build_dir)
            if out_dir.exists():
                shutil.rmtree(out_dir)


class TestGroupSizeValues:
    """Test that group_size values are calculated correctly for different configurations."""

    def test_multiseed_group_size_calculation(self):
        """Verify multiseed groupTuple size parameter."""
        # With 2 init sims and 2 generations (single daughters)
        # sims_per_seed = 2, total = 2 * 2 = 4 per variant
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=2,
            single_daughters=True,
            analysis_config={"multiseed": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Extract size parameter from groupTuple
        match = re.search(r"groupTuple\(by: \[2, 3\], size: (\d+)", workflow_str)
        assert match, "Could not find multiseed groupTuple"
        size = int(match.group(1))

        # sims_per_seed (2) * n_init_sims (2) = 4
        assert size == 4

    def test_multigeneration_group_size_calculation(self):
        """Verify multigeneration groupTuple size parameter."""
        # With 3 generations (single daughters)
        # sims_per_seed = 3
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=3,
            single_daughters=True,
            analysis_config={"multigeneration": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Extract size parameter
        match = re.search(r"groupTuple\(by: \[2, 3, 4\], size: (\d+)", workflow_str)
        assert match, "Could not find multigeneration groupTuple"
        size = int(match.group(1))

        # sims_per_seed = generations = 3
        assert size == 3

    def test_multidaughter_generation_size_mapping(self):
        """Verify multidaughter generationSize mapping for full tree."""
        # With 3 generations (both daughters)
        # gen 1: 1 cell, gen 2: 2 cells, gen 3: 4 cells
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=1,
            generations=3,
            single_daughters=False,
            analysis_config={"multidaughter": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Should have generationSize = [1: 1, 2: 2, 3: 4]
        assert "generationSize = [1: 1, 2: 2, 3: 4]" in workflow_str

    def test_full_tree_sims_per_seed(self):
        """Verify sims_per_seed = 2^generations - 1 for full tree."""
        # With 4 generations (both daughters)
        # sims_per_seed = 2^4 - 1 = 15
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=1,
            generations=4,
            single_daughters=False,
            analysis_config={"multigeneration": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # Extract size parameter
        match = re.search(r"groupTuple\(by: \[2, 3, 4\], size: (\d+)", workflow_str)
        assert match, "Could not find multigeneration groupTuple"
        size = int(match.group(1))

        # sims_per_seed = 2^4 - 1 = 15
        assert size == 15

    def test_multivariant_group_size_calculation(self):
        """Verify multivariant uses .sum() to aggregate group sizes across variants."""
        # With 2 init sims and 2 generations (single daughters)
        # sims_per_seed = 2, total per variant = 2 * 2 = 4
        # multivariant groups all variants together, summing their counts
        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=2,
            generations=2,
            single_daughters=True,
            analysis_config={"multivariant": {"test": {}}},
        )

        workflow_str = "\n".join(sim_workflow)

        # First groupTuple should have size = sims_per_seed * n_init_sims
        # The multivariant channel template starts with simCh.groupTuple(...)
        match = re.search(
            r"simCh\s*\.groupTuple\(by: \[2, 3\], size: (\d+)",
            workflow_str,
        )
        assert match, "Could not find first multivariant groupTuple"
        first_size = int(match.group(1))

        # sims_per_seed (2) * n_init_sims (2) = 4
        assert first_size == 4

        # Should include .size() in first map
        assert "it[0].size()" in workflow_str

        # Second grouping should use .sum() to aggregate sizes
        assert "it[4].sum()" in workflow_str

        # Verify the second groupTuple groups by experiment_id only
        assert "groupTuple(by: [2])" in workflow_str

        # Verify multiVariantCh is set at the end
        assert "set { multiVariantCh }" in workflow_str

    @pytest.mark.parametrize(
        "generations,n_init_sims,single_daughters",
        [
            (1, 1, True),  # Minimal case
            (2, 2, True),  # Single daughters
            (2, 2, False),  # Full tree
            (3, 4, True),  # Larger single daughters
            (3, 2, False),  # Larger full tree
        ],
    )
    def test_group_size_formulas(self, generations, n_init_sims, single_daughters):
        """Verify group_size calculations match expected formulas across configurations.

        Formula summary (assuming N_VARIANTS variants):
        - sims_per_seed = generations (single_daughters) or 2^generations - 1 (full tree)
        - multiseed size = sims_per_seed * n_init_sims (per variant)
        - multigeneration size = sims_per_seed (per variant-seed pair)
        - multivariant total = N_VARIANTS * sims_per_seed * n_init_sims
        """
        # Calculate expected values
        if single_daughters:
            sims_per_seed = generations
        else:
            sims_per_seed = 2**generations - 1

        expected_multiseed_size = sims_per_seed * n_init_sims
        expected_multigen_size = sims_per_seed
        # Total sims = num_variants * expected_multiseed_size
        # (verified in stub tests with MOCK_NUM_VARIANTS=3)

        # Generate workflow with all multi-* analyses
        analysis_config = {
            "multiseed": {"test": {}},
            "multigeneration": {"test": {}},
            "multivariant": {"test": {}},
        }
        if not single_daughters:
            analysis_config["multidaughter"] = {"test": {}}

        sim_imports, sim_workflow = generate_lineage(
            seed=0,
            n_init_sims=n_init_sims,
            generations=generations,
            single_daughters=single_daughters,
            analysis_config=analysis_config,
        )

        workflow_str = "\n".join(sim_workflow)

        # Verify multiseed/multivariant first groupTuple size
        match = re.search(
            r"groupTuple\(by: \[2, 3\], size: (\d+)",
            workflow_str,
        )
        assert match, "Could not find groupTuple(by: [2, 3])"
        actual_multiseed_size = int(match.group(1))
        assert actual_multiseed_size == expected_multiseed_size, (
            f"multiseed size mismatch: expected {expected_multiseed_size}, "
            f"got {actual_multiseed_size} "
            f"(generations={generations}, n_init_sims={n_init_sims}, "
            f"single_daughters={single_daughters})"
        )

        # Verify multigeneration groupTuple size
        match = re.search(
            r"groupTuple\(by: \[2, 3, 4\], size: (\d+)",
            workflow_str,
        )
        assert match, "Could not find groupTuple(by: [2, 3, 4])"
        actual_multigen_size = int(match.group(1))
        assert actual_multigen_size == expected_multigen_size, (
            f"multigeneration size mismatch: expected {expected_multigen_size}, "
            f"got {actual_multigen_size}"
        )

        # Verify multidaughter generationSize mapping for full tree
        if not single_daughters:
            expected_gen_sizes = [f"{g + 1}: {2**g}" for g in range(generations)]
            expected_gen_size_map = "[" + ", ".join(expected_gen_sizes) + "]"
            assert expected_gen_size_map in workflow_str, (
                f"Expected generationSize = {expected_gen_size_map} not found"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
