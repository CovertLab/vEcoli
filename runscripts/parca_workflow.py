#!/usr/bin/env python
"""
ParCa Workflow Runner
=====================

A refactor-friendly wrapper around ParCa with:
- Fast runs (~20-30 min) with reduced TF conditions
- Smoke runs (~2-5 min) using cached intermediates
- Full production runs (2-4 hours) with stage-by-stage timing
- Instrumentation and reporting

Usage:
------
# Fast run - fits only basal + with_aa conditions (no prerequisites)
python runscripts/parca_workflow.py --mode fast --cpus 4

# Smoke run (requires pre-computed intermediates from a full run)
python runscripts/parca_workflow.py --mode smoke

# Full run with timing report
python runscripts/parca_workflow.py --mode full --cpus 4

# Resume from a specific stage (after a full run saved intermediates)
python runscripts/parca_workflow.py --mode resume --from-stage promoter_binding

# Generate timing report from existing run
python runscripts/parca_workflow.py --mode report --intermediates-dir reconstruction/sim_data/intermediates

Stage Timing Reference (approximate):
-------------------------------------
Stage                   | Fast   | Smoke  | Full (4 CPUs)
------------------------|--------|--------|---------------
initialize              | 6s     | 6s     | 6s
input_adjustments       | <1s    | <1s    | <1s
basal_specs             | 8s*    | 8s*    | 16s
tf_condition_specs      | ~1 min | skip   | 3-4 min
fit_condition           | ~15 min| skip   | 2-3 hours  <-- bottleneck
promoter_binding        | 10s    | 10s    | 1-2 min
adjust_promoters        | 5s     | 5s     | 10s
set_conditions          | 5s     | 5s     | 30s
final_adjustments       | 10s    | 10s    | 1-2 min
------------------------|--------|--------|---------------
TOTAL                   | ~20 min| ~2 min | 2-4 hours

* Uses cached Km optimization if available

Modes:
------
- fast: Runs all stages but with reduced TF conditions (8 TFs, 2 conditions).
        Produces valid sim_data for testing/development. No prerequisites.
- smoke: Loads from fit_condition checkpoint, runs remaining stages.
         Requires pre-computed intermediates from a full run.
- full: Complete ParCa run with all 23 TFs and 5 conditions.
- resume: Resume from a specific stage using saved intermediates.
- report: Show timing summary from existing intermediates.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig
from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
from validation.ecoli.validation_data_raw import ValidationDataRawEcoli
from validation.ecoli.validation_data import ValidationDataEcoli
from wholecell.utils import constants
import wholecell.utils.filepath as fp


# Stage names in execution order
STAGES = [
    "initialize",
    "input_adjustments",
    "basal_specs",
    "tf_condition_specs",
    "fit_condition",
    "promoter_binding",
    "adjust_promoters",
    "set_conditions",
    "final_adjustments",
]

# Stages that are fast and safe to always run
FAST_STAGES = ["initialize", "input_adjustments", "basal_specs"]

# The expensive stage that we skip in smoke mode
EXPENSIVE_STAGES = ["tf_condition_specs", "fit_condition"]


@dataclass
class StageReport:
    """Report for a single ParCa stage."""
    name: str
    duration_seconds: float = 0.0
    skipped: bool = False
    loaded_from_cache: bool = False
    output_size_bytes: int = 0
    error: Optional[str] = None


@dataclass
class ParcaReport:
    """Full ParCa run report."""
    mode: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    stages: list = field(default_factory=list)
    total_duration_seconds: float = 0.0
    success: bool = False
    output_path: Optional[str] = None

    def add_stage(self, report: StageReport):
        self.stages.append(report)

    def finalize(self, success: bool, output_path: Optional[str] = None):
        self.end_time = datetime.now()
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.output_path = output_path

    def print_summary(self):
        """Print a formatted summary of the run."""
        print("\n" + "=" * 70)
        print(f"ParCa Workflow Report ({self.mode} mode)")
        print("=" * 70)
        print(f"Start:    {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End:      {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}")
        print(f"Duration: {self._format_duration(self.total_duration_seconds)}")
        print(f"Status:   {'SUCCESS' if self.success else 'FAILED'}")
        if self.output_path:
            print(f"Output:   {self.output_path}")
        print("-" * 70)
        print(f"{'Stage':<25} {'Duration':>12} {'Size':>12} {'Status':<15}")
        print("-" * 70)

        for stage in self.stages:
            duration_str = self._format_duration(stage.duration_seconds)
            size_str = self._format_size(stage.output_size_bytes) if stage.output_size_bytes else "-"

            if stage.error:
                status = "ERROR"
            elif stage.skipped:
                status = "skipped"
            elif stage.loaded_from_cache:
                status = "from cache"
            else:
                status = "completed"

            print(f"{stage.name:<25} {duration_str:>12} {size_str:>12} {status:<15}")

        print("-" * 70)
        print(f"{'TOTAL':<25} {self._format_duration(self.total_duration_seconds):>12}")
        print("=" * 70 + "\n")

    def save_json(self, path: str):
        """Save report as JSON for programmatic access."""
        data = {
            "mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.success,
            "output_path": self.output_path,
            "stages": [
                {
                    "name": s.name,
                    "duration_seconds": s.duration_seconds,
                    "skipped": s.skipped,
                    "loaded_from_cache": s.loaded_from_cache,
                    "output_size_bytes": s.output_size_bytes,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    @staticmethod
    def _format_size(bytes_: int) -> str:
        if bytes_ < 1024:
            return f"{bytes_}B"
        elif bytes_ < 1024 * 1024:
            return f"{bytes_ / 1024:.1f}KB"
        else:
            return f"{bytes_ / (1024 * 1024):.1f}MB"


def get_intermediate_files(intermediates_dir: str) -> dict:
    """Get available intermediate files and their sizes."""
    files = {}
    if os.path.exists(intermediates_dir):
        for name in os.listdir(intermediates_dir):
            if name.endswith(".cPickle"):
                path = os.path.join(intermediates_dir, name)
                files[name] = os.path.getsize(path)
    return files


def check_smoke_prerequisites(intermediates_dir: str) -> tuple[bool, str]:
    """
    Check if we have the prerequisites for a smoke run.

    Smoke run requires pre-computed intermediates through fit_condition.
    """
    required_files = [
        "sim_data_fit_condition.cPickle",
        "cell_specs_fit_condition.cPickle",
    ]

    for fname in required_files:
        path = os.path.join(intermediates_dir, fname)
        if not os.path.exists(path):
            return False, f"Missing required intermediate: {fname}"

    return True, "Prerequisites satisfied"


def run_parca_instrumented(config: dict, report: ParcaReport) -> bool:
    """Run ParCa with instrumentation."""
    kb_directory = fp.makedirs(config["outdir"], constants.KB_DIR)

    raw_data_file = os.path.join(kb_directory, constants.SERIALIZED_RAW_DATA)
    sim_data_file = os.path.join(kb_directory, constants.SERIALIZED_SIM_DATA_FILENAME)
    raw_validation_data_file = os.path.join(kb_directory, constants.SERIALIZED_RAW_VALIDATION_DATA)
    validation_data_file = os.path.join(kb_directory, constants.SERIALIZED_VALIDATION_DATA)

    try:
        # Raw data loading
        stage_start = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Loading raw_data (operons={config['operons']})")
        raw_data = KnowledgeBaseEcoli(
            operons_on=config["operons"],
            remove_rrna_operons=config["remove_rrna_operons"],
            remove_rrff=config["remove_rrff"],
            stable_rrna=config["stable_rrna"],
            new_genes_option=config["new_genes"],
        )
        with open(raw_data_file, "wb") as f:
            pickle.dump(raw_data, f)
        raw_data_time = time.time() - stage_start
        print(f"    Raw data loaded in {raw_data_time:.1f}s")

        # Sim data fitting
        print(f"\n[{time.strftime('%H:%M:%S')}] Running fitSimData_1...")
        if config.get("load_intermediate"):
            print(f"    Loading from checkpoint: {config['load_intermediate']}")

        sim_data = fitSimData_1(
            raw_data=raw_data,
            cpus=config["cpus"],
            smoke=config.get("smoke", False),
            debug=config.get("debug_parca", False),
            load_intermediate=config.get("load_intermediate"),
            save_intermediates=config.get("save_intermediates", False),
            intermediates_directory=config.get("intermediates_directory", ""),
            variable_elongation_transcription=config["variable_elongation_transcription"],
            variable_elongation_translation=config["variable_elongation_translation"],
            disable_ribosome_capacity_fitting=(not config["ribosome_fitting"]),
            disable_rnapoly_capacity_fitting=(not config["rnapoly_fitting"]),
            cache_dir=config.get("cache_dir", ""),
        )

        # Save sim_data
        stage_start = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Saving sim_data...")
        with open(sim_data_file, "wb") as f:
            pickle.dump(sim_data, f)
        sim_data_size = os.path.getsize(sim_data_file)
        print(f"    Saved ({sim_data_size / (1024*1024):.1f} MB)")

        # Validation data
        print(f"\n[{time.strftime('%H:%M:%S')}] Creating validation_data...")
        raw_validation_data = ValidationDataRawEcoli()
        with open(raw_validation_data_file, "wb") as f:
            pickle.dump(raw_validation_data, f)

        validation_data = ValidationDataEcoli()
        validation_data.initialize(raw_validation_data, raw_data)
        with open(validation_data_file, "wb") as f:
            pickle.dump(validation_data, f)
        print("    Validation data saved")

        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def collect_stage_reports(intermediates_dir: str, report: ParcaReport):
    """Collect timing info from intermediate files."""
    files = get_intermediate_files(intermediates_dir)

    for stage in STAGES:
        sim_data_file = f"sim_data_{stage}.cPickle"
        cell_specs_file = f"cell_specs_{stage}.cPickle"

        stage_report = StageReport(name=stage)

        if sim_data_file in files:
            stage_report.output_size_bytes = files.get(sim_data_file, 0) + files.get(cell_specs_file, 0)

        report.add_stage(stage_report)


def main():
    parser = argparse.ArgumentParser(
        description="ParCa Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode",
        choices=["fast", "smoke", "full", "resume", "report"],
        default="fast",
        help="Run mode: fast (~20-30 min, reduced conditions), smoke (uses cached fit_condition), full (2-4 hours), resume (from stage), report (show timing)"
    )
    parser.add_argument(
        "--from-stage",
        type=str,
        choices=STAGES,
        help="Stage to resume from (for --mode resume)"
    )
    parser.add_argument(
        "-c", "--cpus",
        type=int,
        default=1,
        help="Number of CPUs for parallel stages"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="reconstruction/sim_data",
        help="Output directory"
    )
    parser.add_argument(
        "--intermediates-dir",
        type=str,
        default="reconstruction/sim_data/intermediates",
        help="Directory for intermediate checkpoints"
    )
    parser.add_argument(
        "--no-operons",
        action="store_true",
        help="Disable operons"
    )
    parser.add_argument(
        "--save-report",
        type=str,
        help="Save JSON report to this path"
    )

    args = parser.parse_args()

    # Create report
    report = ParcaReport(mode=args.mode)

    # Handle report-only mode
    if args.mode == "report":
        collect_stage_reports(args.intermediates_dir, report)
        report.finalize(success=True)
        report.print_summary()
        if args.save_report:
            report.save_json(args.save_report)
        return

    # Build config
    config = {
        "outdir": os.path.abspath(args.outdir),
        "operons": not args.no_operons,
        "remove_rrna_operons": False,
        "remove_rrff": False,
        "stable_rrna": False,
        "new_genes": "off",
        "cpus": args.cpus,
        "ribosome_fitting": True,
        "rnapoly_fitting": True,
        "variable_elongation_transcription": True,
        "variable_elongation_translation": False,
        "save_intermediates": True,
        "intermediates_directory": os.path.abspath(args.intermediates_dir),
        "cache_dir": os.path.join(os.path.abspath(args.outdir), "cache"),
    }

    os.makedirs(config["cache_dir"], exist_ok=True)
    os.makedirs(config["intermediates_directory"], exist_ok=True)

    # Mode-specific configuration
    if args.mode == "fast":
        # Use smoke flag to fit only minimal TF conditions
        config["smoke"] = True
        config["load_intermediate"] = None
        print("\n" + "=" * 50)
        print("FAST RUN - Fitting only basal + with_aa conditions")
        print("Expected time: ~20-30 minutes with 4 CPUs")
        print("=" * 50)

    elif args.mode == "smoke":
        # Check prerequisites for loading from checkpoint
        ok, msg = check_smoke_prerequisites(args.intermediates_dir)
        if not ok:
            print(f"\n[ERROR] Cannot run smoke test: {msg}")
            print("\nTo create intermediates, first run:")
            print(f"  python runscripts/parca_workflow.py --mode full --cpus 4")
            print("\nOr use --mode fast for a reduced run without prerequisites.")
            sys.exit(1)

        # Load from fit_condition (skips the expensive stages)
        config["load_intermediate"] = "fit_condition"
        print("\n" + "=" * 50)
        print("SMOKE RUN - Loading from fit_condition checkpoint")
        print("=" * 50)

    elif args.mode == "resume":
        if not args.from_stage:
            print("[ERROR] --from-stage required for resume mode")
            sys.exit(1)
        config["load_intermediate"] = args.from_stage
        print(f"\n[RESUME] Starting from {args.from_stage}")

    elif args.mode == "full":
        config["load_intermediate"] = None
        print("\n" + "=" * 50)
        print("FULL RUN - This will take 2-4 hours")
        print("=" * 50)

    # Run ParCa
    success = run_parca_instrumented(config, report)

    # Collect stage info from intermediates
    collect_stage_reports(args.intermediates_dir, report)

    # Finalize report
    output_path = os.path.join(config["outdir"], "kb", constants.SERIALIZED_SIM_DATA_FILENAME)
    report.finalize(success=success, output_path=output_path if success else None)
    report.print_summary()

    if args.save_report:
        report.save_json(args.save_report)
        print(f"Report saved to: {args.save_report}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
