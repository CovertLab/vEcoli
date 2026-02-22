#!/usr/bin/env python
"""
ParCa Pipeline as Process-Bigraph Steps
========================================

Runs the full ParCa parameter calculator using process-bigraph Step/Composite
infrastructure.  This is functionally equivalent to ``parca_workflow.py`` but
orchestrates stages via the process-bigraph dependency engine.

Usage::

    # Fast run (reduced TF conditions)
    python runscripts/parca_bigraph.py --mode fast --cpus 4

    # Full production run
    python runscripts/parca_bigraph.py --mode full --cpus 4

    # With custom output directory
    python runscripts/parca_bigraph.py --mode fast --cpus 4 -o reconstruction/sim_data
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from reconstruction.ecoli.parca.composite import run_parca
from wholecell.utils import constants
import wholecell.utils.filepath as fp


def main():
    parser = argparse.ArgumentParser(
        description="ParCa Pipeline as Process-Bigraph Steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="Run mode: fast (~20-30 min, reduced TF conditions via debug flag), "
             "full (2-4 hours, all conditions)",
    )
    parser.add_argument(
        "-c", "--cpus",
        type=int,
        default=1,
        help="Number of CPUs for parallel stages",
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="reconstruction/sim_data",
        help="Output directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for Km optimization (default: <outdir>/cache)",
    )
    parser.add_argument(
        "--no-operons",
        action="store_true",
        help="Disable operons in raw data",
    )

    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    kb_directory = fp.makedirs(outdir, constants.KB_DIR)
    cache_dir = args.cache_dir or os.path.join(outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # --- Load raw data ---
    print(f"\n{'=' * 60}")
    print(f"ParCa Bigraph Pipeline ({args.mode} mode)")
    print(f"{'=' * 60}")

    t0 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading raw_data (operons={not args.no_operons})")
    raw_data = KnowledgeBaseEcoli(
        operons_on=not args.no_operons,
        remove_rrna_operons=False,
        remove_rrff=False,
        stable_rrna=False,
        new_genes_option="off",
    )
    raw_data_file = os.path.join(kb_directory, constants.SERIALIZED_RAW_DATA)
    with open(raw_data_file, "wb") as f:
        pickle.dump(raw_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Raw data loaded in {time.time() - t0:.1f}s")

    # --- Build pipeline kwargs ---
    kwargs = {
        'cpus': args.cpus,
        'debug': args.mode == 'fast',
        'cache_dir': cache_dir,
        'variable_elongation_transcription': True,
        'variable_elongation_translation': False,
        'disable_ribosome_capacity_fitting': False,
        'disable_rnapoly_capacity_fitting': False,
    }

    # --- Run pipeline ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Running ParCa pipeline as bigraph Steps...")
    t1 = time.time()
    sim_data = run_parca(raw_data, **kwargs)
    pipeline_time = time.time() - t1
    print(f"\n[{time.strftime('%H:%M:%S')}] Pipeline completed in {pipeline_time:.1f}s")

    # --- Save sim_data ---
    sim_data_file = os.path.join(kb_directory, constants.SERIALIZED_SIM_DATA_FILENAME)
    print(f"\n[{time.strftime('%H:%M:%S')}] Saving sim_data...")
    with open(sim_data_file, "wb") as f:
        pickle.dump(sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    sim_data_size = os.path.getsize(sim_data_file)
    print(f"    Saved ({sim_data_size / (1024 * 1024):.1f} MB)")

    # --- Summary ---
    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Output: {sim_data_file}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
