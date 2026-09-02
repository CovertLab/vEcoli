"""
Utility for building a ``variants`` config block from a CSV of objective
weights, and optionally launching the full workflow with that config.

The core function :func:`csv_to_variants_config` can be imported and used
programmatically.  The CLI merges the result into a base config JSON and
passes it to ``workflow.py``, so no manual JSON editing is required.

Usage (CLI):
    # Generate and immediately run the workflow with all CSV rows as variants:
    python runscripts/variants_or_json_from_csv.py \\
        --config configs/metabolism_redux_classic_variant.json \\
        --csv "notebooks/Heena notebooks/Metabolism_New Genes/\\
pareto_results_init_selective_2000samples/objective_weights_batch_apr.csv" \\
        --outdir out/variants_apr

    # Only run specific rows by Index value:
    python runscripts/variants_or_json_from_csv.py \\
        --config configs/..json \\
        --csv path/to/weights.csv \\
        --outdir out/variants_subset \\
        --indices 282 424 728

    # Just write the merged config JSON without running the workflow:
    uvenv runscripts/variants_or_json_from_csv.py \\
        --config configs/metabolism_redux_classic_variants.json \\
        --csv path/to/weights.csv \\
        --outdir out/variants_apr \\
        --write-config-only merged_config.json

Usage (programmatic):
    from runscripts.variants_or_json_from_csv import csv_to_variants_config

    variants_block = csv_to_variants_config("path/to/weights.csv")
    # variants_block can be assigned to config["variants"] directly
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANT_NAME = "metabolism_redux_classic_objective_weights"
CSV_COLUMN_MAP = {
    "homeostatic": "lambda_hom",
    "secretion": "lambda_sec",
    "efficiency": "lambda_eff",
    "kinetics": "lambda_kin",
    "diversity": "lambda_div",
}


def csv_to_variants_config(
    csv_path: str | Path,
    indices: list[str | int] | None = None,
) -> dict[str, Any]:
    """
    Read a CSV of objective weights and return a ``variants`` config dict
    ready to be placed under the ``"variants"`` key of a workflow config JSON.

    The returned dict uses ``"op": "zip"`` so that each CSV row becomes one
    variant simulation, with all five weights swept together.

    Args:
        csv_path: Path to CSV with columns ``lambda_sec``, ``lambda_eff``,
            ``lambda_kin``, ``lambda_div`` (and optionally ``Index``).
        indices: If given, only include rows whose ``Index`` column matches
            one of these values.

    Returns:
        Dictionary of the form::

            {
                "metabolism_redux_classic_objective_weights": {
                    "secretion":   {"value": [...]},
                    "efficiency":  {"value": [...]},
                    "kinetics":    {"value": [...]},
                    "diversity":   {"value": [...]},
                    "homeostatic": {"value": [...]},
                    "op": "zip",
                }
            }
    """
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    df = pd.read_csv(csv_path)

    missing = [col for col in CSV_COLUMN_MAP.values() if col not in df.columns]
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}")

    if indices is not None:
        if "Index" not in df.columns:
            sys.exit("--indices requires an 'Index' column in the CSV.")
        df = df[df["Index"].astype(str).isin([str(i) for i in indices])]
        if df.empty:
            sys.exit(f"No rows matched the provided --indices: {indices}")

    if df.empty:
        sys.exit("CSV produced no rows to process.")

    param_block: dict[str, Any] = {
        weight_key: {"value": df[csv_col].astype(float).tolist()}
        for weight_key, csv_col in CSV_COLUMN_MAP.items()
    }
    param_block["op"] = "zip"

    return {VARIANT_NAME: param_block}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a variants config from a CSV of objective weights and run "
            "the workflow. Equivalent to manually writing the 'variants' block "
            "in a config JSON."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to base workflow config JSON (e.g. configs/metabolism_redux_classic_variant.json).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Path to CSV with columns lambda_sec, lambda_eff, lambda_kin, "
            "lambda_div (and optionally Index)."
        ),
    )
    parser.add_argument(
        "--outdir",
        "-o",
        required=True,
        help="Output directory passed to the workflow (sets emitter_arg.out_dir).",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        help="Optional: only process rows whose Index column matches these values.",
    )

    parser.add_argument(
        "--write-config-only",
        metavar="OUTPUT_PATH",
        default=None,
        help=(
            "Write the merged config JSON to OUTPUT_PATH and exit without "
            "running the workflow. Useful for inspection or manual runs."
        ),
    )
    args = parser.parse_args()

    # Resolve base config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Build the variants block from CSV and inject it
    variants_block = csv_to_variants_config(
        args.csv,
        indices=args.indices,
    )
    n_variants = len(next(iter(variants_block[VARIANT_NAME].values()))["value"])
    print(f"Built {n_variants} variants from CSV.")

    config["variants"] = variants_block
    config.setdefault("emitter_arg", {})["out_dir"] = args.outdir

    if args.write_config_only:
        out_path = Path(args.write_config_only)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / "configs" / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Merged config written to {out_path}")
        print("Run the workflow with:")
        print(f"  uvenv runscripts/workflow.py --config {out_path}")
        return

    # Write a temporary config and hand off to workflow.py
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="variants_from_csv_",
        delete=False,
    ) as tmp:
        json.dump(config, tmp, indent=2)
        tmp_path = tmp.name

    try:
        workflow_script = str(REPO_ROOT / "runscripts" / "workflow.py")
        cmd = [sys.executable, workflow_script, "--config", tmp_path]
        print(f"Launching workflow with {n_variants} variants...")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        sys.exit(result.returncode)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
