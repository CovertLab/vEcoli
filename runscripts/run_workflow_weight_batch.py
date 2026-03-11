"""
Run the existing Nextflow workflow N times, once per row of objective weights in a CSV.
Each run uses the same base workflow config with process_configs objective_weights overridden.

Usage:
  python runscripts/run_workflow_weight_batch.py --config path/to/workflow_config.json --csv path/to/weights.csv
  python runscripts/run_workflow_weight_batch.py --config workflow_config.json --csv weights.csv --indices 313 171 542
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

# Default paths relative to repo root
DEFAULT_WORKFLOW_CONFIG = (
    "configs/metabolism_redux_classic.json"  # or your actual workflow config path
)
RUNSCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNSCRIPT_DIR.parent


def objective_weights_from_row(row, homeostatic=1.0):
    return {
        "secretion": float(row["lambda_sec"]),
        "efficiency": float(row["lambda_eff"]),
        "kinetics": float(row["lambda_kin"]),
        "diversity": float(row["lambda_div"]),
        "homeostatic": homeostatic,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run existing Nextflow workflow once per CSV row of objective weights."
    )
    parser.add_argument(
        "--config", required=True, help="Path to base workflow config JSON."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with lambda_sec, lambda_eff, lambda_kin, lambda_div (and optionally Index).",
    )
    parser.add_argument(
        "--indices", nargs="*", help="Optional: run only rows with these Index values."
    )
    parser.add_argument(
        "--rows",
        nargs="*",
        type=int,
        help="Optional: run only these 0-based row indices.",
    )
    parser.add_argument(
        "--experiment-id-prefix",
        default="metabolism_weights",
        help="Prefix for experiment_id; suffix will be row Index or row index.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Base out dir for emitter_arg.out_dir (default: keep from base config).",
    )
    args = parser.parse_args()

    # Resolve config path: if relative, treat as relative to repo root
    config_path_arg = Path(args.config)
    if not config_path_arg.is_absolute():
        config_path_arg = REPO_ROOT / args.config
    if not config_path_arg.exists():
        sys.exit(f"Config file not found: {config_path_arg}")

    import pandas as pd

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / args.csv
    if not csv_path.exists():
        sys.exit(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for c in ["lambda_sec", "lambda_eff", "lambda_kin", "lambda_div"]:
        if c not in df.columns:
            sys.exit(f"CSV missing column: {c}")

    if args.indices is not None:
        if "Index" not in df.columns:
            sys.exit("--indices requires an 'Index' column in the CSV.")
        df = df[df["Index"].astype(str).isin([str(i) for i in args.indices])]
    if args.rows is not None:
        df = df.iloc[args.rows]

    with open(config_path_arg) as f:
        base_config = json.load(f)

    repo_root = REPO_ROOT
    n = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        row_id = row.get("Index", idx)
        if isinstance(row_id, (int, float)):
            row_id = int(row_id)
        exp_id = f"{args.experiment_id_prefix}_{row_id}"

        config = copy.deepcopy(base_config)
        config.setdefault("process_configs", {})
        config["process_configs"].setdefault("ecoli-metabolism-redux-classic", {})
        config["process_configs"]["ecoli-metabolism-redux-classic"][
            "objective_weights"
        ] = objective_weights_from_row(row)
        config["experiment_id"] = exp_id
        if args.out_dir is not None:
            config.setdefault("emitter_arg", {})["out_dir"] = str(
                Path(args.out_dir) / exp_id
            )

        per_run_config_path = (
            Path(repo_root) / "nextflow_temp" / exp_id / "workflow_config.json"
        )
        per_run_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_run_config_path, "w") as f:
            json.dump(config, f, indent=2)

        cmd = [
            sys.executable,
            str(RUNSCRIPT_DIR / "workflow.py"),
            "--config",
            str(per_run_config_path),
        ]
        print(
            f"[{i + 1}/{n}] Running workflow for row Index={row_id} (experiment_id={exp_id}) ..."
        )
        try:
            result = subprocess.run(cmd, cwd=str(repo_root))
            if result.returncode != 0:
                sys.exit(result.returncode)
        except Exception:
            raise
    print("Done.")


if __name__ == "__main__":
    main()
