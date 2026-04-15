"""
Sensitivity campaign meta-runner.

Takes a JSON campaign spec, generates a set of perturbed RNA-seq datasets in
``$ECOLI_SOURCES/data/perturbations/`` via the operator library in
``ecoli-sources/processing/perturbations.py``, and emits a Nextflow multi-parca
config that runs parca + sims on all of them.

Does not launch Nextflow itself; the emitted config is run with the existing
``runscripts/workflow.py`` entrypoint.

See ``.claude/plans/dataset-sensitivity-exploration.md`` Part 4 for the 10k-sim
experiment design. This runner handles one sub-campaign (one operator family
over one source dataset); the higher-level driver composes many sub-campaigns.

Campaign spec JSON
------------------
::

    {
        "name": "pilot_expression_noise",
        "source_dataset_id": "vecoli_m9_glucose_minus_aas",
        "operator": "add_log_normal_noise",      # unary or binary
        "binary_partner": null,                   # required for binary operators
        "param_grid": {
            "sigma": [0.0, 0.1, 0.2, 0.4, 0.8],
            "seed":  [0, 1, 2, 3, 4]
        },
        "include_source_as_baseline": true,       # add parca_variant for the source
        "base_config": "configs/test_multi_parca.json",
        "sim": { "generations": 2, "n_init_sims": 2 }
    }

``param_grid`` is a Cartesian product. For every point, one perturbed dataset
is generated and one entry is added to ``parca_variants``.

Usage
-----
    uv run runscripts/run_sensitivity_campaign.py --spec <path-to-spec.json>

    # then:
    uv run runscripts/workflow.py --config configs/campaigns/<name>.json
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd

from wholecell.io.ingestion import resolve_ecoli_sources_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS_CONFIG_DIR = REPO_ROOT / "configs" / "campaigns"


def _ecoli_sources_dir() -> Path:
    resolved = resolve_ecoli_sources_path("$ECOLI_SOURCES")
    if resolved is None:
        raise RuntimeError("could not resolve $ECOLI_SOURCES")
    return Path(resolved)


def _load_manifest(ecoli_sources_dir: Path) -> pd.DataFrame:
    return pd.read_csv(ecoli_sources_dir / "data" / "manifest.tsv", sep="\t")


def _import_perturbations(ecoli_sources_dir: Path):
    """Import the perturbations module from the sibling ecoli-sources repo."""
    path = str(ecoli_sources_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    # Also ensure ecoli-sources/processing is importable as a package
    from processing import perturbations  # type: ignore
    return perturbations


def _cartesian_grid(param_grid: dict) -> list[dict]:
    """Expand a dict-of-lists into a list of parameter dicts."""
    keys = list(param_grid)
    value_lists = [param_grid[k] if isinstance(param_grid[k], list) else [param_grid[k]]
                   for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _generate_datasets(spec: dict, perturbations, manifest: pd.DataFrame,
                       ecoli_sources_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    operator = spec["operator"]
    data_dir = str(ecoli_sources_dir / "data")
    grid = _cartesian_grid(spec["param_grid"])

    is_binary = operator in perturbations.BINARY_OPERATORS
    generated_ids: list[str] = []

    if is_binary:
        partner = spec.get("binary_partner")
        if not partner:
            raise ValueError(f"operator {operator!r} is binary; "
                             f"spec must set 'binary_partner'")
        for params in grid:
            manifest, did = perturbations.make_binary_perturbation_variant(
                manifest,
                (spec["source_dataset_id"], partner),
                operator,
                params,
                data_dir,
            )
            generated_ids.append(did)
    else:
        for params in grid:
            manifest, did = perturbations.make_perturbation_variant(
                manifest,
                spec["source_dataset_id"],
                operator,
                params,
                data_dir,
            )
            generated_ids.append(did)

    return manifest, generated_ids


def _build_config(spec: dict, generated_ids: list[str]) -> dict:
    base_path = spec.get("base_config", "configs/test_multi_parca.json")
    with open(REPO_ROOT / base_path) as f:
        config = json.load(f)

    # Header
    config["experiment_id"] = spec["name"]

    # Sim overrides
    for k, v in spec.get("sim", {}).items():
        config[k] = v

    # Parca-options basal: source dataset
    config.setdefault("parca_options", {})
    config["parca_options"]["rnaseq_basal_dataset_id"] = spec["source_dataset_id"]
    config["parca_options"]["rnaseq_manifest_path"] = "$ECOLI_SOURCES/data/manifest.tsv"

    # Variants
    variants = []
    if spec.get("include_source_as_baseline", True):
        variants.append({"rnaseq_basal_dataset_id": spec["source_dataset_id"]})
    for did in generated_ids:
        variants.append({"rnaseq_basal_dataset_id": did})
    config["parca_variants"] = variants

    return config


def run_campaign(spec_path: str, *, dry_run: bool = False) -> str:
    with open(spec_path) as f:
        spec = json.load(f)

    name = spec["name"]
    ecoli_sources = _ecoli_sources_dir()
    manifest = _load_manifest(ecoli_sources)

    if dry_run:
        grid = _cartesian_grid(spec["param_grid"])
        print(f"[dry-run] campaign {name!r}: operator={spec['operator']!r} "
              f"source={spec['source_dataset_id']!r}")
        print(f"[dry-run] would generate {len(grid)} perturbed datasets and "
              f"a multi-parca config with "
              f"{len(grid) + (1 if spec.get('include_source_as_baseline', True) else 0)} "
              f"parca_variants")
        return ""

    perturbations = _import_perturbations(ecoli_sources)

    initial_rows = len(manifest)
    manifest, generated_ids = _generate_datasets(
        spec, perturbations, manifest, ecoli_sources
    )
    print(f"Generated {len(generated_ids)} perturbed datasets "
          f"({initial_rows} -> {len(manifest)} manifest rows)")

    config = _build_config(spec, generated_ids)

    CAMPAIGNS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAMPAIGNS_CONFIG_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    # Also write a sidecar that records the spec + generated ids for
    # post-hoc traceability.
    sidecar_path = CAMPAIGNS_CONFIG_DIR / f"{name}.campaign.json"
    with open(sidecar_path, "w") as f:
        json.dump({
            "spec": spec,
            "generated_dataset_ids": generated_ids,
            "include_source_as_baseline": spec.get("include_source_as_baseline", True),
        }, f, indent=2)

    print(f"Wrote Nextflow config: {out_path}")
    print(f"Wrote campaign sidecar: {sidecar_path}")
    print(f"Next: uv run runscripts/workflow.py --config {out_path.relative_to(REPO_ROOT)}")
    return str(out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--spec", required=True, help="Path to campaign JSON spec")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be generated, don't write anything")
    args = p.parse_args(argv)

    run_campaign(args.spec, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
