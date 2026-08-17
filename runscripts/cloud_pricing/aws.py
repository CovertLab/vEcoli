#!/usr/bin/env python3
"""
Check AWS EC2 spot prices across all regions and sort by $/physical core.

Physical core adjustment:
- Graviton instances (family endswith "g" or family == "a1") use $/vCPU as-is.
- AMD *7a/*8a (family endswith "a" with generation 7 or 8) use $/vCPU as-is.
- Everything else doubles $/vCPU to estimate $/physical core.

Usage:
    python aws_spot_prices.py [--top N] [--region-filter REGEX]
        [--instance-filter REGEX] [--profile PROFILE]
        [--regions REGION1,REGION2,...] [--max-workers N]

Prerequisites:
    pip install boto3 "botocore[crt]"
    AWS credentials configured (env vars, ~/.aws, or IAM role)
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import boto3


CHUNK_SIZE = 10000
PRODUCT_DESCRIPTIONS = ["Linux/UNIX"]


@dataclass
class SpotPrice:
    region: str
    instance_type: str
    vcpus: int
    price_per_hour: float
    best_az: str

    @property
    def price_per_vcpu(self) -> float:
        return self.price_per_hour / self.vcpus if self.vcpus else 0.0

    @property
    def physical_core_multiplier(self) -> int:
        return 1 if is_physical_core_mapping(self.instance_type) else 2

    @property
    def price_per_physical_core(self) -> float:
        return self.price_per_vcpu * self.physical_core_multiplier


def is_physical_core_mapping(instance_type: str) -> bool:
    """
    Heuristic for 1 vCPU == 1 physical core.
    - Graviton: family endswith "g" or family == "a1"
    - AMD Genoa/Bergamo: *7a or *8a (family endswith "a" with generation 7/8)
    """
    family = instance_type.split(".")[0]
    if family == "a1" or family.endswith("g"):
        return True
    if family.endswith("a"):
        match = re.search(r"(\d+)", family)
        if match and match.group(1) in {"7", "8"}:
            return True
    return False


def chunked(items: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def get_regions(session: boto3.Session) -> list[str]:
    try:
        ec2 = session.client("ec2", region_name="us-east-1")
        response = ec2.describe_regions(AllRegions=False)
        return sorted(r["RegionName"] for r in response["Regions"])
    except Exception:
        return ["us-gov-west-1", "us-gov-east-1"]


def get_instance_types(client, instance_filter: re.Pattern | None) -> dict[str, int]:
    paginator = client.get_paginator("describe_instance_types")
    instance_map: dict[str, int] = {}
    for page in paginator.paginate():
        for it in page["InstanceTypes"]:
            instance_type = it["InstanceType"]
            if instance_type.startswith("t"):
                continue
            if instance_filter and not instance_filter.search(instance_type):
                continue
            vcpus = it.get("VCpuInfo", {}).get("DefaultVCpus") or 0
            if vcpus:
                instance_map[instance_type] = vcpus
    return instance_map


def get_spot_prices_for_region(
    session: boto3.Session,
    region: str,
    instance_map: dict[str, int],
) -> list[SpotPrice]:
    client = session.client("ec2", region_name=region)
    start_time = datetime.now(timezone.utc)

    best_prices: dict[str, tuple[float, str]] = {}

    for chunk in chunked(sorted(instance_map.keys()), CHUNK_SIZE):
        paginator = client.get_paginator("describe_spot_price_history")
        for page in paginator.paginate(
            StartTime=start_time,
            InstanceTypes=chunk,
            ProductDescriptions=PRODUCT_DESCRIPTIONS,
        ):
            for entry in page["SpotPriceHistory"]:
                instance_type = entry["InstanceType"]
                price = float(entry["SpotPrice"])
                az = entry["AvailabilityZone"]

                current = best_prices.get(instance_type)
                if current is None or price < current[0]:
                    best_prices[instance_type] = (price, az)

    results: list[SpotPrice] = []
    for instance_type, (price, az) in best_prices.items():
        vcpus = instance_map.get(instance_type, 0)
        if vcpus:
            results.append(
                SpotPrice(
                    region=region,
                    instance_type=instance_type,
                    vcpus=vcpus,
                    price_per_hour=price,
                    best_az=az,
                )
            )

    return results


def fetch_region_prices(
    profile_name: str | None,
    region: str,
    instance_filter: re.Pattern | None,
) -> list[SpotPrice]:
    # Create a client per thread to avoid shared client contention.
    session = boto3.Session(profile_name=profile_name)
    client = session.client("ec2", region_name=region)
    instance_map = get_instance_types(client, instance_filter)
    if not instance_map:
        return []
    return get_spot_prices_for_region(session, region, instance_map)


def format_table(prices: list[SpotPrice], top: int) -> str:
    sorted_prices = sorted(prices, key=lambda p: p.price_per_physical_core)
    lines = [
        "\nSorted by $/physical core (spot)",
        "=" * 120,
        (
            f"{'Rank':<5} {'Region':<15} {'Instance Type':<18} "
            f"{'vCPUs':<6} {'$/hr':<10} {'$/vCPU':<12} {'$/core(adj)':<13} {'Best AZ':<10}"
        ),
        "-" * 120,
    ]

    for i, p in enumerate(sorted_prices[:top], 1):
        lines.append(
            f"{i:<5} {p.region:<15} {p.instance_type:<18} "
            f"{p.vcpus:<6} ${p.price_per_hour:<9.6f} "
            f"${p.price_per_vcpu:<11.6f} ${p.price_per_physical_core:<12.6f} {p.best_az:<10}"
        )

    return "\n".join(lines)


def family_from_instance_type(instance_type: str) -> str:
    return instance_type.split(".")[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check AWS EC2 spot prices across all regions"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top results to show"
    )
    parser.add_argument(
        "--region-filter",
        type=str,
        default=None,
        help="Regex filter for regions (e.g., 'us-' for US regions)",
    )
    parser.add_argument(
        "--instance-filter",
        type=str,
        default=None,
        help="Regex filter for instance types (e.g., '^c7a\\.')",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile name (defaults to boto3/AWS config resolution)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="Comma-separated list of regions to query (overrides discovery)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel region workers (defaults to min(8, cpu count, regions))",
    )
    args = parser.parse_args()

    region_filter = re.compile(args.region_filter) if args.region_filter else None
    instance_filter = re.compile(args.instance_filter) if args.instance_filter else None

    session = boto3.Session(profile_name=args.profile)
    if args.regions:
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    else:
        regions = get_regions(session)
    if region_filter:
        regions = [r for r in regions if region_filter.search(r)]

    if not regions:
        print("No regions to query after filtering.")
        return

    default_workers = min(8, os.cpu_count() or 1, len(regions))
    max_workers = args.max_workers or default_workers

    all_prices: list[SpotPrice] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_region_prices, args.profile, region, instance_filter
            ): region
            for region in regions
        }
        for future in as_completed(futures):
            region_prices = future.result()
            if region_prices:
                all_prices.extend(region_prices)

    if not all_prices:
        print("No spot prices found. Check AWS credentials and permissions.")
        return

    print(
        f"Found {len(all_prices)} price points across {len(set(p.region for p in all_prices))} regions"
    )
    print(format_table(all_prices, args.top))

    families = sorted({family_from_instance_type(p.instance_type) for p in all_prices})
    if families:
        print("\n" + "=" * 120)
        print("SUMMARY: Best option per family (by $/physical core)")
        print("-" * 120)
        for family in families:
            family_prices = [
                p
                for p in all_prices
                if family_from_instance_type(p.instance_type) == family
            ]
            if not family_prices:
                continue
            best = min(family_prices, key=lambda p: p.price_per_physical_core)
            print(
                f"  {family:<8} Best: {best.instance_type:<18} {best.region:<15} "
                f"${best.price_per_physical_core:.6f}/core/hr  "
                f"${best.price_per_vcpu:.6f}/vCPU/hr  "
                f"(x{best.physical_core_multiplier})"
            )


if __name__ == "__main__":
    main()
