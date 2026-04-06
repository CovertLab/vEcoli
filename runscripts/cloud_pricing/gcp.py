#!/usr/bin/env python3
"""
Check latest generation GCP spot VM prices across all regions and sort by
$/vCPU and $/physical core. T2D instances are included despite using older CPUs
because they can be very cost-effective for our sim workloads.

Physical core adjustments map $/vCPU to $/physical core:
- Families where 1 vCPU == 1 physical core: any family ending in "a" (ARM), h4d, t2d
- All others: assume 2 vCPU per physical core

Usage:
    python check-spot-prices.py [--top N] [--region-filter PATTERN]
        [--instance-type TYPE] [--project PROJECT_ID]
        [--family-filter FAMILY1,FAMILY2,...]

Prerequisites:
    pip install google-cloud-billing google-cloud-compute
    gcloud auth application-default login

API Reference:
    https://cloud.google.com/billing/docs/how-to/get-pricing-information-api
    https://cloud.google.com/compute/docs/reference/rest/v1/machineTypes
"""

import argparse
import re
import sys
from dataclasses import dataclass

# Physical core multipliers (price_per_physical_core = price_per_vcpu * multiplier)
CORE_MULTIPLIERS = {
    "n4d": 2,
    "c4d": 2,
    "h4d": 1,
    "h3": 1,
    "n4": 2,
    "c4": 2,
    "c4a": 1,
    "n4a": 1,
    "t2d": 1,
}

# Machine families to check
MACHINE_FAMILIES = list(CORE_MULTIPLIERS.keys())

# Compute Engine service name
COMPUTE_ENGINE_SERVICE = "services/6F81-5844-456A"


@dataclass
class MachineTypeSpec:
    """Specs for a specific machine type from Compute Engine API."""

    name: str  # e.g., "n4d-standard-2"
    family: str  # e.g., "n4d"
    instance_type: str  # e.g., "standard"
    vcpus: int
    ram_gb: float

    @property
    def ram_per_vcpu(self) -> float:
        return self.ram_gb / self.vcpus if self.vcpus > 0 else 0


@dataclass
class SpotPrice:
    region: str
    family: str
    instance_type: str  # standard, highcpu, highmem
    price_per_hour: float  # Total $/hr for reference instance
    vcpus: int  # vCPUs in reference instance
    ram_gb: float  # Actual RAM from Compute Engine API
    core_multiplier: int

    @property
    def price_per_vcpu(self) -> float:
        """Price per vCPU-hour."""
        return self.price_per_hour / self.vcpus if self.vcpus > 0 else 0

    @property
    def price_per_physical_core(self) -> float:
        """Price per physical core-hour."""
        return self.price_per_vcpu * self.core_multiplier

    @property
    def cores_per_dollar(self) -> float:
        """Physical cores per dollar-hour (higher is better)."""
        return (
            1.0 / self.price_per_physical_core
            if self.price_per_physical_core > 0
            else 0
        )

    @property
    def machine_type(self) -> str:
        """Full machine type name."""
        return f"{self.family}-{self.instance_type}-{self.vcpus}"

    @property
    def ram_per_vcpu(self) -> float:
        """RAM per vCPU in GB."""
        return self.ram_gb / self.vcpus if self.vcpus > 0 else 0


def extract_price_from_sku(sku) -> tuple[float, list[str]]:
    """
    Extract hourly price and regions from a SKU.
    Returns (price_per_hour, [regions]).
    """
    regions = list(sku.service_regions)

    if not sku.pricing_info:
        return 0.0, regions

    pricing = sku.pricing_info[0]
    if not pricing.pricing_expression or not pricing.pricing_expression.tiered_rates:
        return 0.0, regions

    # Get the first tier (base price)
    tier = pricing.pricing_expression.tiered_rates[0]
    unit_price = tier.unit_price

    # Convert to float (units + nanos/1e9)
    price = float(unit_price.units) + (float(unit_price.nanos) / 1e9)

    return price, regions


def get_machine_type_specs(
    project: str, zone: str = "us-central1-a"
) -> dict[str, MachineTypeSpec]:
    """
    Fetch machine type specifications from Compute Engine API.

    Returns a dict of machine_type_name -> MachineTypeSpec
    """
    try:
        from google.cloud import compute_v1
    except ImportError:
        print(
            "Warning: google-cloud-compute not installed, using estimated RAM values."
        )
        print("Run: pip install google-cloud-compute")
        return {}

    specs = {}

    try:
        client = compute_v1.MachineTypesClient()

        # List machine types in the zone
        request = compute_v1.ListMachineTypesRequest(
            project=project,
            zone=zone,
        )

        for mt in client.list(request=request):
            name = mt.name  # e.g., "n4d-standard-2"

            # Parse family and instance type from name
            # Format: {family}-{type}-{vcpus} e.g., "n4d-standard-2"
            parts = name.split("-")
            if len(parts) >= 3:
                family = parts[0].lower()
                instance_type = parts[1].lower()

                # Only care about our target families
                if family not in MACHINE_FAMILIES:
                    continue

                # Only care about standard/highcpu/highmem
                if instance_type not in ("standard", "highcpu", "highmem"):
                    continue

                specs[name] = MachineTypeSpec(
                    name=name,
                    family=family,
                    instance_type=instance_type,
                    vcpus=mt.guest_cpus,
                    ram_gb=mt.memory_mb / 1024.0,
                )

        print(f"  Loaded {len(specs)} machine type specs from Compute Engine API")

    except Exception as e:
        print(f"Warning: Could not fetch machine type specs: {e}")
        return {}

    return specs


def extract_vcpus_from_description(desc: str) -> int:
    """Extract vCPU count from SKU description."""
    # Try "N vCPU" pattern
    match = re.search(r"(\d+)\s*VCPU", desc.upper())
    if match:
        return int(match.group(1))

    # Try "STANDARD-N" or "HIGHCPU-N" pattern
    match = re.search(r"(?:STANDARD|HIGHCPU|HIGHMEM)-(\d+)", desc.upper())
    if match:
        return int(match.group(1))

    return 0


def get_spot_prices_from_api(
    instance_type: str = "standard",
    project: str | None = None,
) -> list[SpotPrice]:
    """
    Fetch spot prices using CPU+RAM component pricing and real machine specs.

    The Cloud Billing API provides per-vCPU and per-GB-RAM pricing for each family.
    We combine this with actual machine specs from Compute Engine API to calculate
    the total price for predefined instance types.

    Args:
        instance_type: Type of instance (standard, highcpu, highmem)
        project: GCP project ID for fetching machine type specs (optional)
    """
    try:
        from google.cloud import billing_v1
    except ImportError:
        print("Error: google-cloud-billing not installed.")
        print("Run: pip install google-cloud-billing")
        return []

    prices = []

    print(f"Fetching spot prices for {instance_type} instances...\n")

    # Get actual machine type specs from Compute Engine API
    machine_specs: dict[str, MachineTypeSpec] = {}
    if project:
        machine_specs = get_machine_type_specs(project)
    else:
        # Try to get project from gcloud config
        try:
            import subprocess

            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                project = result.stdout.strip()
                machine_specs = get_machine_type_specs(project)
        except Exception:
            pass

    # Build fallback RAM estimates if we couldn't get specs
    fallback_ram_per_vcpu = {
        "standard": {family: 4 for family in MACHINE_FAMILIES},
        "highcpu": {family: 2 for family in MACHINE_FAMILIES},
        "highmem": {family: 8 for family in MACHINE_FAMILIES},
    }

    try:
        client = billing_v1.CloudCatalogClient()
    except Exception as e:
        print(f"Error initializing billing client: {e}")
        print("\nTo authenticate, run:")
        print("  gcloud auth application-default login")
        return []

    # Build a map of family -> resource_type -> region -> price
    # resource_type is either "cpu" or "ram"
    family_prices: dict[str, dict[str, dict[str, float]]] = {
        f: {"cpu": {}, "ram": {}} for f in MACHINE_FAMILIES
    }

    # List all SKUs once and filter in memory
    request = billing_v1.ListSkusRequest(
        parent=COMPUTE_ENGINE_SERVICE,
        currency_code="USD",
    )

    try:
        sku_count = 0
        matched_count = 0

        for sku in client.list_skus(request=request):
            sku_count += 1
            desc = sku.description.upper()

            # Skip if not spot/preemptible
            if "SPOT" not in desc and "PREEMPTIBLE" not in desc:
                continue

            # Check each family
            for family in MACHINE_FAMILIES:
                family_upper = family.upper()

                # Must contain the family name
                if family_upper not in desc:
                    continue

                # Precise matching to avoid e.g. "N4" matching "N4D"
                if f" {family_upper} " not in f" {desc} ":
                    if not desc.startswith(f"{family_upper} "):
                        continue

                # Determine if this is CPU or RAM (component pricing)
                is_cpu = "CORE" in desc or "VCPU" in desc
                is_ram = "RAM" in desc

                # Skip if neither or both (we want exact match)
                if not (is_cpu ^ is_ram):
                    continue

                # Skip predefined instance SKUs (we want component pricing)
                # Component SKUs don't have CUSTOM but also don't have instance size numbers
                if "CUSTOM" in desc:
                    # Skip custom extended, but keep regular custom (which is component pricing)
                    if "EXTENDED" in desc:
                        continue

                # Extract price and regions
                price, regions = extract_price_from_sku(sku)
                if price <= 0:
                    continue

                resource_type = "cpu" if is_cpu else "ram"

                for region in regions:
                    # Only keep lowest price per region
                    current = family_prices[family][resource_type].get(region)
                    if current is None or price < current:
                        family_prices[family][resource_type][region] = price

                matched_count += 1
                break  # Don't match multiple families for same SKU

        print(
            f"  Scanned {sku_count} SKUs, matched {matched_count} component spot pricing entries"
        )

    except Exception as e:
        print(f"  Error fetching SKUs: {e}")
        return []

    # Report per-family results
    for family in MACHINE_FAMILIES:
        cpu_count = len(family_prices[family]["cpu"])
        ram_count = len(family_prices[family]["ram"])
        print(f"  {family.upper()}: {cpu_count} CPU regions, {ram_count} RAM regions")

    # Use reference vCPU count (2 is common smallest size)
    ref_vcpus = 2

    # Combine CPU and RAM prices into SpotPrice objects for each family/region
    for family in MACHINE_FAMILIES:
        cpu_prices = family_prices[family]["cpu"]
        ram_prices = family_prices[family]["ram"]

        # Get the actual RAM for this machine type
        machine_type_name = f"{family}-{instance_type}-{ref_vcpus}"
        if machine_type_name in machine_specs:
            ram_gb = machine_specs[machine_type_name].ram_gb
        else:
            # Use fallback estimate
            ram_per_vcpu = fallback_ram_per_vcpu.get(instance_type, {}).get(family, 4.0)
            ram_gb = ref_vcpus * ram_per_vcpu

        for region in cpu_prices:
            if region in ram_prices:
                # Calculate total price: (cpu_price * vcpus) + (ram_price * ram_gb)
                cpu_price = cpu_prices[region]
                ram_price = ram_prices[region]
                total_price = (cpu_price * ref_vcpus) + (ram_price * ram_gb)

                prices.append(
                    SpotPrice(
                        region=region,
                        family=family,
                        instance_type=instance_type,
                        price_per_hour=total_price,
                        vcpus=ref_vcpus,
                        ram_gb=ram_gb,
                        core_multiplier=CORE_MULTIPLIERS[family],
                    )
                )

    return prices


def format_table(
    prices: list[SpotPrice], sort_by: str = "price_per_vcpu", top: int = 20
) -> str:
    """Format prices as a table."""
    instance_type = prices[0].instance_type if prices else "standard"
    if sort_by == "price_per_vcpu":
        sorted_prices = sorted(prices, key=lambda p: p.price_per_vcpu)
        header = f"Sorted by $/vCPU ({instance_type} instances)"
    elif sort_by == "price_per_physical_core":
        sorted_prices = sorted(prices, key=lambda p: p.price_per_physical_core)
        header = "Sorted by $/Physical Core (lower is better)"
    else:
        sorted_prices = sorted(prices, key=lambda p: -p.cores_per_dollar)
        header = "Sorted by Physical Cores/$ (higher is better)"

    lines = [
        f"\n{header}",
        "=" * 110,
        f"{'Rank':<5} {'Region':<20} {'Machine Type':<18} {'RAM/vCPU':<10} {'$/vCPU/hr':<12} {'$/core/hr':<12} {'Cores/$':<12}",
        "-" * 110,
    ]

    for i, p in enumerate(sorted_prices[:top], 1):
        lines.append(
            f"{i:<5} {p.region:<20} {p.machine_type:<18} "
            f"{p.ram_per_vcpu:<10.2f} "
            f"${p.price_per_vcpu:<11.6f} ${p.price_per_physical_core:<11.6f} {p.cores_per_dollar:<12.3f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check GCP spot VM prices using Cloud Billing Catalog API"
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
        "--family-filter",
        type=str,
        default=None,
        help="Comma-separated list of families to check (e.g., 'n4d,c4d')",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        choices=["standard", "highcpu", "highmem"],
        default="standard",
        help="Instance type: standard, highcpu, or highmem",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP project ID for fetching actual RAM specs (uses gcloud default if not specified)",
    )
    args = parser.parse_args()

    # Optionally filter families
    global MACHINE_FAMILIES
    if args.family_filter:
        MACHINE_FAMILIES = [f.strip().lower() for f in args.family_filter.split(",")]
        for f in MACHINE_FAMILIES:
            if f not in CORE_MULTIPLIERS:
                print(f"Error: Unknown family '{f}'")
                sys.exit(1)

    # Fetch prices from API
    prices = get_spot_prices_from_api(
        instance_type=args.instance_type, project=args.project
    )

    if not prices:
        print("\nNo prices found. Possible issues:")
        print("  1. Run: gcloud auth application-default login")
        print("  2. Ensure you have billing API access")
        print("  3. The machine families may not have spot pricing yet")
        sys.exit(1)

    # Filter by region if specified
    if args.region_filter:
        pattern = re.compile(args.region_filter)
        prices = [p for p in prices if pattern.search(p.region)]

    # Filter by family if specified
    if args.family_filter:
        prices = [p for p in prices if p.family in MACHINE_FAMILIES]

    print(
        f"\nFound {len(prices)} price points across {len(set(p.region for p in prices))} regions"
    )

    # Print tables
    print(format_table(prices, "price_per_vcpu", args.top))
    print()
    print(format_table(prices, "price_per_physical_core", args.top))

    # Summary stats
    print("\n" + "=" * 110)
    print(f"SUMMARY: Best {args.instance_type} option per family (by $/vCPU)")
    print("-" * 110)
    for family in sorted(set(p.family for p in prices)):
        family_prices = [p for p in prices if p.family == family]
        if family_prices:
            best = min(family_prices, key=lambda p: p.price_per_vcpu)
            print(
                f"  {best.machine_type:<18} Best: {best.region:<20} "
                f"${best.price_per_vcpu:.5f}/vCPU/hr  "
                f"${best.price_per_physical_core:.5f}/core/hr  "
                f"(core x{best.core_multiplier})"
            )

    print("\n" + "=" * 110)
    print(f"SUMMARY: Best {args.instance_type} option per family (by Cores/$)")
    print("-" * 110)
    best_overall = max(prices, key=lambda p: p.cores_per_dollar)
    for family in sorted(set(p.family for p in prices)):
        family_prices = [p for p in prices if p.family == family]
        if family_prices:
            best = max(family_prices, key=lambda p: p.cores_per_dollar)
            marker = " <-- BEST OVERALL" if best == best_overall else ""
            print(
                f"  {best.machine_type:<18} Best: {best.region:<20} "
                f"Cores/$: {best.cores_per_dollar:>7.3f}  "
                f"${best.price_per_physical_core:.5f}/core/hr{marker}"
            )


if __name__ == "__main__":
    main()
