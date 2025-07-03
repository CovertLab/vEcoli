"""
Process vulnerability data from comma-separated JSON format.

This script processes JSON data containing package vulnerability information,
generates a markdown report with vulnerability details, and creates a shell
script to apply package upgrades using uv.

Expected JSON format:
{
    "name": "package_name",
    "version": "current_version",
    "vulns": [
        {
            "id": "VULNERABILITY_ID",
            "fix_versions": ["fixed_version"],
            "aliases": ["ALIAS1", "ALIAS2"],
            "description": "Vulnerability description"
        }
    ]
}
"""

import os
import json
import sys
from typing import Any
from datetime import datetime
import argparse
from packaging.version import Version


def generate_markdown_report(packages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Generate a markdown report of vulnerabilities and upgrades."""

    markdown = f"""# Security Vulnerability Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Found vulnerabilities in **{len(packages)}** packages requiring updates.

## Package Upgrades Overview

| Package | Current Version | Recommended Version | Vulnerabilities |
|---------|----------------|-------------------|-----------------|
"""

    # Package summary table
    upgrade_commands = []

    for pkg in packages:
        name = pkg.get("name", "Unknown")
        current_version = pkg.get("version", "Unknown")
        vulns = pkg.get("vulns", [])

        # Find the highest fix version across all vulnerabilities
        all_fix_versions = []
        vuln_count = len(vulns)

        for vuln in vulns:
            fix_versions = vuln.get("fix_versions", [])
            all_fix_versions.extend([Version(v) for v in fix_versions if v])

        recommended_version = max(all_fix_versions) if all_fix_versions else "Unknown"

        markdown += f"| **{name}** | {current_version} | **{recommended_version}** | {vuln_count} |\n"

        if recommended_version != "Unknown":
            upgrade_commands.append(f'-P "{name}=={recommended_version}"')

    markdown += "\n## Detailed Vulnerability Information\n\n"

    # Detailed vulnerability information
    for pkg in packages:
        name = pkg.get("name", "Unknown")
        current_version = pkg.get("version", "Unknown")
        vulns = pkg.get("vulns", [])

        markdown += f"### {name} (v{current_version})\n\n"

        if not vulns:
            markdown += "No specific vulnerability details available.\n\n"
            continue

        markdown += "| Vulnerability ID | Fix Versions | Aliases |\n"
        markdown += "|-----------------|-------------|---------|\n"

        for vuln in vulns:
            vuln_id = vuln.get("id", "Unknown")
            fix_versions = ", ".join(vuln.get("fix_versions", ["Unknown"]))
            aliases = ", ".join(vuln.get("aliases", []))

            markdown += f"| {vuln_id} | {fix_versions} | {aliases} |\n"

        markdown += "\n"

    markdown += """
## Recommended Actions

1. Review the vulnerability details above.
2. Close and reopen this PR to trigger CI/CD tests.
3. Approve and merge the PR if everything looks good.

---
*This report was generated automatically. Please verify all upgrades before applying.*
"""

    return markdown, upgrade_commands


def main():
    parser = argparse.ArgumentParser(
        description="Process vulnerability data and generate reports"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Input file with comma-separated JSONs (default: stdin)",
    )
    parser.add_argument(
        "--output-md", default="vulnerability_report.md", help="Output markdown file"
    )
    parser.add_argument(
        "--output-sh",
        default="apply_security_upgrades.sh",
        help="Output shell script file",
    )

    args = parser.parse_args()

    # Read input data
    if args.input_file:
        try:
            with open(args.input_file, "r") as f:
                input_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Reading from stdin... (Ctrl+D to end)")
        input_data = json.load(sys.stdin)

    if not input_data:
        print("Error: No input data provided.", file=sys.stderr)
        sys.exit(1)

    # Process the data
    packages = [pkg for pkg in input_data["dependencies"] if pkg["vulns"]]

    print(f"📋 Detected {len(packages)} vulnerable packages")

    # Generate markdown report
    markdown_content, upgrade_commands = generate_markdown_report(packages)
    with open(args.output_md, "w") as f:
        f.write(markdown_content)
    print(f"📄 Markdown report saved to: {args.output_md}")

    # Generate shell script
    script = f"""#!/bin/bash
# Security upgrade script
# Generated automatically from vulnerability analysis

set -e  # Exit on any error

echo "🔒 Applying security upgrades..."
echo "This script will upgrade vulnerable packages using uv lock --upgrade-package"
uv lock {" ".join(upgrade_commands)}

echo "✅ All security upgrades completed successfully!"
"""
    with open(args.output_sh, "w") as f:
        f.write(script)

    # Make script executable
    os.chmod(args.output_sh, 0o755)
    print(f"🔧 Shell script saved to: {args.output_sh} (executable)")

    print("\n✅ Processing complete!")
    print(f"Review the report: {args.output_md}")
    print(f"Apply upgrades: ./{args.output_sh}")


if __name__ == "__main__":
    main()
