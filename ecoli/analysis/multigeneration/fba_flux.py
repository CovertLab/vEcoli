"""
Visualize FBA reaction fluxes over time for specified reactions with net flux calculation across multiple variants.

Supports two visualization modes:
1. 'grid' mode: Each row represents a variant, each column represents a reaction
2. 'stacked' mode: Each reaction gets its own chart, variants shown as different colored lines

You can specify the reactions and layout using parameters:
    "fba_flux": {
        # Required: specify BioCyc reaction IDs to visualize
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify layout mode ('grid' or 'stacked')
        # Default: 'stacked'
        "layout": "stacked"  # or "grid"
        }

This script is the dummy version of ecoli.analysis.multivariant.fba_flux, you can turn to origin file for more detail
"""

from ecoli.analysis.multivariant.fba_flux import plot

__all__ = ["plot"]
