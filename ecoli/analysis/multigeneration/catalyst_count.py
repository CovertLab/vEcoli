"""
Visualize catalyst counts over time for specified BioCyc reactions across generations.
For each specific BioCyc ID reaction, this scripts will add all the catalysts which catalyse it:
```number of catalysts = sum(number of catalysts[i])```

Supports two visualization modes:
1. 'grid' mode: Each row represents a variant, each column represents a reaction's catalysts
2. 'stacked' mode: Each reaction's catalysts get their own chart, variants shown as different colored lines

You can specify the reactions and layout using parameters:
    "catalyst_count": {
        # Required: specify BioCyc reaction IDs to visualize
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify generations to visualize
        # If not specified, all generations will be used
        "generation": [1, 2, ...],
        # Optional: specify layout mode ('grid' or 'stacked')
        # Default: 'stacked'
        "layout": "stacked"  # or "grid"
        }

This script is the dummy version of ecoli.analysis.multivariant.catalyst_count, you can turn to origin file for more detail
"""

from ecoli.analysis.multivariant.catalyst_count import plot

__all__ = ["plot"]
