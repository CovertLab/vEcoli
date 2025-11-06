"""
Analysis script toolbox functions
"""

from __future__ import annotations

import os


from wholecell.utils import filepath

LOW_RES_DIR = "low_res_plots"
SVG_DIR = "svg_plots"
HTML_DIR = "html_plots"
LOW_RES_DPI = 120


def exportFigure(
    plt,
    plotOutDir,
    plotOutFileName,
    metadata=None,
    transparent=False,
    dpi=LOW_RES_DPI,
    extension=None,
):
    if metadata is not None and "analysis_type" in metadata:
        analysis_type = metadata["analysis_type"]

        if analysis_type == "single":
            # Format metadata signature for single gen figure
            metadata_signature = "_".join(
                [
                    str(metadata["time"])[:13],
                    str(metadata["variant_function"]),
                    str(metadata["variant_index"]),
                    "Seed",
                    str(metadata["seed"]),
                    "Gen",
                    str(metadata["gen"]) + "/" + str(int(metadata["total_gens"]) - 1),
                    "Githash",
                    str(metadata["git_hash"])[:10],
                    "Desc",
                    str(metadata["description"]),
                ]
            )

        elif analysis_type == "multigen":
            # Format metadata signature for multi gen figure
            metadata_signature = "_".join(
                [
                    str(metadata["time"][:13]),
                    str(metadata["variant_function"]),
                    str(metadata["variant_index"]),
                    "Seed",
                    str(metadata["seed"]),
                    str(metadata["total_gens"]),
                    "gens",
                    "Githash",
                    str(metadata["git_hash"])[:10],
                    "Desc",
                    str(metadata["description"]),
                ]
            )

        elif analysis_type == "cohort":
            # Format metadata signature for cohort figure
            metadata_signature = "_".join(
                [
                    str(metadata["time"][:13]),
                    str(metadata["variant_function"]),
                    str(metadata["variant_index"]),
                    str(metadata["total_gens"]),
                    "gens",
                    "Githash",
                    str(metadata["git_hash"])[:10],
                    "Desc",
                    str(metadata["description"]),
                ]
            )

        elif analysis_type == "variant":
            # Format metadata signature for variant figure
            metadata_signature = "_".join(
                [
                    str(metadata["time"][:13]),
                    str(metadata["total_variants"]),
                    "variants",
                    str(metadata["total_gens"]),
                    "gens",
                    "Githash",
                    str(metadata["git_hash"])[:10],
                    "Desc",
                    str(metadata["description"]),
                ]
            )

        elif analysis_type == "parca":
            # Format metadata signature for parca figure
            metadata_signature = "_".join(
                [
                    str(metadata["time"][:13]),
                    "Githash",
                    str(metadata["git_hash"])[:10],
                    "Desc",
                    str(metadata["description"]),
                ]
            )

        elif analysis_type == "comparison":
            # Format metadata signature for a comparison figure
            metadata_signature = "_".join(
                [
                    str(metadata["time"][:13]),
                    str(metadata["total_variants"]),
                    "variants",
                    str(metadata["total_gens"]),
                    "gens",
                    "Githash",
                    str(metadata["git_hash"])[:10],
                    "Desc",
                    str(metadata["description"]),
                ]
            )

        else:
            # raise ValueError('Unknown analysis_type {}'.format(analysis_type))
            metadata_signature = "_"

        # Add metadata signature to the bottom of the plot
        # Don't accidentally trigger $TeX formatting$.
        metadata_signature = metadata_signature.replace("$", "")
        plt.figtext(0, 0, metadata_signature, size=8)

    # Make folders for holding alternate types of images
    filepath.makedirs(plotOutDir, LOW_RES_DIR)
    filepath.makedirs(plotOutDir, SVG_DIR)

    # Save images
    if extension:
        # Only save one type in main analysis directory if extension is given
        plt.savefig(
            os.path.join(plotOutDir, plotOutFileName + extension),
            dpi=dpi,
            transparent=transparent,
        )
    else:
        # Save all image types
        plt.savefig(
            os.path.join(plotOutDir, plotOutFileName + ".pdf"), transparent=transparent
        )
        plt.savefig(
            os.path.join(plotOutDir, SVG_DIR, plotOutFileName + ".svg"),
            transparent=transparent,
        )
        plt.savefig(
            os.path.join(plotOutDir, LOW_RES_DIR, plotOutFileName + ".png"),
            dpi=dpi,
            transparent=transparent,
        )


def _remove_first(remove_first: bool):
    """Slice to remove the first time point entry if remove_first is True"""
    return slice(1, None) if remove_first else slice(None)
