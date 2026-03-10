import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # RNA-seq Data QC

        Compare experimental RNA-seq data against reference datasets using the
        `wholecell.io.data_qc` module.
        """
    )
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from wholecell.io.data_qc import (
        compare_rnaseq_tables,
        load_essential_genes,
        load_gene_annotations,
    )
    from wholecell.io.ingestion import ingest_transcriptome

    return (
        Path,
        compare_rnaseq_tables,
        ingest_transcriptome,
        load_essential_genes,
        load_gene_annotations,
        np,
        pd,
    )


@app.cell
def _(Path):
    MANIFEST_PATH = Path("experimental_data/rnaseq/manifest.tsv")
    REF_DATASET_ID = "ref_0001"
    EXPT_DATASET_ID = "gbw_0001"
    return EXPT_DATASET_ID, MANIFEST_PATH, REF_DATASET_ID


@app.cell
def _(load_essential_genes, load_gene_annotations):
    gene_annotations = load_gene_annotations()
    essential_genes = load_essential_genes()
    return essential_genes, gene_annotations


@app.cell
def _(EXPT_DATASET_ID, MANIFEST_PATH, REF_DATASET_ID, ingest_transcriptome):
    ref_tpm, ref_meta = ingest_transcriptome(MANIFEST_PATH, REF_DATASET_ID)
    expt_tpm, expt_meta = ingest_transcriptome(MANIFEST_PATH, EXPT_DATASET_ID)
    return expt_meta, expt_tpm, ref_meta, ref_tpm


@app.cell
def _(expt_meta, mo, ref_meta):
    mo.md(
        f"""
    ## Datasets

    | | Reference | Experimental |
    |---|---|---|
    | **Dataset ID** | `{ref_meta["dataset_id"]}` | `{expt_meta["dataset_id"]}` |
    | **Description** | {ref_meta["dataset_description"]} | {expt_meta["dataset_description"]} |
    | **Source** | {ref_meta["data_source"]} | {expt_meta["data_source"]} |
    | **Strain** | {ref_meta.get("strain", "N/A")} | {expt_meta.get("strain", "N/A")} |
    | **Condition** | {ref_meta.get("condition", "N/A")} | {expt_meta.get("condition", "N/A")} |
    """
    )
    return


@app.cell
def _(
    compare_rnaseq_tables,
    essential_genes,
    expt_tpm,
    gene_annotations,
    ref_tpm,
):
    result = compare_rnaseq_tables(
        ref_tpm,
        expt_tpm,
        gene_annotations=gene_annotations,
        essential_genes=essential_genes,
    )
    return (result,)


@app.cell(hide_code=True)
def _(mo, result):
    stats = result.summary_stats
    mo.md(
        f"""
        ## Summary Statistics

        | Metric | Value |
        |--------|-------|
        | Genes in reference | {stats["n_genes_ref_total"]:,} |
        | Genes in experimental | {stats["n_genes_expt_total"]:,} |
        | Genes matched | {stats["n_genes_matched"]:,} |
        | Genes only in reference | {stats["n_genes_only_in_ref"]:,} |
        | Genes only in experimental | {stats["n_genes_only_in_expt"]:,} |
        | Reference coverage | {stats["pct_ref_covered"]:.1f}% |
        | Experimental coverage | {stats["pct_expt_covered"]:.1f}% |
        | Pearson r | {stats["pearson_r"]:.4f} |
        | Spearman r | {stats["spearman_r"]:.4f} |
        | RMSE | {stats["rmse"]:.2f} |
        | Mean log2 fold change | {stats["log2_fold_change_mean"]:.3f} |
        | Std log2 fold change | {stats["log2_fold_change_std"]:.3f} |
        | | |
        | **Essential genes matched** | {stats.get("n_essential_genes_matched", "N/A"):,} / {stats.get("n_essential_genes_total", "N/A"):,} |
        | Essential only in reference | {stats.get("n_essential_only_in_ref", "N/A")} |
        | Essential only in experimental | {stats.get("n_essential_only_in_expt", "N/A")} |
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""## Comparison Table (first 20 rows)""")
    return


@app.cell
def _(result):
    result.comparison_table.head(20)
    return


@app.cell
def _(result):
    ## essential genes
    result.comparison_table[result.comparison_table["gene_essential"]].sort_values(
        "expt_tpm"
    )
    return


@app.cell
def _(np, result):
    result.comparison_table[
        (result.comparison_table["gene_essential"])
        & (np.isnan(result.comparison_table["expt_tpm"]))
    ].sort_values("expt_tpm")
    return


@app.cell
def _(mo, result):
    _n_only_ref = len(result.genes_only_in_ref)
    _n_only_expt = len(result.genes_only_in_expt)
    mo.md(
        f"""
        ## Missing Genes

        **Genes only in reference** ({_n_only_ref} total): 
        `{", ".join(result.genes_only_in_ref[:10])}{"..." if _n_only_ref > 10 else ""}`

        **Genes only in experimental** ({_n_only_expt} total):
        `{", ".join(result.genes_only_in_expt[:10])}{"..." if _n_only_expt > 10 else ""}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Interactive Scatter Plot: Reference vs Experimental TPM

    Hover over points to see gene ID and name. Essential genes are shown in red.
    """
    )
    return


@app.cell(hide_code=True)
def _(np, pd, result):
    import altair as alt

    matched = result.comparison_table.dropna(subset=["ref_tpm", "expt_tpm"]).copy()
    matched["log10_ref_tpm"] = np.log10(matched["ref_tpm"] + 1)
    matched["log10_expt_tpm"] = np.log10(matched["expt_tpm"] + 1)
    matched["essentiality"] = matched["gene_essential"].map(
        {True: "Essential", False: "Non-essential"}
    )

    scatter = (
        alt.Chart(matched)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X(
                "log10_ref_tpm:Q",
                title="log10(Reference TPM + 1)",
                scale=alt.Scale(domain=[0, 5]),
            ),
            y=alt.Y(
                "log10_expt_tpm:Q",
                title="log10(Experimental TPM + 1)",
                scale=alt.Scale(domain=[0, 5]),
            ),
            color=alt.Color(
                "essentiality:N",
                title="Gene Type",
                scale=alt.Scale(
                    domain=["Essential", "Non-essential"], range=["#d62728", "#1f77b4"]
                ),
            ),
            tooltip=[
                "gene_id:N",
                "gene_name:N",
                "ref_tpm:Q",
                "expt_tpm:Q",
                "essentiality:N",
            ],
        )
    )

    identity_line = (
        alt.Chart(pd.DataFrame({"x": [0, 5], "y": [0, 5]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )

    chart = (
        (identity_line + scatter)
        .properties(width=600, height=600, title="RNA-seq TPM Comparison")
        .configure_axis(grid=True)
    )

    chart
    return (matched,)


@app.cell
def _(mo):
    mo.md("""## Log2 Fold Change Distribution""")
    return


@app.cell
def _(matched, np):
    import matplotlib.pyplot as plt

    log2_fc = np.log2((matched["expt_tpm"] + 1) / (matched["ref_tpm"] + 1))

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(log2_fc, bins=100, edgecolor="none", alpha=0.7)
    ax2.axvline(0, color="r", linestyle="--", linewidth=1)
    ax2.axvline(
        log2_fc.mean(),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {log2_fc.mean():.2f}",
    )
    ax2.set_xlabel("log2(Experimental / Reference)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Log2 Fold Changes")
    ax2.legend()

    plt.tight_layout()
    fig2
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""TODO: add in variance-aware comparative analyses (e.g. fill in and use the optional tpm_std columns)"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Consider... 

    - if/how/when to implement data correction or flexibilization. For example, ParCa fails on an attempt to ingest new data. How to troubleshoot?
       - Identify candidate problematic datapoints, e.g. by most different from reference, perhaps weighted by essentiality or pathway designation
       - Options to remediate
          - drop sets of problematic datapoints (revert to reference measurement) until issue resolves; how to generalize?
          - "flex" data in a rational way with respect to its variance. E.g. "what's the minimal number of genes I have to modify within n*[std] to successfully integrate"?
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
