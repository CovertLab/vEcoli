"""
Pandera schemas for experimental RNA-seq data.

Canonical format: one file per condition, two required columns (gene_id, tpm_mean)
plus optional tpm_std. Sample metadata is stored in a separate manifest so that
condition semantics (strain, media, is_basal, etc.) are not encoded in column
headers.
"""

import pandera.pandas as pa


# ---------------------------------------------------------------------------
# TPM table: one file per condition (gene_id, tpm_mean [, tpm_std])
# ---------------------------------------------------------------------------

RnaseqTpmTableSchema = pa.DataFrameSchema(
    name="rnaseq_tpm_table",
    columns={
        "gene_id": pa.Column(
            dtype=str,
            unique=True,
            nullable=False,
            description="Gene identifier; must match reference gene set (e.g. EcoCyc id like EG10001).",
        ),
        "tpm_mean": pa.Column(
            float,
            nullable=False,
            checks=[
                pa.Check.greater_than_or_equal_to(0),
                # TODO: add check to ensure tpm_mean sums to 1 million (perhaps just give a warning if not)
            ],
            description="Mean TPM (transcripts per million) for this gene in this condition.",
        ),
        "tpm_std": pa.Column(
            float,
            nullable=True,
            required=False,
            checks=[
                pa.Check.greater_than_or_equal_to(0),
            ],
            description="Optional: standard deviation of TPM across replicates.",
        ),
    },
    strict="filter",  # allow extra columns but validate required ones
    coerce=True,
    description=(
        "RNA-seq TPM table. One file per sample/condition. "
        "Required columns: gene_id, tpm_mean. Optional: tpm_std."
    ),
)


# ---------------------------------------------------------------------------
# Sample manifest: maps sample_id to file path and metadata
# ---------------------------------------------------------------------------

RnaseqSamplesManifestSchema = pa.DataFrameSchema(
    name="rnaseq_samples_manifest",
    columns={
        "dataset_id": pa.Column(
            dtype=str,
            unique=True,
            nullable=False,
            description="Unique identifier for this dataset, corresponding to the file name of the TPM table.",
        ),
        "dataset_description": pa.Column(
            dtype=str,
            nullable=False,
            description="Description of the dataset (e.g. 'exp96546: MG1655 in M9 glucose, average of 3h and 4h timepoints').",
        ),
        "file_path": pa.Column(
            dtype=str,
            nullable=False,
            description="Path to the TPM table file (relative to manifest or absolute).",
        ),
        "data_source": pa.Column(
            dtype=str,
            nullable=False,
            description="Source of the data (e.g. 'Ginkgo', 'PNNL').",
        ),
        "data_source_experiment_id": pa.Column(
            dtype=str,
            nullable=True,
            required=False,
            description="Experiment identifier from the data source (e.g. 'exp96546').",
        ),
        "data_source_date": pa.Column(
            dtype=str,
            nullable=True,
            required=False,
            description="Date of the experiment from the data source (e.g. '2026-01-01').",
        ),
        "strain": pa.Column(
            dtype=str,
            nullable=True,
            required=False,
            description="Optional: strain descriptor (e.g. 'MG1655 rph+').",
        ),
        "condition": pa.Column(
            dtype=str,
            nullable=True,
            required=False,
            description="Optional: cultivation condition descriptor (e.g. 'Modified_M9_N_Fe').",
        ),
        # "is_basal": pa.Column(
        #     bool,
        #     nullable=True,
        #     required=False,
        #     description="Optional: True if this sample is the default basal condition for ParCa.",
        # ),
    },
    strict="filter",
    coerce=True,
    description=(
        "Manifest of RNA-seq datasets: dataset_id, dataset_description, file_path, data_source, data_source_experiment_id, data_source_date, strain, condition, and optional "
        "metadata."
    ),
)
