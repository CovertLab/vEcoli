# Experimental data schemas

Pandera schemas for the ParCa data integration layer. These define the **canonical formats** for experimental inputs so that ingestion is validated and datasets can be substituted like-for-like.

## RNA-seq TPM tables

**Canonical format:** One file per condition. Two required columns:

| Column     | Type  | Required | Description |
|------------|--------|----------|-------------|
| `gene_id`  | string | yes      | Gene identifier; must match reference gene set (e.g. EcoCyc id like `EG10001`). Must be unique (one row per gene). |
| `tpm_mean` | float  | yes      | Mean TPM (transcripts per million) for this gene in this condition. Must be ≥ 0. |
| `tpm_std`  | float  | no       | Optional: standard deviation of TPM across replicates. Must be ≥ 0 if present. |

- File naming suggestion: `rnaseq_<dataset>_<sample_id>_tpm.tsv`
- Extra columns (e.g. gene symbol) are allowed; the schema uses `strict="filter"` and validates only the columns above.

### Validation

```python
import pandas as pd
from reconstruction.ecoli.experimental_data.schemas import RnaseqTpmTableSchema

df = pd.read_csv("path/to/rnaseq_exp96546_MG1655_M9_tpm.tsv", sep="\t")
validated = RnaseqTpmTableSchema.validate(df)
```

## RNA-seq sample manifest

Maps datasets to TPM table paths and metadata. One row per dataset. Used by ParCa config and for QC.

| Column                     | Type  | Required | Description |
|----------------------------|--------|----------|-------------|
| `dataset_id`               | string | yes      | Unique identifier for this dataset; typically matches the TPM table file name. |
| `dataset_description`      | string | yes      | Description of the dataset (e.g. "exp96546: MG1655 in M9 glucose, average of 3h and 4h timepoints"). |
| `file_path`                | string | yes      | Path to the TPM table file (relative to manifest or absolute). |
| `data_source`              | string | yes      | Source of the data (e.g. "Ginkgo", "PNNL"). |
| `data_source_experiment_id`| string | no       | Experiment identifier from the data source (e.g. "exp96546"). |
| `data_source_date`         | string | no       | Date of the experiment from the data source (e.g. "2026-01-01"). |
| `strain`                   | string | no       | Strain descriptor (e.g. "MG1655 rph+"). |
| `condition`                | string | no       | Cultivation condition descriptor (e.g. "Modified_M9_N_Fe"). |

Extra columns are allowed (`strict="filter"`).

### Validation

```python
import pandas as pd
from reconstruction.ecoli.experimental_data.schemas import RnaseqSamplesManifestSchema

manifest = pd.read_csv("path/to/rnaseq_samples.tsv", sep="\t")
validated = RnaseqSamplesManifestSchema.validate(manifest)
```

## Dependencies

- `pandas`
- `pandera` (added to project dependencies in `pyproject.toml`)
