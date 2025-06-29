# h5ad2cbioportaldb

**⚠️  Work in Progress**

A Python CLI tool for importing h5ad single-cell files into cBioPortal's
ClickHouse database. This enables queries across
bulk and single cell sequencing data.

## Features

- **cBioPortal Integration**: Direct integration with cBioPortal ClickHouse database schema
- **Sample/Patient Mapping**: Flexible sample/patient mapping strategies with automatic fallbacks
- **SPARSE Columns**: Efficient storage using ClickHouse SPARSE columns for expression matrices
- **Cell Type Harmonization**: Map cell types to standard ontologies (Cell Ontology, UBERON)
- **Cross-Analysis Queries**: Compare bulk RNA-seq vs single-cell expression data
- **Comprehensive Validation**: Validate mappings, gene symbols, and data quality
- **Flexible Configuration**: YAML-based configuration with sensible defaults
- **Production Ready**: Comprehensive error handling, logging, and testing

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/h5ad2cbioportaldb.git
cd h5ad2cbioportaldb

# Install with uv
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install h5ad2cbioportaldb
```

## Quick Start

### 1. Configure Database Connection

Copy the example configuration:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your cBioPortal ClickHouse connection details:

```yaml
cbioportal:
  clickhouse:
    host: your-clickhouse-host
    port: 9000
    database: your-database
    username: your-username
    password: your-password
```

### 2. Generate Mapping Templates

First, generate mapping templates to understand your data:

```bash
h5ad2cbioportaldb generate-mapping-template \
  --file your_data.h5ad \
  --sample-obs-column sample_id \
  --patient-obs-column donor_id \
  --study-id skcm_tcga \
  --output-dir templates/
```

This creates:
- `sample_mapping_template.csv` - Map h5ad samples to cBioPortal samples
- `patient_mapping_template.csv` - Map h5ad patients to cBioPortal patients  
- `skcm_tcga_existing_samples.csv` - Reference of existing cBioPortal samples
- `skcm_tcga_existing_patients.csv` - Reference of existing cBioPortal patients

### 3. Complete the Mappings

Edit the template files to map your data:

**sample_mapping.csv**:
```csv
h5ad_sample_id,cbioportal_sample_id
MELANOMA_01,skcm_tcga_TCGA-BF-A1PU-01
MELANOMA_02,skcm_tcga_TCGA-BF-A1PV-01
MELANOMA_03,  # Leave empty if no mapping exists
```

**patient_mapping.csv**:
```csv
h5ad_patient_id,cbioportal_patient_id
DONOR_01,skcm_tcga_TCGA-BF-A1PU
DONOR_02,skcm_tcga_TCGA-BF-A1PV
```

### 4. Validate Mappings

```bash
h5ad2cbioportaldb validate-mappings \
  --study-id skcm_tcga \
  --sample-mapping sample_mapping.csv \
  --patient-mapping patient_mapping.csv
```

### 5. Import Dataset

```bash
h5ad2cbioportaldb import \
  --file your_data.h5ad \
  --dataset-id sc_skcm_001 \
  --study-id skcm_tcga \
  --cell-type-column leiden \
  --sample-obs-column sample_id \
  --sample-mapping sample_mapping.csv \
  --patient-obs-column donor_id \
  --patient-mapping patient_mapping.csv \
  --description "Single-cell RNA-seq from SKCM patients"
```

## Mapping Strategies

The tool uses the following mapping strategies to handle various scenarios:

### 1. Direct Sample Match
- h5ad sample → existing cBioPortal sample
- **Best case**: Direct integration with existing bulk data

### 2. Patient-Only Match + Synthetic Samples
- h5ad sample → missing, but patient exists
- **Action**: Creates synthetic sample ID (e.g., `PATIENT_001-SC`)
- **Benefit**: Enables patient-level analysis

### 3. No Mapping
- Neither sample nor patient found
- **Action**: Stores cells without cBioPortal links
- **Use case**: Exploratory analysis of new cohorts

### 4. Configurable Behavior

```yaml
mapping:
  strategy: "flexible"  # "strict", "patient_only", "flexible"
  create_synthetic_samples: true
  synthetic_sample_suffix: "SC"
  allow_unmapped_cells: true
```

## Database Schema

The tool creates these tables in your cBioPortal database:

```sql
-- Dataset metadata
scRNA_datasets (dataset_id, name, cancer_study_identifier, ...)

-- Cell-level data with flexible mapping
scRNA_cells (dataset_id, cell_id, sample_unique_id, patient_unique_id, ...)

-- Gene mapping to cBioPortal genes
scRNA_dataset_genes (dataset_id, gene_idx, hugo_gene_symbol, entrez_gene_id)

-- Expression data using SPARSE columns
scRNA_expression_matrix (dataset_id, cell_id, gene_idx, matrix_type, count SPARSE)

-- Embeddings (UMAP, t-SNE, PCA)
scRNA_cell_embeddings (dataset_id, cell_id, embedding_type, dimension_idx, value)

-- Cell type harmonization
scRNA_cell_type_ontology (cell_type_id, cell_type_name, ontology, ...)
```

## Advanced Usage

### Cell Type Harmonization

Harmonize cell types to Cell Ontology:

```bash
# Auto-harmonization using built-in mappings
h5ad2cbioportaldb harmonize \
  --dataset sc_skcm_001 \
  --ontology CL

# Custom harmonization with mapping file
h5ad2cbioportaldb harmonize \
  --dataset sc_skcm_001 \
  --ontology CL \
  --mapping-file custom_cell_types.csv
```

### Cross-Analysis Queries

Compare bulk vs single-cell expression:

```bash
h5ad2cbioportaldb query compare-expression \
  --gene TP53 \
  --study skcm_tcga \
  --sc-dataset sc_skcm_001 \
  --output tp53_comparison.csv
```

Get cell type summary:

```bash
h5ad2cbioportaldb query cell-type-summary \
  --sc-dataset sc_skcm_001
```

### Export Subsets

Export filtered data back to h5ad:

```bash
h5ad2cbioportaldb export \
  --dataset sc_skcm_001 \
  --output t_cells_subset.h5ad \
  --cell-types "T cell,CD4+ T cell,CD8+ T cell" \
  --genes "TP53,BRCA1,EGFR"
```

### List Data

```bash
# List datasets in study
h5ad2cbioportaldb list datasets --study skcm_tcga

# List all studies
h5ad2cbioportaldb list studies

# List cell types in study
h5ad2cbioportaldb list cell-types --study skcm_tcga
```

## Configuration Reference

### Database Configuration

```yaml
cbioportal:
  clickhouse:
    host: localhost
    port: 9000
    database: my_database
    username: default
    password: ""
    secure: false
    timeout: 30
```

### Import Settings

```yaml
import:
  table_prefix: "scRNA_"
  auto_map_genes: true
  validate_mappings: true
  batch_size: 10000
  max_memory_usage: "4GB"
```

### Mapping Configuration

```yaml
mapping:
  strategy: "flexible"  # "strict", "patient_only", "flexible"
  create_synthetic_samples: true
  synthetic_sample_suffix: "SC"
  allow_unmapped_cells: true
  require_patient_mapping: false
```

### Validation Settings

```yaml
validation:
  check_study_exists: true
  warn_unmapped_genes: true
  warn_missing_mappings: true
  min_cells_per_sample: 10
  max_genes_per_dataset: 50000
```

## Example Workflows

### Workflow 1: New Study Integration

```bash
# 1. Generate templates
h5ad2cbioportaldb generate-mapping-template \
  --file new_study.h5ad \
  --sample-obs-column sample \
  --patient-obs-column patient \
  --study-id new_study \
  --output-dir mappings/

# 2. Complete mappings (manual step)
# Edit mappings/sample_mapping_template.csv
# Edit mappings/patient_mapping_template.csv

# 3. Validate
h5ad2cbioportaldb validate-mappings \
  --study-id new_study \
  --sample-mapping mappings/sample_mapping.csv \
  --patient-mapping mappings/patient_mapping.csv

# 4. Import
h5ad2cbioportaldb import \
  --file new_study.h5ad \
  --dataset-id sc_new_001 \
  --study-id new_study \
  --cell-type-column cell_type \
  --sample-obs-column sample \
  --patient-obs-column patient \
  --sample-mapping mappings/sample_mapping.csv \
  --patient-mapping mappings/patient_mapping.csv
```

### Workflow 2: Existing Study Enhancement

```bash
# Import into existing study with patient-level mapping
h5ad2cbioportaldb import \
  --file additional_samples.h5ad \
  --dataset-id sc_skcm_002 \
  --study-id skcm_tcga \
  --cell-type-column leiden \
  --patient-obs-column patient_id \
  --patient-mapping patient_mapping.csv \
  --description "Additional single-cell samples"

# Harmonize cell types
h5ad2cbioportaldb harmonize \
  --dataset sc_skcm_002 \
  --ontology CL

# Compare with bulk data
h5ad2cbioportaldb query compare-expression \
  --gene TP53 \
  --study skcm_tcga \
  --sc-dataset sc_skcm_002 \
  --output analysis/tp53_bulk_vs_sc.csv
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/your-org/h5ad2cbioportaldb.git
cd h5ad2cbioportaldb
uv pip install -e ".[dev]"
```

### Run Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires ClickHouse)
pytest tests/integration/ -m integration

# All tests
pytest

# With coverage
pytest --cov=h5ad2cbioportaldb --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking (if using mypy)
mypy src/
```

### Create Test Data

```bash
cd tests/fixtures/
python create_test_data.py
```

## Performance Considerations

### Large Datasets

For datasets with >100k cells or >20k genes:

```yaml
import:
  batch_size: 50000  # Increase batch size
  max_memory_usage: "8GB"  # Increase memory limit

expression:
  min_expression_threshold: 0.1  # Filter low expression
  compression: "zstd"  # Use compression
```

### Memory Usage

- **Expression matrices**: Use SPARSE columns automatically
- **Batch processing**: Configurable batch sizes
- **Memory monitoring**: Built-in memory usage tracking

### Query Performance

- **Indexed columns**: All key columns are indexed
- **Partitioning**: Consider partitioning by study_id for large deployments
- **Materialized views**: Create for common query patterns

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Solution: Check ClickHouse host, port, and credentials in config.yaml
   ```

2. **Study Not Found**
   ```
   Solution: Verify study exists in cBioPortal: h5ad2cbioportaldb list studies
   ```

3. **Gene Mapping Issues**
   ```
   Solution: Use --warn-unmapped-genes to see which genes weren't found
   ```

4. **Memory Issues**
   ```
   Solution: Reduce batch_size or increase max_memory_usage in config
   ```

### Logging

Enable debug logging:

```bash
h5ad2cbioportaldb --verbose import [options...]
```

Or in config.yaml:
```yaml
logging:
  level: "DEBUG"
  file: "import.log"
```

### Validation Errors

Always run validation before import:

```bash
h5ad2cbioportaldb validate-mappings \
  --study-id your_study \
  --sample-mapping your_samples.csv \
  --patient-mapping your_patients.csv
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
