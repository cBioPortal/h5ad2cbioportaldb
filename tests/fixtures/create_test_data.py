"""Create test data fixtures for h5ad2cbioportaldb tests."""

import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse


def create_sample_h5ad_file(output_path: str, n_cells: int = 1000, n_genes: int = 2000) -> None:
    """Create sample h5ad file for testing."""
    
    # Create realistic single-cell expression data
    # Use negative binomial distribution to simulate counts
    X = sparse.random(n_cells, n_genes, density=0.1, format='csr', dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, size=len(X.data)).astype(np.float32)
    
    # Create gene names (mix of real and fake genes)
    real_genes = [
        "TP53", "BRCA1", "EGFR", "MYC", "KRAS", "PIK3CA", "PTEN", "RB1", "APC", "VHL",
        "CDKN2A", "NRAS", "BRAF", "IDH1", "TERT", "ARID1A", "KMT2D", "CREBBP", "EP300", "ASXL1",
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD19", "CD20", "CD14", "CD68",
        "PTPRC", "NCAM1", "THY1", "ITGAM", "ITGAX", "CCR7", "SELL", "IL7R", "KLRB1", "GNLY",
        "PRF1", "GZMB", "FOXP3", "IL2RA", "CTLA4", "PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX"
    ]
    
    gene_names = real_genes[:min(len(real_genes), n_genes)]
    gene_names.extend([f"GENE_{i:04d}" for i in range(len(gene_names), n_genes)])
    
    # Create cell metadata with realistic cell types and sample/patient structure
    cell_types = np.random.choice([
        "CD4+ T cell", "CD8+ T cell", "B cell", "NK cell", "monocyte", "dendritic cell",
        "neutrophil", "eosinophil", "mast cell", "epithelial cell", "fibroblast", "endothelial cell"
    ], size=n_cells, p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.03, 0.02, 0.01])
    
    # Create hierarchical sample/patient structure
    n_patients = max(5, n_cells // 200)
    n_samples_per_patient = np.random.choice([1, 2, 3], size=n_patients, p=[0.5, 0.3, 0.2])
    
    sample_patient_mapping = {}
    sample_ids = []
    patient_ids = []
    
    for patient_idx in range(n_patients):
        patient_id = f"PATIENT_{patient_idx:03d}"
        n_samples = n_samples_per_patient[patient_idx]
        
        for sample_idx in range(n_samples):
            if n_samples == 1:
                sample_id = f"{patient_id}_TUMOR"
            else:
                sample_types = ["PRIMARY", "METASTATIC", "NORMAL"]
                sample_id = f"{patient_id}_{sample_types[sample_idx]}"
            
            sample_patient_mapping[sample_id] = patient_id
            sample_ids.append(sample_id)
    
    # Assign cells to samples
    cell_sample_ids = np.random.choice(sample_ids, size=n_cells)
    cell_patient_ids = [sample_patient_mapping[sid] for sid in cell_sample_ids]
    
    # Create additional metadata
    obs = pd.DataFrame({
        "cell_type": cell_types,
        "sample_id": cell_sample_ids,
        "patient_id": cell_patient_ids,
        "n_genes_detected": np.sum(X > 0, axis=1).A1,
        "total_counts": np.sum(X, axis=1).A1,
        "mt_frac": np.random.beta(1, 10, size=n_cells),  # Mitochondrial fraction
        "ribo_frac": np.random.beta(2, 8, size=n_cells),  # Ribosomal fraction
        "doublet_score": np.random.beta(1, 20, size=n_cells),  # Doublet detection score
        "phase": np.random.choice(["G1", "S", "G2M"], size=n_cells, p=[0.6, 0.2, 0.2]),
    })
    obs.index = [f"CELL_{i:06d}" for i in range(n_cells)]
    
    # Create gene metadata
    var = pd.DataFrame({
        "gene_name": gene_names,
        "gene_type": np.random.choice(
            ["protein_coding", "lncRNA", "miRNA", "pseudogene"], 
            size=n_genes, 
            p=[0.8, 0.1, 0.05, 0.05]
        ),
        "highly_variable": np.random.choice([True, False], size=n_genes, p=[0.2, 0.8]),
        "mean_expression": np.array(X.mean(axis=0)).flatten(),
        "dispersion": np.random.gamma(2, 0.5, size=n_genes),
    })
    var.index = gene_names
    
    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    
    # Add dimensionality reduction results
    adata.obsm["X_pca"] = np.random.normal(0, 1, size=(n_cells, 50))
    adata.obsm["X_umap"] = np.random.normal(0, 3, size=(n_cells, 2))
    adata.obsm["X_tsne"] = np.random.normal(0, 10, size=(n_cells, 2))
    
    # Add normalized expression layer
    adata.layers["normalized"] = X.copy().astype(np.float32)
    adata.layers["normalized"].data = np.log1p(adata.layers["normalized"].data)
    
    # Add scaled expression layer  
    adata.layers["scaled"] = X.copy().astype(np.float32)
    # Simple scaling (subtract mean, divide by std)
    means = np.array(adata.layers["scaled"].mean(axis=0)).flatten()
    stds = np.array(np.sqrt(adata.layers["scaled"].multiply(adata.layers["scaled"]).mean(axis=0) - np.square(means))).flatten()
    stds[stds == 0] = 1  # Avoid division by zero
    
    # Add metadata
    adata.uns["dataset_info"] = {
        "dataset_type": "scRNA-seq",
        "technology": "10X Genomics",
        "tissue": "PBMC",
        "condition": "healthy_control",
        "processing_date": "2024-01-15",
        "reference_genome": "GRCh38",
        "created_for": "h5ad2cbioportaldb testing"
    }
    
    adata.uns["processing_log"] = [
        "Raw count matrix loaded",
        "Quality control metrics calculated", 
        "Doublets detected and removed",
        "Normalization applied",
        "Highly variable genes identified",
        "PCA computed",
        "UMAP embedding computed",
        "t-SNE embedding computed",
        "Cell type annotation performed"
    ]
    
    # Save to file
    adata.write_h5ad(output_path)
    print(f"Created sample h5ad file: {output_path}")
    print(f"  - {n_cells} cells, {n_genes} genes")
    print(f"  - {len(np.unique(cell_types))} cell types")
    print(f"  - {len(sample_ids)} samples from {n_patients} patients")


def create_sample_mapping_files(output_dir: str) -> None:
    """Create sample mapping files for testing."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Sample mapping - maps h5ad sample IDs to cBioPortal sample IDs
    sample_mapping = pd.DataFrame({
        "h5ad_sample_id": [
            "PATIENT_000_TUMOR", "PATIENT_001_TUMOR", "PATIENT_002_TUMOR",
            "PATIENT_003_PRIMARY", "PATIENT_003_METASTATIC",
            "PATIENT_004_TUMOR", "UNMAPPED_SAMPLE"
        ],
        "cbioportal_sample_id": [
            "SKCM_PATIENT_000_01", "SKCM_PATIENT_001_01", "SKCM_PATIENT_002_01", 
            "SKCM_PATIENT_003_01", "SKCM_PATIENT_003_02",
            "SKCM_PATIENT_004_01", ""  # Empty mapping to test handling
        ]
    })
    
    sample_file = output_path / "sample_mapping.csv"
    sample_mapping.to_csv(sample_file, index=False)
    
    # Patient mapping - maps h5ad patient IDs to cBioPortal patient IDs  
    patient_mapping = pd.DataFrame({
        "h5ad_patient_id": [
            "PATIENT_000", "PATIENT_001", "PATIENT_002", "PATIENT_003", "PATIENT_004"
        ],
        "cbioportal_patient_id": [
            "SKCM_PATIENT_000", "SKCM_PATIENT_001", "SKCM_PATIENT_002", 
            "SKCM_PATIENT_003", "SKCM_PATIENT_004"
        ]
    })
    
    patient_file = output_path / "patient_mapping.csv"
    patient_mapping.to_csv(patient_file, index=False)
    
    # Cell type mapping for harmonization testing
    cell_type_mapping = pd.DataFrame({
        "original_cell_type": [
            "CD4+ T cell", "CD8+ T cell", "B cell", "NK cell", "monocyte", 
            "dendritic cell", "neutrophil", "epithelial cell", "fibroblast"
        ],
        "harmonized_cell_type_id": [
            "CL:0000624", "CL:0000625", "CL:0000236", "CL:0000623", "CL:0000576",
            "CL:0000451", "CL:0000775", "CL:0000066", "CL:0000057"
        ],
        "cell_type_name": [
            "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell", 
            "B cell", "natural killer cell", "monocyte", "dendritic cell", 
            "neutrophil", "epithelial cell", "fibroblast"
        ],
        "confidence": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8]
    })
    
    cell_type_file = output_path / "cell_type_mapping.csv"
    cell_type_mapping.to_csv(cell_type_file, index=False)
    
    print(f"Created mapping files in: {output_dir}")
    print(f"  - {sample_file}")
    print(f"  - {patient_file}")
    print(f"  - {cell_type_file}")


def create_test_cbioportal_sql(output_path: str) -> None:
    """Create SQL script to set up test cBioPortal database."""
    
    sql_content = """
-- Test cBioPortal database setup
-- This script creates the minimal tables needed for testing

CREATE DATABASE IF NOT EXISTS test_cbioportal;
USE test_cbioportal;

-- Cancer studies
CREATE TABLE IF NOT EXISTS cancer_study (
    cancer_study_identifier String,
    name String,
    description String,
    pmid String,
    citation String,
    groups String,
    status Int32,
    import_date DateTime
) ENGINE = MergeTree() ORDER BY cancer_study_identifier;

-- Insert test study
INSERT INTO cancer_study VALUES (
    'skcm_tcga',
    'Skin Cutaneous Melanoma (TCGA, Firehose Legacy)',
    'TCGA Skin Cutaneous Melanoma dataset',
    '25079317',
    'TCGA, Nature 2015',
    'PUBLIC',
    0,
    now()
);

-- Patients
CREATE TABLE IF NOT EXISTS patient (
    patient_unique_id String,
    cancer_study_identifier String
) ENGINE = MergeTree() ORDER BY (cancer_study_identifier, patient_unique_id);

-- Insert test patients
INSERT INTO patient VALUES
('SKCM_PATIENT_000', 'skcm_tcga'),
('SKCM_PATIENT_001', 'skcm_tcga'),
('SKCM_PATIENT_002', 'skcm_tcga'),
('SKCM_PATIENT_003', 'skcm_tcga'),
('SKCM_PATIENT_004', 'skcm_tcga');

-- Samples
CREATE TABLE IF NOT EXISTS sample (
    sample_unique_id String,
    patient_unique_id String,
    cancer_study_identifier String,
    sample_type String
) ENGINE = MergeTree() ORDER BY (cancer_study_identifier, sample_unique_id);

-- Insert test samples
INSERT INTO sample VALUES
('SKCM_PATIENT_000_01', 'SKCM_PATIENT_000', 'skcm_tcga', 'Primary Tumor'),
('SKCM_PATIENT_001_01', 'SKCM_PATIENT_001', 'skcm_tcga', 'Primary Tumor'),
('SKCM_PATIENT_002_01', 'SKCM_PATIENT_002', 'skcm_tcga', 'Primary Tumor'),
('SKCM_PATIENT_003_01', 'SKCM_PATIENT_003', 'skcm_tcga', 'Primary Tumor'),
('SKCM_PATIENT_003_02', 'SKCM_PATIENT_003', 'skcm_tcga', 'Metastatic'),
('SKCM_PATIENT_004_01', 'SKCM_PATIENT_004', 'skcm_tcga', 'Primary Tumor');

-- Genes
CREATE TABLE IF NOT EXISTS gene (
    entrez_gene_id Int64,
    hugo_gene_symbol String,
    gene_alias String,
    type String
) ENGINE = MergeTree() ORDER BY entrez_gene_id;

-- Insert test genes
INSERT INTO gene VALUES
(7157, 'TP53', 'p53', 'protein-coding'),
(672, 'BRCA1', 'BRCA1', 'protein-coding'),
(1956, 'EGFR', 'ERBB1', 'protein-coding'),
(4609, 'MYC', 'c-Myc', 'protein-coding'),
(3845, 'KRAS', 'KRAS2', 'protein-coding'),
(5290, 'PIK3CA', 'PI3K', 'protein-coding'),
(5728, 'PTEN', 'PTEN', 'protein-coding'),
(5925, 'RB1', 'RB', 'protein-coding'),
(324, 'APC', 'APC', 'protein-coding'),
(7428, 'VHL', 'VHL', 'protein-coding');

-- Genetic profiles
CREATE TABLE IF NOT EXISTS genetic_profile (
    genetic_profile_id String,
    stable_id String,
    cancer_study_identifier String,
    genetic_alteration_type String,
    datatype String,
    name String,
    description String,
    show_profile_in_analysis_tab Bool
) ENGINE = MergeTree() ORDER BY genetic_profile_id;

-- Insert test genetic profile
INSERT INTO genetic_profile VALUES (
    'skcm_tcga_rna_seq_v2_mrna',
    'rna_seq_v2_mrna',
    'skcm_tcga',
    'MRNA_EXPRESSION',
    'CONTINUOUS',
    'mRNA Expression (RNA Seq V2 RSEM)',
    'Expression levels from RNA-Seq by Expectation Maximization',
    true
);

-- Genetic alterations (bulk RNA-seq data)
CREATE TABLE IF NOT EXISTS genetic_alteration_derived (
    genetic_profile_id String,
    sample_unique_id String,
    patient_unique_id String,
    hugo_gene_symbol String,
    entrez_gene_id Int64,
    cancer_study_identifier String,
    profile_type String,
    alteration_value String
) ENGINE = MergeTree() ORDER BY (cancer_study_identifier, hugo_gene_symbol, sample_unique_id);

-- Insert sample bulk expression data
INSERT INTO genetic_alteration_derived VALUES
('skcm_tcga_rna_seq_v2_mrna', 'SKCM_PATIENT_000_01', 'SKCM_PATIENT_000', 'TP53', 7157, 'skcm_tcga', 'rna_seq_v2_mrna', '12.5'),
('skcm_tcga_rna_seq_v2_mrna', 'SKCM_PATIENT_001_01', 'SKCM_PATIENT_001', 'TP53', 7157, 'skcm_tcga', 'rna_seq_v2_mrna', '8.3'),
('skcm_tcga_rna_seq_v2_mrna', 'SKCM_PATIENT_002_01', 'SKCM_PATIENT_002', 'TP53', 7157, 'skcm_tcga', 'rna_seq_v2_mrna', '15.1'),
('skcm_tcga_rna_seq_v2_mrna', 'SKCM_PATIENT_000_01', 'SKCM_PATIENT_000', 'BRCA1', 672, 'skcm_tcga', 'rna_seq_v2_mrna', '7.2'),
('skcm_tcga_rna_seq_v2_mrna', 'SKCM_PATIENT_001_01', 'SKCM_PATIENT_001', 'BRCA1', 672, 'skcm_tcga', 'rna_seq_v2_mrna', '9.8'),
('skcm_tcga_rna_seq_v2_mrna', 'SKCM_PATIENT_002_01', 'SKCM_PATIENT_002', 'BRCA1', 672, 'skcm_tcga', 'rna_seq_v2_mrna', '6.5');
"""
    
    with open(output_path, 'w') as f:
        f.write(sql_content)
    
    print(f"Created test cBioPortal SQL script: {output_path}")


def main():
    """Create all test fixtures."""
    
    fixtures_dir = Path(__file__).parent
    
    # Create sample h5ad files
    print("Creating sample h5ad files...")
    create_sample_h5ad_file(str(fixtures_dir / "sample_data.h5ad"), n_cells=1000, n_genes=500)
    create_sample_h5ad_file(str(fixtures_dir / "large_sample_data.h5ad"), n_cells=5000, n_genes=2000)
    
    # Create mapping files
    print("Creating mapping files...")
    create_sample_mapping_files(str(fixtures_dir))
    
    # Create SQL script
    print("Creating test database SQL script...")
    create_test_cbioportal_sql(str(fixtures_dir / "test_cbioportal_db.sql"))
    
    print("All test fixtures created successfully!")


if __name__ == "__main__":
    main()