"""Pytest configuration and fixtures for h5ad2cbioportaldb tests."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Generator

import anndata
import numpy as np
import pandas as pd
import pytest
from testcontainers.clickhouse import ClickHouseContainer

from h5ad2cbioportaldb.cbioportal.client import CBioPortalClient
from h5ad2cbioportaldb.cbioportal.integration import CBioPortalIntegration
from h5ad2cbioportaldb.harmonizer import CellTypeHarmonizer
from h5ad2cbioportaldb.importer import H5adImporter
from h5ad2cbioportaldb.mapper import CBioPortalMapper
from h5ad2cbioportaldb.querier import CBioPortalQuerier
from h5ad2cbioportaldb.validator import CBioPortalValidator


@pytest.fixture(scope="session")
def clickhouse_container() -> Generator[ClickHouseContainer, None, None]:
    """Start ClickHouse container for integration tests."""
    with ClickHouseContainer("clickhouse/clickhouse-server:latest") as container:
        yield container


@pytest.fixture(scope="session")
def test_config() -> Dict:
    """Test configuration."""
    return {
        "cbioportal": {
            "clickhouse": {
                "host": "localhost",
                "port": 9000,
                "database": "test_cbioportal",
                "username": "default",
                "password": "",
                "secure": False,
                "timeout": 30,
            }
        },
        "import": {
            "table_prefix": "scRNA_",
            "auto_map_genes": True,
            "validate_mappings": True,
            "batch_size": 1000,
        },
        "mapping": {
            "strategy": "flexible",
            "create_synthetic_samples": True,
            "synthetic_sample_suffix": "SC",
            "allow_unmapped_cells": True,
            "require_patient_mapping": False,
        },
        "harmonization": {
            "default_ontology": "CL",
            "confidence_threshold": 0.8,
            "auto_harmonize": True,
        },
        "validation": {
            "check_study_exists": True,
            "warn_unmapped_genes": True,
            "warn_missing_mappings": True,
            "report_mapping_stats": True,
            "min_cells_per_sample": 10,
            "max_genes_per_dataset": 50000,
        },
    }


@pytest.fixture
def test_config_with_container(clickhouse_container, test_config) -> Dict:
    """Test configuration using ClickHouse container."""
    config = test_config.copy()
    config["cbioportal"]["clickhouse"].update({
        "host": clickhouse_container.get_container_host_ip(),
        "port": clickhouse_container.get_exposed_port(9000),
    })
    return config


@pytest.fixture
def mock_client() -> CBioPortalClient:
    """Mock cBioPortal client for unit tests."""
    class MockClient:
        def __init__(self):
            self.database = "test_cbioportal"
            self.table_prefix = "scRNA_"
            
        def query(self, sql, parameters=None):
            # Return empty DataFrame for most queries
            return pd.DataFrame()
            
        def command(self, sql, parameters=None):
            pass
            
        def insert_dataframe(self, table, df, batch_size=10000):
            pass
            
        def bulk_insert(self, table, data, columns, batch_size=10000):
            pass
            
        def table_exists(self, table):
            return True
            
        def create_tables_if_not_exist(self):
            pass
            
        def get_existing_studies(self):
            return pd.DataFrame({
                "cancer_study_identifier": ["test_study"],
                "name": ["Test Study"],
                "description": ["Test study for unit tests"]
            })
            
        def get_existing_samples(self, study_id=None):
            return pd.DataFrame({
                "sample_unique_id": ["SAMPLE_001", "SAMPLE_002"],
                "patient_unique_id": ["PATIENT_001", "PATIENT_001"],
                "cancer_study_identifier": ["test_study", "test_study"]
            })
            
        def get_existing_patients(self, study_id=None):
            return pd.DataFrame({
                "patient_unique_id": ["PATIENT_001", "PATIENT_002"],
                "cancer_study_identifier": ["test_study", "test_study"]
            })
            
        def get_existing_genes(self, gene_symbols=None):
            genes = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]
            if gene_symbols:
                genes = [g for g in genes if g in gene_symbols]
            return pd.DataFrame({
                "hugo_gene_symbol": genes,
                "entrez_gene_id": [7157, 672, 1956, 4609, 3845][:len(genes)]
            })
            
        def close(self):
            pass
    
    return MockClient()


@pytest.fixture
def sample_adata() -> anndata.AnnData:
    """Create sample AnnData object for testing."""
    n_obs = 1000
    n_vars = 500
    
    # Create expression matrix
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create gene names
    gene_names = [f"GENE_{i:03d}" for i in range(n_vars)]
    # Include some real gene names for testing
    gene_names[:5] = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]
    
    # Create cell metadata
    cell_types = np.random.choice([
        "T cell", "B cell", "NK cell", "monocyte", "dendritic cell"
    ], size=n_obs)
    
    sample_ids = np.random.choice([
        "SAMPLE_001", "SAMPLE_002", "SAMPLE_003"
    ], size=n_obs)
    
    patient_ids = np.random.choice([
        "PATIENT_001", "PATIENT_002"
    ], size=n_obs)
    
    obs = pd.DataFrame({
        "cell_type": cell_types,
        "sample_id": sample_ids,
        "patient_id": patient_ids,
        "n_genes": np.sum(X > 0, axis=1),
        "total_counts": np.sum(X, axis=1),
    })
    obs.index = [f"CELL_{i:04d}" for i in range(n_obs)]
    
    # Create gene metadata
    var = pd.DataFrame({
        "gene_name": gene_names,
        "highly_variable": np.random.choice([True, False], size=n_vars),
    })
    var.index = gene_names
    
    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    
    # Add some embeddings
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_obs, 2))
    adata.obsm["X_pca"] = np.random.normal(0, 1, size=(n_obs, 50))
    
    # Add metadata
    adata.uns["dataset_info"] = {
        "source": "test_data",
        "date_created": "2024-01-01",
        "processing_notes": "Generated for testing"
    }
    
    return adata


@pytest.fixture
def sample_h5ad_file(sample_adata) -> Generator[str, None, None]:
    """Create temporary h5ad file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        sample_adata.write(f.name)
        yield f.name
    Path(f.name).unlink()


@pytest.fixture
def sample_mapping_files() -> Generator[Dict[str, str], None, None]:
    """Create sample mapping files for testing."""
    # Sample mapping
    sample_mapping = pd.DataFrame({
        "h5ad_sample_id": ["SAMPLE_001", "SAMPLE_002", "SAMPLE_003"],
        "cbioportal_sample_id": ["SAMPLE_001", "SAMPLE_002", ""]
    })
    
    # Patient mapping
    patient_mapping = pd.DataFrame({
        "h5ad_patient_id": ["PATIENT_001", "PATIENT_002"],
        "cbioportal_patient_id": ["PATIENT_001", "PATIENT_002"]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as sample_file:
        sample_mapping.to_csv(sample_file.name, index=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as patient_file:
            patient_mapping.to_csv(patient_file.name, index=False)
            
            yield {
                "sample_mapping": sample_file.name,
                "patient_mapping": patient_file.name
            }
    
    Path(sample_file.name).unlink()
    Path(patient_file.name).unlink()


@pytest.fixture
def test_client(mock_client, test_config) -> CBioPortalClient:
    """Get test client."""
    return mock_client


@pytest.fixture
def test_mapper(test_client, test_config) -> CBioPortalMapper:
    """Get test mapper."""
    return CBioPortalMapper(test_client, test_config["mapping"])


@pytest.fixture
def test_validator(test_client, test_config) -> CBioPortalValidator:
    """Get test validator."""
    return CBioPortalValidator(test_client, test_config["validation"])


@pytest.fixture
def test_harmonizer(test_client, test_config) -> CellTypeHarmonizer:
    """Get test harmonizer."""
    return CellTypeHarmonizer(test_client, test_config["harmonization"])


@pytest.fixture
def test_importer(test_client, test_config) -> H5adImporter:
    """Get test importer."""
    return H5adImporter(test_client, test_config["import"])


@pytest.fixture
def test_querier(test_client) -> CBioPortalQuerier:
    """Get test querier."""
    return CBioPortalQuerier(test_client)


@pytest.fixture
def test_integration(test_client, test_config) -> CBioPortalIntegration:
    """Get test integration."""
    return CBioPortalIntegration(test_client, test_config["import"])


# Integration test fixtures (using real ClickHouse)
@pytest.fixture
def integration_client(test_config_with_container) -> Generator[CBioPortalClient, None, None]:
    """Get integration test client with real ClickHouse."""
    client = CBioPortalClient(test_config_with_container["cbioportal"])
    
    # Create test database and tables
    client.command("CREATE DATABASE IF NOT EXISTS test_cbioportal")
    client.create_tables_if_not_exist()
    
    # Insert test data
    _populate_test_data(client)
    
    yield client
    
    # Cleanup
    client.command("DROP DATABASE IF EXISTS test_cbioportal")
    client.close()


def _populate_test_data(client: CBioPortalClient) -> None:
    """Populate test database with sample data."""
    
    # Create test study
    client.command("""
    CREATE TABLE IF NOT EXISTS test_cbioportal.cancer_study (
        cancer_study_identifier String,
        name String,
        description String
    ) ENGINE = Memory
    """)
    
    client.command("""
    INSERT INTO test_cbioportal.cancer_study VALUES 
    ('test_study', 'Test Study', 'Test study for integration tests')
    """)
    
    # Create test samples
    client.command("""
    CREATE TABLE IF NOT EXISTS test_cbioportal.sample (
        sample_unique_id String,
        patient_unique_id String,
        cancer_study_identifier String,
        sample_type String
    ) ENGINE = Memory
    """)
    
    client.command("""
    INSERT INTO test_cbioportal.sample VALUES 
    ('SAMPLE_001', 'PATIENT_001', 'test_study', 'Primary Tumor'),
    ('SAMPLE_002', 'PATIENT_001', 'test_study', 'Metastatic'),
    ('SAMPLE_003', 'PATIENT_002', 'test_study', 'Primary Tumor')
    """)
    
    # Create test patients
    client.command("""
    CREATE TABLE IF NOT EXISTS test_cbioportal.patient (
        patient_unique_id String,
        cancer_study_identifier String
    ) ENGINE = Memory
    """)
    
    client.command("""
    INSERT INTO test_cbioportal.patient VALUES 
    ('PATIENT_001', 'test_study'),
    ('PATIENT_002', 'test_study')
    """)
    
    # Create test genes
    client.command("""
    CREATE TABLE IF NOT EXISTS test_cbioportal.gene (
        hugo_gene_symbol String,
        entrez_gene_id Int64
    ) ENGINE = Memory
    """)
    
    client.command("""
    INSERT INTO test_cbioportal.gene VALUES 
    ('TP53', 7157),
    ('BRCA1', 672),
    ('EGFR', 1956),
    ('MYC', 4609),
    ('KRAS', 3845)
    """)
    
    # Create test bulk expression data
    client.command("""
    CREATE TABLE IF NOT EXISTS test_cbioportal.genetic_alteration_derived (
        hugo_gene_symbol String,
        cancer_study_identifier String,
        patient_unique_id String,
        sample_unique_id String,
        profile_type String,
        alteration_value String
    ) ENGINE = Memory
    """)
    
    # Insert some test expression values
    bulk_expression_data = []
    genes = ['TP53', 'BRCA1', 'EGFR']
    patients = ['PATIENT_001', 'PATIENT_002']
    samples = ['SAMPLE_001', 'SAMPLE_002', 'SAMPLE_003']
    
    for gene in genes:
        for i, patient in enumerate(patients):
            for j, sample in enumerate(samples[:2] if i == 0 else samples[2:]):
                value = np.random.lognormal(2, 1)  # Random expression value
                bulk_expression_data.append(
                    f"('{gene}', 'test_study', '{patient}', '{sample}', 'rna_seq_v2_mrna', '{value:.2f}')"
                )
    
    if bulk_expression_data:
        client.command(f"""
        INSERT INTO test_cbioportal.genetic_alteration_derived VALUES 
        {', '.join(bulk_expression_data)}
        """)


# Benchmark fixtures
@pytest.fixture
def large_adata() -> anndata.AnnData:
    """Create large AnnData object for performance testing."""
    n_obs = 10000
    n_vars = 2000
    
    # Create sparse expression matrix for realism
    from scipy import sparse
    
    X = sparse.random(n_obs, n_vars, density=0.1, format='csr', dtype=np.float32)
    X.data = np.random.negative_binomial(5, 0.3, size=len(X.data)).astype(np.float32)
    
    # Create metadata
    cell_types = np.random.choice([
        "T cell", "B cell", "NK cell", "monocyte", "dendritic cell",
        "neutrophil", "eosinophil", "mast cell", "epithelial cell", "fibroblast"
    ], size=n_obs)
    
    obs = pd.DataFrame({
        "cell_type": cell_types,
        "sample_id": np.random.choice([f"SAMPLE_{i:03d}" for i in range(50)], size=n_obs),
        "patient_id": np.random.choice([f"PATIENT_{i:03d}" for i in range(20)], size=n_obs),
    })
    obs.index = [f"CELL_{i:06d}" for i in range(n_obs)]
    
    var = pd.DataFrame(index=[f"GENE_{i:04d}" for i in range(n_vars)])
    
    return anndata.AnnData(X=X, obs=obs, var=var)


# Property-based testing fixtures
@pytest.fixture
def hypothesis_settings():
    """Hypothesis settings for property-based tests."""
    from hypothesis import settings
    return settings(max_examples=50, deadline=5000)  # 5 second deadline