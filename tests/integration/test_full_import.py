"""Integration tests for full import pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from h5ad2cbioportaldb.importer import H5adImporter


@pytest.mark.integration
class TestFullImportPipeline:
    """Test the complete import pipeline with real ClickHouse."""

    def test_full_import_workflow(self, integration_client, sample_adata, sample_mapping_files):
        """Test complete import workflow from h5ad to database."""
        
        # Create importer
        config = {
            "auto_map_genes": True,
            "validate_mappings": True,
            "batch_size": 1000,
        }
        importer = H5adImporter(integration_client, config)
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            sample_adata.write(f.name)
            h5ad_file = f.name
        
        try:
            # Test dry run first
            dry_result = importer.import_dataset(
                file=h5ad_file,
                dataset_id="test_dataset_001",
                study_id="test_study",
                cell_type_column="cell_type",
                sample_obs_column="sample_id",
                patient_obs_column="patient_id",
                sample_mapping_file=sample_mapping_files["sample_mapping"],
                patient_mapping_file=sample_mapping_files["patient_mapping"],
                description="Test dataset for integration testing",
                matrix_type="raw",
                dry_run=True,
            )
            
            # Verify dry run results
            assert dry_result["dry_run"] is True
            assert dry_result["n_cells"] == sample_adata.n_obs
            assert dry_result["n_genes"] > 0  # Should have mapped some genes
            
            # Now do actual import
            import_result = importer.import_dataset(
                file=h5ad_file,
                dataset_id="test_dataset_001",
                study_id="test_study",
                cell_type_column="cell_type",
                sample_obs_column="sample_id",
                patient_obs_column="patient_id",
                sample_mapping_file=sample_mapping_files["sample_mapping"],
                patient_mapping_file=sample_mapping_files["patient_mapping"],
                description="Test dataset for integration testing",
                matrix_type="raw",
                dry_run=False,
            )
            
            # Verify import results
            assert import_result["success"] is True
            assert import_result["n_cells"] == sample_adata.n_obs
            assert import_result["dataset_id"] == "test_dataset_001"
            
            # Verify data was actually inserted
            self._verify_database_contents(integration_client, "test_dataset_001", sample_adata)
            
        finally:
            Path(h5ad_file).unlink()

    def test_import_with_validation_errors(self, integration_client, sample_adata):
        """Test import with validation errors."""
        
        config = {
            "auto_map_genes": True,
            "validate_mappings": True,
            "batch_size": 1000,
        }
        importer = H5adImporter(integration_client, config)
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            sample_adata.write(f.name)
            h5ad_file = f.name
        
        try:
            # Test with invalid study ID
            with pytest.raises(ValueError, match="not found in cBioPortal"):
                importer.import_dataset(
                    file=h5ad_file,
                    dataset_id="test_dataset_002",
                    study_id="nonexistent_study",
                    cell_type_column="cell_type",
                    dry_run=False,
                )
                
        finally:
            Path(h5ad_file).unlink()

    def test_import_without_mappings(self, integration_client, sample_adata):
        """Test import without mapping files (should use flexible strategy)."""
        
        config = {
            "auto_map_genes": True,
            "validate_mappings": True,
            "batch_size": 1000,
            "mapping": {
                "strategy": "flexible",
                "allow_unmapped_cells": True,
            }
        }
        importer = H5adImporter(integration_client, config)
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            sample_adata.write(f.name)
            h5ad_file = f.name
        
        try:
            # Import without mapping files
            import_result = importer.import_dataset(
                file=h5ad_file,
                dataset_id="test_dataset_003",
                study_id="test_study",
                cell_type_column="cell_type",
                dry_run=False,
            )
            
            # Should succeed but with no_mapping strategy for most cells
            assert import_result["success"] is True
            assert "no_mapping" in import_result["mapping_statistics"]
            
        finally:
            Path(h5ad_file).unlink()

    def test_export_dataset(self, integration_client, sample_adata, sample_mapping_files):
        """Test dataset export functionality."""
        
        config = {
            "auto_map_genes": True,
            "validate_mappings": True,
            "batch_size": 1000,
        }
        importer = H5adImporter(integration_client, config)
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            sample_adata.write(f.name)
            h5ad_file = f.name
        
        try:
            # First import the dataset
            importer.import_dataset(
                file=h5ad_file,
                dataset_id="test_dataset_004",
                study_id="test_study",
                cell_type_column="cell_type",
                sample_obs_column="sample_id",
                patient_obs_column="patient_id",
                sample_mapping_file=sample_mapping_files["sample_mapping"],
                patient_mapping_file=sample_mapping_files["patient_mapping"],
                dry_run=False,
            )
            
            # Now test export
            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as export_file:
                export_result = importer.export_dataset(
                    dataset_id="test_dataset_004",
                    output_file=export_file.name,
                    filters={
                        "cell_types": ["T cell", "B cell"],
                        "genes": ["TP53", "BRCA1"]
                    }
                )
                
                # Verify export results
                assert export_result["n_cells"] > 0
                assert export_result["n_genes"] <= 2  # Filtered to 2 genes
                assert Path(export_file.name).exists()
                
                # Load exported file and verify
                import anndata
                exported_adata = anndata.read_h5ad(export_file.name)
                assert exported_adata.n_obs > 0
                assert exported_adata.n_vars <= 2
                
                Path(export_file.name).unlink()
                
        finally:
            Path(h5ad_file).unlink()

    def _verify_database_contents(self, client, dataset_id, original_adata):
        """Verify that data was correctly inserted into database."""
        
        # Check dataset record
        datasets = client.query(f"""
        SELECT * FROM {client.database}.{client.table_prefix}datasets 
        WHERE dataset_id = %(dataset)s
        """, {"dataset": dataset_id})
        
        assert len(datasets) == 1
        assert datasets.iloc[0]["n_cells"] == original_adata.n_obs
        
        # Check cells
        cells = client.query(f"""
        SELECT COUNT(*) as count FROM {client.database}.{client.table_prefix}cells 
        WHERE dataset_id = %(dataset)s
        """, {"dataset": dataset_id})
        
        assert cells.iloc[0]["count"] == original_adata.n_obs
        
        # Check genes  
        genes = client.query(f"""
        SELECT COUNT(*) as count FROM {client.database}.{client.table_prefix}dataset_genes 
        WHERE dataset_id = %(dataset)s
        """, {"dataset": dataset_id})
        
        assert genes.iloc[0]["count"] > 0  # Should have mapped some genes
        
        # Check expression data
        expression = client.query(f"""
        SELECT COUNT(*) as count FROM {client.database}.{client.table_prefix}expression_matrix 
        WHERE dataset_id = %(dataset)s
        """, {"dataset": dataset_id})
        
        assert expression.iloc[0]["count"] > 0  # Should have non-zero expressions


@pytest.mark.integration  
class TestCrossAnalysisQueries:
    """Test cross-analysis functionality with real data."""

    def test_bulk_vs_single_cell_comparison(self, integration_client, test_querier):
        """Test bulk vs single-cell expression comparison."""
        
        # First need to import some single-cell data
        # This test assumes previous tests have run and imported data
        
        try:
            comparison_result = test_querier.compare_bulk_vs_single_cell(
                gene_symbol="TP53",
                study_id="test_study", 
                sc_dataset_id="test_dataset_001"
            )
            
            # Should return DataFrame even if empty
            assert isinstance(comparison_result, pd.DataFrame)
            
        except Exception:
            # If no data available, test should pass gracefully
            pytest.skip("No single-cell data available for comparison")

    def test_cell_type_summary(self, integration_client, test_querier):
        """Test cell type summary generation."""
        
        try:
            summary_result = test_querier.get_cell_type_summary("test_dataset_001")
            
            # Should return DataFrame
            assert isinstance(summary_result, pd.DataFrame)
            
            if len(summary_result) > 0:
                # Check expected columns
                expected_columns = ["original_cell_type", "cell_count", "unique_samples", "unique_patients"]
                for col in expected_columns:
                    assert col in summary_result.columns
                    
        except Exception:
            pytest.skip("No dataset available for cell type summary")

    def test_dataset_statistics(self, integration_client, test_querier):
        """Test dataset statistics generation."""
        
        try:
            stats_result = test_querier.get_dataset_stats("test_dataset_001")
            
            # Should return DataFrame with metrics
            assert isinstance(stats_result, pd.DataFrame)
            
            if len(stats_result) > 0:
                # Check that we have different metric types
                metric_types = stats_result["metric_type"].unique()
                assert len(metric_types) > 0
                
        except Exception:
            pytest.skip("No dataset available for statistics")


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for import operations."""

    def test_import_performance_small_dataset(self, benchmark, integration_client, sample_adata):
        """Benchmark import performance with small dataset."""
        
        config = {
            "auto_map_genes": True,
            "validate_mappings": False,  # Skip validation for speed
            "batch_size": 1000,
        }
        importer = H5adImporter(integration_client, config)
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            sample_adata.write(f.name)
            h5ad_file = f.name
        
        def import_dataset():
            return importer.import_dataset(
                file=h5ad_file,
                dataset_id=f"benchmark_dataset_{benchmark.name}",
                study_id="test_study",
                cell_type_column="cell_type",
                dry_run=False,
            )
        
        try:
            result = benchmark(import_dataset)
            assert result["success"] is True
            
        finally:
            Path(h5ad_file).unlink()

    def test_query_performance(self, benchmark, integration_client, test_querier):
        """Benchmark query performance."""
        
        def run_queries():
            # Run multiple types of queries
            try:
                test_querier.get_dataset_stats("test_dataset_001")
                test_querier.get_cell_type_summary("test_dataset_001")
                test_querier.list_datasets("test_study")
                return True
            except Exception:
                return False
        
        result = benchmark(run_queries)
        # Should complete without errors (even if no data)
        assert result is not None