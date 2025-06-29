"""Unit tests for CBioPortalValidator."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from h5ad2cbioportaldb.validator import CBioPortalValidator


class TestCBioPortalValidator:
    """Test CBioPortalValidator functionality."""

    def test_initialization(self, test_validator):
        """Test validator initialization."""
        assert test_validator.check_study_exists is True
        assert test_validator.warn_unmapped_genes is True
        assert test_validator.min_cells_per_sample == 10

    def test_validate_study_integration_valid_study(self, test_validator):
        """Test study validation with valid study."""
        result = test_validator.validate_study_integration("test_study")
        
        assert result["valid"] is True
        assert "study_info" in result
        assert len(result["errors"]) == 0

    def test_validate_study_integration_invalid_study(self, test_validator):
        """Test study validation with invalid study."""
        result = test_validator.validate_study_integration("nonexistent_study")
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0]

    def test_validate_sample_mapping_file_valid(self, test_validator):
        """Test validation of valid sample mapping file."""
        mapping_data = pd.DataFrame({
            "h5ad_sample_id": ["S001", "S002", "S003"],
            "cbioportal_sample_id": ["SAMPLE_001", "SAMPLE_002", "SAMPLE_999"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            mapping_data.to_csv(f.name, index=False)
            
            result = test_validator._validate_sample_mapping_file("test_study", f.name)
            
            assert result["valid"] is True
            assert "mapping_stats" in result
            assert result["mapping_stats"]["total_mappings"] == 3
            
        Path(f.name).unlink()

    def test_validate_sample_mapping_file_missing_columns(self, test_validator):
        """Test validation with missing required columns."""
        mapping_data = pd.DataFrame({
            "wrong_column": ["S001", "S002"],
            "another_wrong": ["SAMPLE_001", "SAMPLE_002"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            mapping_data.to_csv(f.name, index=False)
            
            result = test_validator._validate_sample_mapping_file("test_study", f.name)
            
            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert "Missing required columns" in result["errors"][0]
            
        Path(f.name).unlink()

    def test_validate_patient_mapping_file_valid(self, test_validator):
        """Test validation of valid patient mapping file."""
        mapping_data = pd.DataFrame({
            "h5ad_patient_id": ["P001", "P002"],
            "cbioportal_patient_id": ["PATIENT_001", "PATIENT_002"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            mapping_data.to_csv(f.name, index=False)
            
            result = test_validator._validate_patient_mapping_file("test_study", f.name)
            
            assert result["valid"] is True
            assert "mapping_stats" in result
            assert result["mapping_stats"]["total_mappings"] == 2
            
        Path(f.name).unlink()

    def test_validate_h5ad_file_valid(self, test_validator, sample_h5ad_file):
        """Test validation of valid h5ad file."""
        result = test_validator.validate_h5ad_file(sample_h5ad_file)
        
        assert result["valid"] is True
        assert "file_info" in result
        assert result["file_info"]["n_obs"] == 1000
        assert result["file_info"]["n_vars"] == 500
        assert "data_quality" in result

    def test_validate_h5ad_file_nonexistent(self, test_validator):
        """Test validation of nonexistent h5ad file."""
        result = test_validator.validate_h5ad_file("/nonexistent/file.h5ad")
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_check_data_quality(self, test_validator, sample_adata):
        """Test data quality checking."""
        quality_metrics = test_validator._check_data_quality(sample_adata)
        
        assert "expression_matrix" in quality_metrics
        assert "cells" in quality_metrics
        assert "genes" in quality_metrics
        
        expr_metrics = quality_metrics["expression_matrix"]
        assert "total_values" in expr_metrics
        assert "non_zero_values" in expr_metrics
        assert "sparsity" in expr_metrics
        assert expr_metrics["sparsity"] >= 0 and expr_metrics["sparsity"] <= 1

    def test_validate_metadata_completeness(self, test_validator, sample_adata):
        """Test metadata completeness validation."""
        result = {"warnings": []}
        
        test_validator._validate_metadata_completeness(sample_adata, result)
        
        # Should have found cell_type column, so fewer warnings
        warnings_about_cell_types = [w for w in result["warnings"] if "cell type" in w.lower()]
        assert len(warnings_about_cell_types) == 0  # Should find cell_type column

    def test_validate_gene_symbols_valid(self, test_validator):
        """Test validation of valid gene symbols."""
        gene_symbols = ["TP53", "BRCA1", "EGFR"]
        
        result = test_validator.validate_gene_symbols(gene_symbols)
        
        assert result["validation_passed"] is True
        assert len(result["valid"]) == 3
        assert len(result["invalid"]) == 0
        assert result["coverage_percentage"] == 100.0

    def test_validate_gene_symbols_mixed(self, test_validator):
        """Test validation of mixed valid/invalid gene symbols."""
        gene_symbols = ["TP53", "FAKE_GENE", "BRCA1"]
        
        result = test_validator.validate_gene_symbols(gene_symbols)
        
        assert result["validation_passed"] is False
        assert len(result["valid"]) == 2
        assert len(result["invalid"]) == 1
        assert "FAKE_GENE" in result["invalid"]
        assert result["coverage_percentage"] < 100.0

    def test_validate_gene_symbols_empty(self, test_validator):
        """Test validation of empty gene list."""
        result = test_validator.validate_gene_symbols([])
        
        assert result["validation_passed"] is True
        assert result["total"] == 0

    def test_validate_dataset_constraints_min_cells(self, test_validator):
        """Test validation of minimum cells per sample constraint."""
        dataset_data = {
            "cell_sample_mapping": {
                "CELL_001": {"sample_id": "SAMPLE_001"},
                "CELL_002": {"sample_id": "SAMPLE_001"},
                "CELL_003": {"sample_id": "SAMPLE_002"},  # Only 1 cell
            }
        }
        
        result = test_validator.validate_dataset_constraints(dataset_data)
        
        assert "constraint_checks" in result
        assert "min_cells_per_sample" in result["constraint_checks"]
        
        check_result = result["constraint_checks"]["min_cells_per_sample"]
        assert check_result["samples_below_threshold"] == 1  # SAMPLE_002
        assert len(result["warnings"]) > 0

    def test_validate_dataset_constraints_max_genes(self, test_validator):
        """Test validation of maximum genes constraint."""
        dataset_data = {"n_genes": 60000}  # Exceeds default max of 50000
        
        result = test_validator.validate_dataset_constraints(dataset_data)
        
        assert "constraint_checks" in result
        assert "max_genes_per_dataset" in result["constraint_checks"]
        
        check_result = result["constraint_checks"]["max_genes_per_dataset"]
        assert check_result["exceeds_threshold"] is True
        assert len(result["warnings"]) > 0

    def test_generate_validation_summary(self, test_validator):
        """Test validation summary generation."""
        validation_result = {
            "valid": False,
            "warnings": ["Warning 1", "Warning 2"],
            "errors": ["Error 1"],
            "sample_validation": {
                "mapping_stats": {
                    "total_mappings": 10,
                    "valid_cbioportal_samples": 8,
                    "invalid_cbioportal_samples": 2
                }
            },
            "patient_validation": {
                "mapping_stats": {
                    "total_mappings": 5,
                    "valid_cbioportal_patients": 5,
                    "invalid_cbioportal_patients": 0
                }
            }
        }
        
        summary = test_validator._generate_validation_summary("test_study", validation_result)
        
        assert summary["study_id"] == "test_study"
        assert summary["overall_valid"] is False
        assert summary["total_warnings"] == 2
        assert summary["total_errors"] == 1
        assert "sample_mapping" in summary
        assert "patient_mapping" in summary


class TestValidationEdgeCases:
    """Test validation edge cases and error handling."""

    @given(
        n_obs=st.integers(min_value=0, max_value=100),
        n_vars=st.integers(min_value=0, max_value=100)
    )
    def test_data_quality_various_sizes(self, test_validator, n_obs, n_vars, hypothesis_settings):
        """Test data quality validation with various dataset sizes."""
        import anndata
        
        if n_obs == 0 or n_vars == 0:
            # Skip empty datasets as they would fail basic validation
            return
        
        # Create minimal AnnData object
        X = np.random.random((n_obs, n_vars))
        adata = anndata.AnnData(X=X)
        
        quality_metrics = test_validator._check_data_quality(adata)
        
        # Should always return the basic structure
        assert "expression_matrix" in quality_metrics
        if "expression_matrix" in quality_metrics and not quality_metrics.get("error"):
            assert "total_values" in quality_metrics["expression_matrix"]
            assert quality_metrics["expression_matrix"]["total_values"] == n_obs * n_vars

    def test_file_validation_edge_cases(self, test_validator):
        """Test file validation with edge cases."""
        # Test with empty file path
        result = test_validator.validate_h5ad_file("")
        assert result["valid"] is False
        
        # Test with directory instead of file
        with tempfile.TemporaryDirectory() as temp_dir:
            result = test_validator.validate_h5ad_file(temp_dir)
            assert result["valid"] is False

    def test_mapping_file_edge_cases(self, test_validator):
        """Test mapping file validation with edge cases."""
        
        # Test with empty mapping file
        empty_df = pd.DataFrame({
            "h5ad_sample_id": [],
            "cbioportal_sample_id": []
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            empty_df.to_csv(f.name, index=False)
            
            result = test_validator._validate_sample_mapping_file("test_study", f.name)
            
            # Should handle empty file gracefully
            assert result["valid"] is True
            assert result["mapping_stats"]["total_mappings"] == 0
            
        Path(f.name).unlink()

    def test_constraint_validation_edge_cases(self, test_validator):
        """Test constraint validation with edge cases."""
        
        # Test with no cell mapping data
        result = test_validator.validate_dataset_constraints({})
        
        assert result["valid"] is True
        assert "constraint_checks" in result

    @given(gene_count=st.integers(min_value=0, max_value=1000))
    def test_gene_validation_various_counts(self, test_validator, gene_count, hypothesis_settings):
        """Test gene validation with various gene counts."""
        
        # Generate fake gene symbols
        gene_symbols = [f"FAKE_GENE_{i:03d}" for i in range(gene_count)]
        
        result = test_validator.validate_gene_symbols(gene_symbols)
        
        assert result["total"] == gene_count
        assert len(result["valid"]) + len(result["invalid"]) == gene_count
        
        if gene_count == 0:
            assert result["validation_passed"] is True
        else:
            # All fake genes should be invalid
            assert len(result["invalid"]) == gene_count
            assert result["validation_passed"] is False