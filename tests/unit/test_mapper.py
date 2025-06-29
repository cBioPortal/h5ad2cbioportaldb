"""Unit tests for CBioPortalMapper."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from h5ad2cbioportaldb.mapper import CBioPortalMapper


class TestCBioPortalMapper:
    """Test CBioPortalMapper functionality."""

    def test_initialization(self, test_mapper):
        """Test mapper initialization."""
        assert test_mapper.strategy == "flexible"
        assert test_mapper.create_synthetic_samples is True
        assert test_mapper.synthetic_sample_suffix == "SC"

    def test_load_sample_mapping_valid(self, test_mapper):
        """Test loading valid sample mapping file."""
        mapping_data = pd.DataFrame({
            "h5ad_sample_id": ["S001", "S002", "S003"],
            "cbioportal_sample_id": ["SAMPLE_001", "SAMPLE_002", ""]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            mapping_data.to_csv(f.name, index=False)
            
            mapping = test_mapper._load_sample_mapping(f.name)
            
            # Should exclude empty mapping
            assert len(mapping) == 2
            assert mapping["S001"] == "SAMPLE_001"
            assert mapping["S002"] == "SAMPLE_002"
            assert "S003" not in mapping
            
        Path(f.name).unlink()

    def test_load_sample_mapping_invalid_columns(self, test_mapper):
        """Test loading sample mapping with invalid columns."""
        mapping_data = pd.DataFrame({
            "wrong_column": ["S001", "S002"],
            "another_wrong": ["SAMPLE_001", "SAMPLE_002"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            mapping_data.to_csv(f.name, index=False)
            
            with pytest.raises(ValueError, match="Missing required columns"):
                test_mapper._load_sample_mapping(f.name)
                
        Path(f.name).unlink()

    def test_load_patient_mapping_valid(self, test_mapper):
        """Test loading valid patient mapping file."""
        mapping_data = pd.DataFrame({
            "h5ad_patient_id": ["P001", "P002"],
            "cbioportal_patient_id": ["PATIENT_001", "PATIENT_002"]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
            mapping_data.to_csv(f.name, index=False)
            
            mapping = test_mapper._load_patient_mapping(f.name)
            
            assert len(mapping) == 2
            assert mapping["P001"] == "PATIENT_001"
            assert mapping["P002"] == "PATIENT_002"
            
        Path(f.name).unlink()

    def test_determine_mapping_strategy_direct_sample_match(self, test_mapper):
        """Test direct sample matching strategy."""
        import pandas as pd
        
        cell_obs = pd.Series({
            "sample_id": "S001",
            "patient_id": "P001"
        })
        
        sample_mapping = {"S001": "SAMPLE_001"}
        patient_mapping = {"P001": "PATIENT_001"}
        existing_samples = {"SAMPLE_001": "PATIENT_001"}
        existing_patients = {"PATIENT_001": "PATIENT_001"}
        
        strategy, sample_id, patient_id = test_mapper._determine_mapping_strategy(
            cell_obs, "sample_id", "patient_id", sample_mapping, patient_mapping,
            existing_samples, existing_patients, "test_study"
        )
        
        assert strategy == "direct_sample_match"
        assert sample_id == "SAMPLE_001"
        assert patient_id == "PATIENT_001"

    def test_determine_mapping_strategy_synthetic_sample(self, test_mapper):
        """Test synthetic sample creation strategy."""
        import pandas as pd
        
        cell_obs = pd.Series({
            "sample_id": "S999",  # Non-existent sample
            "patient_id": "P001"
        })
        
        sample_mapping = {"S999": "SAMPLE_999"}  # Maps to non-existent sample
        patient_mapping = {"P001": "PATIENT_001"}
        existing_samples = {"SAMPLE_001": "PATIENT_001"}  # SAMPLE_999 not here
        existing_patients = {"PATIENT_001": "PATIENT_001"}
        
        strategy, sample_id, patient_id = test_mapper._determine_mapping_strategy(
            cell_obs, "sample_id", "patient_id", sample_mapping, patient_mapping,
            existing_samples, existing_patients, "test_study"
        )
        
        assert strategy == "synthetic_sample_created"
        assert sample_id == "PATIENT_001-SC"
        assert patient_id == "PATIENT_001"

    def test_determine_mapping_strategy_no_mapping(self, test_mapper):
        """Test no mapping strategy."""
        import pandas as pd
        
        cell_obs = pd.Series({
            "sample_id": "S999",
            "patient_id": "P999"
        })
        
        sample_mapping = {}
        patient_mapping = {}
        existing_samples = {"SAMPLE_001": "PATIENT_001"}
        existing_patients = {"PATIENT_001": "PATIENT_001"}
        
        strategy, sample_id, patient_id = test_mapper._determine_mapping_strategy(
            cell_obs, "sample_id", "patient_id", sample_mapping, patient_mapping,
            existing_samples, existing_patients, "test_study"
        )
        
        assert strategy == "no_mapping"
        assert sample_id is None
        assert patient_id is None

    def test_resolve_mappings(self, test_mapper, sample_adata, sample_mapping_files):
        """Test full mapping resolution."""
        resolved_mappings, strategies = test_mapper.resolve_mappings(
            sample_adata,
            "test_study",
            sample_obs_column="sample_id",
            patient_obs_column="patient_id",
            sample_mapping_file=sample_mapping_files["sample_mapping"],
            patient_mapping_file=sample_mapping_files["patient_mapping"]
        )
        
        # Check that we got mappings for all cells
        assert len(resolved_mappings) == sample_adata.n_obs
        
        # Check strategy statistics
        assert sum(strategies.values()) == sample_adata.n_obs
        assert "direct_sample_match" in strategies
        assert "synthetic_sample_created" in strategies or "no_mapping" in strategies

    def test_generate_mapping_templates(self, test_mapper, sample_h5ad_file):
        """Test mapping template generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files_created = test_mapper.generate_mapping_templates(
                sample_h5ad_file,
                "sample_id",
                "patient_id", 
                "test_study",
                temp_dir
            )
            
            # Check that template files were created
            assert len(files_created) >= 2  # At least sample and patient templates
            
            for file_path in files_created:
                assert Path(file_path).exists()
                
                if "sample_mapping" in file_path:
                    df = pd.read_csv(file_path)
                    assert "h5ad_sample_id" in df.columns
                    assert "cbioportal_sample_id" in df.columns
                    assert len(df) > 0
                    
                elif "patient_mapping" in file_path:
                    df = pd.read_csv(file_path)
                    assert "h5ad_patient_id" in df.columns
                    assert "cbioportal_patient_id" in df.columns
                    assert len(df) > 0

    def test_validate_mappings_valid(self, test_mapper, sample_mapping_files):
        """Test validation of valid mappings."""
        result = test_mapper.validate_mappings(
            "test_study",
            sample_mapping_files["sample_mapping"],
            sample_mapping_files["patient_mapping"]
        )
        
        assert result["valid"] is True
        assert "sample_validation" in result
        assert "patient_validation" in result

    def test_validate_mappings_invalid_study(self, test_mapper, sample_mapping_files):
        """Test validation with invalid study."""
        result = test_mapper.validate_mappings(
            "nonexistent_study",
            sample_mapping_files["sample_mapping"],
            sample_mapping_files["patient_mapping"]
        )
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0]

    def test_get_mapping_summary(self, test_mapper):
        """Test mapping summary generation."""
        resolved_mappings = {
            "CELL_001": {"strategy": "direct_sample_match", "sample_id": "SAMPLE_001", "patient_id": "PATIENT_001"},
            "CELL_002": {"strategy": "synthetic_sample_created", "sample_id": "PATIENT_001-SC", "patient_id": "PATIENT_001"},
            "CELL_003": {"strategy": "no_mapping", "sample_id": None, "patient_id": None},
        }
        
        summary = test_mapper.get_mapping_summary(resolved_mappings)
        
        assert summary["total_cells"] == 3
        assert summary["strategy_breakdown"]["direct_sample_match"] == 1
        assert summary["strategy_breakdown"]["synthetic_sample_created"] == 1
        assert summary["strategy_breakdown"]["no_mapping"] == 1
        assert summary["unique_samples_mapped"] == 2  # SAMPLE_001 and PATIENT_001-SC
        assert summary["unique_patients_mapped"] == 1  # PATIENT_001


class TestMappingStrategies:
    """Test different mapping strategies with property-based testing."""

    @given(
        sample_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20),
        patient_ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10)
    )
    def test_mapping_completeness(self, test_mapper, sample_ids, patient_ids, hypothesis_settings):
        """Test that mapping always produces results for all input cells."""
        # Create mock cell observations
        import pandas as pd
        
        n_cells = len(sample_ids)
        cell_observations = []
        
        for i in range(n_cells):
            cell_obs = pd.Series({
                "sample_id": sample_ids[i % len(sample_ids)],
                "patient_id": patient_ids[i % len(patient_ids)]
            })
            cell_observations.append(cell_obs)
        
        # Mock mappings and existing entities
        sample_mapping = {}
        patient_mapping = {}
        existing_samples = {}
        existing_patients = {}
        
        # Test each mapping strategy
        strategies_used = set()
        
        for cell_obs in cell_observations:
            strategy, sample_id, patient_id = test_mapper._determine_mapping_strategy(
                cell_obs, "sample_id", "patient_id", sample_mapping, patient_mapping,
                existing_samples, existing_patients, "test_study"
            )
            
            # Should always return a valid strategy
            assert strategy in ["direct_sample_match", "patient_only_match", "synthetic_sample_created", "no_mapping"]
            strategies_used.add(strategy)
        
        # Should use no_mapping strategy for unmapped data
        assert "no_mapping" in strategies_used

    def test_synthetic_sample_id_generation(self, test_mapper):
        """Test synthetic sample ID generation."""
        from h5ad2cbioportaldb.cbioportal.schema import CBioPortalSchema
        
        schema = CBioPortalSchema(test_mapper.client)
        
        # Test basic generation
        synthetic_id = schema.generate_synthetic_sample_id("PATIENT_001", "SC")
        assert synthetic_id == "PATIENT_001-SC"
        
        # Test with different suffix
        synthetic_id = schema.generate_synthetic_sample_id("PATIENT_001", "SCRNA")
        assert synthetic_id == "PATIENT_001-SCRNA"