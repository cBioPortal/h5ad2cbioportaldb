"""Intelligent sample/patient mapping for cBioPortal integration."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import pandas as pd
import yaml

from .cbioportal.client import CBioPortalClient
from .cbioportal.schema import CBioPortalSchema


logger = logging.getLogger(__name__)


class CBioPortalMapper:
    """Handle intelligent sample/patient mapping with flexible strategies."""

    def __init__(self, client: CBioPortalClient, config: Dict[str, Any]) -> None:
        """Initialize mapper with client and configuration."""
        self.client = client
        self.config = config
        self.schema = CBioPortalSchema(client)
        
        # Mapping configuration
        self.strategy = config.get("strategy", "flexible")
        self.create_synthetic_samples = config.get("create_synthetic_samples", False)  # Changed default to False
        self.synthetic_sample_suffix = config.get("synthetic_sample_suffix", "SC")
        self.allow_unmapped_cells = config.get("allow_unmapped_cells", True)
        self.require_patient_mapping = config.get("require_patient_mapping", False)

    def resolve_mappings(
        self,
        adata: anndata.AnnData,
        study_id: str,
        sample_obs_column: Optional[str] = None,
        patient_obs_column: Optional[str] = None,
        sample_mapping_file: Optional[str] = None,
        patient_mapping_file: Optional[str] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        """Intelligently resolve sample/patient mappings with multiple strategies."""
        
        # Load mapping files
        sample_mapping = self._load_sample_mapping(sample_mapping_file) if sample_mapping_file else {}
        patient_mapping = self._load_patient_mapping(patient_mapping_file) if patient_mapping_file else {}
        
        # Get existing cBioPortal entities
        existing_samples = self._get_existing_samples(study_id)
        existing_patients = self._get_existing_patients(study_id)
        
        logger.info(f"Found {len(existing_samples)} existing samples and {len(existing_patients)} existing patients in study {study_id}")
        if patient_mapping:
            logger.info(f"Patient mapping has {len(patient_mapping)} entries")
            logger.info(f"Patient mapping keys: {list(patient_mapping.keys())[:5]}...")
            logger.info(f"Patient mapping values: {list(patient_mapping.values())[:5]}...")
            logger.info(f"Existing patients: {list(existing_patients.keys())[:5]}...")
            
            # Check for overlaps
            mapped_patients = set(patient_mapping.values())
            existing_patient_ids = set(existing_patients.keys())
            overlap = mapped_patients & existing_patient_ids
            logger.info(f"Overlap between mapped and existing patients: {len(overlap)} out of {len(mapped_patients)}")
            if len(overlap) < len(mapped_patients):
                missing = mapped_patients - existing_patient_ids
                logger.warning(f"Mapped patients not found in cBioPortal: {list(missing)[:5]}...")
        
        # Sample a few cells to see their donor_id values
        if patient_obs_column and patient_obs_column in adata.obs.columns:
            sample_donor_ids = adata.obs[patient_obs_column].value_counts().head(5)
            logger.info(f"Sample donor_id values from h5ad: {sample_donor_ids.index.tolist()}")
            logger.info(f"Sample donor_id counts: {sample_donor_ids.values.tolist()}")
        
        # Initialize strategy counters
        strategies = {
            "direct_sample_match": 0,
            "patient_only_match": 0,
            "synthetic_sample_created": 0,
            "no_mapping": 0,
        }
        
        resolved_mappings = {}
        synthetic_samples_created = set()  # Track unique synthetic samples
        
        for cell_id in adata.obs.index:
            cell_obs = adata.obs.loc[cell_id]
            
            strategy, sample_id, patient_id = self._determine_mapping_strategy(
                cell_obs,
                sample_obs_column,
                patient_obs_column,
                sample_mapping,
                patient_mapping,
                existing_samples,
                existing_patients,
                study_id,
            )
            
            strategies[strategy] += 1
            resolved_mappings[cell_id] = {
                "strategy": strategy,
                "sample_id": sample_id,
                "patient_id": patient_id,
            }
            
            # Track unique synthetic samples
            if strategy == "synthetic_sample_created" and sample_id:
                synthetic_samples_created.add(sample_id)
        
        # Log mapping statistics
        self._log_mapping_statistics(strategies, len(adata.obs), len(synthetic_samples_created))
        
        # Optionally create synthetic samples for patient-only matches
        if self.config.get("create_synthetic_samples_for_patients", False):
            resolved_mappings = self._create_synthetic_samples_for_patients(
                resolved_mappings, strategies
            )
        
        return resolved_mappings, strategies

    def update_existing_mappings(
        self,
        dataset_id: str,
        study_id: str,
        sample_mapping_file: Optional[str] = None,
        patient_mapping_file: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Update mappings for existing dataset when new samples/patients are added to cBioPortal."""
        
        logger.info(f"Updating mappings for dataset {dataset_id}")
        
        # Load new mapping files
        sample_mapping = self._load_sample_mapping(sample_mapping_file) if sample_mapping_file else {}
        patient_mapping = self._load_patient_mapping(patient_mapping_file) if patient_mapping_file else {}
        
        # Get current cBioPortal entities (may have new samples/patients)
        existing_samples = self._get_existing_samples(study_id)
        existing_patients = self._get_existing_patients(study_id)
        
        logger.info(f"Current cBioPortal state: {len(existing_samples)} samples, {len(existing_patients)} patients")
        
        # Get cells that currently have no mapping or synthetic samples
        unmapped_cells = self.client.query(f"""
        SELECT cell_id, mapping_strategy, sample_unique_id, patient_unique_id, obs_data
        FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
          AND (mapping_strategy IN ('no_mapping', 'synthetic_sample_created') 
               OR sample_unique_id IS NULL)
        """, {"dataset": dataset_id})
        
        if len(unmapped_cells) == 0:
            return {
                "total_cells_checked": 0,
                "cells_updated": 0,
                "new_direct_matches": 0,
                "new_patient_matches": 0,
                "still_unmapped": 0
            }
        
        logger.info(f"Found {len(unmapped_cells)} cells to potentially re-map")
        
        # Process each unmapped cell
        updates = []
        stats = {
            "new_direct_matches": 0,
            "new_patient_matches": 0,
            "still_unmapped": 0
        }
        
        for _, cell_row in unmapped_cells.iterrows():
            cell_id = cell_row["cell_id"]
            current_strategy = cell_row["mapping_strategy"]
            
            # Parse obs data to get original sample/patient IDs
            try:
                import json
                obs_data = json.loads(cell_row["obs_data"])
            except Exception as e:
                logger.warning(f"Could not parse obs_data for cell {cell_id}: {e}")
                stats["still_unmapped"] += 1
                continue
            
            # Try to find new mapping
            new_strategy, new_sample_id, new_patient_id = self._find_new_mapping(
                obs_data, sample_mapping, patient_mapping, existing_samples, existing_patients
            )
            
            # Check if mapping improved
            if self._is_mapping_better(current_strategy, new_strategy):
                updates.append({
                    "cell_id": cell_id,
                    "old_strategy": current_strategy,
                    "new_strategy": new_strategy,
                    "new_sample_id": new_sample_id,
                    "new_patient_id": new_patient_id
                })
                
                if new_strategy == "direct_sample_match":
                    stats["new_direct_matches"] += 1
                elif new_strategy in ["patient_only_match", "synthetic_sample_created"]:
                    stats["new_patient_matches"] += 1
            else:
                stats["still_unmapped"] += 1
        
        # Apply updates if not dry run
        if not dry_run and updates:
            self._apply_mapping_updates(dataset_id, updates)
        
        return {
            "total_cells_checked": len(unmapped_cells),
            "cells_updated": len(updates),
            "updates": updates if dry_run else [],
            **stats
        }

    def _find_new_mapping(
        self,
        obs_data: Dict[str, Any],
        sample_mapping: Dict[str, str],
        patient_mapping: Dict[str, str],
        existing_samples: Dict[str, str],
        existing_patients: Dict[str, str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Find new mapping for a cell using updated cBioPortal data."""
        
        # Extract original IDs from obs_data
        h5ad_sample_id = obs_data.get("sample_id") or obs_data.get("author_sample_id")
        h5ad_patient_id = obs_data.get("patient_id") or obs_data.get("donor_id")
        
        # Convert to string if present
        h5ad_sample_id = str(h5ad_sample_id) if h5ad_sample_id else None
        h5ad_patient_id = str(h5ad_patient_id) if h5ad_patient_id else None
        
        # Apply mappings
        mapped_sample_id = sample_mapping.get(h5ad_sample_id) if h5ad_sample_id else None
        mapped_patient_id = patient_mapping.get(h5ad_patient_id) if h5ad_patient_id else None
        
        # Try direct sample match first
        if mapped_sample_id and mapped_sample_id in existing_samples:
            return "direct_sample_match", mapped_sample_id, existing_samples[mapped_sample_id]
        
        # Try patient match
        if mapped_patient_id and mapped_patient_id in existing_patients:
            return "patient_only_match", None, mapped_patient_id
        
        # Try direct patient ID match (no mapping file)
        if h5ad_patient_id and h5ad_patient_id in existing_patients:
            return "patient_only_match", None, h5ad_patient_id
        
        return "no_mapping", None, None

    def _is_mapping_better(self, old_strategy: str, new_strategy: str) -> bool:
        """Check if new mapping strategy is better than old one."""
        strategy_rank = {
            "direct_sample_match": 4,
            "synthetic_sample_created": 3,
            "patient_only_match": 2,
            "no_mapping": 1
        }
        
        return strategy_rank.get(new_strategy, 0) > strategy_rank.get(old_strategy, 0)

    def _apply_mapping_updates(self, dataset_id: str, updates: List[Dict[str, Any]]) -> None:
        """Apply mapping updates to database."""
        
        for update in updates:
            try:
                self.client.command(f"""
                ALTER TABLE {self.client.database}.{self.client.table_prefix}cells
                UPDATE 
                    mapping_strategy = %(new_strategy)s,
                    sample_unique_id = %(new_sample_id)s,
                    patient_unique_id = %(new_patient_id)s
                WHERE dataset_id = %(dataset)s 
                  AND cell_id = %(cell_id)s
                """, {
                    "new_strategy": update["new_strategy"],
                    "new_sample_id": update["new_sample_id"],
                    "new_patient_id": update["new_patient_id"],
                    "dataset": dataset_id,
                    "cell_id": update["cell_id"]
                })
                
                logger.debug(f"Updated cell {update['cell_id']}: {update['old_strategy']} -> {update['new_strategy']}")
                
            except Exception as e:
                logger.error(f"Failed to update cell {update['cell_id']}: {e}")

        logger.info(f"Applied {len(updates)} mapping updates")

    def _determine_mapping_strategy(
        self,
        cell_obs: pd.Series,
        sample_obs_column: Optional[str],
        patient_obs_column: Optional[str],
        sample_mapping: Dict[str, str],
        patient_mapping: Dict[str, str],
        existing_samples: Dict[str, str],
        existing_patients: Dict[str, str],
        study_id: str,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Determine the best mapping strategy for a single cell."""
        
        # Extract sample and patient IDs from obs
        h5ad_sample_id = None
        h5ad_patient_id = None
        
        if sample_obs_column and sample_obs_column in cell_obs:
            h5ad_sample_id = str(cell_obs[sample_obs_column])
        
        if patient_obs_column and patient_obs_column in cell_obs:
            h5ad_patient_id = str(cell_obs[patient_obs_column])
        
        # Apply mappings
        mapped_sample_id = sample_mapping.get(h5ad_sample_id) if h5ad_sample_id else None
        mapped_patient_id = patient_mapping.get(h5ad_patient_id) if h5ad_patient_id else None
        
        # Strategy 1: Direct sample match
        if mapped_sample_id and mapped_sample_id in existing_samples:
            return "direct_sample_match", mapped_sample_id, existing_samples[mapped_sample_id]
        
        # Debug logging for first few cells
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
        
        if self._debug_count <= 3:  # Debug first 3 cells
            logger.info(f"DEBUG Cell {self._debug_count}: h5ad_sample_id='{h5ad_sample_id}', h5ad_patient_id='{h5ad_patient_id}'")
            logger.info(f"DEBUG Cell {self._debug_count}: mapped_sample_id='{mapped_sample_id}', mapped_patient_id='{mapped_patient_id}'")
            logger.info(f"DEBUG Cell {self._debug_count}: patient_id in existing? {mapped_patient_id in existing_patients if mapped_patient_id else False}")
            if mapped_patient_id:
                logger.info(f"DEBUG Cell {self._debug_count}: available patient keys: {list(existing_patients.keys())[:5]}...")
        
        # Strategy 2: Patient-only match (mapped patient ID)
        if mapped_patient_id and mapped_patient_id in existing_patients:
            return "patient_only_match", None, mapped_patient_id
        
        # Strategy 2b: Direct patient ID match (no mapping file case)
        if h5ad_patient_id and h5ad_patient_id in existing_patients:
            return "patient_only_match", None, h5ad_patient_id
        
        # Strategy 3: No mapping (if allowed)
        if self.allow_unmapped_cells:
            return "no_mapping", None, None
        
        # Strategy 4: Strict mode failure
        if self.strategy == "strict":
            raise ValueError(f"No valid mapping found for cell with sample={h5ad_sample_id}, patient={h5ad_patient_id}")
        
        return "no_mapping", None, None

    def _load_sample_mapping(self, mapping_file: str) -> Dict[str, str]:
        """Load sample mapping from CSV file."""
        try:
            df = pd.read_csv(mapping_file)
            if "h5ad_sample_id" not in df.columns or "cbioportal_sample_id" not in df.columns:
                raise ValueError("Sample mapping file must have 'h5ad_sample_id' and 'cbioportal_sample_id' columns")
            
            # Remove empty mappings
            df = df.dropna(subset=["cbioportal_sample_id"])
            df = df[df["cbioportal_sample_id"].str.strip() != ""]
            
            mapping = dict(zip(df["h5ad_sample_id"].astype(str), df["cbioportal_sample_id"].astype(str)))
            logger.info(f"Loaded {len(mapping)} sample mappings from {mapping_file}")
            return mapping
            
        except Exception as e:
            logger.error(f"Failed to load sample mapping from {mapping_file}: {e}")
            raise

    def _load_patient_mapping(self, mapping_file: str) -> Dict[str, str]:
        """Load patient mapping from CSV file."""
        try:
            df = pd.read_csv(mapping_file)
            if "h5ad_patient_id" not in df.columns or "cbioportal_patient_id" not in df.columns:
                raise ValueError("Patient mapping file must have 'h5ad_patient_id' and 'cbioportal_patient_id' columns")
            
            # Remove empty mappings
            df = df.dropna(subset=["cbioportal_patient_id"])
            df = df[df["cbioportal_patient_id"].str.strip() != ""]
            
            mapping = dict(zip(df["h5ad_patient_id"].astype(str), df["cbioportal_patient_id"].astype(str)))
            logger.info(f"Loaded {len(mapping)} patient mappings from {mapping_file}")
            return mapping
            
        except Exception as e:
            logger.error(f"Failed to load patient mapping from {mapping_file}: {e}")
            raise

    def _get_existing_samples(self, study_id: str) -> Dict[str, str]:
        """Get existing samples for study as dict: sample_id -> patient_id."""
        try:
            samples_df = self.client.get_existing_samples(study_id)
            return dict(zip(samples_df["sample_unique_id"], samples_df["patient_unique_id"]))
        except Exception as e:
            logger.error(f"Failed to get existing samples for study {study_id}: {e}")
            return {}

    def _get_existing_patients(self, study_id: str) -> Dict[str, str]:
        """Get existing patients for study as dict: patient_id -> patient_id."""
        try:
            patients_df = self.client.get_existing_patients(study_id)
            # Use both patient_unique_id and patient_stable_id for matching
            patient_dict = {}
            for _, row in patients_df.iterrows():
                unique_id = row["patient_unique_id"] 
                stable_id = row.get("patient_stable_id", unique_id)
                # Map both IDs to the unique_id (which is used in database)
                patient_dict[unique_id] = unique_id
                if stable_id != unique_id:
                    patient_dict[stable_id] = unique_id
            return patient_dict
        except Exception as e:
            logger.error(f"Failed to get existing patients for study {study_id}: {e}")
            return {}

    def _log_mapping_statistics(self, strategies: Dict[str, int], total_cells: int, unique_synthetic_samples: int = 0) -> None:
        """Log detailed mapping statistics."""
        logger.info("=== Mapping Statistics ===")
        logger.info(f"Total cells: {total_cells}")
        
        for strategy, count in strategies.items():
            percentage = (count / total_cells) * 100 if total_cells > 0 else 0
            logger.info(f"{strategy}: {count} cells ({percentage:.1f}%)")
        
        # Warnings for potentially problematic mappings
        if strategies["no_mapping"] > 0:
            logger.warning(f"{strategies['no_mapping']} cells have no cBioPortal mapping")
        
        if strategies["synthetic_sample_created"] > 0:
            logger.info(f"{unique_synthetic_samples} unique synthetic samples will be created (for {strategies['synthetic_sample_created']} cells)")

    def generate_mapping_templates(
        self,
        h5ad_file: str,
        sample_obs_column: Optional[str],
        patient_obs_column: Optional[str],
        study_id: str,
        output_dir: str,
    ) -> List[str]:
        """Generate mapping template files for user completion."""
        
        logger.info(f"Generating mapping templates from {h5ad_file}")
        
        # Fast metadata-only read to avoid loading expression matrix
        try:
            # Load only obs (metadata) for speed
            adata = anndata.read_h5ad(h5ad_file, backed='r')  # Read-only mode
            obs_df = adata.obs.copy()  # Copy just the metadata
            adata.file.close()  # Close file handle immediately
            logger.info(f"Fast metadata read: {len(obs_df)} cells")
        except Exception as e:
            logger.warning(f"Fast read failed ({e}), using standard read")
            # Fallback to normal read
            adata = anndata.read_h5ad(h5ad_file)
            obs_df = adata.obs
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        files_created = []
        
        # Generate sample mapping template
        if sample_obs_column and sample_obs_column in obs_df.columns:
            unique_samples = obs_df[sample_obs_column].dropna().unique()
            sample_template = pd.DataFrame({
                "h5ad_sample_id": unique_samples,
                "cbioportal_sample_id": [""] * len(unique_samples),
            })
            
            sample_file = output_path / "sample_mapping_template.csv"
            sample_template.to_csv(sample_file, index=False)
            files_created.append(str(sample_file))
            logger.info(f"Created sample mapping template: {sample_file} ({len(unique_samples)} samples)")
        
        # Generate patient mapping template
        if patient_obs_column and patient_obs_column in obs_df.columns:
            unique_patients = obs_df[patient_obs_column].dropna().unique()
            patient_template = pd.DataFrame({
                "h5ad_patient_id": unique_patients,
                "cbioportal_patient_id": [""] * len(unique_patients),
            })
            
            patient_file = output_path / "patient_mapping_template.csv"
            patient_template.to_csv(patient_file, index=False)
            files_created.append(str(patient_file))
            logger.info(f"Created patient mapping template: {patient_file} ({len(unique_patients)} patients)")
        
        # Generate reference files with existing cBioPortal IDs
        self._create_reference_files(study_id, output_path, files_created)
        
        # Generate dataset configuration file
        self._create_dataset_config(
            h5ad_file, study_id, sample_obs_column, patient_obs_column, 
            output_path, files_created, obs_df
        )
        
        return files_created

    def _create_reference_files(
        self,
        study_id: str,
        output_path: Path,
        files_created: List[str],
    ) -> None:
        """Create reference files with existing cBioPortal IDs."""
        
        # Existing samples reference
        try:
            existing_samples = self.client.get_existing_samples(study_id)
            if len(existing_samples) > 0:
                sample_ref_file = output_path / f"{study_id}_existing_samples.csv"
                existing_samples[["sample_unique_id", "patient_unique_id", "sample_type"]].to_csv(
                    sample_ref_file, index=False
                )
                files_created.append(str(sample_ref_file))
                logger.info(f"Created sample reference file: {sample_ref_file}")
        except Exception as e:
            logger.warning(f"Could not create sample reference file: {e}")
        
        # Existing patients reference
        try:
            existing_patients = self.client.get_existing_patients(study_id)
            if len(existing_patients) > 0:
                patient_ref_file = output_path / f"{study_id}_existing_patients.csv"
                existing_patients[["patient_unique_id"]].to_csv(patient_ref_file, index=False)
                files_created.append(str(patient_ref_file))
                logger.info(f"Created patient reference file: {patient_ref_file}")
        except Exception as e:
            logger.warning(f"Could not create patient reference file: {e}")

    def _create_dataset_config(
        self,
        h5ad_file: str,
        study_id: str,
        sample_obs_column: Optional[str],
        patient_obs_column: Optional[str],
        output_path: Path,
        files_created: List[str],
        obs_df: pd.DataFrame,
    ) -> None:
        """Create dataset configuration file for easy reuse."""
        
        # Detect potential cell type columns
        potential_cell_type_columns = []
        for col in obs_df.columns:
            if any(keyword in col.lower() for keyword in ['cell_type', 'celltype', 'cluster', 'annotation']):
                potential_cell_type_columns.append(col)
        
        # Get sample column info
        sample_info = {}
        if sample_obs_column and sample_obs_column in obs_df.columns:
            unique_samples = obs_df[sample_obs_column].nunique()
            sample_info = {
                "column": sample_obs_column,
                "unique_count": int(unique_samples),
                "example_values": obs_df[sample_obs_column].value_counts().head(3).index.tolist()
            }
        
        # Get patient column info  
        patient_info = {}
        if patient_obs_column and patient_obs_column in obs_df.columns:
            unique_patients = obs_df[patient_obs_column].nunique()
            patient_info = {
                "column": patient_obs_column,
                "unique_count": int(unique_patients),
                "example_values": obs_df[patient_obs_column].value_counts().head(3).index.tolist()
            }
        
        config = {
            "dataset": {
                "h5ad_file": Path(h5ad_file).name,
                "study_id": study_id,
                "description": f"Single-cell dataset for {study_id}",
                "total_cells": len(obs_df),
                "matrix_type": "raw"
            },
            "mapping": {
                "sample_obs_column": sample_obs_column,
                "patient_obs_column": patient_obs_column,
                "sample_mapping_file": "sample_mapping_template.csv" if sample_obs_column else None,
                "patient_mapping_file": "patient_mapping_template.csv" if patient_obs_column else None
            },
            "annotation": {
                "potential_cell_type_columns": potential_cell_type_columns,
                "recommended_cell_type_column": potential_cell_type_columns[0] if potential_cell_type_columns else None
            },
            "sample_info": sample_info,
            "patient_info": patient_info,
            "available_obs_columns": list(obs_df.columns),
            "commands": {
                "debug_mappings": f"h5ad2cbioportaldb debug-mappings --config dataset_config.yaml",
                "validate_mappings": f"h5ad2cbioportaldb validate-mappings --config dataset_config.yaml",
                "import_dataset": f"h5ad2cbioportaldb import-dataset --config dataset_config.yaml --dataset-id YOUR_DATASET_ID"
            }
        }
        
        # Save configuration
        config_file = output_path / "dataset_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        files_created.append(str(config_file))
        logger.info(f"Created dataset configuration: {config_file}")

    def _create_synthetic_samples_for_patients(
        self,
        resolved_mappings: Dict[str, Dict[str, Any]],
        strategies: Dict[str, int],
    ) -> Dict[str, Dict[str, Any]]:
        """Convert patient_only_match entries to synthetic_sample_created if configured."""
        
        logger.info("Creating synthetic samples for patient-only matches...")
        
        synthetic_samples_created = set()
        updated_mappings = {}
        
        for cell_id, mapping in resolved_mappings.items():
            if mapping["strategy"] == "patient_only_match" and mapping["patient_id"]:
                # Create synthetic sample ID
                synthetic_sample_id = self.schema.generate_synthetic_sample_id(
                    mapping["patient_id"], self.synthetic_sample_suffix
                )
                
                # Update mapping
                updated_mappings[cell_id] = {
                    "strategy": "synthetic_sample_created",
                    "sample_id": synthetic_sample_id,
                    "patient_id": mapping["patient_id"],
                }
                
                synthetic_samples_created.add(synthetic_sample_id)
            else:
                # Keep original mapping
                updated_mappings[cell_id] = mapping
        
        # Update strategy counts
        patient_only_count = strategies["patient_only_match"]
        strategies["patient_only_match"] = 0
        strategies["synthetic_sample_created"] = patient_only_count
        
        logger.info(f"Created {len(synthetic_samples_created)} unique synthetic samples")
        
        return updated_mappings

    def validate_mappings(
        self,
        study_id: str,
        sample_mapping_file: Optional[str] = None,
        patient_mapping_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate mapping files against cBioPortal study."""
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "sample_validation": {},
            "patient_validation": {},
        }
        
        # Validate study exists
        if not self.schema.validate_study_exists(study_id):
            validation_results["valid"] = False
            validation_results["errors"].append(f"Study {study_id} not found in cBioPortal")
            return validation_results
        
        # Validate sample mappings
        if sample_mapping_file:
            try:
                sample_mapping = self._load_sample_mapping(sample_mapping_file)
                mapped_sample_ids = list(sample_mapping.values())
                
                sample_validation = self.schema.validate_sample_ids(study_id, mapped_sample_ids)
                validation_results["sample_validation"] = sample_validation
                
                if sample_validation["missing"] > 0:
                    validation_results["warnings"].append(
                        f"{sample_validation['missing']} mapped samples not found in study {study_id}"
                    )
                    
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Failed to validate sample mappings: {e}")
        
        # Validate patient mappings
        if patient_mapping_file:
            try:
                patient_mapping = self._load_patient_mapping(patient_mapping_file)
                mapped_patient_ids = list(patient_mapping.values())
                
                patient_validation = self.schema.validate_patient_ids(study_id, mapped_patient_ids)
                validation_results["patient_validation"] = patient_validation
                
                if patient_validation["missing"] > 0:
                    validation_results["warnings"].append(
                        f"{patient_validation['missing']} mapped patients not found in study {study_id}"
                    )
                    
            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Failed to validate patient mappings: {e}")
        
        return validation_results

    def get_mapping_summary(self, resolved_mappings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of resolved mappings."""
        
        strategies = {}
        samples_mapped = set()
        patients_mapped = set()
        
        for cell_mapping in resolved_mappings.values():
            strategy = cell_mapping["strategy"]
            strategies[strategy] = strategies.get(strategy, 0) + 1
            
            if cell_mapping["sample_id"]:
                samples_mapped.add(cell_mapping["sample_id"])
            if cell_mapping["patient_id"]:
                patients_mapped.add(cell_mapping["patient_id"])
        
        return {
            "total_cells": len(resolved_mappings),
            "strategy_breakdown": strategies,
            "unique_samples_mapped": len(samples_mapped),
            "unique_patients_mapped": len(patients_mapped),
            "mapping_coverage": {
                "cells_with_samples": sum(1 for m in resolved_mappings.values() if m["sample_id"]),
                "cells_with_patients": sum(1 for m in resolved_mappings.values() if m["patient_id"]),
            },
        }