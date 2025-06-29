"""Intelligent sample/patient mapping for cBioPortal integration."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import pandas as pd

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
        self.create_synthetic_samples = config.get("create_synthetic_samples", True)
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
        
        # Initialize strategy counters
        strategies = {
            "direct_sample_match": 0,
            "patient_only_match": 0,
            "synthetic_sample_created": 0,
            "no_mapping": 0,
        }
        
        resolved_mappings = {}
        
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
        
        # Log mapping statistics
        self._log_mapping_statistics(strategies, len(adata.obs))
        
        return resolved_mappings, strategies

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
        
        # Strategy 2: Patient-only match (with synthetic sample creation)
        if mapped_patient_id and mapped_patient_id in existing_patients:
            if self.create_synthetic_samples:
                synthetic_sample_id = self.schema.generate_synthetic_sample_id(
                    mapped_patient_id, self.synthetic_sample_suffix
                )
                return "synthetic_sample_created", synthetic_sample_id, mapped_patient_id
            else:
                return "patient_only_match", None, mapped_patient_id
        
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
            return {pid: pid for pid in patients_df["patient_unique_id"]}
        except Exception as e:
            logger.error(f"Failed to get existing patients for study {study_id}: {e}")
            return {}

    def _log_mapping_statistics(self, strategies: Dict[str, int], total_cells: int) -> None:
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
            logger.info(f"{strategies['synthetic_sample_created']} synthetic samples will be created")

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