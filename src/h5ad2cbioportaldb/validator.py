"""cBioPortal validation utilities."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .cbioportal.client import CBioPortalClient
from .cbioportal.schema import CBioPortalSchema


logger = logging.getLogger(__name__)


class CBioPortalValidator:
    """Validate data and mappings against cBioPortal database."""

    def __init__(self, client: CBioPortalClient, config: Dict[str, Any]) -> None:
        """Initialize validator with client and configuration."""
        self.client = client
        self.config = config
        self.schema = CBioPortalSchema(client)
        
        # Validation configuration
        self.check_study_exists = config.get("check_study_exists", True)
        self.warn_unmapped_genes = config.get("warn_unmapped_genes", True)
        self.warn_missing_mappings = config.get("warn_missing_mappings", True)
        self.report_mapping_stats = config.get("report_mapping_stats", True)
        self.min_cells_per_sample = config.get("min_cells_per_sample", 10)
        self.max_genes_per_dataset = config.get("max_genes_per_dataset", 50000)

    def validate_study_integration(
        self,
        study_id: str,
        sample_mapping_file: Optional[str] = None,
        patient_mapping_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate that study exists and mappings are valid."""
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "study_info": {},
            "sample_validation": {},
            "patient_validation": {},
            "summary": {},
        }
        
        try:
            # Validate study exists
            if self.check_study_exists:
                study_info = self.schema.get_study_info(study_id)
                if not study_info:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Study {study_id} not found in cBioPortal")
                    return validation_result
                
                validation_result["study_info"] = study_info
                logger.info(f"Study validation passed: {study_id}")
            
            # Validate sample mappings
            if sample_mapping_file:
                sample_validation = self._validate_sample_mapping_file(study_id, sample_mapping_file)
                validation_result["sample_validation"] = sample_validation
                
                if not sample_validation["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(sample_validation["errors"])
                
                validation_result["warnings"].extend(sample_validation["warnings"])
            
            # Validate patient mappings
            if patient_mapping_file:
                patient_validation = self._validate_patient_mapping_file(study_id, patient_mapping_file)
                validation_result["patient_validation"] = patient_validation
                
                if not patient_validation["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(patient_validation["errors"])
                
                validation_result["warnings"].extend(patient_validation["warnings"])
            
            # Generate summary
            validation_result["summary"] = self._generate_validation_summary(
                study_id, validation_result
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
            return validation_result

    def _validate_sample_mapping_file(
        self, study_id: str, sample_mapping_file: str
    ) -> Dict[str, Any]:
        """Validate sample mapping file."""
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "file_path": sample_mapping_file,
            "mapping_stats": {},
        }
        
        try:
            # Load mapping file
            mapping_df = pd.read_csv(sample_mapping_file)
            
            # Check required columns
            required_columns = ["h5ad_sample_id", "cbioportal_sample_id"]
            missing_columns = [col for col in required_columns if col not in mapping_df.columns]
            
            if missing_columns:
                result["valid"] = False
                result["errors"].append(f"Missing required columns: {missing_columns}")
                return result
            
            # Remove empty mappings
            original_count = len(mapping_df)
            mapping_df = mapping_df.dropna(subset=["cbioportal_sample_id"])
            mapping_df = mapping_df[mapping_df["cbioportal_sample_id"].str.strip() != ""]
            
            if len(mapping_df) < original_count:
                result["warnings"].append(
                    f"Removed {original_count - len(mapping_df)} empty mappings"
                )
            
            # Validate mapped sample IDs exist in cBioPortal
            mapped_sample_ids = mapping_df["cbioportal_sample_id"].unique().tolist()
            sample_validation = self.schema.validate_sample_ids(study_id, mapped_sample_ids)
            
            result["mapping_stats"] = {
                "total_mappings": len(mapping_df),
                "unique_h5ad_samples": mapping_df["h5ad_sample_id"].nunique(),
                "unique_cbioportal_samples": len(mapped_sample_ids),
                "valid_cbioportal_samples": len(sample_validation["valid"]),
                "invalid_cbioportal_samples": len(sample_validation["invalid"]),
            }
            
            if sample_validation["missing"] > 0:
                result["warnings"].append(
                    f"{sample_validation['missing']} mapped samples not found in study {study_id}"
                )
                if self.warn_missing_mappings:
                    result["warnings"].append(
                        f"Missing samples: {sample_validation['invalid'][:5]}..."
                    )
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate sample mapping file: {e}")
            return result

    def _validate_patient_mapping_file(
        self, study_id: str, patient_mapping_file: str
    ) -> Dict[str, Any]:
        """Validate patient mapping file."""
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "file_path": patient_mapping_file,
            "mapping_stats": {},
        }
        
        try:
            # Load mapping file
            mapping_df = pd.read_csv(patient_mapping_file)
            
            # Check required columns
            required_columns = ["h5ad_patient_id", "cbioportal_patient_id"]
            missing_columns = [col for col in required_columns if col not in mapping_df.columns]
            
            if missing_columns:
                result["valid"] = False
                result["errors"].append(f"Missing required columns: {missing_columns}")
                return result
            
            # Remove empty mappings
            original_count = len(mapping_df)
            mapping_df = mapping_df.dropna(subset=["cbioportal_patient_id"])
            mapping_df = mapping_df[mapping_df["cbioportal_patient_id"].str.strip() != ""]
            
            if len(mapping_df) < original_count:
                result["warnings"].append(
                    f"Removed {original_count - len(mapping_df)} empty mappings"
                )
            
            # Validate mapped patient IDs exist in cBioPortal
            mapped_patient_ids = mapping_df["cbioportal_patient_id"].unique().tolist()
            patient_validation = self.schema.validate_patient_ids(study_id, mapped_patient_ids)
            
            result["mapping_stats"] = {
                "total_mappings": len(mapping_df),
                "unique_h5ad_patients": mapping_df["h5ad_patient_id"].nunique(),
                "unique_cbioportal_patients": len(mapped_patient_ids),
                "valid_cbioportal_patients": len(patient_validation["valid"]),
                "invalid_cbioportal_patients": len(patient_validation["invalid"]),
            }
            
            if patient_validation["missing"] > 0:
                result["warnings"].append(
                    f"{patient_validation['missing']} mapped patients not found in study {study_id}"
                )
                if self.warn_missing_mappings:
                    result["warnings"].append(
                        f"Missing patients: {patient_validation['invalid'][:5]}..."
                    )
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate patient mapping file: {e}")
            return result

    def _generate_validation_summary(
        self, study_id: str, validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate validation summary."""
        
        summary = {
            "study_id": study_id,
            "overall_valid": validation_result["valid"],
            "total_warnings": len(validation_result["warnings"]),
            "total_errors": len(validation_result["errors"]),
        }
        
        # Add sample mapping summary
        if "sample_validation" in validation_result:
            sample_stats = validation_result["sample_validation"].get("mapping_stats", {})
            if sample_stats:
                summary["sample_mapping"] = {
                    "total_mappings": sample_stats.get("total_mappings", 0),
                    "valid_samples": sample_stats.get("valid_cbioportal_samples", 0),
                    "invalid_samples": sample_stats.get("invalid_cbioportal_samples", 0),
                }
        
        # Add patient mapping summary
        if "patient_validation" in validation_result:
            patient_stats = validation_result["patient_validation"].get("mapping_stats", {})
            if patient_stats:
                summary["patient_mapping"] = {
                    "total_mappings": patient_stats.get("total_mappings", 0),
                    "valid_patients": patient_stats.get("valid_cbioportal_patients", 0),
                    "invalid_patients": patient_stats.get("invalid_cbioportal_patients", 0),
                }
        
        return summary

    def validate_h5ad_file(self, file_path: str) -> Dict[str, Any]:
        """Validate h5ad file structure and content."""
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "file_info": {},
            "data_quality": {},
        }
        
        try:
            import anndata
            
            # Load file
            adata = anndata.read_h5ad(file_path)
            
            # Basic file information
            result["file_info"] = {
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "obs_columns": list(adata.obs.columns),
                "var_columns": list(adata.var.columns),
                "uns_keys": list(adata.uns.keys()) if adata.uns else [],
                "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
                "layers_keys": list(adata.layers.keys()) if adata.layers else [],
            }
            
            # Validate basic requirements
            if adata.n_obs == 0:
                result["valid"] = False
                result["errors"].append("File contains no cells")
            
            if adata.n_vars == 0:
                result["valid"] = False
                result["errors"].append("File contains no genes")
            
            # Check for reasonable data size
            if adata.n_vars > self.max_genes_per_dataset:
                result["warnings"].append(
                    f"Large number of genes ({adata.n_vars}) may impact performance"
                )
            
            # Data quality checks
            result["data_quality"] = self._check_data_quality(adata)
            
            # Check for required metadata
            self._validate_metadata_completeness(adata, result)
            
            return result
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate h5ad file: {e}")
            return result

    def _check_data_quality(self, adata) -> Dict[str, Any]:
        """Check data quality metrics."""
        
        import numpy as np
        from scipy import sparse
        
        quality_metrics = {}
        
        try:
            # Expression matrix checks
            X = adata.X
            if sparse.issparse(X):
                X = X.toarray()
            
            quality_metrics["expression_matrix"] = {
                "total_values": X.size,
                "non_zero_values": np.count_nonzero(X),
                "sparsity": 1 - (np.count_nonzero(X) / X.size),
                "mean_expression": float(np.mean(X)),
                "max_expression": float(np.max(X)),
                "min_expression": float(np.min(X)),
            }
            
            # Cell-level metrics
            cells_per_gene = np.sum(X > 0, axis=0)
            genes_per_cell = np.sum(X > 0, axis=1)
            
            quality_metrics["cells"] = {
                "mean_genes_per_cell": float(np.mean(genes_per_cell)),
                "median_genes_per_cell": float(np.median(genes_per_cell)),
                "cells_with_few_genes": int(np.sum(genes_per_cell < 200)),
            }
            
            # Gene-level metrics
            quality_metrics["genes"] = {
                "mean_cells_per_gene": float(np.mean(cells_per_gene)),
                "median_cells_per_gene": float(np.median(cells_per_gene)),
                "rarely_expressed_genes": int(np.sum(cells_per_gene < 3)),
            }
            
        except Exception as e:
            logger.warning(f"Could not compute data quality metrics: {e}")
            quality_metrics["error"] = str(e)
        
        return quality_metrics

    def _validate_metadata_completeness(self, adata, result: Dict[str, Any]) -> None:
        """Validate metadata completeness."""
        
        # Check for common cell annotation columns
        expected_obs_columns = ["cell_type", "leiden", "seurat_clusters", "sample", "donor", "patient"]
        available_obs_columns = set(adata.obs.columns)
        
        cell_type_columns = [col for col in expected_obs_columns[:3] if col in available_obs_columns]
        if not cell_type_columns:
            result["warnings"].append(
                "No obvious cell type annotation column found. Consider using --cell-type-column"
            )
        
        # Check for sample/patient columns
        sample_columns = [col for col in expected_obs_columns[3:] if col in available_obs_columns]
        if not sample_columns:
            result["warnings"].append(
                "No sample/patient identifier columns found. Mapping may be limited."
            )
        
        # Check for embeddings
        expected_embeddings = ["X_umap", "X_tsne", "X_pca"]
        available_embeddings = [emb for emb in expected_embeddings if emb in adata.obsm]
        
        if not available_embeddings:
            result["warnings"].append(
                "No standard embeddings (UMAP, t-SNE, PCA) found in obsm"
            )

    def validate_gene_symbols(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """Validate gene symbols against cBioPortal gene database."""
        
        if not gene_symbols:
            return {"valid": [], "invalid": [], "total": 0, "validation_passed": True}
        
        gene_validation = self.schema.validate_gene_symbols(gene_symbols)
        
        result = {
            "valid": gene_validation["valid"],
            "invalid": gene_validation["invalid"],
            "total": gene_validation["total"],
            "found": gene_validation["found"],
            "missing": gene_validation["missing"],
            "validation_passed": gene_validation["missing"] == 0,
            "coverage_percentage": (gene_validation["found"] / gene_validation["total"]) * 100,
        }
        
        if self.warn_unmapped_genes and gene_validation["missing"] > 0:
            logger.warning(
                f"Gene validation: {gene_validation['missing']} of {gene_validation['total']} "
                f"genes not found in cBioPortal ({result['coverage_percentage']:.1f}% coverage)"
            )
        
        return result

    def validate_dataset_constraints(self, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset against configured constraints."""
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "constraint_checks": {},
        }
        
        # Check minimum cells per sample
        if "cell_sample_mapping" in dataset_data:
            sample_cell_counts = {}
            for cell_mapping in dataset_data["cell_sample_mapping"].values():
                sample_id = cell_mapping.get("sample_id")
                if sample_id:
                    sample_cell_counts[sample_id] = sample_cell_counts.get(sample_id, 0) + 1
            
            low_cell_samples = [
                sample_id for sample_id, count in sample_cell_counts.items()
                if count < self.min_cells_per_sample
            ]
            
            result["constraint_checks"]["min_cells_per_sample"] = {
                "threshold": self.min_cells_per_sample,
                "samples_below_threshold": len(low_cell_samples),
                "sample_cell_counts": sample_cell_counts,
            }
            
            if low_cell_samples:
                result["warnings"].append(
                    f"{len(low_cell_samples)} samples have fewer than {self.min_cells_per_sample} cells"
                )
        
        # Check maximum genes per dataset
        if "n_genes" in dataset_data:
            n_genes = dataset_data["n_genes"]
            result["constraint_checks"]["max_genes_per_dataset"] = {
                "threshold": self.max_genes_per_dataset,
                "actual_genes": n_genes,
                "exceeds_threshold": n_genes > self.max_genes_per_dataset,
            }
            
            if n_genes > self.max_genes_per_dataset:
                result["warnings"].append(
                    f"Dataset has {n_genes} genes, exceeding recommended maximum of {self.max_genes_per_dataset}"
                )
        
        return result