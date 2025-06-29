"""cBioPortal schema helpers and validation."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .client import CBioPortalClient


logger = logging.getLogger(__name__)


class CBioPortalSchema:
    """Helper class for cBioPortal schema operations and validation."""

    def __init__(self, client: CBioPortalClient) -> None:
        """Initialize with cBioPortal client."""
        self.client = client

    def validate_study_exists(self, study_id: str) -> bool:
        """Validate that a study exists in cBioPortal."""
        try:
            studies = self.client.query(
                "SELECT cancer_study_identifier FROM cancer_study WHERE cancer_study_identifier = %(study)s",
                {"study": study_id}
            )
            return len(studies) > 0
        except Exception as e:
            logger.error(f"Failed to validate study {study_id}: {e}")
            return False

    def get_study_info(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a study."""
        try:
            result = self.client.query(
                "SELECT * FROM cancer_study WHERE cancer_study_identifier = %(study)s",
                {"study": study_id}
            )
            if len(result) > 0:
                return result.iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get study info for {study_id}: {e}")
            return None

    def validate_sample_ids(self, study_id: str, sample_ids: List[str]) -> Dict[str, Any]:
        """Validate sample IDs against existing samples in study."""
        try:
            if not sample_ids:
                return {"valid": [], "invalid": [], "total": 0}

            placeholders = ",".join([f"'{sid}'" for sid in sample_ids])
            existing = self.client.query(f"""
                SELECT sample_unique_id 
                FROM sample 
                WHERE cancer_study_identifier = '{study_id}' 
                  AND sample_unique_id IN ({placeholders})
            """)
            
            existing_ids = set(existing["sample_unique_id"].tolist())
            input_ids = set(sample_ids)
            
            return {
                "valid": list(existing_ids),
                "invalid": list(input_ids - existing_ids),
                "total": len(sample_ids),
                "found": len(existing_ids),
                "missing": len(input_ids - existing_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate sample IDs: {e}")
            return {"valid": [], "invalid": sample_ids, "total": len(sample_ids)}

    def validate_patient_ids(self, study_id: str, patient_ids: List[str]) -> Dict[str, Any]:
        """Validate patient IDs against existing patients in study."""
        try:
            if not patient_ids:
                return {"valid": [], "invalid": [], "total": 0}

            placeholders = ",".join([f"'{pid}'" for pid in patient_ids])
            existing = self.client.query(f"""
                SELECT patient_unique_id 
                FROM patient 
                WHERE cancer_study_identifier = '{study_id}' 
                  AND patient_unique_id IN ({placeholders})
            """)
            
            existing_ids = set(existing["patient_unique_id"].tolist())
            input_ids = set(patient_ids)
            
            return {
                "valid": list(existing_ids),
                "invalid": list(input_ids - existing_ids),
                "total": len(patient_ids),
                "found": len(existing_ids),
                "missing": len(input_ids - existing_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate patient IDs: {e}")
            return {"valid": [], "invalid": patient_ids, "total": len(patient_ids)}

    def validate_gene_symbols(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """Validate gene symbols against cBioPortal gene table."""
        try:
            if not gene_symbols:
                return {"valid": [], "invalid": [], "total": 0}

            placeholders = ",".join([f"'{gene}'" for gene in gene_symbols])
            existing = self.client.query(f"""
                SELECT hugo_gene_symbol, entrez_gene_id 
                FROM gene 
                WHERE hugo_gene_symbol IN ({placeholders})
            """)
            
            existing_genes = set(existing["hugo_gene_symbol"].tolist())
            input_genes = set(gene_symbols)
            
            return {
                "valid": list(existing_genes),
                "invalid": list(input_genes - existing_genes),
                "total": len(gene_symbols),
                "found": len(existing_genes),
                "missing": len(input_genes - existing_genes),
                "gene_info": existing.to_dict("records")
            }
            
        except Exception as e:
            logger.error(f"Failed to validate gene symbols: {e}")
            return {"valid": [], "invalid": gene_symbols, "total": len(gene_symbols)}

    def get_sample_patient_mapping(self, study_id: str) -> pd.DataFrame:
        """Get sample to patient mapping for a study."""
        try:
            return self.client.query("""
                SELECT 
                    s.sample_unique_id,
                    s.patient_unique_id,
                    s.sample_type,
                    p.patient_unique_id as patient_id_check
                FROM sample s
                LEFT JOIN patient p ON s.patient_unique_id = p.patient_unique_id 
                    AND s.cancer_study_identifier = p.cancer_study_identifier
                WHERE s.cancer_study_identifier = %(study)s
            """, {"study": study_id})
        except Exception as e:
            logger.error(f"Failed to get sample-patient mapping for {study_id}: {e}")
            return pd.DataFrame()

    def generate_synthetic_sample_id(self, patient_id: str, suffix: str = "SC") -> str:
        """Generate synthetic sample ID for single-cell data."""
        return f"{patient_id}-{suffix}"

    def validate_synthetic_sample_ids(self, study_id: str, synthetic_ids: List[str]) -> List[str]:
        """Check if synthetic sample IDs would conflict with existing samples."""
        try:
            placeholders = ",".join([f"'{sid}'" for sid in synthetic_ids])
            existing = self.client.query(f"""
                SELECT sample_unique_id 
                FROM sample 
                WHERE cancer_study_identifier = '{study_id}' 
                  AND sample_unique_id IN ({placeholders})
            """)
            
            conflicts = set(existing["sample_unique_id"].tolist())
            return [sid for sid in synthetic_ids if sid in conflicts]
            
        except Exception as e:
            logger.error(f"Failed to validate synthetic sample IDs: {e}")
            return synthetic_ids

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get single-cell dataset information."""
        try:
            table_name = f"{self.client.table_prefix}datasets"
            result = self.client.query(
                f"SELECT * FROM {self.client.database}.{table_name} WHERE dataset_id = %(dataset)s",
                {"dataset": dataset_id}
            )
            if len(result) > 0:
                return result.iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_id}: {e}")
            return None

    def list_datasets(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """List all single-cell datasets, optionally filtered by study."""
        try:
            table_name = f"{self.client.table_prefix}datasets"
            if study_id:
                return self.client.query(
                    f"SELECT * FROM {self.client.database}.{table_name} WHERE cancer_study_identifier = %(study)s",
                    {"study": study_id}
                )
            return self.client.query(f"SELECT * FROM {self.client.database}.{table_name}")
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return pd.DataFrame()

    def get_expression_profiles(self, study_id: str) -> pd.DataFrame:
        """Get available expression profiles for a study."""
        try:
            return self.client.query("""
                SELECT 
                    genetic_profile_id,
                    genetic_alteration_type,
                    datatype,
                    profile_name,
                    profile_description
                FROM genetic_profile 
                WHERE cancer_study_identifier = %(study)s
                  AND genetic_alteration_type = 'MRNA_EXPRESSION'
            """, {"study": study_id})
        except Exception as e:
            logger.error(f"Failed to get expression profiles for {study_id}: {e}")
            return pd.DataFrame()

    def check_table_exists(self, table_name: str) -> bool:
        """Check if a specific table exists."""
        return self.client.table_exists(table_name)

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a table."""
        try:
            return self.client.query(
                "SELECT * FROM system.columns WHERE database = %(db)s AND table = %(table)s",
                {"db": self.client.database, "table": table_name}
            )
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return pd.DataFrame()

    def estimate_table_size(self, table_name: str) -> Dict[str, Any]:
        """Estimate table size and row count."""
        try:
            size_info = self.client.query(
                "SELECT * FROM system.parts WHERE database = %(db)s AND table = %(table)s",
                {"db": self.client.database, "table": table_name}
            )
            
            if len(size_info) == 0:
                return {"rows": 0, "bytes": 0, "compressed_bytes": 0}
            
            return {
                "rows": size_info["rows"].sum(),
                "bytes": size_info["bytes_on_disk"].sum(),
                "compressed_bytes": size_info["data_compressed_bytes"].sum(),
                "parts": len(size_info)
            }
        except Exception as e:
            logger.error(f"Failed to estimate size for table {table_name}: {e}")
            return {"rows": 0, "bytes": 0, "compressed_bytes": 0}