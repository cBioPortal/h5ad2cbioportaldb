"""cBioPortal-specific integration utilities."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .client import CBioPortalClient
from .schema import CBioPortalSchema


logger = logging.getLogger(__name__)


class CBioPortalIntegration:
    """Handle cBioPortal-specific integration tasks."""

    def __init__(self, client: CBioPortalClient, config: Dict[str, Any]) -> None:
        """Initialize with client and configuration."""
        self.client = client
        self.config = config
        self.schema = CBioPortalSchema(client)

    def prepare_dataset_record(
        self,
        dataset_id: str,
        name: str,
        study_id: str,
        description: str,
        n_cells: int,
        n_genes: int,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare dataset record for insertion."""
        return {
            "dataset_id": dataset_id,
            "name": name,
            "cancer_study_identifier": study_id,
            "description": description or "",
            "n_cells": n_cells,
            "n_genes": n_genes,
            "imported_at": datetime.now(),
            "file_path": file_path,
            "metadata": json.dumps(metadata),
        }

    def prepare_cell_records(
        self,
        dataset_id: str,
        cell_data: pd.DataFrame,
        mapping_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare cell records for insertion."""
        records = []
        
        for idx, row in cell_data.iterrows():
            cell_id = str(idx)
            mapping_info = mapping_results.get(cell_id, {})
            
            record = {
                "dataset_id": dataset_id,
                "cell_id": cell_id,
                "cell_barcode": row.get("barcode", cell_id),
                "sample_unique_id": mapping_info.get("sample_id"),
                "patient_unique_id": mapping_info.get("patient_id"),
                "original_cell_type": row.get("cell_type", "Unknown"),
                "harmonized_cell_type_id": None,
                "harmonization_confidence": None,
                "mapping_strategy": mapping_info.get("strategy", "no_mapping"),
                "obs_data": json.dumps(row.to_dict()),
            }
            records.append(record)
            
        return records

    def prepare_gene_records(
        self,
        dataset_id: str,
        gene_symbols: List[str],
        gene_mapping: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Prepare gene records for insertion."""
        records = []
        
        for idx, gene_symbol in enumerate(gene_symbols):
            gene_info = gene_mapping.get(gene_symbol, {})
            
            record = {
                "dataset_id": dataset_id,
                "gene_idx": idx,
                "hugo_gene_symbol": gene_symbol,
                "entrez_gene_id": gene_info.get("entrez_gene_id"),
            }
            records.append(record)
            
        return records

    def prepare_expression_data(
        self,
        dataset_id: str,
        expression_matrix: sparse.csr_matrix,
        cell_ids: List[str],
        matrix_type: str = "raw",
        min_expression: float = 0.0,
    ) -> Tuple[List[List[Any]], List[str]]:
        """Prepare expression data for bulk insertion using SPARSE columns."""
        data = []
        columns = ["dataset_id", "cell_id", "gene_idx", "matrix_type", "count"]
        
        # Convert to COO format for efficient iteration
        coo_matrix = expression_matrix.tocoo()
        
        for i, j, value in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if value > min_expression:
                data.append([
                    dataset_id,
                    cell_ids[i],
                    int(j),
                    matrix_type,
                    float(value)
                ])
        
        logger.info(f"Prepared {len(data)} non-zero expression values")
        return data, columns

    def prepare_embedding_data(
        self,
        dataset_id: str,
        embeddings: Dict[str, np.ndarray],
        cell_ids: List[str],
    ) -> Tuple[List[List[Any]], List[str]]:
        """Prepare embedding data for insertion."""
        data = []
        columns = ["dataset_id", "cell_id", "embedding_type", "dimension_idx", "value"]
        
        for embedding_type, embedding_matrix in embeddings.items():
            for cell_idx, cell_id in enumerate(cell_ids):
                for dim_idx in range(embedding_matrix.shape[1]):
                    value = embedding_matrix[cell_idx, dim_idx]
                    if not np.isnan(value):
                        data.append([
                            dataset_id,
                            cell_id,
                            embedding_type,
                            int(dim_idx),
                            float(value)
                        ])
        
        logger.info(f"Prepared {len(data)} embedding values")
        return data, columns

    def insert_dataset(self, dataset_record: Dict[str, Any]) -> None:
        """Insert dataset record."""
        df = pd.DataFrame([dataset_record])
        table_name = f"{self.client.table_prefix}datasets"
        self.client.insert_dataframe(table_name, df)

    def insert_cells(self, cell_records: List[Dict[str, Any]]) -> None:
        """Insert cell records."""
        df = pd.DataFrame(cell_records)
        table_name = f"{self.client.table_prefix}cells"
        self.client.insert_dataframe(table_name, df)

    def insert_genes(self, gene_records: List[Dict[str, Any]]) -> None:
        """Insert gene records."""
        df = pd.DataFrame(gene_records)
        table_name = f"{self.client.table_prefix}dataset_genes"
        self.client.insert_dataframe(table_name, df)

    def insert_expression_matrix(
        self,
        expression_data: List[List[Any]],
        columns: List[str],
        batch_size: int = 50000,
    ) -> None:
        """Insert expression matrix data."""
        table_name = f"{self.client.table_prefix}expression_matrix"
        self.client.bulk_insert(table_name, expression_data, columns, batch_size)

    def insert_embeddings(
        self,
        embedding_data: List[List[Any]],
        columns: List[str],
        batch_size: int = 50000,
    ) -> None:
        """Insert embedding data."""
        table_name = f"{self.client.table_prefix}cell_embeddings"
        self.client.bulk_insert(table_name, embedding_data, columns, batch_size)

    def create_synthetic_samples(
        self,
        study_id: str,
        patient_sample_mapping: Dict[str, str],
        suffix: str = "SC",
    ) -> Dict[str, str]:
        """Create synthetic sample IDs and validate they don't conflict."""
        synthetic_samples = {}
        
        for patient_id in patient_sample_mapping.keys():
            synthetic_id = self.schema.generate_synthetic_sample_id(patient_id, suffix)
            synthetic_samples[patient_id] = synthetic_id
        
        # Check for conflicts
        conflicts = self.schema.validate_synthetic_sample_ids(
            study_id, list(synthetic_samples.values())
        )
        
        if conflicts:
            logger.warning(f"Synthetic sample ID conflicts detected: {conflicts}")
            # Generate alternative IDs
            for patient_id, sample_id in synthetic_samples.items():
                if sample_id in conflicts:
                    counter = 1
                    while True:
                        alternative_id = f"{patient_id}-{suffix}{counter}"
                        if alternative_id not in conflicts:
                            synthetic_samples[patient_id] = alternative_id
                            break
                        counter += 1
        
        return synthetic_samples

    def get_bulk_expression_comparison(
        self,
        study_id: str,
        gene_symbol: str,
        sc_dataset_id: str,
    ) -> pd.DataFrame:
        """Get bulk vs single-cell expression comparison."""
        return self.client.query("""
        WITH bulk_expr AS (
            SELECT 
                patient_unique_id,
                AVG(CAST(alteration_value AS Float32)) as avg_bulk_expression,
                COUNT(*) as bulk_samples
            FROM genetic_alteration_derived 
            WHERE hugo_gene_symbol = %(gene)s
              AND cancer_study_identifier = %(study)s
              AND profile_type = 'rna_seq_v2_mrna'
            GROUP BY patient_unique_id
        ),
        sc_expr AS (
            SELECT 
                sc.patient_unique_id,
                sc.original_cell_type,
                AVG(expr.count) as avg_sc_expression,
                COUNT(*) as num_cells
            FROM {sc_expr_table} expr
            JOIN {sc_cells_table} sc ON expr.dataset_id = sc.dataset_id AND expr.cell_id = sc.cell_id
            JOIN {sc_genes_table} sg ON expr.dataset_id = sg.dataset_id AND expr.gene_idx = sg.gene_idx
            WHERE sg.hugo_gene_symbol = %(gene)s
              AND sc.dataset_id = %(sc_dataset)s
              AND sc.patient_unique_id IS NOT NULL
            GROUP BY sc.patient_unique_id, sc.original_cell_type
        )
        SELECT 
            COALESCE(b.patient_unique_id, s.patient_unique_id) as patient_id,
            b.avg_bulk_expression,
            b.bulk_samples,
            s.original_cell_type,
            s.avg_sc_expression,
            s.num_cells
        FROM bulk_expr b
        FULL OUTER JOIN sc_expr s ON b.patient_unique_id = s.patient_unique_id
        ORDER BY patient_id, original_cell_type
        """.format(
            sc_expr_table=f"{self.client.database}.{self.client.table_prefix}expression_matrix",
            sc_cells_table=f"{self.client.database}.{self.client.table_prefix}cells",
            sc_genes_table=f"{self.client.database}.{self.client.table_prefix}dataset_genes"
        ), {
            "gene": gene_symbol,
            "study": study_id,
            "sc_dataset": sc_dataset_id
        })

    def get_cell_type_summary(self, dataset_id: str) -> pd.DataFrame:
        """Get cell type summary for dataset."""
        return self.client.query(f"""
        SELECT 
            original_cell_type,
            harmonized_cell_type_id,
            mapping_strategy,
            COUNT(*) as cell_count,
            COUNT(DISTINCT sample_unique_id) as unique_samples,
            COUNT(DISTINCT patient_unique_id) as unique_patients,
            AVG(harmonization_confidence) as avg_confidence
        FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
        GROUP BY original_cell_type, harmonized_cell_type_id, mapping_strategy
        ORDER BY cell_count DESC
        """, {"dataset": dataset_id})

    def get_dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        try:
            # Basic stats
            cells_stats = self.client.query(f"""
            SELECT 
                COUNT(*) as total_cells,
                COUNT(DISTINCT original_cell_type) as unique_cell_types,
                COUNT(DISTINCT sample_unique_id) as mapped_samples,
                COUNT(DISTINCT patient_unique_id) as mapped_patients,
                COUNT(CASE WHEN sample_unique_id IS NOT NULL THEN 1 END) as cells_with_samples,
                COUNT(CASE WHEN patient_unique_id IS NOT NULL THEN 1 END) as cells_with_patients
            FROM {self.client.database}.{self.client.table_prefix}cells
            WHERE dataset_id = %(dataset)s
            """, {"dataset": dataset_id})
            
            # Expression stats
            expr_stats = self.client.query(f"""
            SELECT 
                COUNT(DISTINCT gene_idx) as total_genes,
                COUNT(*) as total_expression_values,
                AVG(count) as avg_expression,
                MAX(count) as max_expression
            FROM {self.client.database}.{self.client.table_prefix}expression_matrix
            WHERE dataset_id = %(dataset)s
            """, {"dataset": dataset_id})
            
            # Mapping strategy breakdown
            mapping_stats = self.client.query(f"""
            SELECT 
                mapping_strategy,
                COUNT(*) as cell_count
            FROM {self.client.database}.{self.client.table_prefix}cells
            WHERE dataset_id = %(dataset)s
            GROUP BY mapping_strategy
            """, {"dataset": dataset_id})
            
            return {
                "cells": cells_stats.iloc[0].to_dict() if len(cells_stats) > 0 else {},
                "expression": expr_stats.iloc[0].to_dict() if len(expr_stats) > 0 else {},
                "mapping_strategies": mapping_stats.to_dict("records"),
            }
            
        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {e}")
            return {"cells": {}, "expression": {}, "mapping_strategies": []}