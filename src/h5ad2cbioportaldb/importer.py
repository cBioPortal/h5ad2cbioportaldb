"""Main importer for h5ad files into cBioPortal ClickHouse database."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from .cbioportal.client import CBioPortalClient
from .cbioportal.integration import CBioPortalIntegration
from .cbioportal.schema import CBioPortalSchema
from .mapper import CBioPortalMapper


logger = logging.getLogger(__name__)


class H5adImporter:
    """Main importer for h5ad files into cBioPortal database."""

    def __init__(self, client: CBioPortalClient, config: Dict[str, Any]) -> None:
        """Initialize importer with client and configuration."""
        self.client = client
        self.config = config
        self.schema = CBioPortalSchema(client)
        self.integration = CBioPortalIntegration(client, config)
        self.mapper = CBioPortalMapper(client, config.get("mapping", {}))
        
        # Import configuration
        self.auto_map_genes = config.get("auto_map_genes", True)
        self.validate_mappings = config.get("validate_mappings", True)
        self.batch_size = config.get("batch_size", 10000)
        self.max_memory_usage = config.get("max_memory_usage", "4GB")

    def import_dataset(
        self,
        file: str,
        dataset_id: str,
        study_id: str,
        cell_type_column: Optional[str] = None,
        sample_obs_column: Optional[str] = None,
        patient_obs_column: Optional[str] = None,
        sample_mapping_file: Optional[str] = None,
        patient_mapping_file: Optional[str] = None,
        description: Optional[str] = None,
        matrix_type: str = "raw",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Import h5ad file into cBioPortal database."""
        
        logger.info(f"Starting import of dataset {dataset_id} from {file}")
        
        # Validate inputs
        self._validate_import_inputs(file, dataset_id, study_id, matrix_type)
        
        # Load h5ad file
        adata = self._load_h5ad_file(file)
        logger.info(f"Loaded h5ad file: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Validate study exists
        if not self.schema.validate_study_exists(study_id):
            raise ValueError(f"Study {study_id} not found in cBioPortal")
        
        # Resolve sample/patient mappings
        resolved_mappings, mapping_stats = self.mapper.resolve_mappings(
            adata, study_id, sample_obs_column, patient_obs_column,
            sample_mapping_file, patient_mapping_file
        )
        
        # Map genes to cBioPortal
        gene_mapping = self._map_genes_to_cbioportal(adata, dataset_id)
        
        # Validate mappings if requested
        if self.validate_mappings and not dry_run:
            self._validate_before_import(study_id, resolved_mappings, gene_mapping)
        
        # Prepare data for insertion
        dataset_record = self._prepare_dataset_record(
            adata, dataset_id, study_id, file, description, mapping_stats
        )
        
        cell_records = self._prepare_cell_records(adata, dataset_id, resolved_mappings, cell_type_column)
        gene_records = self._prepare_gene_records(dataset_id, gene_mapping)
        
        # Prepare expression data
        expression_data = self._prepare_expression_data(
            adata, dataset_id, matrix_type, cell_records
        )
        
        # Prepare embeddings if available
        embedding_data = self._prepare_embedding_data(adata, dataset_id, cell_records)
        
        if dry_run:
            return self._generate_dry_run_summary(
                dataset_record, cell_records, gene_records, expression_data, embedding_data
            )
        
        # Create tables if they don't exist
        self.client.create_tables_if_not_exist()
        
        # Insert data
        self._insert_all_data(
            dataset_record, cell_records, gene_records, expression_data, embedding_data
        )
        
        # Generate final summary
        return self._generate_import_summary(dataset_id, adata, mapping_stats)

    def _validate_import_inputs(
        self, file: str, dataset_id: str, study_id: str, matrix_type: str
    ) -> None:
        """Validate import inputs."""
        if not Path(file).exists():
            raise FileNotFoundError(f"H5AD file not found: {file}")
        
        if not dataset_id or not study_id:
            raise ValueError("Dataset ID and study ID are required")
        
        if matrix_type not in ["raw", "normalized", "scaled"]:
            raise ValueError(f"Invalid matrix type: {matrix_type}")
        
        # Check if dataset already exists
        existing_dataset = self.schema.get_dataset_info(dataset_id)
        if existing_dataset:
            raise ValueError(f"Dataset {dataset_id} already exists")

    def _load_h5ad_file(self, file: str) -> anndata.AnnData:
        """Load and validate h5ad file."""
        try:
            adata = anndata.read_h5ad(file)
            
            # Basic validation
            if adata.n_obs == 0:
                raise ValueError("H5AD file contains no cells")
            if adata.n_vars == 0:
                raise ValueError("H5AD file contains no genes")
            
            logger.info(f"H5AD file validation passed: {adata.n_obs} cells, {adata.n_vars} genes")
            return adata
            
        except Exception as e:
            logger.error(f"Failed to load h5ad file {file}: {e}")
            raise

    def _map_genes_to_cbioportal(self, adata: anndata.AnnData, dataset_id: str) -> Dict[str, Any]:
        """Map genes from h5ad to cBioPortal gene table."""
        gene_symbols = adata.var.index.tolist()
        
        # Get existing genes from cBioPortal
        existing_genes_df = self.client.get_existing_genes(gene_symbols)
        existing_genes = {
            row["hugo_gene_symbol"]: {
                "entrez_gene_id": row["entrez_gene_id"],
                "gene_symbol": row["hugo_gene_symbol"]
            }
            for _, row in existing_genes_df.iterrows()
        }
        
        # Create mapping
        mapped_genes = []
        unmapped_genes = []
        
        for idx, gene_symbol in enumerate(gene_symbols):
            if gene_symbol in existing_genes:
                mapped_genes.append({
                    "idx": idx,
                    "symbol": gene_symbol,
                    "entrez_id": existing_genes[gene_symbol]["entrez_gene_id"]
                })
            else:
                unmapped_genes.append(gene_symbol)
        
        logger.info(f"Gene mapping: {len(mapped_genes)} mapped, {len(unmapped_genes)} unmapped")
        
        if unmapped_genes and self.config.get("warn_unmapped_genes", True):
            logger.warning(f"Unmapped genes ({len(unmapped_genes)}): {unmapped_genes[:10]}...")
        
        return {
            "mapped": mapped_genes,
            "unmapped": unmapped_genes,
            "total": len(gene_symbols)
        }

    def _validate_before_import(
        self, study_id: str, resolved_mappings: Dict[str, Any], gene_mapping: Dict[str, Any]
    ) -> None:
        """Perform validation before import."""
        # Check mapping coverage
        unmapped_cells = sum(1 for m in resolved_mappings.values() if m["strategy"] == "no_mapping")
        if unmapped_cells > 0:
            logger.warning(f"{unmapped_cells} cells have no cBioPortal mapping")
        
        # Check gene mapping coverage
        unmapped_genes = len(gene_mapping["unmapped"])
        if unmapped_genes > 0:
            logger.warning(f"{unmapped_genes} genes not found in cBioPortal")

    def _prepare_dataset_record(
        self,
        adata: anndata.AnnData,
        dataset_id: str,
        study_id: str,
        file_path: str,
        description: Optional[str],
        mapping_stats: Dict[str, int],
    ) -> Dict[str, Any]:
        """Prepare dataset record for insertion."""
        metadata = {
            "original_file": Path(file_path).name,
            "import_timestamp": pd.Timestamp.now().isoformat(),
            "mapping_statistics": mapping_stats,
            "h5ad_metadata": {
                "uns_keys": list(adata.uns.keys()) if adata.uns else [],
                "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
                "varm_keys": list(adata.varm.keys()) if adata.varm else [],
                "obs_columns": list(adata.obs.columns),
                "var_columns": list(adata.var.columns),
            }
        }
        
        return self.integration.prepare_dataset_record(
            dataset_id=dataset_id,
            name=f"Single-cell dataset {dataset_id}",
            study_id=study_id,
            description=description or f"Imported from {Path(file_path).name}",
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            file_path=file_path,
            metadata=metadata,
        )

    def _prepare_cell_records(
        self,
        adata: anndata.AnnData,
        dataset_id: str,
        resolved_mappings: Dict[str, Any],
        cell_type_column: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Prepare cell records for insertion."""
        cell_data = adata.obs.copy()
        
        # Add cell type information
        if cell_type_column and cell_type_column in cell_data.columns:
            cell_data["cell_type"] = cell_data[cell_type_column]
        else:
            cell_data["cell_type"] = "Unknown"
        
        # Add barcode information
        cell_data["barcode"] = cell_data.index
        
        return self.integration.prepare_cell_records(dataset_id, cell_data, resolved_mappings)

    def _prepare_gene_records(
        self, dataset_id: str, gene_mapping: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare gene records for insertion."""
        gene_records = []
        
        for gene_info in gene_mapping["mapped"]:
            gene_records.append({
                "dataset_id": dataset_id,
                "gene_idx": gene_info["idx"],
                "hugo_gene_symbol": gene_info["symbol"],
                "entrez_gene_id": gene_info["entrez_id"],
            })
        
        return gene_records

    def _prepare_expression_data(
        self,
        adata: anndata.AnnData,
        dataset_id: str,
        matrix_type: str,
        cell_records: List[Dict[str, Any]],
    ) -> Tuple[List[List[Any]], List[str]]:
        """Prepare expression matrix data for insertion."""
        # Select appropriate matrix
        if matrix_type == "raw" and adata.raw is not None:
            expression_matrix = adata.raw.X
        elif matrix_type == "normalized" and "normalized" in adata.layers:
            expression_matrix = adata.layers["normalized"]
        elif matrix_type == "scaled" and "scaled" in adata.layers:
            expression_matrix = adata.layers["scaled"]
        else:
            expression_matrix = adata.X
        
        # Convert to sparse if needed
        if not sparse.issparse(expression_matrix):
            expression_matrix = sparse.csr_matrix(expression_matrix)
        
        # Get cell IDs
        cell_ids = [record["cell_id"] for record in cell_records]
        
        # Prepare data for insertion
        return self.integration.prepare_expression_data(
            dataset_id, expression_matrix, cell_ids, matrix_type
        )

    def _prepare_embedding_data(
        self,
        adata: anndata.AnnData,
        dataset_id: str,
        cell_records: List[Dict[str, Any]],
    ) -> Optional[Tuple[List[List[Any]], List[str]]]:
        """Prepare embedding data for insertion."""
        if not adata.obsm:
            return None
        
        # Find supported embeddings
        supported_embeddings = ["X_umap", "X_tsne", "X_pca"]
        available_embeddings = {}
        
        for embedding_key in supported_embeddings:
            if embedding_key in adata.obsm:
                embedding_name = embedding_key.replace("X_", "")
                available_embeddings[embedding_name] = adata.obsm[embedding_key]
        
        if not available_embeddings:
            return None
        
        # Get cell IDs
        cell_ids = [record["cell_id"] for record in cell_records]
        
        # Prepare data for insertion
        return self.integration.prepare_embedding_data(dataset_id, available_embeddings, cell_ids)

    def _insert_all_data(
        self,
        dataset_record: Dict[str, Any],
        cell_records: List[Dict[str, Any]],
        gene_records: List[Dict[str, Any]],
        expression_data: Tuple[List[List[Any]], List[str]],
        embedding_data: Optional[Tuple[List[List[Any]], List[str]]],
    ) -> None:
        """Insert all prepared data into database."""
        try:
            # Insert dataset
            logger.info("Inserting dataset record...")
            self.integration.insert_dataset(dataset_record)
            
            # Insert cells
            logger.info(f"Inserting {len(cell_records)} cell records...")
            self.integration.insert_cells(cell_records)
            
            # Insert genes
            logger.info(f"Inserting {len(gene_records)} gene records...")
            self.integration.insert_genes(gene_records)
            
            # Insert expression matrix
            expr_data, expr_columns = expression_data
            logger.info(f"Inserting {len(expr_data)} expression values...")
            self.integration.insert_expression_matrix(expr_data, expr_columns, self.batch_size)
            
            # Insert embeddings if available
            if embedding_data:
                emb_data, emb_columns = embedding_data
                logger.info(f"Inserting {len(emb_data)} embedding values...")
                self.integration.insert_embeddings(emb_data, emb_columns, self.batch_size)
            
            logger.info("All data inserted successfully")
            
        except Exception as e:
            logger.error(f"Data insertion failed: {e}")
            raise

    def _generate_dry_run_summary(
        self,
        dataset_record: Dict[str, Any],
        cell_records: List[Dict[str, Any]],
        gene_records: List[Dict[str, Any]],
        expression_data: Tuple[List[List[Any]], List[str]],
        embedding_data: Optional[Tuple[List[List[Any]], List[str]]],
    ) -> Dict[str, Any]:
        """Generate dry run summary."""
        expr_data, _ = expression_data
        
        summary = {
            "dataset_id": dataset_record["dataset_id"],
            "study_id": dataset_record["cancer_study_identifier"],
            "n_cells": len(cell_records),
            "n_genes": len(gene_records),
            "n_expression_values": len(expr_data),
            "n_embedding_values": len(embedding_data[0]) if embedding_data else 0,
            "dry_run": True,
        }
        
        return summary

    def _generate_import_summary(
        self, dataset_id: str, adata: anndata.AnnData, mapping_stats: Dict[str, int]
    ) -> Dict[str, Any]:
        """Generate final import summary."""
        return {
            "dataset_id": dataset_id,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "mapping_statistics": mapping_stats,
            "success": True,
        }

    def export_dataset(
        self,
        dataset_id: str,
        output_file: str,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Export dataset back to h5ad format with optional filtering."""
        
        logger.info(f"Exporting dataset {dataset_id} to {output_file}")
        
        # Get dataset info
        dataset_info = self.schema.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Build filter conditions
        filter_conditions = []
        if filters:
            if "cell_types" in filters:
                cell_types = "', '".join(filters["cell_types"])
                filter_conditions.append(f"original_cell_type IN ('{cell_types}')")
            
            if "samples" in filters:
                samples = "', '".join(filters["samples"])
                filter_conditions.append(f"sample_unique_id IN ('{samples}')")
        
        where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"
        
        # Get filtered cells
        cells_df = self.client.query(f"""
        SELECT * FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s AND {where_clause}
        """, {"dataset": dataset_id})
        
        if len(cells_df) == 0:
            raise ValueError("No cells found matching filter criteria")
        
        # Get genes
        genes_df = self.client.query(f"""
        SELECT * FROM {self.client.database}.{self.client.table_prefix}dataset_genes
        WHERE dataset_id = %(dataset)s
        ORDER BY gene_idx
        """, {"dataset": dataset_id})
        
        # Apply gene filtering if specified
        if filters and "genes" in filters:
            genes_df = genes_df[genes_df["hugo_gene_symbol"].isin(filters["genes"])]
        
        # Get expression data
        cell_ids = "', '".join(cells_df["cell_id"])
        gene_indices = "', '".join(map(str, genes_df["gene_idx"]))
        
        expr_df = self.client.query(f"""
        SELECT * FROM {self.client.database}.{self.client.table_prefix}expression_matrix
        WHERE dataset_id = %(dataset)s 
          AND cell_id IN ('{cell_ids}')
          AND gene_idx IN ({gene_indices})
        """, {"dataset": dataset_id})
        
        # Reconstruct AnnData object
        adata = self._reconstruct_anndata(cells_df, genes_df, expr_df)
        
        # Save to file
        adata.write_h5ad(output_file)
        
        return {
            "n_cells": len(cells_df),
            "n_genes": len(genes_df),
            "output_file": output_file,
        }

    def _reconstruct_anndata(
        self,
        cells_df: pd.DataFrame,
        genes_df: pd.DataFrame,
        expr_df: pd.DataFrame,
    ) -> anndata.AnnData:
        """Reconstruct AnnData object from database data."""
        
        # Create obs DataFrame
        obs = cells_df.set_index("cell_id")
        obs["obs_data"] = obs["obs_data"].apply(json.loads)
        
        # Create var DataFrame
        var = genes_df.set_index("hugo_gene_symbol")
        
        # Create expression matrix
        n_cells = len(cells_df)
        n_genes = len(genes_df)
        
        # Create cell_id to index mapping
        cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(cells_df["cell_id"])}
        gene_idx_to_pos = {gene_idx: pos for pos, gene_idx in enumerate(genes_df["gene_idx"])}
        
        # Build sparse matrix
        rows = []
        cols = []
        data = []
        
        for _, row in expr_df.iterrows():
            cell_idx = cell_id_to_idx[row["cell_id"]]
            gene_pos = gene_idx_to_pos[row["gene_idx"]]
            
            rows.append(cell_idx)
            cols.append(gene_pos)
            data.append(row["count"])
        
        X = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))
        
        # Create AnnData object
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        
        return adata