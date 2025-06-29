"""ClickHouse client for cBioPortal database operations."""

import logging
from typing import Any, Dict, List, Optional, Union

import clickhouse_connect
import pandas as pd
from clickhouse_connect.driver import Client


logger = logging.getLogger(__name__)


class CBioPortalClient:
    """ClickHouse client for cBioPortal database operations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ClickHouse client with cBioPortal configuration."""
        self.config = config.get("clickhouse", {})
        self.database = self.config.get("database", "cbioportal")
        self.table_prefix = config.get("table_prefix", "scRNA_")
        
        self._client: Optional[Client] = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to ClickHouse."""
        try:
            self._client = clickhouse_connect.get_client(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 9000),
                database=self.database,
                username=self.config.get("username", "default"),
                password=self.config.get("password", ""),
                secure=self.config.get("secure", False),
                connect_timeout=self.config.get("timeout", 30),
            )
            
            # Test connection
            self._client.command("SELECT 1")
            logger.info(f"Connected to ClickHouse database: {self.database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    @property
    def client(self) -> Client:
        """Get ClickHouse client instance."""
        if self._client is None:
            self._connect()
        return self._client

    def query(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        try:
            if parameters:
                result = self.client.query_df(sql, parameters=parameters)
            else:
                result = self.client.query_df(sql)
            logger.debug(f"Query executed successfully: {sql[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Query failed: {sql[:100]}... Error: {e}")
            raise

    def command(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Execute command without returning results."""
        try:
            if parameters:
                self.client.command(sql, parameters=parameters)
            else:
                self.client.command(sql)
            logger.debug(f"Command executed successfully: {sql[:100]}...")
        except Exception as e:
            logger.error(f"Command failed: {sql[:100]}... Error: {e}")
            raise

    def insert_dataframe(
        self,
        table: str,
        df: pd.DataFrame,
        batch_size: int = 10000,
    ) -> None:
        """Insert DataFrame into table in batches."""
        try:
            full_table_name = f"{self.database}.{table}"
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                self.client.insert_df(full_table_name, batch)
                logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} rows")
                
            logger.info(f"Successfully inserted {len(df)} rows into {full_table_name}")
            
        except Exception as e:
            logger.error(f"Insert failed for table {table}: {e}")
            raise

    def bulk_insert(
        self,
        table: str,
        data: List[List[Any]],
        columns: List[str],
        batch_size: int = 10000,
    ) -> None:
        """Bulk insert data into table."""
        try:
            full_table_name = f"{self.database}.{table}"
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                self.client.insert(full_table_name, batch_data, column_names=columns)
                logger.debug(f"Bulk inserted batch {i//batch_size + 1}: {len(batch_data)} rows")
                
            logger.info(f"Successfully bulk inserted {len(data)} rows into {full_table_name}")
            
        except Exception as e:
            logger.error(f"Bulk insert failed for table {table}: {e}")
            raise

    def table_exists(self, table: str) -> bool:
        """Check if table exists in database."""
        try:
            result = self.query(
                "SELECT 1 FROM system.tables WHERE database = %(db)s AND name = %(table)s",
                {"db": self.database, "table": table}
            )
            return len(result) > 0
        except Exception:
            return False

    def create_tables_if_not_exist(self) -> None:
        """Create single-cell tables if they don't exist."""
        tables = [
            self._get_datasets_table_ddl(),
            self._get_cells_table_ddl(),
            self._get_genes_table_ddl(),
            self._get_ontology_table_ddl(),
            self._get_embeddings_table_ddl(),
            self._get_expression_table_ddl(),
        ]
        
        for ddl in tables:
            try:
                self.command(ddl)
                logger.debug(f"Table creation executed: {ddl[:100]}...")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create table: {e}")
                    raise

    def _get_datasets_table_ddl(self) -> str:
        """Get DDL for scRNA_datasets table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_prefix}datasets (
            dataset_id String,
            name String,
            cancer_study_identifier String,
            description String,
            n_cells UInt32,
            n_genes UInt32,
            imported_at DateTime,
            file_path String,
            metadata String
        ) ENGINE = SharedMergeTree() ORDER BY dataset_id
        """

    def _get_cells_table_ddl(self) -> str:
        """Get DDL for scRNA_cells table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_prefix}cells (
            dataset_id String,
            cell_id String,
            cell_barcode String,
            sample_unique_id Nullable(String),
            patient_unique_id Nullable(String),
            original_cell_type String,
            harmonized_cell_type_id Nullable(String),
            harmonization_confidence Nullable(Float32),
            mapping_strategy String,
            obs_data String
        ) ENGINE = SharedMergeTree() ORDER BY (dataset_id, cell_id)
        """

    def _get_genes_table_ddl(self) -> str:
        """Get DDL for scRNA_dataset_genes table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_prefix}dataset_genes (
            dataset_id String,
            gene_idx UInt32,
            hugo_gene_symbol String,
            entrez_gene_id Nullable(Int64)
        ) ENGINE = SharedMergeTree() ORDER BY (dataset_id, gene_idx)
        """

    def _get_ontology_table_ddl(self) -> str:
        """Get DDL for scRNA_cell_type_ontology table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_prefix}cell_type_ontology (
            cell_type_id String,
            cell_type_name String,
            ontology String,
            ontology_id String,
            parent_id Nullable(String),
            level UInt8,
            synonyms Array(String)
        ) ENGINE = SharedMergeTree() ORDER BY (ontology, cell_type_id)
        """

    def _get_embeddings_table_ddl(self) -> str:
        """Get DDL for scRNA_cell_embeddings table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_prefix}cell_embeddings (
            dataset_id String,
            cell_id String,
            embedding_type String,
            dimension_idx UInt16,
            value Float32
        ) ENGINE = SharedMergeTree() ORDER BY (dataset_id, cell_id, embedding_type, dimension_idx)
        """

    def _get_expression_table_ddl(self) -> str:
        """Get DDL for scRNA_expression_matrix table with SPARSE columns."""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{self.table_prefix}expression_matrix (
            dataset_id String,
            cell_id String,
            gene_idx UInt32,
            matrix_type String,
            count Float32 SPARSE
        ) ENGINE = SharedMergeTree() ORDER BY (dataset_id, cell_id, gene_idx, matrix_type)
        """

    def get_existing_studies(self) -> pd.DataFrame:
        """Get all existing cancer studies."""
        return self.query("SELECT * FROM cancer_study")

    def get_existing_samples(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """Get existing samples, optionally filtered by study."""
        if study_id:
            return self.query(
                "SELECT * FROM sample_derived WHERE cancer_study_identifier = %(study)s",
                {"study": study_id}
            )
        return self.query("SELECT * FROM sample_derived")

    def get_existing_patients(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """Get existing patients, optionally filtered by study."""
        if study_id:
            return self.query("""
                SELECT DISTINCT 
                    patient_unique_id,
                    patient_stable_id,
                    cancer_study_identifier,
                    patient_internal_id
                FROM sample_derived 
                WHERE cancer_study_identifier = %(study)s
            """, {"study": study_id})
        return self.query("""
            SELECT DISTINCT 
                patient_unique_id,
                patient_stable_id,
                cancer_study_identifier,
                patient_internal_id
            FROM sample_derived
        """)

    def get_existing_genes(self, gene_symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get existing genes, optionally filtered by symbols."""
        if gene_symbols:
            placeholders = ",".join([f"'{symbol}'" for symbol in gene_symbols])
            return self.query(f"SELECT * FROM gene WHERE hugo_gene_symbol IN ({placeholders})")
        return self.query("SELECT * FROM gene")

    def get_bulk_expression(self, study_id: str, gene_symbol: str) -> pd.DataFrame:
        """Get bulk RNA-seq expression data for a gene in a study."""
        return self.query("""
        SELECT 
            gad.sample_unique_id,
            sd.patient_unique_id,
            gad.alteration_value
        FROM genetic_alteration_derived gad
        JOIN sample_derived sd ON gad.sample_unique_id = sd.sample_unique_id
        WHERE gad.cancer_study_identifier = %(study)s
          AND gad.hugo_gene_symbol = %(gene)s
          AND gad.profile_type = 'rna_seq_mrna'
        """, {"study": study_id, "gene": gene_symbol})

    def close(self) -> None:
        """Close database connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("ClickHouse connection closed")