"""Cross-analysis queries for bulk vs single-cell comparisons."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .cbioportal.client import CBioPortalClient
from .cbioportal.schema import CBioPortalSchema


logger = logging.getLogger(__name__)


class CBioPortalQuerier:
    """Query engine for cross-analysis between bulk and single-cell data."""

    def __init__(self, client: CBioPortalClient) -> None:
        """Initialize querier with cBioPortal client."""
        self.client = client
        self.schema = CBioPortalSchema(client)

    def compare_bulk_vs_single_cell(
        self,
        gene_symbol: str,
        study_id: str,
        sc_dataset_id: str,
        cell_type_filter: Optional[List[str]] = None,
        correlation_method: str = "pearson",
    ) -> pd.DataFrame:
        """Compare bulk RNA-seq vs single-cell expression for a gene."""
        
        logger.info(f"Comparing bulk vs single-cell expression for {gene_symbol} in {study_id}")
        
        # Build cell type filter
        cell_type_condition = ""
        if cell_type_filter:
            cell_types = "', '".join(cell_type_filter)
            cell_type_condition = f"AND sc.original_cell_type IN ('{cell_types}')"
        
        # Execute comparison query
        comparison_df = self.client.query(f"""
        WITH bulk_expr AS (
            SELECT 
                gad.patient_unique_id,
                AVG(CAST(gad.alteration_value AS Float32)) as avg_bulk_expression,
                COUNT(*) as bulk_samples,
                STDDEV(CAST(gad.alteration_value AS Float32)) as bulk_stddev
            FROM genetic_alteration_derived gad
            WHERE gad.hugo_gene_symbol = %(gene)s
              AND gad.cancer_study_identifier = %(study)s
              AND gad.profile_type = 'rna_seq_v2_mrna'
              AND gad.alteration_value IS NOT NULL
              AND gad.alteration_value != 'NaN'
            GROUP BY gad.patient_unique_id
        ),
        sc_expr AS (
            SELECT 
                sc.patient_unique_id,
                sc.original_cell_type,
                AVG(expr.count) as avg_sc_expression,
                STDDEV(expr.count) as sc_stddev,
                COUNT(*) as num_cells,
                MIN(expr.count) as min_expression,
                MAX(expr.count) as max_expression,
                quantile(0.5)(expr.count) as median_expression
            FROM {self.client.database}.{self.client.table_prefix}expression_matrix expr
            JOIN {self.client.database}.{self.client.table_prefix}cells sc 
                ON expr.dataset_id = sc.dataset_id AND expr.cell_id = sc.cell_id
            JOIN {self.client.database}.{self.client.table_prefix}dataset_genes sg 
                ON expr.dataset_id = sg.dataset_id AND expr.gene_idx = sg.gene_idx
            WHERE sg.hugo_gene_symbol = %(gene)s
              AND sc.dataset_id = %(sc_dataset)s
              AND sc.patient_unique_id IS NOT NULL
              {cell_type_condition}
            GROUP BY sc.patient_unique_id, sc.original_cell_type
        )
        SELECT 
            COALESCE(b.patient_unique_id, s.patient_unique_id) as patient_id,
            b.avg_bulk_expression,
            b.bulk_stddev,
            b.bulk_samples,
            s.original_cell_type,
            s.avg_sc_expression,
            s.sc_stddev,
            s.median_expression,
            s.min_expression,
            s.max_expression,
            s.num_cells,
            CASE 
                WHEN b.avg_bulk_expression IS NOT NULL AND s.avg_sc_expression IS NOT NULL 
                THEN abs(b.avg_bulk_expression - s.avg_sc_expression)
                ELSE NULL 
            END as expression_difference,
            CASE 
                WHEN b.avg_bulk_expression > 0 AND s.avg_sc_expression > 0 
                THEN log2(s.avg_sc_expression / b.avg_bulk_expression)
                ELSE NULL 
            END as log2_fold_change
        FROM bulk_expr b
        FULL OUTER JOIN sc_expr s ON b.patient_unique_id = s.patient_unique_id
        ORDER BY patient_id, original_cell_type
        """, {
            "gene": gene_symbol,
            "study": study_id,
            "sc_dataset": sc_dataset_id
        })
        
        # Calculate correlation if both bulk and single-cell data available
        if len(comparison_df) > 0:
            comparison_df = self._add_correlation_metrics(comparison_df, correlation_method)
        
        logger.info(f"Comparison completed: {len(comparison_df)} records")
        return comparison_df

    def _add_correlation_metrics(
        self, 
        comparison_df: pd.DataFrame, 
        correlation_method: str
    ) -> pd.DataFrame:
        """Add correlation metrics to comparison results."""
        
        # Filter to rows with both bulk and single-cell data
        both_data = comparison_df[
            (comparison_df["avg_bulk_expression"].notna()) & 
            (comparison_df["avg_sc_expression"].notna())
        ].copy()
        
        if len(both_data) < 2:
            comparison_df["correlation"] = None
            comparison_df["p_value"] = None
            return comparison_df
        
        try:
            from scipy.stats import pearsonr, spearmanr
            
            if correlation_method == "pearson":
                corr, p_val = pearsonr(both_data["avg_bulk_expression"], both_data["avg_sc_expression"])
            elif correlation_method == "spearman":
                corr, p_val = spearmanr(both_data["avg_bulk_expression"], both_data["avg_sc_expression"])
            else:
                corr, p_val = None, None
            
            comparison_df["correlation"] = corr
            comparison_df["p_value"] = p_val
            
        except Exception as e:
            logger.warning(f"Could not calculate correlation: {e}")
            comparison_df["correlation"] = None
            comparison_df["p_value"] = None
        
        return comparison_df

    def get_cell_type_summary(self, dataset_id: str) -> pd.DataFrame:
        """Get comprehensive cell type summary for dataset."""
        
        return self.client.query(f"""
        WITH cell_counts AS (
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
        ),
        expression_stats AS (
            SELECT 
                sc.original_cell_type,
                COUNT(DISTINCT expr.gene_idx) as expressed_genes,
                AVG(expr.count) as avg_expression,
                quantile(0.5)(expr.count) as median_expression,
                STDDEV(expr.count) as expression_stddev
            FROM {self.client.database}.{self.client.table_prefix}expression_matrix expr
            JOIN {self.client.database}.{self.client.table_prefix}cells sc 
                ON expr.dataset_id = sc.dataset_id AND expr.cell_id = sc.cell_id
            WHERE sc.dataset_id = %(dataset)s
              AND expr.count > 0
            GROUP BY sc.original_cell_type
        )
        SELECT 
            cc.*,
            es.expressed_genes,
            es.avg_expression,
            es.median_expression,
            es.expression_stddev,
            cc.cell_count * 100.0 / SUM(cc.cell_count) OVER() as percentage_of_total
        FROM cell_counts cc
        LEFT JOIN expression_stats es ON cc.original_cell_type = es.original_cell_type
        ORDER BY cc.cell_count DESC
        """, {"dataset": dataset_id})

    def get_dataset_stats(self, dataset_id: str) -> pd.DataFrame:
        """Get comprehensive dataset statistics."""
        
        # Get basic dataset info
        dataset_info = self.schema.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Get detailed statistics
        stats = self.client.query(f"""
        SELECT 
            'cells' as metric_type,
            'total_cells' as metric_name,
            CAST(COUNT(*) AS String) as metric_value
        FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
        
        UNION ALL
        
        SELECT 
            'cells' as metric_type,
            'unique_cell_types' as metric_name,
            CAST(COUNT(DISTINCT original_cell_type) AS String) as metric_value
        FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
        
        UNION ALL
        
        SELECT 
            'mapping' as metric_type,
            'cells_with_samples' as metric_name,
            CAST(COUNT(CASE WHEN sample_unique_id IS NOT NULL THEN 1 END) AS String) as metric_value
        FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
        
        UNION ALL
        
        SELECT 
            'mapping' as metric_type,
            'cells_with_patients' as metric_name,
            CAST(COUNT(CASE WHEN patient_unique_id IS NOT NULL THEN 1 END) AS String) as metric_value
        FROM {self.client.database}.{self.client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
        
        UNION ALL
        
        SELECT 
            'expression' as metric_type,
            'total_genes' as metric_name,
            CAST(COUNT(DISTINCT gene_idx) AS String) as metric_value
        FROM {self.client.database}.{self.client.table_prefix}dataset_genes
        WHERE dataset_id = %(dataset)s
        
        UNION ALL
        
        SELECT 
            'expression' as metric_type,
            'non_zero_expressions' as metric_name,
            CAST(COUNT(*) AS String) as metric_value
        FROM {self.client.database}.{self.client.table_prefix}expression_matrix
        WHERE dataset_id = %(dataset)s
          AND count > 0
        """, {"dataset": dataset_id})
        
        return stats

    def get_gene_expression_profile(
        self,
        gene_symbol: str,
        dataset_id: str,
        groupby: str = "cell_type",
        min_cells: int = 10,
    ) -> pd.DataFrame:
        """Get gene expression profile across cell types or samples."""
        
        if groupby == "cell_type":
            group_column = "sc.original_cell_type"
        elif groupby == "sample":
            group_column = "sc.sample_unique_id"
        elif groupby == "patient":
            group_column = "sc.patient_unique_id"
        else:
            raise ValueError("groupby must be 'cell_type', 'sample', or 'patient'")
        
        return self.client.query(f"""
        SELECT 
            {group_column} as group_id,
            COUNT(*) as total_cells,
            COUNT(CASE WHEN expr.count > 0 THEN 1 END) as expressing_cells,
            COUNT(CASE WHEN expr.count > 0 THEN 1 END) * 100.0 / COUNT(*) as expression_percentage,
            AVG(expr.count) as mean_expression,
            STDDEV(expr.count) as stddev_expression,
            quantile(0.25)(expr.count) as q25_expression,
            quantile(0.5)(expr.count) as median_expression,
            quantile(0.75)(expr.count) as q75_expression,
            MAX(expr.count) as max_expression
        FROM {self.client.database}.{self.client.table_prefix}cells sc
        LEFT JOIN {self.client.database}.{self.client.table_prefix}expression_matrix expr 
            ON sc.dataset_id = expr.dataset_id AND sc.cell_id = expr.cell_id
        LEFT JOIN {self.client.database}.{self.client.table_prefix}dataset_genes sg 
            ON expr.dataset_id = sg.dataset_id AND expr.gene_idx = sg.gene_idx
        WHERE sc.dataset_id = %(dataset)s
          AND sg.hugo_gene_symbol = %(gene)s
          AND {group_column} IS NOT NULL
        GROUP BY {group_column}
        HAVING COUNT(*) >= %(min_cells)s
        ORDER BY mean_expression DESC
        """, {
            "dataset": dataset_id,
            "gene": gene_symbol,
            "min_cells": min_cells
        })

    def find_marker_genes(
        self,
        dataset_id: str,
        cell_type: str,
        top_n: int = 50,
        min_fold_change: float = 1.5,
        min_expression: float = 1.0,
    ) -> pd.DataFrame:
        """Find marker genes for a specific cell type."""
        
        return self.client.query(f"""
        WITH target_expression AS (
            SELECT 
                sg.hugo_gene_symbol,
                sg.entrez_gene_id,
                AVG(expr.count) as target_mean_expr,
                COUNT(CASE WHEN expr.count > %(min_expr)s THEN 1 END) as target_expressing_cells,
                COUNT(*) as target_total_cells
            FROM {self.client.database}.{self.client.table_prefix}cells sc
            JOIN {self.client.database}.{self.client.table_prefix}expression_matrix expr 
                ON sc.dataset_id = expr.dataset_id AND sc.cell_id = expr.cell_id
            JOIN {self.client.database}.{self.client.table_prefix}dataset_genes sg 
                ON expr.dataset_id = sg.dataset_id AND expr.gene_idx = sg.gene_idx
            WHERE sc.dataset_id = %(dataset)s
              AND sc.original_cell_type = %(cell_type)s
            GROUP BY sg.hugo_gene_symbol, sg.entrez_gene_id
        ),
        other_expression AS (
            SELECT 
                sg.hugo_gene_symbol,
                AVG(expr.count) as other_mean_expr,
                COUNT(CASE WHEN expr.count > %(min_expr)s THEN 1 END) as other_expressing_cells,
                COUNT(*) as other_total_cells
            FROM {self.client.database}.{self.client.table_prefix}cells sc
            JOIN {self.client.database}.{self.client.table_prefix}expression_matrix expr 
                ON sc.dataset_id = expr.dataset_id AND sc.cell_id = expr.cell_id
            JOIN {self.client.database}.{self.client.table_prefix}dataset_genes sg 
                ON expr.dataset_id = sg.dataset_id AND expr.gene_idx = sg.gene_idx
            WHERE sc.dataset_id = %(dataset)s
              AND sc.original_cell_type != %(cell_type)s
            GROUP BY sg.hugo_gene_symbol
        )
        SELECT 
            te.hugo_gene_symbol,
            te.entrez_gene_id,
            te.target_mean_expr,
            oe.other_mean_expr,
            te.target_mean_expr / GREATEST(oe.other_mean_expr, 0.1) as fold_change,
            te.target_expressing_cells * 100.0 / te.target_total_cells as target_expression_pct,
            oe.other_expressing_cells * 100.0 / oe.other_total_cells as other_expression_pct,
            te.target_expressing_cells * 100.0 / te.target_total_cells - 
                oe.other_expressing_cells * 100.0 / oe.other_total_cells as expression_pct_diff
        FROM target_expression te
        JOIN other_expression oe ON te.hugo_gene_symbol = oe.hugo_gene_symbol
        WHERE te.target_mean_expr >= %(min_expr)s
          AND te.target_mean_expr / GREATEST(oe.other_mean_expr, 0.1) >= %(min_fold)s
        ORDER BY fold_change DESC
        LIMIT %(top_n)s
        """, {
            "dataset": dataset_id,
            "cell_type": cell_type,
            "min_expr": min_expression,
            "min_fold": min_fold_change,
            "top_n": top_n
        })

    def compare_datasets(
        self,
        dataset1_id: str,
        dataset2_id: str,
        comparison_type: str = "cell_types",
    ) -> pd.DataFrame:
        """Compare two datasets by cell types or other metrics."""
        
        if comparison_type == "cell_types":
            return self.client.query(f"""
            WITH dataset1_types AS (
                SELECT 
                    original_cell_type,
                    COUNT(*) as dataset1_cells
                FROM {self.client.database}.{self.client.table_prefix}cells
                WHERE dataset_id = %(dataset1)s
                GROUP BY original_cell_type
            ),
            dataset2_types AS (
                SELECT 
                    original_cell_type,
                    COUNT(*) as dataset2_cells
                FROM {self.client.database}.{self.client.table_prefix}cells
                WHERE dataset_id = %(dataset2)s
                GROUP BY original_cell_type
            )
            SELECT 
                COALESCE(d1.original_cell_type, d2.original_cell_type) as cell_type,
                COALESCE(d1.dataset1_cells, 0) as dataset1_cells,
                COALESCE(d2.dataset2_cells, 0) as dataset2_cells,
                CASE 
                    WHEN d1.dataset1_cells IS NULL THEN 'Only in dataset2'
                    WHEN d2.dataset2_cells IS NULL THEN 'Only in dataset1'
                    ELSE 'In both datasets'
                END as presence
            FROM dataset1_types d1
            FULL OUTER JOIN dataset2_types d2 ON d1.original_cell_type = d2.original_cell_type
            ORDER BY GREATEST(COALESCE(d1.dataset1_cells, 0), COALESCE(d2.dataset2_cells, 0)) DESC
            """, {"dataset1": dataset1_id, "dataset2": dataset2_id})
        
        else:
            raise ValueError("Only 'cell_types' comparison currently supported")

    def list_datasets(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """List all datasets, optionally filtered by study."""
        return self.schema.list_datasets(study_id)

    def list_studies(self) -> pd.DataFrame:
        """List all available studies."""
        return self.client.get_existing_studies()

    def list_cell_types(self, study_id: Optional[str] = None) -> pd.DataFrame:
        """List all cell types, optionally filtered by study."""
        
        study_condition = ""
        params = {}
        
        if study_id:
            study_condition = """
            JOIN {datasets_table} d ON sc.dataset_id = d.dataset_id
            WHERE d.cancer_study_identifier = %(study)s
            """.format(datasets_table=f"{self.client.database}.{self.client.table_prefix}datasets")
            params["study"] = study_id
        
        return self.client.query(f"""
        SELECT 
            original_cell_type,
            harmonized_cell_type_id,
            COUNT(DISTINCT dataset_id) as datasets_count,
            COUNT(*) as total_cells,
            AVG(harmonization_confidence) as avg_confidence
        FROM {self.client.database}.{self.client.table_prefix}cells sc
        {study_condition}
        GROUP BY original_cell_type, harmonized_cell_type_id
        ORDER BY total_cells DESC
        """, params)

    def get_expression_correlation_matrix(
        self,
        dataset_id: str,
        gene_list: List[str],
        cell_type: Optional[str] = None,
        min_cells: int = 50,
    ) -> pd.DataFrame:
        """Calculate expression correlation matrix for a set of genes."""
        
        cell_type_condition = ""
        if cell_type:
            cell_type_condition = f"AND sc.original_cell_type = '{cell_type}'"
        
        # Get expression data for specified genes
        expr_data = self.client.query(f"""
        SELECT 
            sc.cell_id,
            sg.hugo_gene_symbol,
            expr.count as expression
        FROM {self.client.database}.{self.client.table_prefix}cells sc
        JOIN {self.client.database}.{self.client.table_prefix}expression_matrix expr 
            ON sc.dataset_id = expr.dataset_id AND sc.cell_id = expr.cell_id
        JOIN {self.client.database}.{self.client.table_prefix}dataset_genes sg 
            ON expr.dataset_id = sg.dataset_id AND expr.gene_idx = sg.gene_idx
        WHERE sc.dataset_id = %(dataset)s
          AND sg.hugo_gene_symbol IN ({gene_placeholders})
          {cell_type_condition}
        """.format(
            gene_placeholders=",".join([f"'{gene}'" for gene in gene_list]),
            cell_type_condition=cell_type_condition
        ), {"dataset": dataset_id})
        
        if len(expr_data) < min_cells:
            raise ValueError(f"Insufficient cells ({len(expr_data)}) for correlation analysis")
        
        # Pivot to create gene x cell matrix
        pivot_df = expr_data.pivot(index="cell_id", columns="hugo_gene_symbol", values="expression")
        pivot_df = pivot_df.fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = pivot_df.corr()
        
        return correlation_matrix