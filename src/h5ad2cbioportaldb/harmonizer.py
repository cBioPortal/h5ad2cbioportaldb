"""Cell type harmonization to standard ontologies."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .cbioportal.client import CBioPortalClient


logger = logging.getLogger(__name__)


class CellTypeHarmonizer:
    """Harmonize cell types to standard ontologies."""
    
    # Built-in cell type mappings for common single-cell datasets
    DEFAULT_MAPPINGS = {
        "CL": {
            # T cells
            "T cell": "CL:0000084",
            "T cells": "CL:0000084", 
            "CD4+ T cell": "CL:0000624",
            "CD4 T cell": "CL:0000624",
            "CD8+ T cell": "CL:0000625",
            "CD8 T cell": "CL:0000625",
            "regulatory T cell": "CL:0000815",
            "Treg": "CL:0000815",
            "memory T cell": "CL:0000813",
            "naive T cell": "CL:0000898",
            
            # B cells
            "B cell": "CL:0000236",
            "B cells": "CL:0000236",
            "plasma cell": "CL:0000786",
            "memory B cell": "CL:0000787",
            "naive B cell": "CL:0000788",
            
            # Myeloid cells
            "monocyte": "CL:0000576",
            "macrophage": "CL:0000235",
            "dendritic cell": "CL:0000451",
            "neutrophil": "CL:0000775",
            "eosinophil": "CL:0000771",
            "basophil": "CL:0000767",
            
            # NK cells
            "NK cell": "CL:0000623",
            "natural killer cell": "CL:0000623",
            
            # Other immune cells
            "mast cell": "CL:0000097",
            
            # Epithelial cells
            "epithelial cell": "CL:0000066",
            "keratinocyte": "CL:0000312",
            
            # Endothelial cells
            "endothelial cell": "CL:0000115",
            
            # Fibroblasts
            "fibroblast": "CL:0000057",
            
            # Stem cells
            "stem cell": "CL:0000034",
            "hematopoietic stem cell": "CL:0000037",
            
            # Cancer cells
            "cancer cell": "CL:0001063",
            "tumor cell": "CL:0001063",
            "malignant cell": "CL:0001063",
        }
    }
    
    # Confidence scoring weights
    CONFIDENCE_WEIGHTS = {
        "exact_match": 1.0,
        "case_insensitive_match": 0.95,
        "substring_match": 0.8,
        "fuzzy_match": 0.7,
        "manual_mapping": 1.0,
    }

    def __init__(self, client: CBioPortalClient, config: Dict[str, Any]) -> None:
        """Initialize harmonizer with client and configuration."""
        self.client = client
        self.config = config
        
        # Harmonization configuration
        self.default_ontology = config.get("default_ontology", "CL")
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.auto_harmonize = config.get("auto_harmonize", True)
        self.ontology_sources = config.get("ontology_sources", {
            "CL": "Cell Ontology",
            "UBERON": "Uber-anatomy ontology"
        })

    def harmonize_dataset(
        self,
        dataset_id: str,
        target_ontology: str = "CL",
        custom_mapping_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Harmonize cell types for an entire dataset."""
        
        logger.info(f"Starting cell type harmonization for dataset {dataset_id}")
        
        # Get current cell types from dataset
        current_cell_types = self._get_dataset_cell_types(dataset_id)
        if not current_cell_types:
            raise ValueError(f"No cell types found for dataset {dataset_id}")
        
        # Load custom mappings if provided
        custom_mappings = {}
        if custom_mapping_file:
            custom_mappings = self._load_custom_mappings(custom_mapping_file)
        
        # Harmonize cell types
        harmonization_results = self._harmonize_cell_types(
            current_cell_types, target_ontology, custom_mappings
        )
        
        # Update database with harmonized types
        if harmonization_results["mappings"]:
            updated_cells = self._update_harmonized_cell_types(
                dataset_id, harmonization_results["mappings"]
            )
        else:
            updated_cells = 0
        
        # Generate summary
        summary = {
            "dataset_id": dataset_id,
            "target_ontology": target_ontology,
            "original_cell_types": len(current_cell_types),
            "harmonized_cell_types": len(harmonization_results["mappings"]),
            "updated_cells": updated_cells,
            "confidence_threshold": self.confidence_threshold,
            "high_confidence_mappings": sum(
                1 for mapping in harmonization_results["mappings"].values()
                if mapping["confidence"] >= self.confidence_threshold
            ),
            "low_confidence_mappings": sum(
                1 for mapping in harmonization_results["mappings"].values()
                if mapping["confidence"] < self.confidence_threshold
            ),
            "unmapped_cell_types": harmonization_results["unmapped"],
        }
        
        logger.info(f"Harmonization completed: {summary}")
        return summary

    def _get_dataset_cell_types(self, dataset_id: str) -> List[str]:
        """Get unique cell types from dataset."""
        try:
            result = self.client.query(f"""
            SELECT DISTINCT original_cell_type 
            FROM {self.client.database}.{self.client.table_prefix}cells
            WHERE dataset_id = %(dataset)s 
              AND original_cell_type IS NOT NULL 
              AND original_cell_type != ''
            """, {"dataset": dataset_id})
            
            return result["original_cell_type"].tolist()
            
        except Exception as e:
            logger.error(f"Failed to get cell types for dataset {dataset_id}: {e}")
            return []

    def _load_custom_mappings(self, mapping_file: str) -> Dict[str, Dict[str, Any]]:
        """Load custom cell type mappings from file."""
        try:
            mapping_df = pd.read_csv(mapping_file)
            
            required_columns = ["original_cell_type", "harmonized_cell_type_id", "confidence"]
            missing_columns = [col for col in required_columns if col not in mapping_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in mapping file: {missing_columns}")
            
            custom_mappings = {}
            for _, row in mapping_df.iterrows():
                custom_mappings[row["original_cell_type"]] = {
                    "cell_type_id": row["harmonized_cell_type_id"],
                    "confidence": float(row["confidence"]),
                    "method": "manual_mapping",
                    "cell_type_name": row.get("cell_type_name", ""),
                }
            
            logger.info(f"Loaded {len(custom_mappings)} custom mappings from {mapping_file}")
            return custom_mappings
            
        except Exception as e:
            logger.error(f"Failed to load custom mappings from {mapping_file}: {e}")
            return {}

    def _harmonize_cell_types(
        self,
        cell_types: List[str],
        target_ontology: str,
        custom_mappings: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Harmonize list of cell types to target ontology."""
        
        mappings = {}
        unmapped = []
        
        # Get existing ontology mappings from database
        existing_mappings = self._get_existing_ontology_mappings(target_ontology)
        
        for cell_type in cell_types:
            # Check custom mappings first
            if cell_type in custom_mappings:
                mappings[cell_type] = custom_mappings[cell_type]
                continue
            
            # Try to find mapping
            mapping = self._find_cell_type_mapping(cell_type, target_ontology, existing_mappings)
            
            if mapping:
                mappings[cell_type] = mapping
            else:
                unmapped.append(cell_type)
        
        return {
            "mappings": mappings,
            "unmapped": unmapped,
            "target_ontology": target_ontology,
        }

    def _find_cell_type_mapping(
        self,
        cell_type: str,
        target_ontology: str,
        existing_mappings: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find the best mapping for a cell type."""
        
        # Strategy 1: Exact match in existing mappings
        if cell_type in existing_mappings:
            mapping = existing_mappings[cell_type].copy()
            mapping["confidence"] = self.CONFIDENCE_WEIGHTS["exact_match"]
            mapping["method"] = "exact_match"
            return mapping
        
        # Strategy 2: Case-insensitive match
        cell_type_lower = cell_type.lower()
        for existing_type, mapping_info in existing_mappings.items():
            if existing_type.lower() == cell_type_lower:
                mapping = mapping_info.copy()
                mapping["confidence"] = self.CONFIDENCE_WEIGHTS["case_insensitive_match"]
                mapping["method"] = "case_insensitive_match"
                return mapping
        
        # Strategy 3: Default mappings
        if target_ontology in self.DEFAULT_MAPPINGS:
            default_mappings = self.DEFAULT_MAPPINGS[target_ontology]
            
            # Exact match in defaults
            if cell_type in default_mappings:
                return {
                    "cell_type_id": default_mappings[cell_type],
                    "confidence": self.CONFIDENCE_WEIGHTS["exact_match"],
                    "method": "default_exact_match",
                    "cell_type_name": cell_type,
                }
            
            # Case-insensitive match in defaults
            for default_type, ontology_id in default_mappings.items():
                if default_type.lower() == cell_type_lower:
                    return {
                        "cell_type_id": ontology_id,
                        "confidence": self.CONFIDENCE_WEIGHTS["case_insensitive_match"],
                        "method": "default_case_insensitive_match",
                        "cell_type_name": default_type,
                    }
        
        # Strategy 4: Fuzzy matching (simplified version)
        best_match = self._fuzzy_match_cell_type(cell_type, existing_mappings)
        if best_match:
            return best_match
        
        return None

    def _fuzzy_match_cell_type(
        self,
        cell_type: str,
        existing_mappings: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Perform fuzzy matching against existing mappings."""
        
        cell_type_words = set(cell_type.lower().split())
        best_score = 0
        best_match = None
        
        for existing_type, mapping_info in existing_mappings.items():
            existing_words = set(existing_type.lower().split())
            
            # Simple Jaccard similarity
            intersection = len(cell_type_words & existing_words)
            union = len(cell_type_words | existing_words)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity > best_score and similarity >= 0.5:  # Minimum 50% similarity
                    best_score = similarity
                    best_match = {
                        "cell_type_id": mapping_info["cell_type_id"],
                        "confidence": similarity * self.CONFIDENCE_WEIGHTS["fuzzy_match"],
                        "method": "fuzzy_match",
                        "cell_type_name": mapping_info.get("cell_type_name", existing_type),
                        "similarity_score": similarity,
                    }
        
        return best_match

    def _get_existing_ontology_mappings(self, ontology: str) -> Dict[str, Dict[str, Any]]:
        """Get existing ontology mappings from database."""
        try:
            result = self.client.query(f"""
            SELECT 
                cell_type_name,
                cell_type_id,
                ontology_id,
                synonyms
            FROM {self.client.database}.{self.client.table_prefix}cell_type_ontology
            WHERE ontology = %(ontology)s
            """, {"ontology": ontology})
            
            mappings = {}
            for _, row in result.iterrows():
                mappings[row["cell_type_name"]] = {
                    "cell_type_id": row["cell_type_id"],
                    "ontology_id": row["ontology_id"],
                    "cell_type_name": row["cell_type_name"],
                }
                
                # Add synonyms as additional mappings
                if row["synonyms"]:
                    try:
                        synonyms = json.loads(row["synonyms"]) if isinstance(row["synonyms"], str) else row["synonyms"]
                        for synonym in synonyms:
                            mappings[synonym] = mappings[row["cell_type_name"]].copy()
                    except Exception:
                        pass  # Skip malformed synonyms
            
            logger.info(f"Loaded {len(mappings)} existing {ontology} mappings")
            return mappings
            
        except Exception as e:
            logger.warning(f"Could not load existing {ontology} mappings: {e}")
            return {}

    def _update_harmonized_cell_types(
        self,
        dataset_id: str,
        mappings: Dict[str, Dict[str, Any]],
    ) -> int:
        """Update database with harmonized cell types."""
        
        updated_count = 0
        
        try:
            for original_cell_type, mapping_info in mappings.items():
                # Only update if confidence meets threshold
                if mapping_info["confidence"] >= self.confidence_threshold:
                    self.client.command(f"""
                    ALTER TABLE {self.client.database}.{self.client.table_prefix}cells
                    UPDATE 
                        harmonized_cell_type_id = %(cell_type_id)s,
                        harmonization_confidence = %(confidence)s
                    WHERE dataset_id = %(dataset)s 
                      AND original_cell_type = %(original_type)s
                    """, {
                        "cell_type_id": mapping_info["cell_type_id"],
                        "confidence": mapping_info["confidence"],
                        "dataset": dataset_id,
                        "original_type": original_cell_type,
                    })
                    
                    # Count updated cells
                    count_result = self.client.query(f"""
                    SELECT COUNT(*) as count
                    FROM {self.client.database}.{self.client.table_prefix}cells
                    WHERE dataset_id = %(dataset)s 
                      AND original_cell_type = %(original_type)s
                    """, {
                        "dataset": dataset_id,
                        "original_type": original_cell_type,
                    })
                    
                    if len(count_result) > 0:
                        updated_count += count_result.iloc[0]["count"]
            
            logger.info(f"Updated {updated_count} cells with harmonized cell types")
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to update harmonized cell types: {e}")
            return updated_count

    def populate_ontology_database(
        self,
        ontology: str,
        ontology_data: List[Dict[str, Any]],
    ) -> int:
        """Populate cell type ontology database."""
        
        try:
            # Prepare ontology records
            records = []
            for entry in ontology_data:
                record = {
                    "cell_type_id": entry["id"],
                    "cell_type_name": entry["name"],
                    "ontology": ontology,
                    "ontology_id": entry.get("ontology_id", entry["id"]),
                    "parent_id": entry.get("parent_id"),
                    "level": entry.get("level", 0),
                    "synonyms": json.dumps(entry.get("synonyms", [])),
                }
                records.append(record)
            
            # Insert records
            df = pd.DataFrame(records)
            table_name = f"{self.client.table_prefix}cell_type_ontology"
            self.client.insert_dataframe(table_name, df)
            
            logger.info(f"Populated {len(records)} {ontology} ontology entries")
            return len(records)
            
        except Exception as e:
            logger.error(f"Failed to populate {ontology} ontology: {e}")
            return 0

    def generate_harmonization_report(self, dataset_id: str) -> Dict[str, Any]:
        """Generate detailed harmonization report for dataset."""
        
        try:
            # Get harmonization statistics
            stats = self.client.query(f"""
            SELECT 
                original_cell_type,
                harmonized_cell_type_id,
                AVG(harmonization_confidence) as avg_confidence,
                COUNT(*) as cell_count
            FROM {self.client.database}.{self.client.table_prefix}cells
            WHERE dataset_id = %(dataset)s
            GROUP BY original_cell_type, harmonized_cell_type_id
            ORDER BY cell_count DESC
            """, {"dataset": dataset_id})
            
            # Calculate summary metrics
            total_cells = stats["cell_count"].sum()
            harmonized_cells = stats[stats["harmonized_cell_type_id"].notna()]["cell_count"].sum()
            high_confidence_cells = stats[
                (stats["avg_confidence"] >= self.confidence_threshold) & 
                (stats["harmonized_cell_type_id"].notna())
            ]["cell_count"].sum()
            
            report = {
                "dataset_id": dataset_id,
                "total_cells": int(total_cells),
                "harmonized_cells": int(harmonized_cells),
                "harmonization_rate": harmonized_cells / total_cells if total_cells > 0 else 0,
                "high_confidence_cells": int(high_confidence_cells),
                "high_confidence_rate": high_confidence_cells / total_cells if total_cells > 0 else 0,
                "cell_type_breakdown": stats.to_dict("records"),
                "confidence_threshold": self.confidence_threshold,
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate harmonization report: {e}")
            return {"error": str(e)}

    def export_mapping_template(
        self,
        dataset_id: str,
        output_file: str,
        target_ontology: str = "CL",
    ) -> str:
        """Export cell type mapping template for manual curation."""
        
        try:
            # Get current cell types
            cell_types = self._get_dataset_cell_types(dataset_id)
            
            # Create template DataFrame
            template_data = []
            for cell_type in cell_types:
                template_data.append({
                    "original_cell_type": cell_type,
                    "harmonized_cell_type_id": "",  # To be filled by user
                    "cell_type_name": "",  # To be filled by user
                    "confidence": 1.0,  # Manual mappings get full confidence
                    "notes": "",
                })
            
            template_df = pd.DataFrame(template_data)
            template_df.to_csv(output_file, index=False)
            
            logger.info(f"Exported mapping template with {len(cell_types)} cell types to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export mapping template: {e}")
            raise