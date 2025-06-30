"""Command-line interface for h5ad2cbioportaldb."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from .cbioportal.client import CBioPortalClient
from .harmonizer import CellTypeHarmonizer
from .importer import H5adImporter
from .mapper import CBioPortalMapper
from .querier import CBioPortalQuerier
from .validator import CBioPortalValidator


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        click.echo(f"Config file not found: {config_path}")
        click.echo("Using default configuration. Create config.yaml from config.yaml.example for customization.")
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_config(dataset_config_path: str) -> dict:
    """Load dataset-specific configuration from YAML file."""
    config_path = Path(dataset_config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """h5ad2cbioportaldb: Import h5ad single-cell files into cBioPortal ClickHouse database (in-memory import only)."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    
    log_level = "DEBUG" if verbose else ctx.obj["config"].get("logging", {}).get("level", "INFO")
    setup_logging(log_level)


@cli.group(name="import")
@click.pass_context
def import_group(ctx: click.Context):
    """Import workflow: prepare TSVs and load into ClickHouse."""
    pass


@import_group.command("prepare")
@click.option("--file", "-f", type=click.Path(exists=True), required=True, help="Path to h5ad file")
@click.option("--dataset-id", required=True, help="Unique dataset identifier")
@click.option("--study-id", required=True, help="cBioPortal study identifier")
@click.option("--output-dir", required=True, type=click.Path(), help="Directory to write TSV files")
@click.option("--cell-type-column", help="Column name for cell types in obs")
@click.option("--sample-obs-column", help="Column name for sample IDs in obs")
@click.option("--patient-obs-column", help="Column name for patient IDs in obs")
@click.option("--sample-mapping", type=click.Path(exists=True), help="CSV file mapping h5ad samples to cBioPortal")
@click.option("--patient-mapping", type=click.Path(exists=True), help="CSV file mapping h5ad patients to cBioPortal")
@click.option("--description", help="Dataset description")
@click.option("--matrix-type", default="raw", help="Expression matrix type to import")
@click.option("--dry-run", is_flag=True, help="Validate without importing")
@click.option("--overwrite", is_flag=True, help="Overwrite existing dataset")
@click.option("--batch-size", default=10000, show_default=True, type=int, help="Batch size for expression matrix export")
@click.pass_context
def prepare(
    ctx: click.Context,
    file: str,
    dataset_id: str,
    study_id: str,
    output_dir: str,
    cell_type_column: Optional[str],
    sample_obs_column: Optional[str],
    patient_obs_column: Optional[str],
    sample_mapping: Optional[str],
    patient_mapping: Optional[str],
    description: Optional[str],
    matrix_type: str,
    dry_run: bool,
    overwrite: bool,
    batch_size: int,
):
    """Prepare (generate) TSV files from h5ad file for later import into ClickHouse."""
    config = ctx.obj["config"]
    client = CBioPortalClient(config.get("cbioportal", {}))
    importer = H5adImporter(client, config.get("import", {}))
    importer.prepare_tsvs(
        file=file,
        dataset_id=dataset_id,
        study_id=study_id,
        output_dir=output_dir,
        cell_type_column=cell_type_column,
        sample_obs_column=sample_obs_column,
        patient_obs_column=patient_obs_column,
        sample_mapping_file=sample_mapping,
        patient_mapping_file=patient_mapping,
        description=description,
        matrix_type=matrix_type,
        dry_run=dry_run,
        overwrite=overwrite,
        batch_size=batch_size,
    )


@import_group.command("clickhouse")
@click.option("--parquet-dir", required=True, type=click.Path(exists=True), help="Directory containing Parquet files to import")
@click.option("--dataset-id", required=True, help="Unique dataset identifier")
@click.option("--study-id", required=True, help="cBioPortal study identifier")
@click.pass_context
def clickhouse(
    ctx: click.Context,
    parquet_dir: str,
    dataset_id: str,
    study_id: str,
):
    """Load prepared Parquet files into ClickHouse database."""
    config = ctx.obj["config"]
    client = CBioPortalClient(config.get("cbioportal", {}))
    importer = H5adImporter(client, config.get("import", {}))
    importer.load_parquets_to_clickhouse(
        parquet_dir=parquet_dir,
        dataset_id=dataset_id,
        study_id=study_id,
    )


@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), help="Path to h5ad file")
@click.option("--dataset-id", help="Unique dataset identifier")
@click.option("--study-id", help="cBioPortal study identifier")
@click.option("--cell-type-column", help="Column name for cell types in obs")
@click.option("--sample-obs-column", help="Column name for sample IDs in obs")
@click.option("--patient-obs-column", help="Column name for patient IDs in obs")
@click.option("--sample-mapping", type=click.Path(exists=True), help="CSV file mapping h5ad samples to cBioPortal")
@click.option("--patient-mapping", type=click.Path(exists=True), help="CSV file mapping h5ad patients to cBioPortal")
@click.option("--description", help="Dataset description")
@click.option("--matrix-type", default="raw", help="Expression matrix type to import")
@click.option("--dry-run", is_flag=True, help="Validate without importing")
@click.option("--overwrite", is_flag=True, help="Overwrite existing dataset")
@click.option("--dataset-config", type=click.Path(exists=True), help="Dataset configuration YAML file")
@click.pass_context
def import_dataset(
    ctx: click.Context,
    file: Optional[str],
    dataset_id: Optional[str],
    study_id: Optional[str],
    cell_type_column: Optional[str],
    sample_obs_column: Optional[str],
    patient_obs_column: Optional[str],
    sample_mapping: Optional[str],
    patient_mapping: Optional[str],
    description: Optional[str],
    matrix_type: str,
    dry_run: bool,
    overwrite: bool,
    dataset_config: Optional[str],
) -> None:
    """Import h5ad file into cBioPortal database."""
    config = ctx.obj["config"]
    
    # Load dataset config if provided
    if dataset_config:
        dataset_cfg = load_dataset_config(dataset_config)
        # Override parameters with dataset config values if not provided
        if not file:
            file = str(Path(dataset_config).parent / dataset_cfg["dataset"]["h5ad_file"])
        if not study_id:
            study_id = dataset_cfg["dataset"]["study_id"]
        if not description:
            description = dataset_cfg["dataset"].get("description")
        if not cell_type_column:
            cell_type_column = dataset_cfg["annotation"].get("recommended_cell_type_column")
        if not sample_obs_column:
            sample_obs_column = dataset_cfg["mapping"].get("sample_obs_column")
        if not patient_obs_column:
            patient_obs_column = dataset_cfg["mapping"].get("patient_obs_column")
        if not sample_mapping:
            sample_mapping_file = dataset_cfg["mapping"].get("sample_mapping_file")
            if sample_mapping_file:
                sample_mapping = str(Path(dataset_config).parent / sample_mapping_file)
        if not patient_mapping:
            patient_mapping_file = dataset_cfg["mapping"].get("patient_mapping_file")
            if patient_mapping_file:
                patient_mapping = str(Path(dataset_config).parent / patient_mapping_file)
        if matrix_type == "raw":  # Only override if using default
            matrix_type = dataset_cfg["dataset"].get("matrix_type", "raw")
    
    # Validate required parameters
    if not file or not dataset_id or not study_id:
        click.echo("Error: --file, --dataset-id, and --study-id are required (or use --dataset-config)", err=True)
        sys.exit(1)
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        importer = H5adImporter(client, config.get("import", {}))
        
        result = importer.import_dataset(
            file=file,
            dataset_id=dataset_id,
            study_id=study_id,
            cell_type_column=cell_type_column,
            sample_obs_column=sample_obs_column,
            patient_obs_column=patient_obs_column,
            sample_mapping_file=sample_mapping,
            patient_mapping_file=patient_mapping,
            description=description,
            matrix_type=matrix_type,
            dry_run=dry_run,
            overwrite=overwrite,
        )
        
        if dry_run:
            click.echo("✓ Validation completed successfully")
            if 'n_genes_total' in result:
                click.echo(f"Would import {result['n_cells']:,} cells, {result['n_genes']:,} genes ({result['n_genes_unmapped']:,} genes skipped)")
            else:
                click.echo(f"Would import {result['n_cells']:,} cells, {result['n_genes']:,} genes")
        else:
            click.echo(f"✓ Successfully imported dataset {dataset_id}")
            if 'n_genes_total' in result:
                click.echo(f"Imported {result['n_cells']:,} cells, {result['n_genes']:,} genes ({result['n_genes_unmapped']:,} genes skipped)")
            else:
                click.echo(f"Imported {result['n_cells']:,} cells, {result['n_genes']:,} genes")
            
    except Exception as e:
        click.echo(f"✗ Import failed: {e}", err=True)
        sys.exit(1)


@cli.command("generate-mapping-template")
@click.option("--file", "-f", required=True, type=click.Path(exists=True), help="Path to h5ad file")
@click.option("--sample-obs-column", help="Column name for sample IDs in obs")
@click.option("--patient-obs-column", help="Column name for patient IDs in obs")
@click.option("--study-id", required=True, help="cBioPortal study identifier")
@click.option("--output-dir", "-o", default="templates", help="Output directory for templates")
@click.pass_context
def generate_mapping_template(
    ctx: click.Context,
    file: str,
    sample_obs_column: Optional[str],
    patient_obs_column: Optional[str],
    study_id: str,
    output_dir: str,
) -> None:
    """Generate mapping template files for user completion."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        mapper = CBioPortalMapper(client, config.get("mapping", {}))
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        files_created = mapper.generate_mapping_templates(
            file, sample_obs_column, patient_obs_column, study_id, str(output_path)
        )
        
        click.echo(f"✓ Generated mapping templates in {output_path}")
        for created_file in files_created:
            click.echo(f"  - {created_file}")
            
    except Exception as e:
        click.echo(f"✗ Template generation failed: {e}", err=True)
        sys.exit(1)


@cli.command("debug-mappings")
@click.option("--file", "-f", type=click.Path(exists=True), help="Path to h5ad file")
@click.option("--study-id", help="cBioPortal study identifier")
@click.option("--sample-obs-column", help="Column name for sample IDs in obs")
@click.option("--patient-obs-column", help="Column name for patient IDs in obs")
@click.option("--sample-mapping", type=click.Path(exists=True), help="Sample mapping CSV file")
@click.option("--patient-mapping", type=click.Path(exists=True), help="Patient mapping CSV file")
@click.option("--dataset-config", type=click.Path(exists=True), help="Dataset configuration YAML file")
@click.pass_context
def debug_mappings(
    ctx: click.Context,
    file: Optional[str],
    study_id: Optional[str],
    sample_obs_column: Optional[str],
    patient_obs_column: Optional[str],
    sample_mapping: Optional[str],
    patient_mapping: Optional[str],
    dataset_config: Optional[str],
) -> None:
    """Debug mapping issues by showing detailed information about h5ad data vs cBioPortal data."""
    config = ctx.obj["config"]
    
    # Load dataset config if provided
    if dataset_config:
        dataset_cfg = load_dataset_config(dataset_config)
        # Override parameters with dataset config values if not provided
        if not file:
            file = str(Path(dataset_config).parent / dataset_cfg["dataset"]["h5ad_file"])
        if not study_id:
            study_id = dataset_cfg["dataset"]["study_id"]
        if not sample_obs_column:
            sample_obs_column = dataset_cfg["mapping"].get("sample_obs_column")
        if not patient_obs_column:
            patient_obs_column = dataset_cfg["mapping"].get("patient_obs_column")
        if not sample_mapping:
            sample_mapping_file = dataset_cfg["mapping"].get("sample_mapping_file")
            if sample_mapping_file:
                sample_mapping = str(Path(dataset_config).parent / sample_mapping_file)
        if not patient_mapping:
            patient_mapping_file = dataset_cfg["mapping"].get("patient_mapping_file")
            if patient_mapping_file:
                patient_mapping = str(Path(dataset_config).parent / patient_mapping_file)
    
    # Validate required parameters
    if not file or not study_id:
        click.echo("Error: --file and --study-id are required (or use --dataset-config)", err=True)
        sys.exit(1)
    
    try:
        import anndata
        client = CBioPortalClient(config.get("cbioportal", {}))
        mapper = CBioPortalMapper(client, config.get("mapping", {}))
        
        click.echo("=== DEBUGGING MAPPING ISSUES ===")
        
        # Load h5ad file (fast metadata-only read)
        click.echo(f"📁 Loading h5ad file: {file}")
        try:
            adata = anndata.read_h5ad(file, backed='r')  # Read-only mode
            obs_df = adata.obs.copy()  # Copy just the metadata
            adata.file.close()  # Close file handle immediately
            click.echo(f"   Cells: {len(obs_df)}, Genes: {adata.n_vars}")
        except Exception as e:
            click.echo(f"   Fast read failed ({e}), using standard read")
            adata = anndata.read_h5ad(file)
            obs_df = adata.obs
            click.echo(f"   Cells: {adata.n_obs}, Genes: {adata.n_vars}")
        
        # Check obs columns
        click.echo(f"📋 Available obs columns: {list(obs_df.columns)}")
        
        # Check sample/patient columns
        if sample_obs_column:
            if sample_obs_column in obs_df.columns:
                sample_values = obs_df[sample_obs_column].value_counts().head(10)
                click.echo(f"🧬 Sample column '{sample_obs_column}' - Top 10 values:")
                for val, count in sample_values.items():
                    click.echo(f"   {val}: {count} cells")
            else:
                click.echo(f"❌ Sample column '{sample_obs_column}' not found!")
        
        if patient_obs_column:
            if patient_obs_column in obs_df.columns:
                patient_values = obs_df[patient_obs_column].value_counts().head(10)
                click.echo(f"👤 Patient column '{patient_obs_column}' - Top 10 values:")
                for val, count in patient_values.items():
                    click.echo(f"   {val}: {count} cells")
            else:
                click.echo(f"❌ Patient column '{patient_obs_column}' not found!")
        
        # Check mapping files
        if patient_mapping:
            patient_map = mapper._load_patient_mapping(patient_mapping)
            click.echo(f"🗂️  Patient mapping file - {len(patient_map)} entries:")
            for i, (h5ad_id, cbio_id) in enumerate(list(patient_map.items())[:5]):
                click.echo(f"   {h5ad_id} → {cbio_id}")
            if len(patient_map) > 5:
                click.echo(f"   ... and {len(patient_map) - 5} more")
        
        # Check cBioPortal data
        existing_patients = mapper._get_existing_patients(study_id)
        click.echo(f"🏥 cBioPortal patients in study {study_id} - {len(existing_patients)} found:")
        for patient_id in list(existing_patients.keys())[:10]:
            click.echo(f"   {patient_id}")
        if len(existing_patients) > 10:
            click.echo(f"   ... and {len(existing_patients) - 10} more")
        
        # Check for matches
        if patient_mapping and patient_obs_column and patient_obs_column in obs_df.columns:
            patient_map = mapper._load_patient_mapping(patient_mapping)
            h5ad_patient_ids = set(obs_df[patient_obs_column].astype(str).unique())
            mapping_keys = set(patient_map.keys())
            mapping_values = set(patient_map.values())
            cbio_patients = set(existing_patients.keys())
            
            click.echo(f"🔍 MATCHING ANALYSIS:")
            click.echo(f"   H5AD patient IDs found: {len(h5ad_patient_ids)}")
            click.echo(f"   Mapping file keys: {len(mapping_keys)}")
            click.echo(f"   H5AD IDs in mapping keys: {len(h5ad_patient_ids & mapping_keys)}")
            click.echo(f"   Mapping values in cBioPortal: {len(mapping_values & cbio_patients)}")
            
            # Show mismatches
            missing_from_mapping = h5ad_patient_ids - mapping_keys
            if missing_from_mapping:
                click.echo(f"❌ H5AD patient IDs not in mapping file ({len(missing_from_mapping)}):")
                for pid in list(missing_from_mapping)[:5]:
                    click.echo(f"   {pid}")
                if len(missing_from_mapping) > 5:
                    click.echo(f"   ... and {len(missing_from_mapping) - 5} more")
            
            missing_from_cbio = mapping_values - cbio_patients
            if missing_from_cbio:
                click.echo(f"❌ Mapped patient IDs not in cBioPortal ({len(missing_from_cbio)}):")
                for pid in list(missing_from_cbio)[:5]:
                    click.echo(f"   {pid}")
                if len(missing_from_cbio) > 5:
                    click.echo(f"   ... and {len(missing_from_cbio) - 5} more")
            
            if not missing_from_mapping and not missing_from_cbio:
                click.echo("✅ All mappings should work!")
        
    except Exception as e:
        click.echo(f"✗ Debug failed: {e}", err=True)
        sys.exit(1)


@cli.command("validate-mappings")
@click.option("--study-id", required=True, help="cBioPortal study identifier")
@click.option("--sample-mapping", type=click.Path(exists=True), help="Sample mapping CSV file")
@click.option("--patient-mapping", type=click.Path(exists=True), help="Patient mapping CSV file")
@click.pass_context
def validate_mappings(
    ctx: click.Context,
    study_id: str,
    sample_mapping: Optional[str],
    patient_mapping: Optional[str],
) -> None:
    """Validate mapping files against cBioPortal study."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        validator = CBioPortalValidator(client, config.get("validation", {}))
        
        result = validator.validate_study_integration(
            study_id, sample_mapping, patient_mapping
        )
        
        if result["valid"]:
            click.echo("✓ All mappings are valid")
        else:
            click.echo("⚠ Validation warnings found:")
            for warning in result["warnings"]:
                click.echo(f"  - {warning}")
                
    except Exception as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command("query")
@click.argument("query_type", type=click.Choice(["compare-expression", "cell-type-summary", "dataset-stats"]))
@click.option("--gene", help="Gene symbol for expression queries")
@click.option("--study", help="Study identifier")
@click.option("--sc-dataset", help="Single-cell dataset identifier")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def query(
    ctx: click.Context,
    query_type: str,
    gene: Optional[str],
    study: Optional[str],
    sc_dataset: Optional[str],
    output: Optional[str],
) -> None:
    """Run cross-analysis queries."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        querier = CBioPortalQuerier(client)
        
        if query_type == "compare-expression" and gene and study and sc_dataset:
            result = querier.compare_bulk_vs_single_cell(gene, study, sc_dataset)
        elif query_type == "cell-type-summary" and sc_dataset:
            result = querier.get_cell_type_summary(sc_dataset)
        elif query_type == "dataset-stats" and sc_dataset:
            result = querier.get_dataset_stats(sc_dataset)
        else:
            raise ValueError(f"Invalid query parameters for {query_type}")
        
        if output:
            result.to_csv(output, index=False)
            click.echo(f"✓ Results saved to {output}")
        else:
            click.echo(result.to_string())
            
    except Exception as e:
        click.echo(f"✗ Query failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dataset", required=True, help="Dataset identifier")
@click.option("--ontology", default="CL", help="Target ontology (CL, UBERON)")
@click.option("--mapping-file", type=click.Path(exists=True), help="Custom cell type mapping file")
@click.pass_context
def harmonize(
    ctx: click.Context,
    dataset: str,
    ontology: str,
    mapping_file: Optional[str],
) -> None:
    """Harmonize cell types to standard ontology."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        harmonizer = CellTypeHarmonizer(client, config.get("harmonization", {}))
        
        result = harmonizer.harmonize_dataset(dataset, ontology, mapping_file)
        
        click.echo(f"✓ Harmonized {result['updated_cells']} cells")
        click.echo(f"Confidence > {result['confidence_threshold']}: {result['high_confidence_mappings']}")
        
    except Exception as e:
        click.echo(f"✗ Harmonization failed: {e}", err=True)
        sys.exit(1)


@cli.command("list")
@click.argument("list_type", type=click.Choice(["datasets", "studies", "cell-types"]))
@click.option("--study", help="Filter by study identifier")
@click.pass_context
def list_items(ctx: click.Context, list_type: str, study: Optional[str]) -> None:
    """List datasets, studies, or cell types."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        querier = CBioPortalQuerier(client)
        
        if list_type == "datasets":
            result = querier.list_datasets(study)
        elif list_type == "studies":
            result = querier.list_studies()
        elif list_type == "cell-types":
            result = querier.list_cell_types(study)
        
        click.echo(result.to_string())
        
    except Exception as e:
        click.echo(f"✗ List failed: {e}", err=True)
        sys.exit(1)


@cli.command("update-mappings")
@click.option("--dataset", required=True, help="Dataset identifier to update")
@click.option("--study-id", required=True, help="Study identifier")
@click.option("--sample-mapping", type=click.Path(exists=True), help="Updated sample mapping file")
@click.option("--patient-mapping", type=click.Path(exists=True), help="Updated patient mapping file") 
@click.option("--dry-run", is_flag=True, help="Preview changes without applying")
@click.pass_context
def update_mappings(
    ctx: click.Context,
    dataset: str,
    study_id: str,
    sample_mapping: Optional[str],
    patient_mapping: Optional[str],
    dry_run: bool,
) -> None:
    """Update sample/patient mappings for existing dataset with new cBioPortal data."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        mapper = CBioPortalMapper(client, config.get("mapping", {}))
        
        # Check if dataset exists
        from .cbioportal.schema import CBioPortalSchema
        schema = CBioPortalSchema(client)
        dataset_info = schema.get_dataset_info(dataset)
        
        if not dataset_info:
            click.echo(f"Dataset {dataset} not found")
            return
        
        # Get current unmapped cells
        current_unmapped = client.query(f"""
        SELECT cell_id, mapping_strategy, sample_unique_id, patient_unique_id
        FROM {client.database}.{client.table_prefix}cells
        WHERE dataset_id = %(dataset)s
          AND (mapping_strategy = 'no_mapping' OR sample_unique_id IS NULL)
        """, {"dataset": dataset})
        
        if len(current_unmapped) == 0:
            click.echo("No unmapped cells found - all cells already have mappings")
            return
        
        click.echo(f"Found {len(current_unmapped)} unmapped cells to potentially update")
        
        # Get updated cBioPortal entities
        existing_samples = mapper._get_existing_samples(study_id)
        existing_patients = mapper._get_existing_patients(study_id)
        
        # Load new mapping files if provided
        sample_mapping_dict = mapper._load_sample_mapping(sample_mapping) if sample_mapping else {}
        patient_mapping_dict = mapper._load_patient_mapping(patient_mapping) if patient_mapping else {}
        
        # Update mappings using the mapper
        result = mapper.update_existing_mappings(
            dataset_id=dataset,
            study_id=study_id,
            sample_mapping_file=sample_mapping,
            patient_mapping_file=patient_mapping,
            dry_run=dry_run
        )
        
        # Report results
        click.echo(f"✓ Mapping update completed:")
        click.echo(f"  - Cells checked: {result['total_cells_checked']}")
        click.echo(f"  - New direct sample matches: {result['new_direct_matches']}")
        click.echo(f"  - New patient matches: {result['new_patient_matches']}")
        click.echo(f"  - Still unmapped: {result['still_unmapped']}")
        
        if dry_run:
            click.echo(f"  - Would update: {result['cells_updated']} cells")
            if result['updates']:
                click.echo("  Preview of changes:")
                for update in result['updates'][:5]:  # Show first 5
                    click.echo(f"    {update['cell_id']}: {update['old_strategy']} → {update['new_strategy']}")
                if len(result['updates']) > 5:
                    click.echo(f"    ... and {len(result['updates']) - 5} more")
        else:
            click.echo(f"  - Updated: {result['cells_updated']} cells")
            
    except Exception as e:
        click.echo(f"✗ Update failed: {e}", err=True)
        sys.exit(1)


@cli.command("delete")
@click.option("--dataset", required=True, help="Dataset identifier to remove")
@click.option("--force", is_flag=True, help="Force removal without confirmation")
@click.pass_context
def delete(ctx: click.Context, dataset: str, force: bool) -> None:
    """Delete a dataset and all associated data."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        
        # Check if dataset exists
        from .cbioportal.schema import CBioPortalSchema
        schema = CBioPortalSchema(client)
        dataset_info = schema.get_dataset_info(dataset)
        
        if not dataset_info:
            click.echo(f"Dataset {dataset} not found")
            return
        
        # Confirm deletion
        if not force:
            click.echo(f"Dataset info: {dataset_info['name']} ({dataset_info['n_cells']} cells, {dataset_info['n_genes']} genes)")
            if not click.confirm(f"Are you sure you want to delete dataset {dataset}?"):
                click.echo("Cancelled")
                return
        
        # Delete all data for this dataset
        tables_to_clean = [
            f"{client.table_prefix}datasets",
            f"{client.table_prefix}cells", 
            f"{client.table_prefix}dataset_genes",
            f"{client.table_prefix}expression_matrix",
            f"{client.table_prefix}cell_embeddings"
        ]
        
        for table in tables_to_clean:
            try:
                client.command(f"""
                DELETE FROM {client.database}.{table} 
                WHERE dataset_id = %(dataset)s
                """, {"dataset": dataset})
                click.echo(f"Cleaned {table}")
            except Exception as e:
                click.echo(f"Warning: Could not clean {table}: {e}")
        
        click.echo(f"✓ Dataset {dataset} deleted successfully")
        
    except Exception as e:
        click.echo(f"✗ Delete failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--dataset", required=True, help="Dataset identifier")
@click.option("--output", "-o", required=True, help="Output h5ad file path")
@click.option("--cell-types", help="Comma-separated list of cell types to export")
@click.option("--samples", help="Comma-separated list of sample IDs to export")
@click.option("--genes", help="Comma-separated list of genes to export")
@click.pass_context
def export(
    ctx: click.Context,
    dataset: str,
    output: str,
    cell_types: Optional[str],
    samples: Optional[str],
    genes: Optional[str],
) -> None:
    """Export subset of dataset back to h5ad format."""
    config = ctx.obj["config"]
    
    try:
        client = CBioPortalClient(config.get("cbioportal", {}))
        importer = H5adImporter(client, config.get("import", {}))
        
        filters = {}
        if cell_types:
            filters["cell_types"] = [ct.strip() for ct in cell_types.split(",")]
        if samples:
            filters["samples"] = [s.strip() for s in samples.split(",")]
        if genes:
            filters["genes"] = [g.strip() for g in genes.split(",")]
        
        result = importer.export_dataset(dataset, output, filters)
        
        click.echo(f"✓ Exported {result['n_cells']} cells to {output}")
        
    except Exception as e:
        click.echo(f"✗ Export failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()