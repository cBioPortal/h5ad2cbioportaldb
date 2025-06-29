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


@click.group()
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """h5ad2cbioportaldb: Import h5ad single-cell files into cBioPortal ClickHouse database."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    
    log_level = "DEBUG" if verbose else ctx.obj["config"].get("logging", {}).get("level", "INFO")
    setup_logging(log_level)


@cli.command()
@click.option("--file", "-f", required=True, type=click.Path(exists=True), help="Path to h5ad file")
@click.option("--dataset-id", required=True, help="Unique dataset identifier")
@click.option("--study-id", required=True, help="cBioPortal study identifier")
@click.option("--cell-type-column", help="Column name for cell types in obs")
@click.option("--sample-obs-column", help="Column name for sample IDs in obs")
@click.option("--patient-obs-column", help="Column name for patient IDs in obs")
@click.option("--sample-mapping", type=click.Path(exists=True), help="CSV file mapping h5ad samples to cBioPortal")
@click.option("--patient-mapping", type=click.Path(exists=True), help="CSV file mapping h5ad patients to cBioPortal")
@click.option("--description", help="Dataset description")
@click.option("--matrix-type", default="raw", help="Expression matrix type to import")
@click.option("--dry-run", is_flag=True, help="Validate without importing")
@click.pass_context
def import_dataset(
    ctx: click.Context,
    file: str,
    dataset_id: str,
    study_id: str,
    cell_type_column: Optional[str],
    sample_obs_column: Optional[str],
    patient_obs_column: Optional[str],
    sample_mapping: Optional[str],
    patient_mapping: Optional[str],
    description: Optional[str],
    matrix_type: str,
    dry_run: bool,
) -> None:
    """Import h5ad file into cBioPortal database."""
    config = ctx.obj["config"]
    
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
        )
        
        if dry_run:
            click.echo("✓ Validation completed successfully")
            click.echo(f"Would import {result['n_cells']} cells, {result['n_genes']} genes")
        else:
            click.echo(f"✓ Successfully imported dataset {dataset_id}")
            click.echo(f"Imported {result['n_cells']} cells, {result['n_genes']} genes")
            
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