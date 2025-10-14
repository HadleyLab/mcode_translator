"""
Patients Commands

Commands for processing patient data through the mCODE pipeline:
fetch → process → summarize

Supports flexible pipeline execution with flag combinations.
"""

from pathlib import Path
import sys
from typing import Optional

from rich.console import Console
import typer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="patients",
    help="Patient data processing operations",
    add_completion=True,
)


@app.command("pipeline")
def patients_pipeline(
    # Download flags
    download_archive: bool = typer.Option(False, "--download-archive", help="Download synthetic patient data archives"),
    # Fetch flags
    fetch: bool = typer.Option(False, "--fetch", help="Fetch patient data from archives"),
    archive_path: Optional[str] = typer.Option(
        None, help="Path or identifier for patient data archive"
    ),
    patient_id: Optional[str] = typer.Option(None, help="Specific patient ID to fetch"),
    fetch_limit: int = typer.Option(10, help="Maximum number of patients to fetch"),
    # Process flags
    process: bool = typer.Option(
        False, "--process", help="Process patient data with mCODE mapping"
    ),
    input_file: Optional[str] = typer.Option(
        None, help="Path to patient data file (NDJSON format) for processing"
    ),
    trials_criteria: Optional[str] = typer.Option(
        None, help="JSON string of trial eligibility criteria"
    ),
    process_store_memory: bool = typer.Option(
        False, "--store-processed", help="Store processed data in CORE Memory"
    ),
    # Summarize flags
    summarize: bool = typer.Option(
        False, "--summarize", help="Generate natural language summaries"
    ),
    summary_input_file: Optional[str] = typer.Option(
        None, help="Path to processed patient data file for summarization"
    ),
    summary_store_memory: bool = typer.Option(
        False, "--store-summaries", help="Store summaries in CORE Memory"
    ),
    # Output options
    output_file: Optional[str] = typer.Option(None, help="Path to save results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute patient data processing pipeline with flexible stage selection.

    Combine flags to run specific pipeline stages:
    --fetch                    # Only fetch data
    --fetch --process          # Fetch and process
    --process --summarize      # Process and summarize existing data
    --fetch --process --summarize  # Full pipeline
    """
    try:
        # Validate flag combinations
        if not any([fetch, process, summarize]):
            console.print(
                "[red]❌ Must specify at least one pipeline stage: --fetch, --process, or --summarize[/red]"
            )
            raise typer.Exit(1)

        # Import required components
        from src.utils.config import Config
        from src.utils.data_downloader import download_synthetic_patient_archives
        from src.core.dependency_container import get_container

        config = Config()
        container = get_container()
        memory_storage = container.create_memory_storage()

        # Track pipeline results
        pipeline_results = {"fetch": None, "process": None, "summarize": None}

        # Execute fetch stage
        if fetch:
            console.print("[bold blue]📥 Fetching patient data...[/bold blue]")
            if not archive_path:
                console.print("[red]❌ --archive-path required for --fetch[/red]")
                raise typer.Exit(1)

            if download_archive:
                console.print("[bold blue]📥 Checking for and downloading synthetic patient archives...[/bold blue]")
                downloaded_archives = download_synthetic_patient_archives(force_download=False)
                if downloaded_archives:
                    console.print("[green]✅ Archives are ready.[/green]")
                else:
                    console.print("[yellow]⚠️ No new archives were downloaded.[/yellow]")

            from workflows.patients_fetcher import PatientsFetcherWorkflow

            fetcher = PatientsFetcherWorkflow(config, memory_storage)

            fetch_result = fetcher.execute(
                archive_path=archive_path,
                patient_id=patient_id,
                limit=fetch_limit,
                output_path=None,  # We'll pass data to next stage
            )

            if fetch_result.success:
                num_fetched = len(fetch_result.data) if fetch_result.data else 0
                console.print(
                    f"[green]✅ Fetched {num_fetched} patients[/green]"
                )
                if num_fetched > 0 and output_file and not process and not summarize:
                    console.print(f"[blue]💾 Saving fetched patient data to: {output_file}[/blue]")
                    with open(output_file, "w") as f:
                        for patient in fetch_result.data:
                            import json
                            json.dump(patient, f, ensure_ascii=False)
                            f.write("\n")
                    console.print(f"[green]✅ Fetched data saved to {output_file}[/green]")
                pipeline_results["fetch"] = fetch_result
            else:
                console.print(f"[red]❌ Fetch failed: {fetch_result.error_message}[/red]")
                raise typer.Exit(1)

        # Execute process stage
        if process:
            console.print("[bold blue]🔬 Processing patient data...[/bold blue]")

            # Get input data - either from fetch stage or specified file
            if fetch and pipeline_results["fetch"]:
                # Use data from fetch stage
                patients_data = pipeline_results["fetch"].data
                console.print("[blue]📋 Using data from fetch stage[/blue]")
            elif input_file:
                # Load from file
                console.print(f"[blue]📖 Loading from: {input_file}[/blue]")
                patients_data = []
                with open(input_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            import json

                            try:
                                patient = json.loads(line)
                                patients_data.append(patient)
                            except json.JSONDecodeError as e:
                                console.print(f"[yellow]⚠️ Skipping invalid JSON line: {e}[/yellow]")
                console.print(f"[green]✅ Loaded {len(patients_data)} patients[/green]")
            else:
                console.print("[red]❌ Either --fetch or --input-file required for --process[/red]")
                raise typer.Exit(1)

            if not patients_data:
                console.print("[red]❌ No patient data available for processing[/red]")
                raise typer.Exit(1)

            # Parse trials criteria if provided
            trials_criteria_dict = None
            if trials_criteria:
                import json

                try:
                    trials_criteria_dict = json.loads(trials_criteria)
                    console.print("[blue]📋 Trials criteria parsed[/blue]")
                except json.JSONDecodeError as e:
                    console.print(f"[red]❌ Invalid trials criteria JSON: {e}[/red]")
                    raise typer.Exit(1)

            from workflows.patients_processor import PatientsProcessorWorkflow

            processor = PatientsProcessorWorkflow(config, memory_storage)

            process_result = processor.execute(
                patients_data=patients_data,
                trials_criteria=trials_criteria_dict,
                store_in_memory=process_store_memory,
            )

            if process_result.success:
                console.print("[green]✅ Patient processing completed[/green]")
                if process_result.metadata:
                    total = process_result.metadata.get("total_patients", 0)
                    successful = process_result.metadata.get("successful", 0)
                    console.print(
                        f"[green]📊 Processed {successful}/{total} patients successfully[/green]"
                    )
                pipeline_results["process"] = process_result

                # Save processed data if output file specified and not summarizing
                if output_file and not summarize:
                    console.print(f"[blue]💾 Saving processed data to: {output_file}[/blue]")
                    with open(output_file, "w") as f:
                        for patient in process_result.data:
                            import json

                            json.dump(patient, f, ensure_ascii=False)
                            f.write("\n")
                    console.print(f"[green]✅ Processed data saved to {output_file}[/green]")
            else:
                console.print(f"[red]❌ Processing failed: {process_result.error_message}[/red]")
                raise typer.Exit(1)

        # Execute summarize stage
        if summarize:
            console.print("[bold blue]📝 Generating patient summaries...[/bold blue]")

            # Get input data - either from process stage or specified file
            if process and pipeline_results["process"]:
                # Use data from process stage
                patients_data = pipeline_results["process"].data
                console.print("[blue]📋 Using data from process stage[/blue]")
            elif summary_input_file:
                # Load from file
                console.print(f"[blue]📖 Loading from: {summary_input_file}[/blue]")
                patients_data = []
                with open(summary_input_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            import json

                            try:
                                patient = json.loads(line)
                                patients_data.append(patient)
                            except json.JSONDecodeError as e:
                                console.print(f"[yellow]⚠️ Skipping invalid JSON line: {e}[/yellow]")
                console.print(f"[green]✅ Loaded {len(patients_data)} patients[/green]")
            else:
                console.print(
                    "[red]❌ Either --process or --summary-input-file required for --summarize[/red]"
                )
                raise typer.Exit(1)

            if not patients_data:
                console.print("[red]❌ No patient data available for summarization[/red]")
                raise typer.Exit(1)

            from workflows.patients_summarizer import PatientsSummarizerWorkflow

            summarizer = PatientsSummarizerWorkflow(config, memory_storage)

            summarize_result = summarizer.execute(
                patients_data=patients_data, store_in_memory=summary_store_memory
            )

            if summarize_result.success:
                console.print("[green]✅ Patient summarization completed[/green]")
                if summarize_result.metadata:
                    total = summarize_result.metadata.get("total_patients", 0)
                    successful = summarize_result.metadata.get("successful", 0)
                    console.print(
                        f"[green]📊 Summarized {successful}/{total} patients successfully[/green]"
                    )
                pipeline_results["summarize"] = summarize_result
            else:
                console.print(
                    f"[red]❌ Summarization failed: {summarize_result.error_message}[/red]"
                )
                # Don't raise exit here - allow pipeline to continue
                console.print("[yellow]⚠️ Continuing with pipeline despite summarization issues[/yellow]")

        # Save final results if output file specified
        if output_file and pipeline_results["summarize"] and pipeline_results["summarize"].data:
            console.print(f"[blue]💾 Saving results to: {output_file}[/blue]")
            with open(output_file, "w") as f:
                for patient in pipeline_results["summarize"].data:
                    import json

                    json.dump(patient, f, ensure_ascii=False)
                    f.write("\n")
            console.print("[green]✅ Results saved[/green]")

        console.print(
            "[bold green]🎉 Patient pipeline execution completed successfully![/bold green]"
        )

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Pipeline execution failed: {e}[/red]")
        logger.exception("Patient pipeline error")
        raise typer.Exit(1)
