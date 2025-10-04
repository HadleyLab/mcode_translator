"""
Data Commands

Commands for data ingestion, processing, and management operations
including clinical trials, patient data, and bulk operations.
"""

import time
import typer
from rich.console import Console

# Add src to path for imports
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="data",
    help="Data ingestion and management operations",
    add_completion=True,
)


@app.command("ingest-trials")
def ingest_clinical_trials(
    limit: int = typer.Option(50, help="Maximum number of trials to ingest (0 for all)"),
    cancer_type: str = typer.Option("all", help="Filter by cancer type (breast, lung, melanoma, etc.)"),
    phase: str = typer.Option("all", help="Filter by trial phase (I, II, III, I/II, etc.)"),
    status: str = typer.Option("all", help="Filter by trial status (recruiting, active, completed, etc.)"),
    batch_size: int = typer.Option(10, help="Number of trials to process per batch"),
    space_name: str = typer.Option("Clinical Trials Database", help="Target memory space name"),
    engine: str = typer.Option("llm", help="Processing engine: 'regex' (fast, deterministic) or 'llm' (flexible, intelligent)"),
    llm_model: str = typer.Option("deepseek-coder", help="LLM model to use for llm engine"),
    llm_prompt: str = typer.Option("direct_mcode_evidence_based_concise", help="Prompt template for llm engine"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Ingest clinical trials from live ClinicalTrials.gov data into CORE Memory.

    Choose between two processing engines:
    - regex: Fast, deterministic structured data extraction (default)
    - llm: Flexible, intelligent processing with LLM enhancement

    Filters by cancer type, phase, and status, then summarizes and stores in CORE Memory.
    """
    # Validate engine parameter
    if engine not in ["regex", "llm"]:
        console.print(f"[red]❌ Invalid engine: {engine}. Use 'regex' or 'llm'[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]🧪 Ingesting Clinical Trials from Live Data[/bold blue]")
    console.print(f"[blue]📋 Parameters: limit={limit}, cancer_type={cancer_type}, phase={phase}, status={status}[/blue]")
    console.print(f"[blue]🔧 Engine: {engine}[/blue]")
    if engine == "llm":
        console.print(f"[blue]🤖 LLM Model: {llm_model}, Prompt: {llm_prompt}[/blue]")

    try:
        # Import required components
        from config.heysol_config import get_config
        from storage.mcode_memory_storage import OncoCoreMemory
        from workflows.trials_fetcher import TrialsFetcherWorkflow

        # Get configuration
        config = get_config()
        api_key = config.get_api_key()

        if not api_key:
            console.print("[red]❌ No HeySol API key configured[/red]")
            raise typer.Exit(1)

        if verbose:
            console.print(f"[blue]📦 Batch size: {batch_size}[/blue]")
            console.print(f"[blue]🎯 Target space: {space_name}[/blue]")

        # Setup CORE Memory space
        console.print(f"[blue]🏗️ Setting up memory space: {space_name}[/blue]")
        memory = OncoCoreMemory()

        # Check if space exists, create if not
        spaces = memory.api_client.get_spaces()
        space_id = None
        for space in spaces:
            if space.get("name") == space_name:
                space_id = space.get("id")
                space_id_str = str(space_id)[:16] if space_id else "unknown"
                console.print(f"[green]✅ Found existing space: {space_id_str}...[/green]")
                break

        if not space_id:
            space_id = memory.api_client.create_space(space_name, "Live clinical trials database")
            if not space_id:
                console.print(f"[red]❌ Failed to create space: {space_name}[/red]")
                raise typer.Exit(1)
            console.print(f"[green]✅ Created new space: {space_id}[/green]")

        # Initialize workflows
        fetcher = TrialsFetcherWorkflow()
        from workflows.trials_processor import TrialsProcessor
        processor = TrialsProcessor(config)

        # Fetch live trial data using existing workflow
        console.print(f"[blue]🔍 Fetching live trials for condition: {cancer_type}[/blue]")

        # Use TrialsFetcherWorkflow for live data fetching
        fetch_result = fetcher.execute(
            condition=cancer_type if cancer_type != "all" else None,
            limit=limit,
            output_path=None  # Don't save to file, process in memory
        )

        if not fetch_result.success:
            console.print(f"[red]❌ Failed to fetch trials: {fetch_result.error_message}[/red]")
            raise typer.Exit(1)

        trials = fetch_result.data
        console.print(f"[green]✅ Fetched {len(trials)} trials from ClinicalTrials.gov[/green]")

        if not trials:
            console.print(f"[yellow]⚠️ No trials found for cancer type: {cancer_type}[/yellow]")
            raise typer.Exit(1)

        # Filter trials by phase and status if specified
        filtered_trials = []
        for trial in trials:
            protocol_section = trial.get("protocolSection", {})
            design = protocol_section.get("designModule", {})
            status_module = protocol_section.get("statusModule", {})

            # Apply phase filter
            trial_phase = design.get("phase", "")
            if phase != "all" and trial_phase != phase:
                continue

            # Apply status filter
            trial_status = status_module.get("overallStatus", "")
            if status != "all" and trial_status != status:
                continue

            filtered_trials.append(trial)

        # Apply limit after filtering
        if limit > 0:
            filtered_trials = filtered_trials[:limit]

        console.print(f"[green]✅ Filtered to {len(filtered_trials)} trials matching criteria[/green]")

        # Check for existing trials to avoid duplicates
        console.print(f"[blue]🔍 Checking for existing trials in CORE Memory...[/blue]")
        existing_trials = set()
        try:
            # Search for existing trial summaries
            existing_search = memory.search_similar_trials("NCT", limit=1000)

            # Handle both dictionary and string episode formats
            episodes = existing_search.episodes if hasattr(existing_search, 'episodes') else []

            for episode in episodes:
                nct_id = None
                if isinstance(episode, dict):
                    # Try to extract NCT ID from the episode data
                    data = episode.get("data", "")
                    if isinstance(data, str) and "NCT" in data:
                        # Extract NCT ID from summary text
                        start = data.find("NCT")
                        if start != -1:
                            end = data.find(" ", start)
                            if end == -1:
                                end = len(data)
                            nct_id = data[start:end]
                elif isinstance(episode, str) and "NCT" in episode:
                    # Handle string format - extract NCT ID directly from summary text
                    start = episode.find("NCT")
                    if start != -1:
                        end = episode.find(" ", start)
                        if end == -1:
                            end = len(episode)
                        nct_id = episode[start:end]

                if nct_id:
                    existing_trials.add(nct_id)

            console.print(f"[green]✅ Found {len(existing_trials)} existing trials in CORE Memory[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not check existing trials: {e}[/yellow]")
            console.print(f"[blue]💡 Continuing with ingestion (may create duplicates)[/blue]")

        # Process trials in batches
        ingestion_stats = {
            "total_trials": len(filtered_trials),
            "ingested": 0,
            "skipped_duplicates": 0,
            "failed": 0,
            "start_time": time.time(),
        }

        console.print(f"[blue]🚀 Processing {len(filtered_trials)} trials in batches of {batch_size}[/blue]")

        for i in range(0, len(filtered_trials), batch_size):
            batch = filtered_trials[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(filtered_trials) + batch_size - 1) // batch_size

            console.print(f"[blue]📦 Processing batch {batch_num}/{total_batches} ({len(batch)} trials)[/blue]")

            # Filter out duplicates and trials without NCT IDs
            processable_batch = []
            skipped_in_batch = 0

            for trial in batch:
                protocol_section = trial.get("protocolSection", {})
                identification = protocol_section.get("identificationModule", {})
                nct_id = identification.get("nctId", "")

                if not nct_id:
                    console.print(f"[yellow]⚠️ Skipping trial with no NCT ID[/yellow]")
                    ingestion_stats["failed"] += 1
                    continue

                # Check for duplicates
                if nct_id in existing_trials:
                    console.print(f"[blue]   ⏭️ Skipped (duplicate) - {nct_id} already exists[/blue]")
                    ingestion_stats["skipped_duplicates"] += 1
                    skipped_in_batch += 1
                    continue

                processable_batch.append(trial)

            if not processable_batch:
                console.print(f"[yellow]⚠️ No processable trials in this batch[/yellow]")
                continue

            if verbose:
                for trial in processable_batch:
                    protocol_section = trial.get("protocolSection", {})
                    identification = protocol_section.get("identificationModule", {})
                    status_module = protocol_section.get("statusModule", {})
                    nct_id = identification.get("nctId", "")
                    trial_status = status_module.get("overallStatus", "unknown")
                    console.print(f"[blue]   🧪 Processing: {nct_id} ({trial_status})[/blue]")

            try:
                # Use workflow to process the batch
                processing_result = processor.execute(
                    trials_data=processable_batch,
                    engine=engine,
                    model=llm_model,
                    prompt=llm_prompt,
                    workers=1,  # Process sequentially for now
                    store_in_memory=False  # We'll handle storage manually
                )

                if processing_result.success and processing_result.data:
                    processed_trials = processing_result.data

                    for idx, processed_trial in enumerate(processed_trials):
                        if idx >= len(processable_batch):
                            break

                        original_trial = processable_batch[idx]
                        protocol_section = original_trial.get("protocolSection", {})
                        identification = protocol_section.get("identificationModule", {})
                        design = protocol_section.get("designModule", {})
                        status_module = protocol_section.get("statusModule", {})
                        nct_id = identification.get("nctId", "")

                        # Store in CORE Memory
                        mcode_data = {
                            "original_trial_data": original_trial,
                            "trial_metadata": {
                                "nct_id": nct_id,
                                "overall_status": status_module.get("overallStatus", ""),
                                "phase": design.get("phase", ""),
                            },
                            "processing_metadata": processing_result.metadata,
                            "processed_trial": processed_trial
                        }

                        success = memory.store_trial_summary(nct_id, summary)

                        if success:
                            console.print(f"[green]   ✅ Ingested: {nct_id}[/green]")
                            ingestion_stats["ingested"] += 1
                            existing_trials.add(nct_id)
                        else:
                            console.print(f"[red]   ❌ Failed to store: {nct_id}[/red]")
                            ingestion_stats["failed"] += 1
                else:
                    console.print(f"[red]   ❌ Batch processing failed: {processing_result.error_message}[/red]")
                    ingestion_stats["failed"] += len(processable_batch)

            except Exception as e:
                console.print(f"[red]   ❌ Failed to process batch: {e}[/red]")
                ingestion_stats["failed"] += len(processable_batch)

        # Calculate final statistics
        ingestion_stats["end_time"] = time.time()
        ingestion_stats["total_time"] = ingestion_stats["end_time"] - ingestion_stats["start_time"]
        ingestion_stats["success_rate"] = (
            (ingestion_stats["ingested"] / ingestion_stats["total_trials"] * 100)
            if ingestion_stats["total_trials"] > 0 else 0
        )

        # Display results
        console.print("[green]✅ Clinical trials ingestion completed[/green]")
        console.print(f"[green]📊 Ingested: {ingestion_stats['ingested']} trials[/green]")
        console.print(f"[green]❌ Failed: {ingestion_stats['failed']} trials[/green]")
        console.print(f"[green]⏱️ Total time: {ingestion_stats['total_time']:.1f} seconds[/green]")
        console.print(f"[green]💾 All data stored in CORE Memory space: {space_name}[/green]")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Ingestion failed: {e}[/red]")
        logger.exception("Clinical trials ingestion error")
        raise typer.Exit(1)

