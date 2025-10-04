"""
mCODE Translation Commands

Commands for processing and translating clinical data using mCODE standards.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="mcode",
    help="mCODE translation and processing operations",
    add_completion=True,
)






@app.command("summarize")
def summarize_trial(
    trial_id: str = typer.Argument(..., help="Clinical trial NCT ID"),
    output_file: Optional[str] = typer.Option(None, help="Path to save summary"),
    format: str = typer.Option("text", help="Output format (text, json, ndjson)"),
    include_codes: bool = typer.Option(True, help="Include mCODE codes in summary"),
    store_memory: bool = typer.Option(True, help="Store summary in CORE Memory"),
    engine: str = typer.Option("regex", help="Processing engine: 'regex' (fast, deterministic) or 'llm' (flexible, intelligent)"),
    llm_model: str = typer.Option("deepseek-coder", help="LLM model to use for llm engine"),
    llm_prompt: str = typer.Option("direct_mcode_evidence_based_concise", help="Prompt template for llm engine"),
    compare_engines: bool = typer.Option(False, help="Compare both engines and show recommendation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Generate mCODE summary for clinical trial.

    Choose between two processing engines:
    - regex: Fast, deterministic structured data extraction (default)
    - llm: Flexible, intelligent processing with LLM enhancement

    Creates comprehensive mCODE-compliant summaries of clinical trials
    with proper coding and structured information extraction.
    """
    # Validate engine parameter
    if engine not in ["regex", "llm"]:
        console.print(f"[red]❌ Invalid engine: {engine}. Use 'regex' or 'llm'[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]📝 Generating mCODE summary for trial: {trial_id}[/bold blue]")
    console.print(f"[blue]🔧 Engine: {engine}[/blue]")
    if engine == "llm":
        console.print(f"[blue]🤖 LLM Model: {llm_model}, Prompt: {llm_prompt}[/blue]")

    try:
        # Import summarization components
        from storage.mcode_memory_storage import OncoCoreMemory

        if verbose:
            console.print(f"[blue]📋 Output format: {format}[/blue]")
            console.print(f"[blue]📋 Include codes: {include_codes}[/blue]")
            console.print(f"[blue]💾 Memory storage: {'enabled' if store_memory else 'disabled'}[/blue]")

        # Initialize mCODE trial processor
        from services.trial_processor import McodeTrialProcessor
        processor = McodeTrialProcessor(
            default_engine=engine,
            llm_model=llm_model,
            llm_prompt=llm_prompt,
            include_codes=include_codes
        )

        # Handle engine comparison if requested
        if compare_engines:
            console.print("[blue]🔍 Fetching trial data for comparison...[/blue]")
            # TODO: Implement trial data fetching for comparison
            console.print("[yellow]⚠️ Engine comparison not yet implemented[/yellow]")
            console.print(f"[blue]💡 Recommended engine: {processor.recommend_engine({})}[/blue]")
            return

        console.print("[blue]🔍 Fetching trial data...[/blue]")
        # Fetch trial data using TrialsFetcherWorkflow
        from workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
        fetcher = TrialsFetcherWorkflow()
        fetch_result = fetcher.execute(nct_ids=[trial_id], output_path=None)

        if not fetch_result.success:
            console.print(f"[red]❌ Failed to fetch trial {trial_id}: {fetch_result.error_message}[/red]")
            raise typer.Exit(1)

        trials = fetch_result.data
        if not trials:
            console.print(f"[red]❌ Trial {trial_id} not found[/red]")
            raise typer.Exit(1)

        trial_data = trials[0]  # Should be only one trial
        console.print(f"[green]✅ Fetched trial data for {trial_id}[/green]")

        console.print("[blue]📝 Generating mCODE summary...[/blue]")
        # Process trial using unified processor
        import asyncio
        processing_result = asyncio.run(processor.process_trial(trial_data, engine=engine))

        if not processing_result.success:
            console.print(f"[red]❌ Processing failed: {processing_result.error_message}[/red]")
            raise typer.Exit(1)

        summary = processing_result.data
        console.print(f"[green]✅ Generated summary using {engine} engine[/green]")
        if verbose:
            console.print(f"[blue]⏱️ Processing time: {processing_result.processing_time:.2f}s[/blue]")
            if processing_result.metadata:
                console.print(f"[blue]📊 Elements extracted: {processing_result.metadata.get('elements_extracted', 0)}[/blue]")

        # Display summary based on format
        if format == "text":
            console.print(f"\n[bold cyan]📋 Trial Summary:[/bold cyan]")
            console.print(summary)
        elif format == "json":
            import json
            console.print(json.dumps({"trial_id": trial_id, "summary": summary, "engine": engine}, indent=2))
        elif format == "ndjson":
            import json
            console.print(json.dumps({"trial_id": trial_id, "summary": summary, "engine": engine}))

        if store_memory:
            console.print("[blue]💾 Storing in CORE Memory...[/blue]")
            memory = OncoCoreMemory()
            mcode_data = {
                "original_trial_data": trial_data,
                "trial_metadata": {
                    "nct_id": trial_id,
                    "processing_engine": engine,
                    "processing_time": processing_result.processing_time,
                },
                "processing_metadata": processing_result.metadata
            }
            success = memory.store_trial_mcode_summary(trial_id, mcode_data)
            if success:
                console.print("[green]✅ Summary stored in CORE Memory[/green]")
            else:
                console.print("[yellow]⚠️ Failed to store in CORE Memory[/yellow]")

        console.print("[green]✅ Trial summary completed successfully[/green]")

        if output_file:
            with open(output_file, 'w') as f:
                if format == "json":
                    import json
                    json.dump({"trial_id": trial_id, "summary": summary, "engine": engine}, f, indent=2)
                else:
                    f.write(summary)
            console.print(f"[green]💾 Summary saved to: {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Summary generation failed: {e}[/red]")
        logger.exception("Trial summary error")
        raise typer.Exit(1)