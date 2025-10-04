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
    engine: str = typer.Option("llm", help="Processing engine: 'regex' (fast, deterministic) or 'llm' (flexible, intelligent)"),
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

        # Initialize trials processor workflow
        from workflows.trials_processor import TrialsProcessor
        from config.heysol_config import get_config
        config = get_config()
        processor = TrialsProcessor(config)


        console.print("[blue]🔍 Fetching trial data...[/blue]")
        # Fetch trial data using TrialsFetcherWorkflow
        from workflows.trials_fetcher import TrialsFetcherWorkflow
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
        # Process trial using workflow
        processing_result = processor.process_single_trial(
            trial_data,
            engine=engine,
            model=llm_model,
            prompt=llm_prompt
        )

        if not processing_result.success:
            console.print(f"[red]❌ Processing failed: {processing_result.error_message}[/red]")
            raise typer.Exit(1)

        # Use summarizer workflow to generate proper natural language summary
        from workflows.trials_summarizer import TrialsSummarizerWorkflow
        summarizer = TrialsSummarizerWorkflow(config)

        # Get processed trial with mCODE elements
        processed_trial = processing_result.data

        # Generate natural language summary using summarizer workflow
        summary_result = summarizer.process_single_trial(
            processed_trial,
            store_in_memory=False  # Don't store during CLI usage
        )

        if summary_result.success and summary_result.data:
            summary_data = summary_result.data
            # Extract the natural language summary
            summary = summary_data.get("McodeResults", {}).get("natural_language_summary", "Summary not available")
        else:
            summary = "Failed to generate summary"

        console.print(f"[green]✅ Generated summary[/green]")
        if verbose:
            if processing_result.metadata:
                console.print(f"[blue]📊 Processing metadata: {processing_result.metadata}[/blue]")

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
                    "processing_engine": "llm",
                },
                "processing_metadata": processing_result.metadata,
                "processed_trial": processed_trial
            }
            success = memory.store_trial_summary(trial_id, summary)
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