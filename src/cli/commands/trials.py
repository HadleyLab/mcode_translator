"""
Trials Commands

Commands for processing clinical trial data through the mCODE pipeline:
fetch → process → summarize → optimize

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
    name="trials",
    help="Clinical trial data processing operations",
    add_completion=True,
)


@app.command("pipeline")
def trials_pipeline(
    # Fetch flags
    fetch: bool = typer.Option(
        False, "--fetch", help="Fetch trial data from ClinicalTrials.gov API"
    ),
    cancer_type: str = typer.Option("breast", help="Cancer type to search for"),
    phase: str = typer.Option("all", help="Trial phase filter"),
    status: str = typer.Option("all", help="Trial status filter"),
    fetch_limit: int = typer.Option(50, help="Maximum number of trials to fetch"),
    fetch_output_file: Optional[str] = typer.Option(None, help="Path to save fetched trial data"),
    # Process flags
    process: bool = typer.Option(
        False, "--process", help="Process trial data with mCODE extraction"
    ),
    input_file: Optional[str] = typer.Option(
        None, help="Path to trial data file (NDJSON format) for processing"
    ),
    engine: str = typer.Option("llm", help="Processing engine: 'regex' or 'llm'"),
    llm_model: str = typer.Option("deepseek-coder", help="LLM model for processing"),
    llm_prompt: str = typer.Option(
        "direct_mcode_evidence_based_concise", help="Prompt template for LLM processing"
    ),
    workers: int = typer.Option(4, help="Number of concurrent workers for async processing"),
    process_store_memory: bool = typer.Option(
        False, "--store-processed", help="Store processed data in CORE Memory"
    ),
    # Summarize flags
    summarize: bool = typer.Option(
        False, "--summarize", help="Generate natural language summaries"
    ),
    summary_input_file: Optional[str] = typer.Option(
        None, help="Path to processed trial data file for summarization"
    ),
    trial_id: Optional[str] = typer.Option(None, help="Specific trial NCT ID to summarize"),
    summary_store_memory: bool = typer.Option(
        False, "--store-summaries", help="Store summaries in CORE Memory"
    ),
    # Optimize flags
    optimize: bool = typer.Option(False, "--optimize", help="Optimize processing parameters"),
    optimize_input_file: Optional[str] = typer.Option(
        None, help="Path to trial data file for optimization"
    ),
    prompts: Optional[str] = typer.Option(None, help="Comma-separated list of prompts to test"),
    models: Optional[str] = typer.Option(None, help="Comma-separated list of models to test"),
    cv_folds: int = typer.Option(3, help="Number of cross-validation folds"),
    inter_rater: bool = typer.Option(False, help="Run inter-rater reliability analysis"),
    # Output options
    output_file: Optional[str] = typer.Option(None, help="Path to save results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Execute clinical trial data processing pipeline with flexible stage selection.

    Combine flags to run specific pipeline stages:
    --fetch                    # Only fetch data
    --fetch --process          # Fetch and process
    --process --summarize      # Process and summarize existing data
    --fetch --process --summarize  # Full pipeline
    --optimize                 # Optimize processing parameters
    """
    try:
        # Validate flag combinations
        if not any([fetch, process, summarize, optimize]):
            console.print(
                "[red]❌ Must specify at least one pipeline stage: --fetch, --process, --summarize, or --optimize[/red]"
            )
            raise typer.Exit(1)

        # Import required components
        from src.utils.config import Config

        config = Config()

        # Track pipeline results
        pipeline_results = {"fetch": None, "process": None, "summarize": None, "optimize": None}

        # Execute fetch stage
        if fetch:
            console.print("[bold blue]📥 Fetching clinical trial data...[/bold blue]")
            console.print(
                f"[blue]🎯 Cancer type: {cancer_type}, Phase: {phase}, Status: {status}[/blue]"
            )

            from workflows.trials_fetcher import TrialsFetcherWorkflow

            fetcher = TrialsFetcherWorkflow(config)

            fetch_result = fetcher.execute(
                condition=cancer_type if cancer_type != "all" else None,
                limit=fetch_limit,
                output_path=fetch_output_file,
            )

            if fetch_result.success:
                console.print(
                    f"[green]✅ Fetched {len(fetch_result.data) if fetch_result.data else 0} trials[/green]"
                )
                pipeline_results["fetch"] = fetch_result
            else:
                console.print(f"[red]❌ Fetch failed: {fetch_result.error_message}[/red]")
                raise typer.Exit(1)

        # Execute process stage
        if process:
            console.print("[bold blue]🔬 Processing trial data...[/bold blue]")
            console.print(f"[blue]🤖 Engine: {engine}[/blue]")

            # Get input data - either from fetch stage or specified file
            if fetch and pipeline_results["fetch"]:
                # Use data from fetch stage
                trials_data = pipeline_results["fetch"].data
                console.print("[blue]📋 Using data from fetch stage[/blue]")
            elif input_file:
                # Load from file
                console.print(f"[blue]📖 Loading from: {input_file}[/blue]")
                trials_data = []
                with open(input_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            import json

                            try:
                                trial = json.loads(line)
                                trials_data.append(trial)
                            except json.JSONDecodeError as e:
                                console.print(f"[yellow]⚠️ Skipping invalid JSON line: {e}[/yellow]")
                console.print(f"[green]✅ Loaded {len(trials_data)} trials[/green]")
            else:
                console.print("[red]❌ Either --fetch or --input-file required for --process[/red]")
                raise typer.Exit(1)

            if not trials_data:
                console.print("[red]❌ No trial data available for processing[/red]")
                raise typer.Exit(1)

            from workflows.trials_processor import TrialsProcessor

            processor = TrialsProcessor(config)

            process_result = processor.execute(
                trials_data=trials_data,
                engine=engine,
                model=llm_model,
                prompt=llm_prompt,
                workers=workers,  # Use async concurrency
                store_in_memory=process_store_memory,
            )

            if process_result.success:
                console.print("[green]✅ Trial processing completed[/green]")
                if process_result.metadata:
                    total = process_result.metadata.get("total_trials", 0)
                    successful = process_result.metadata.get("successful", 0)
                    console.print(
                        f"[green]📊 Processed {successful}/{total} trials successfully[/green]"
                    )
                pipeline_results["process"] = process_result
            else:
                console.print(f"[red]❌ Processing failed: {process_result.error_message}[/red]")
                raise typer.Exit(1)

        # Execute summarize stage
        if summarize:
            console.print("[bold blue]📝 Generating trial summaries...[/bold blue]")

            # Get input data - either from process stage or specified file
            if process and pipeline_results["process"]:
                # Use data from process stage
                trials_data = pipeline_results["process"].data
                console.print("[blue]📋 Using data from process stage[/blue]")
            elif summary_input_file:
                # Load from file
                console.print(f"[blue]📖 Loading from: {summary_input_file}[/blue]")
                trials_data = []
                with open(summary_input_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            import json

                            try:
                                trial = json.loads(line)
                                trials_data.append(trial)
                            except json.JSONDecodeError as e:
                                console.print(f"[yellow]⚠️ Skipping invalid JSON line: {e}[/yellow]")
                console.print(f"[green]✅ Loaded {len(trials_data)} trials[/green]")
            elif trial_id:
                # Fetch single trial
                console.print(f"[blue]🔍 Fetching single trial: {trial_id}[/blue]")
                from workflows.trials_fetcher import TrialsFetcherWorkflow

                fetcher = TrialsFetcherWorkflow(config)
                fetch_result = fetcher.execute(nct_ids=[trial_id], output_path=None)
                if fetch_result.success and fetch_result.data:
                    trials_data = fetch_result.data
                    console.print(f"[green]✅ Fetched trial {trial_id}[/green]")
                else:
                    console.print(f"[red]❌ Failed to fetch trial {trial_id}[/red]")
                    raise typer.Exit(1)
            else:
                console.print(
                    "[red]❌ Either --process, --summary-input-file, or --trial-id required for --summarize[/red]"
                )
                raise typer.Exit(1)

            if not trials_data:
                console.print("[red]❌ No trial data available for summarization[/red]")
                raise typer.Exit(1)

            from workflows.trials_summarizer import TrialsSummarizerWorkflow

            summarizer = TrialsSummarizerWorkflow(config)

            summarize_result = summarizer.execute(
                trials_data=trials_data, store_in_memory=summary_store_memory
            )

            if summarize_result.success:
                console.print("[green]✅ Trial summarization completed[/green]")
                if summarize_result.metadata:
                    total = summarize_result.metadata.get("total_trials", 0)
                    successful = summarize_result.metadata.get("successful", 0)
                    console.print(
                        f"[green]📊 Summarized {successful}/{total} trials successfully[/green]"
                    )
                pipeline_results["summarize"] = summarize_result
            else:
                console.print(
                    f"[red]❌ Summarization failed: {summarize_result.error_message}[/red]"
                )
                raise typer.Exit(1)

        # Execute optimize stage
        if optimize:
            console.print("[bold blue]🔬 Optimizing trial processing parameters...[/bold blue]")

            if not optimize_input_file:
                console.print("[red]❌ --optimize-input-file required for --optimize[/red]")
                raise typer.Exit(1)

            # Load trial data
            console.print(f"[blue]📖 Loading trials from: {optimize_input_file}[/blue]")
            trials_data = []
            with open(optimize_input_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        import json

                        try:
                            trial = json.loads(line)
                            trials_data.append(trial)
                        except json.JSONDecodeError as e:
                            console.print(f"[yellow]⚠️ Skipping invalid JSON line: {e}[/yellow]")

            if not trials_data:
                console.print("[red]❌ No trial data found in file[/red]")
                raise typer.Exit(1)

            console.print(f"[green]✅ Loaded {len(trials_data)} trials for optimization[/green]")

            from workflows.trials_optimizer import TrialsOptimizerWorkflow

            optimizer = TrialsOptimizerWorkflow(config)

            # Parse prompts and models
            prompts_list = (
                prompts.split(",") if prompts else ["direct_mcode_evidence_based_concise"]
            )
            models_list = models.split(",") if models else ["deepseek-coder"]

            optimize_result = optimizer.execute(
                trials_data=trials_data,
                prompts=prompts_list,
                models=models_list,
                max_combinations=5,  # Default from original command
                cv_folds=cv_folds,
                output_config=None,  # Could add option later
                run_inter_rater_reliability=inter_rater,
            )

            if optimize_result.success:
                console.print("[green]✅ Optimization completed[/green]")
                if optimize_result.metadata:
                    combinations = optimize_result.metadata.get("total_combinations_tested", 0)
                    best_score = optimize_result.metadata.get("best_score", 0)
                    best_combo = optimize_result.metadata.get("best_combination")
                    console.print(f"[green]📊 Tested {combinations} combinations[/green]")
                    if best_combo:
                        console.print(
                            f"[green]🏆 Best: {best_combo['model']} + {best_combo['prompt']} (CV score: {best_score:.3f})[/green]"
                        )
                pipeline_results["optimize"] = optimize_result
            else:
                console.print(f"[red]❌ Optimization failed: {optimize_result.error_message}[/red]")
                raise typer.Exit(1)

        # Save final results if output file specified
        if output_file and pipeline_results.get("summarize") and pipeline_results["summarize"].data:
            console.print(f"[blue]💾 Saving results to: {output_file}[/blue]")
            with open(output_file, "w") as f:
                for trial in pipeline_results["summarize"].data:
                    import json

                    json.dump(trial, f, ensure_ascii=False)
                    f.write("\n")
            console.print("[green]✅ Results saved[/green]")

        console.print(
            "[bold green]🎉 Trial pipeline execution completed successfully![/bold green]"
        )

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Pipeline execution failed: {e}[/red]")
        logger.exception("Trial pipeline error")
        raise typer.Exit(1)
