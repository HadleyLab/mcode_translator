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


@app.command("extract-trial")
def extract_trial(
    trial_id: str = typer.Argument(..., help="Clinical trial NCT ID"),
    output_file: Optional[str] = typer.Option(None, help="Path to save extracted mCODE elements"),
    format: str = typer.Option("json", help="Output format (json, text)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Extract mCODE elements from clinical trial data.

    Uses the TrialExtractor to identify and extract mCODE-relevant
    information from clinical trial protocol sections.
    """
    console.print(f"[bold blue]🔬 Extracting mCODE elements from trial: {trial_id}[/bold blue]")

    try:
        # Import required components
        from workflows.trial_extractor import TrialExtractor

        if verbose:
            console.print(f"[blue]📋 Output format: {format}[/blue]")
            console.print(f"[blue]💾 Output file: {output_file or 'none'}[/blue]")

        # Fetch trial data
        console.print("[blue]🔍 Fetching trial data...[/blue]")
        from workflows.trials_fetcher import TrialsFetcherWorkflow
        from config.heysol_config import get_config
        config = get_config()
        fetcher = TrialsFetcherWorkflow(config)
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

        # Extract mCODE elements
        console.print("[blue]🔬 Extracting mCODE elements...[/blue]")
        extractor = TrialExtractor()
        mcode_elements = extractor.extract_trial_mcode_elements(trial_data)

        console.print(f"[green]✅ Extracted {len(mcode_elements)} mCODE element categories[/green]")

        # Display results based on format
        if format == "json":
            import json
            output_data = {
                "trial_id": trial_id,
                "mcode_elements": mcode_elements,
                "extraction_metadata": {
                    "total_categories": len(mcode_elements),
                    "categories": list(mcode_elements.keys())
                }
            }

            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"[green]💾 mCODE elements saved to: {output_file}[/green]")
            else:
                console.print(json.dumps(output_data, indent=2))

        elif format == "text":
            console.print(f"\n[bold cyan]📋 mCODE Elements for Trial {trial_id}:[/bold cyan]")
            for category, elements in mcode_elements.items():
                console.print(f"\n[cyan]{category}:[/cyan]")
                if isinstance(elements, list):
                    for i, element in enumerate(elements, 1):
                        if isinstance(element, dict):
                            display = element.get('display', str(element))
                            console.print(f"  {i}. {display}")
                        else:
                            console.print(f"  {i}. {element}")
                elif isinstance(elements, dict):
                    display = elements.get('display', str(elements))
                    console.print(f"  {display}")
                else:
                    console.print(f"  {elements}")

            if output_file:
                with open(output_file, 'w') as f:
                    f.write(f"mCODE Elements for Trial {trial_id}:\n\n")
                    for category, elements in mcode_elements.items():
                        f.write(f"{category}:\n")
                        if isinstance(elements, list):
                            for i, element in enumerate(elements, 1):
                                if isinstance(element, dict):
                                    display = element.get('display', str(element))
                                    f.write(f"  {i}. {display}\n")
                                else:
                                    f.write(f"  {i}. {element}\n")
                        elif isinstance(elements, dict):
                            display = elements.get('display', str(elements))
                            f.write(f"  {display}\n")
                        else:
                            f.write(f"  {elements}\n")
                        f.write("\n")
                console.print(f"[green]💾 mCODE elements saved to: {output_file}[/green]")

        if verbose:
            console.print(f"\n[blue]📊 Extraction details: {len(mcode_elements)} categories extracted[/blue]")

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Extraction failed: {e}[/red]")
        logger.exception("Trial extraction error")
        raise typer.Exit(1)


@app.command("optimize-trials")
def optimize_trials(
    input_file: str = typer.Argument(..., help="Path to trial data file (NDJSON format)"),
    prompts: Optional[str] = typer.Option(None, help="Comma-separated list of prompts to test"),
    models: Optional[str] = typer.Option(None, help="Comma-separated list of models to test"),
    max_combinations: int = typer.Option(5, help="Maximum combinations to test"),
    cv_folds: int = typer.Option(3, help="Number of cross-validation folds"),
    output_config: Optional[str] = typer.Option(None, help="Path to save optimal configuration"),
    inter_rater: bool = typer.Option(False, help="Run inter-rater reliability analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Optimize mCODE translation parameters.

    Tests different combinations of prompts and models to find optimal
    settings for mCODE translation using cross-validation.
    """
    console.print("[bold blue]🔬 Optimizing mCODE translation parameters[/bold blue]")

    try:
        # Import required components
        from workflows.trials_optimizer import TrialsOptimizerWorkflow
        from config.heysol_config import get_config
        import json

        # Get configuration
        config = get_config()

        if verbose:
            console.print(f"[blue]📁 Input file: {input_file}[/blue]")
            console.print(f"[blue]📊 CV folds: {cv_folds}[/blue]")
            console.print(f"[blue]🎯 Max combinations: {max_combinations}[/blue]")
            console.print(f"[blue]🤝 Inter-rater analysis: {inter_rater}[/blue]")

        # Parse prompts and models
        prompts_list = prompts.split(",") if prompts else ["direct_mcode_evidence_based_concise"]
        models_list = models.split(",") if models else ["deepseek-coder"]

        if verbose:
            console.print(f"[blue]📝 Prompts: {', '.join(prompts_list)}[/blue]")
            console.print(f"[blue]🤖 Models: {', '.join(models_list)}[/blue]")

        # Load trial data
        console.print("[blue]📖 Loading trial data...[/blue]")
        trials_data = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trial = json.loads(line)
                        trials_data.append(trial)
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]⚠️ Skipping invalid JSON line: {e}[/yellow]")

        if not trials_data:
            console.print("[red]❌ No valid trial data found in file[/red]")
            raise typer.Exit(1)

        console.print(f"[green]✅ Loaded {len(trials_data)} trials[/green]")

        # Initialize optimizer
        optimizer = TrialsOptimizerWorkflow(config)

        # Execute optimization
        result = optimizer.execute(
            trials_data=trials_data,
            prompts=prompts_list,
            models=models_list,
            max_combinations=max_combinations,
            cv_folds=cv_folds,
            output_config=output_config,
            run_inter_rater_reliability=inter_rater
        )

        if result.success:
            console.print("[green]✅ Optimization completed[/green]")

            if result.metadata:
                total_combinations = result.metadata.get("total_combinations_tested", 0)
                successful_tests = result.metadata.get("successful_tests", 0)
                best_score = result.metadata.get("best_score", 0)
                best_combination = result.metadata.get("best_combination")

                console.print(f"[green]📊 Combinations tested: {successful_tests}/{total_combinations}[/green]")
                if best_combination:
                    console.print(f"[green]🏆 Best combination: {best_combination['model']} + {best_combination['prompt']}[/green]")
                    console.print(f"[green]📈 Best CV score: {best_score:.3f}[/green]")

                if result.metadata.get("config_saved"):
                    console.print("[green]💾 Optimal configuration saved[/green]")

                if result.metadata.get("inter_rater_reliability"):
                    console.print("[green]🤝 Inter-rater reliability analysis completed[/green]")

            # Display results summary
            if result.data and verbose:
                console.print(f"\n[blue]📋 Detailed results: {len(result.data)} combinations[/blue]")
                for i, combo_result in enumerate(result.data, 1):
                    if combo_result.get("success"):
                        combo = combo_result.get("combination", {})
                        avg_score = combo_result.get("cv_average_score", 0)
                        console.print(f"  {i}. {combo.get('model')} + {combo.get('prompt')}: {avg_score:.3f}")

        else:
            console.print(f"[red]❌ Optimization failed: {result.error_message}[/red]")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Ensure all dependencies are installed[/yellow]")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print(f"[red]❌ Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Optimization failed: {e}[/red]")
        logger.exception("Trials optimization error")
        raise typer.Exit(1)