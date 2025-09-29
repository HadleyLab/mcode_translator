"""
Trial-related CLI commands for mCODE Translator.

This module contains all commands related to clinical trials:
- fetch_trials
- process_trials
- summarize_trials
- optimize_trials

Migrated from Click to Typer with full type hints and heysol_api_client integration.
"""

from types import SimpleNamespace
from typing import Optional

import typer

from ..config.heysol_config import get_config
from . import (trials_fetcher, trials_optimizer, trials_processor,
               trials_summarizer)

# Create the trials Typer app
app = typer.Typer()


@app.command()
def fetch(
    condition: Optional[str] = typer.Option(
        None, help='Medical condition to search for (e.g., "breast cancer")'
    ),
    nct_id: Optional[str] = typer.Option(
        None, help="Specific NCT ID to fetch (e.g., NCT12345678)"
    ),
    nct_ids: Optional[str] = typer.Option(
        None, help="Comma-separated list of NCT IDs to fetch"
    ),
    limit: int = typer.Option(
        10, help="Maximum number of trials to fetch (default: 10)"
    ),
    output_file: Optional[str] = typer.Option(
        None, help="Output file for trial data (NDJSON format)"
    ),
    model: Optional[str] = typer.Option(None, help="LLM model to use"),
    prompt: str = typer.Option(
        "direct_mcode_evidence_based_concise", help="Prompt template to use"
    ),
    workers: int = typer.Option(0, help="Number of concurrent workers"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    store_in_memory: bool = typer.Option(False, help="Store results in CORE memory"),
) -> None:
    """
    Fetch clinical trials.

    Must specify one of: --condition, --nct-id, or --nct-ids
    """
    if not any([condition, nct_id, nct_ids]):
        typer.echo(
            "Error: Must specify one of: --condition, --nct-id, or --nct-ids", err=True
        )
        raise typer.Exit(1)

    # Get global configuration
    config = get_config()

    # Override with provided values
    if model:
        config.mcode_default_model = model
    if workers > 0:
        config.mcode_workers = workers

    trials_fetcher.fetch_trials_direct(
        condition=condition,
        nct_id=nct_id,
        nct_ids=nct_ids,
        limit=limit,
        output_file=output_file,
        model=config.mcode_default_model,
        prompt=prompt,
        workers=config.mcode_workers,
        log_level=log_level,
        config_file=config_file,
        store_in_memory=store_in_memory,
    )


@app.command()
def process(
    input_file: str = typer.Argument(..., help="Input file containing trial data"),
    output_file: Optional[str] = typer.Option(
        None, help="Output file for processed mCODE data"
    ),
    model: Optional[str] = typer.Option(None, help="LLM model to use"),
    prompt: str = typer.Option(
        "direct_mcode_evidence_based_concise", help="Prompt template to use"
    ),
    workers: int = typer.Option(0, help="Number of concurrent workers"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    store_in_memory: bool = typer.Option(False, help="Store results in CORE memory"),
) -> None:
    """Process clinical trials to mCODE."""

    # Get global configuration
    config = get_config()

    # Override with provided values
    if model:
        config.mcode_default_model = model
    if workers > 0:
        config.mcode_workers = workers

    class Args:
        def __init__(self) -> None:
            self.input_file = input_file
            self.output_file = output_file
            self.model = config.mcode_default_model
            self.prompt = prompt
            self.workers = config.mcode_workers
            self.log_level = log_level
            self.config = config_file
            self.store_in_memory = store_in_memory

    args = Args()
    trials_processor.main(args)  # type: ignore


@app.command()
def summarize(
    input_file: str = typer.Option(
        ..., "--in", help="Input file containing mCODE trial data"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--out", help="Output file for summarized data"
    ),
    model: Optional[str] = typer.Option(None, help="LLM model to use"),
    prompt: str = typer.Option(
        "direct_mcode_evidence_based_concise", help="Prompt template to use"
    ),
    workers: int = typer.Option(0, help="Number of concurrent workers"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    store_in_memory: bool = typer.Option(False, help="Store results in CORE memory"),
) -> None:
    """
    Summarize mCODE trials.

    The input file must be specified with --in
    """
    if not input_file:
        typer.echo("Error: Must specify input file with --in", err=True)
        raise typer.Exit(1)

    # Get global configuration
    config = get_config()

    # Override with provided values
    if model:
        config.mcode_default_model = model
    if workers > 0:
        config.mcode_workers = workers

    args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        model=config.mcode_default_model,
        prompt=prompt,
        workers=config.mcode_workers,
        log_level=log_level,
        config=config_file,
        store_in_memory=store_in_memory,
    )
    trials_summarizer.main(args)  # type: ignore


@app.command()
def optimize(
    trials_file: Optional[str] = typer.Option(
        None, help="Path to trials data file for optimization"
    ),
    cv_folds: int = typer.Option(
        3, help="Number of cross-validation folds (default: 3)"
    ),
    list_models: bool = typer.Option(False, help="List available AI models and exit"),
    list_prompts: bool = typer.Option(
        False, help="List available prompt templates and exit"
    ),
    prompts: Optional[str] = typer.Option(
        None, help="Comma-separated list of prompt templates to test"
    ),
    models: Optional[str] = typer.Option(
        None, help="Comma-separated list of LLM models to test"
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
) -> None:
    """Optimize trial processing parameters."""

    # Get global configuration
    get_config()

    args = SimpleNamespace(
        trials_file=trials_file,
        cv_folds=cv_folds,
        list_models=list_models,
        list_prompts=list_prompts,
        prompts=prompts,
        models=models,
        log_level=log_level,
        config=config_file,
    )
    trials_optimizer.main(args)  # type: ignore


# For backward compatibility, expose the app as 'trials'
trials = app
