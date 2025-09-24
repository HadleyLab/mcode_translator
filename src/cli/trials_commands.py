"""
Trial-related CLI commands for mCODE Translator.

This module contains all Click commands related to clinical trials:
- fetch_trials
- process_trials
- summarize_trials
- optimize_trials
"""

import click
from types import SimpleNamespace

from . import trials_fetcher, trials_processor, trials_summarizer, trials_optimizer


@click.group()
def trials():
    """Clinical trials commands."""
    pass


@trials.command()
@click.option(
    "--condition", help='Medical condition to search for (e.g., "breast cancer")'
)
@click.option("--nct-id", help="Specific NCT ID to fetch (e.g., NCT12345678)")
@click.option("--nct-ids", help="Comma-separated list of NCT IDs to fetch")
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of trials to fetch (default: 10)",
)
@click.option("--out", "output_file", help="Output file for trial data (NDJSON format)")
@click.option("--model", help="LLM model to use")
@click.option(
    "--prompt",
    default="direct_mcode_evidence_based_concise",
    help="Prompt template to use",
)
@click.option("--workers", type=int, default=0, help="Number of concurrent workers")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
def fetch(
    condition,
    nct_id,
    nct_ids,
    limit,
    output_file,
    model,
    prompt,
    workers,
    log_level,
    config,
):
    """Fetch clinical trials."""
    if not any([condition, nct_id, nct_ids]):
        raise click.UsageError(
            "Must specify one of: --condition, --nct-id, or --nct-ids"
        )

    # Create args object for compatibility
    args = SimpleNamespace(
        condition=condition,
        nct_id=nct_id,
        nct_ids=nct_ids,
        limit=limit,
        output_file=output_file,
        model=model,
        prompt=prompt,
        workers=workers,
        log_level=log_level,
        config=config,
    )
    trials_fetcher.main(args)


@trials.command()
@click.argument("input_file")
@click.option("--out", "output_file", help="Output file for processed mCODE data")
@click.option("--model", help="LLM model to use")
@click.option(
    "--prompt",
    default="direct_mcode_evidence_based_concise",
    help="Prompt template to use",
)
@click.option("--workers", type=int, default=0, help="Number of concurrent workers")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.option(
    "--store-in-memory/--no-store-in-memory",
    default=False,
    help="Store results in CORE memory",
)
def process(
    input_file,
    output_file,
    model,
    prompt,
    workers,
    log_level,
    config,
    store_in_memory,
):
    """Process clinical trials to mCODE."""

    class Args:
        def __init__(self):
            self.input_file = input_file
            self.output_file = output_file
            self.model = model
            self.prompt = prompt
            self.workers = workers
            self.log_level = log_level
            self.config = config
            self.store_in_memory = store_in_memory

    args = Args()
    trials_processor.main(args)


@trials.command()
@click.option("--in", "input_file", help="Input file containing mCODE trial data")
@click.option("--out", "output_file", help="Output file for summarized data")
@click.option("--model", help="LLM model to use")
@click.option(
    "--prompt",
    default="direct_mcode_evidence_based_concise",
    help="Prompt template to use",
)
@click.option("--workers", type=int, default=0, help="Number of concurrent workers")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.option(
    "--store-in-memory/--no-store-in-memory",
    default=False,
    help="Store results in CORE memory",
)
def summarize(
    input_file,
    output_file,
    model,
    prompt,
    workers,
    log_level,
    config,
    store_in_memory,
):
    """Summarize mCODE trials."""
    if not input_file:
        raise click.UsageError("Must specify input file with --in")

    args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        model=model,
        prompt=prompt,
        workers=workers,
        log_level=log_level,
        config=config,
        store_in_memory=store_in_memory,
    )
    trials_summarizer.main(args)


@trials.command()
@click.option("--trials-file", help="Path to trials data file for optimization")
@click.option(
    "--cv-folds",
    type=int,
    default=3,
    help="Number of cross-validation folds (default: 3)",
)
@click.option("--list-models", is_flag=True, help="List available AI models and exit")
@click.option(
    "--list-prompts", is_flag=True, help="List available prompt templates and exit"
)
@click.option("--prompts", help="Comma-separated list of prompt templates to test")
@click.option("--models", help="Comma-separated list of LLM models to test")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
def optimize(
    trials_file,
    cv_folds,
    list_models,
    list_prompts,
    prompts,
    models,
    log_level,
    config,
):
    """Optimize trial processing parameters."""

    args = SimpleNamespace(
        trials_file=trials_file,
        cv_folds=cv_folds,
        list_models=list_models,
        list_prompts=list_prompts,
        prompts=prompts,
        models=models,
        log_level=log_level,
        config=config,
    )
    trials_optimizer.main(args)