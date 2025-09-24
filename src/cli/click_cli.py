"""
Click-based CLI for mCODE Translator.

This module provides a Click-based command-line interface for testing purposes.
Click provides better testing utilities compared to argparse.
"""

import click
import sys
import time
from pathlib import Path

# Import our modules
from . import (
    patients_fetcher,
    patients_processor,
    patients_summarizer,
    trials_fetcher,
    trials_optimizer,
    trials_processor,
    trials_summarizer,
)
from .data_downloader import (
    download_archives,
    list_available_archives,
    parse_archive_list,
    get_default_archives_config,
)
from .test_runner import (
    run_unit_tests,
    run_integration_tests,
    run_performance_tests,
    run_all_tests,
    run_coverage_report,
    run_linting,
)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """mCODE Translator CLI - Click-based interface for testing."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
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
@click.pass_context
def fetch_trials(
    ctx,
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
    class Args:
        def __init__(self):
            self.condition = condition
            self.nct_id = nct_id
            self.nct_ids = nct_ids
            self.limit = limit
            self.output_file = output_file
            self.model = model
            self.prompt = prompt
            self.workers = workers
            self.log_level = log_level
            self.config = config

    args = Args()
    trials_fetcher.main(args)


@cli.command()
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
@click.pass_context
def process_trials(
    ctx,
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


@cli.command()
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
@click.pass_context
def summarize_trials(
    ctx,
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
    trials_summarizer.main(args)


@cli.command()
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
@click.pass_context
def optimize_trials(
    ctx,
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

    class Args:
        def __init__(self):
            self.trials_file = trials_file
            self.cv_folds = cv_folds
            self.list_models = list_models
            self.list_prompts = list_prompts
            self.prompts = prompts
            self.models = models
            self.log_level = log_level
            self.config = config

    args = Args()
    trials_optimizer.main(args)


@cli.command()
@click.option(
    "--archive", help="Patient archive identifier (e.g., breast_cancer_10_years)"
)
@click.option("--patient-id", help="Specific patient ID to fetch")
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of patients to fetch (default: 10)",
)
@click.option(
    "--list-archives", is_flag=True, help="List available patient archives and exit"
)
@click.option(
    "--out", "output_file", help="Output file for patient data (NDJSON format)"
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.pass_context
def fetch_patients(
    ctx, archive, patient_id, limit, list_archives, output_file, log_level, config
):
    """Fetch synthetic patients."""

    class Args:
        def __init__(self):
            self.archive = archive
            self.patient_id = patient_id
            self.limit = limit
            self.list_archives = list_archives
            self.output_file = output_file
            self.log_level = log_level
            self.config = config

    args = Args()
    patients_fetcher.main(args)


@cli.command()
@click.option("--in", "input_file", help="Input file containing patient data")
@click.option("--out", "output_file", help="Output file for processed mCODE data")
@click.option(
    "--trials",
    help="Path to NDJSON file containing trial data for eligibility filtering",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.option(
    "--store-in-memory/--no-store-in-memory",
    default=False,
    help="Store results in CORE memory",
)
@click.pass_context
def process_patients(
    ctx, input_file, output_file, trials, log_level, config, store_in_memory
):
    """Process patients to mCODE."""
    if not input_file:
        raise click.UsageError("Must specify input file with --in")

    class Args:
        def __init__(self):
            self.input_file = input_file
            self.output_file = output_file
            self.trials = trials
            self.log_level = log_level
            self.config = config
            self.store_in_memory = store_in_memory

    args = Args()
    patients_processor.main(args)


@cli.command()
@click.option("--in", "input_file", help="Input file containing mCODE patient data")
@click.option("--out", "output_file", help="Output file for summarized data")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run summarization without storing results in CORE Memory",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.option(
    "--store-in-memory/--no-store-in-memory",
    default=False,
    help="Store results in CORE memory",
)
@click.pass_context
def summarize_patients(
    ctx, input_file, output_file, dry_run, log_level, config, store_in_memory
):
    """Summarize mCODE patients."""
    if not input_file:
        raise click.UsageError("Must specify input file with --in")

    class Args:
        def __init__(self):
            self.input_file = input_file
            self.output_file = output_file
            self.dry_run = dry_run
            self.log_level = log_level
            self.config = config
            self.store_in_memory = store_in_memory

    args = Args()
    patients_summarizer.main(args)


@cli.command()
@click.option(
    "--archives",
    help="Comma-separated list of archive names (e.g., breast_cancer_10_years,mixed_cancer_lifetime)",
)
@click.option(
    "--all", "download_all", is_flag=True, help="Download all available archives"
)
@click.option(
    "--list", "list_archives", is_flag=True, help="List available archives and exit"
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Number of concurrent download workers (default: 4)",
)
@click.option("--force", is_flag=True, help="Force re-download of existing archives")
@click.option(
    "--output-dir",
    default="data/synthetic_patients",
    help="Output directory for downloaded archives",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.pass_context
def download_data(
    ctx,
    archives,
    download_all,
    list_archives,
    workers,
    force,
    output_dir,
    log_level,
    config,
):
    """Download data archives."""
    if list_archives:
        list_available_archives()
        return

    # Determine which archives to download
    if download_all:
        archives_config = get_default_archives_config()
        archive_desc = "all available archives"
    else:
        if not archives:
            raise click.UsageError("Must specify --archives or --all")
        archives_config = parse_archive_list(archives)
        archive_count = sum(len(durations) for durations in archives_config.values())
        archive_desc = f"{archive_count} specified archives"

    if not archives_config:
        click.echo("❌ No valid archives specified for download")
        sys.exit(1)

    click.echo("🚀 mCODE Data Downloader")
    click.echo("=" * 30)
    click.echo(f"📦 Downloading: {archive_desc}")
    click.echo(f"🔄 Workers: {workers}")
    click.echo(f"💾 Output: {output_dir}")
    click.echo(f"🔄 Force: {'Yes' if force else 'No'}")
    click.echo()

    try:
        # Perform concurrent download
        start_time = time.time()

        downloaded_paths = download_archives(
            archives_config=archives_config,
            output_dir=output_dir,
            workers=workers,
            force=force,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Report results
        successful = len(downloaded_paths)
        total_requested = sum(len(durations) for durations in archives_config.values())

        click.echo("\n" + "=" * 50)
        click.echo("✅ Download Summary:")
        click.echo(f"   📊 Archives requested: {total_requested}")
        click.echo(f"   ✅ Successfully processed: {successful}")
        click.echo(f"   ⏱️  Total time: {duration:.2f} seconds")
        if total_requested > 0:
            avg_time_per_archive = duration / total_requested
            click.echo(
                f"   📈 Avg time per archive: {avg_time_per_archive:.2f} seconds"
            )

        click.echo("\n📁 Downloaded Archives:")
        for archive_name, path in downloaded_paths.items():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            click.echo(f"   ✅ {archive_name}: {size_mb:.1f} MB")

        if successful < total_requested:
            click.echo(
                f"\n⚠️  {total_requested - successful} archives were skipped (already exist or failed)"
            )
            click.echo("   Use --force to re-download existing archives")

    except KeyboardInterrupt:
        click.echo("\n⏹️  Download cancelled by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Download failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument(
    "suite",
    type=click.Choice(
        ["unit", "integration", "performance", "all", "coverage", "lint"]
    ),
)
@click.option(
    "--live", is_flag=True, help="Run integration tests with live data sources"
)
@click.option("--coverage", is_flag=True, help="Generate coverage reports")
@click.option(
    "--benchmark", is_flag=True, help="Run only benchmark tests in performance suite"
)
@click.option("--fail-fast", is_flag=True, help="Stop on first failure")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.pass_context
def run_tests(ctx, suite, live, coverage, benchmark, fail_fast, log_level, config):
    """Run tests."""
    # Check if we're in the right directory
    if not Path("src").exists():
        click.echo("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Create args object for compatibility
    class Args:
        def __init__(self):
            self.live = live
            self.coverage = coverage
            self.benchmark = benchmark
            self.fail_fast = fail_fast

    args = Args()

    # Run the appropriate test suite
    success = False

    if suite == "unit":
        success = run_unit_tests(args)
    elif suite == "integration":
        success = run_integration_tests(args)
    elif suite == "performance":
        success = run_performance_tests(args)
    elif suite == "all":
        success = run_all_tests(args)
    elif suite == "coverage":
        success = run_coverage_report(args)
    elif suite == "lint":
        success = run_linting(args)

    # Exit with appropriate code
    if success:
        click.echo("✅ All tests passed!")
        sys.exit(0)
    else:
        click.echo("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    cli()
