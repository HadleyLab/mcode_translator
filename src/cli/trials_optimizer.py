#!/usr/bin/env python3
"""
Trials Optimizer - Optimize mCODE translation parameters.

A command-line interface for testing different combinations of prompts and models
to find optimal settings for mCODE translation. Results are saved to local
configuration files only - no CORE Memory storage required.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.shared.cli_utils import McodeCLI
from src.shared.models import WorkflowResult
from src.workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow


class SummaryHandler(FileSystemEventHandler):
    """Event handler for real-time summary of optimization runs."""

    def __init__(self, workflow: "TrialsOptimizerWorkflow") -> None:
        self.workflow = workflow

    def on_created(self, event: Any) -> None:
        if event.is_directory or not event.src_path.endswith(".json"):
            return
        print("\n📊 Real-time summary updated:")
        self.workflow.summarize_benchmark_validations()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for trials optimizer."""
    parser = argparse.ArgumentParser(
        description="Optimize mCODE translation parameters using existing trial data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
   # Optimize using existing trials file
   python -m src.cli.trials_optimizer --trials-file breast_cancer_trials.ndjson --cv-folds 3

   # Test specific prompts and models with cross validation
   python -m src.cli.trials_optimizer --trials-file trials.ndjson --cv-folds 5 --prompts direct_mcode_evidence_based_concise,direct_mcode_evidence_based --models deepseek-coder,gpt-4

   # Save optimal configuration
   python -m src.cli.trials_optimizer --trials-file trials.ndjson --cv-folds 3 --save-config optimal_config.json

   # List available prompts and models
   python -m src.cli.trials_optimizer --list-prompts
   python -m src.cli.trials_optimizer --list-models

   # Verbose output
   python -m src.cli.trials_optimizer --trials-file trials.ndjson --cv-folds 3 --verbose
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)

    # Optimizer-specific arguments
    McodeCLI.add_optimizer_args(parser)

    # Concurrency arguments
    McodeCLI.add_concurrency_args(parser)

    parser.add_argument(
        "--async-queue",
        action="store_true",
        help="Use async task queue instead of threaded (experimental)",
    )

    # Required arguments for file-based optimization
    parser.add_argument(
        "--trials-file", help="Path to NDJSON file containing trial data for testing"
    )

    parser.add_argument("--cv-folds", type=int, help="Number of cross validation folds")

    # Additional arguments
    parser.add_argument(
        "--prompts", help="Comma-separated list of prompt templates to test"
    )

    parser.add_argument(
        "--models",
        type=lambda s: [item.strip() for item in s.split(",")],
        help="Comma-separated list of LLM models to test",
    )

    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt templates and exit",
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    parser.add_argument(
        "--save-mcode-elements",
        help="Save all processed mCODE elements to JSON file for analysis (includes biological analysis report)",
    )

    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for trials optimizer CLI."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Handle list commands (no CORE Memory needed for these)
    if args.list_prompts:
        workflow = TrialsOptimizerWorkflow(config, memory_storage=False)
        prompts = workflow.get_available_prompts()
        print("📝 Available prompt templates:")
        for prompt in prompts:
            print(f"  • {prompt}")
        return

    if args.list_models:
        workflow = TrialsOptimizerWorkflow(config, memory_storage=False)
        models = workflow.get_available_models()
        print("🤖 Available models:")
        for model in models:
            print(f"  • {model}")
        return

    # Validate required arguments for optimization
    if not args.trials_file:
        print("❌ --trials-file is required for optimization")
        sys.exit(1)

    if args.cv_folds is None:
        print("❌ --cv-folds is required for optimization")
        sys.exit(1)

    # Create workflow for optimization (disable CORE Memory - optimizer saves to local files only)
    workflow = TrialsOptimizerWorkflow(config, memory_storage=False)

    # Load trial data from file (required)
    trials_path = Path(args.trials_file)
    if not trials_path.exists():
        print(f"❌ Trials file not found: {trials_path}")
        sys.exit(1)

    trials_data = []  # Initialize to avoid UnboundLocalError
    try:
        import json

        with open(trials_path, "r", encoding="utf-8") as f:
            file_content = f.read().strip()

        # Try to parse as single JSON object first
        try:
            json_data = json.loads(file_content)
            if isinstance(json_data, dict) and "successful_trials" in json_data:
                # Format: {"summary": {...}, "successful_trials": [...]}
                trials_data = json_data["successful_trials"]
            elif isinstance(json_data, dict) and "studies" in json_data:
                # Format: {"studies": [...]}
                trials_data = json_data["studies"]
            elif isinstance(json_data, list):
                # Format: [...]
                trials_data = json_data
            else:
                raise ValueError("Unknown JSON structure")
        except json.JSONDecodeError:
            # Try to parse as NDJSON (one JSON object per line)
            trials_data = []
            for line in file_content.split("\n"):
                line = line.strip()
                if line:
                    trial_data = json.loads(line)
                    trials_data.append(trial_data)

        if not trials_data:
            raise ValueError("No trial data found in file")

        print(f"📋 Loaded {len(trials_data)} trials from file for optimization testing")

    except Exception as e:
        print(f"❌ Failed to load trials data: {e}")
        sys.exit(1)

    # Prepare workflow parameters
    workflow_kwargs = {"trials_data": trials_data, "cv_folds": args.cv_folds}

    # Parse prompts and models - use defaults if not specified
    if args.prompts:
        workflow_kwargs["prompts"] = [p.strip() for p in args.prompts.split(",")]
    else:
        workflow_kwargs["prompts"] = ["direct_mcode_evidence_based_concise"]

    if args.models:
        workflow_kwargs["models"] = args.models
    else:
        workflow_kwargs["models"] = ["deepseek-coder"]

    if args.max_combinations:
        workflow_kwargs["max_combinations"] = args.max_combinations

    if args.save_config:
        workflow_kwargs["output_config"] = args.save_config

    if getattr(args, "save_mcode_elements", None):
        workflow_kwargs["save_mcode_elements"] = args.save_mcode_elements

    # Pass CLI arguments for concurrency configuration
    workflow_kwargs["cli_args"] = args

    # Validate combinations if specified
    if args.prompts and args.models:
        prompts_list = workflow_kwargs["prompts"]
        models_list = workflow_kwargs["models"]

        invalid_combinations = []
        for prompt in prompts_list:
            for model in models_list:
                if not workflow.validate_combination(prompt, model):
                    invalid_combinations.append(f"{model} + {prompt}")

        if invalid_combinations:
            print("❌ Invalid prompt×model combinations:")
            for combo in invalid_combinations:
                print(f"  • {combo}")
            sys.exit(1)

    # Setup real-time summary
    runs_dir = "optimization_runs"
    Path(runs_dir).mkdir(exist_ok=True)
    event_handler = SummaryHandler(workflow)
    observer = Observer()
    observer.schedule(event_handler, runs_dir, recursive=False)
    observer.start()
    print("👀 Watching for new run results in real-time...")

    # Initialize and execute workflow
    try:
        result: WorkflowResult = workflow.execute(**workflow_kwargs)

        if result.success:
            print("✅ Optimization completed successfully!")

            # Print final summary
            print("\n📊 FINAL SUMMARY:")
            workflow.summarize_benchmark_validations()

        else:
            print(f"❌ Optimization failed: {result.error_message}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
