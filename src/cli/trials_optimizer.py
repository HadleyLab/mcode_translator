#!/usr/bin/env python3
"""
Trials Optimizer - Optimize mCODE translation parameters.

A command-line interface for testing different combinations of prompts and models
to find optimal settings for mCODE translation. Results are saved to configuration
files, not to CORE Memory.
"""

import argparse
import sys
from pathlib import Path

from src.shared.cli_utils import McodeCLI
from src.utils.config import Config
from src.workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for trials optimizer."""
    parser = argparse.ArgumentParser(
        description="Optimize mCODE translation parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize with default settings
  python -m src.cli.trials_optimizer

  # Test specific prompts and models
  python -m src.cli.trials_optimizer --prompts direct_mcode_evidence_based,evidence_based_minimal --models gpt-4,claude-3

  # Limit combinations and save results
  python -m src.cli.trials_optimizer --max-combinations 5 --save-config optimal_config.json

  # List available prompts and models
  python -m src.cli.trials_optimizer --list-prompts
  python -m src.cli.trials_optimizer --list-models

  # Verbose output
  python -m src.cli.trials_optimizer --verbose
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)

    # Optimizer-specific arguments
    McodeCLI.add_optimizer_args(parser)

    # Additional arguments
    parser.add_argument(
        "--prompts", help="Comma-separated list of prompt templates to test"
    )

    parser.add_argument("--models", help="Comma-separated list of LLM models to test")

    parser.add_argument(
        "--trials-file",
        help="Path to NDJSON file containing trial data for testing (optional)",
    )

    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt templates and exit",
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    return parser


def main() -> None:
    """Main entry point for trials optimizer CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Handle list commands
    workflow = TrialsOptimizerWorkflow(config)

    if args.list_prompts:
        prompts = workflow.get_available_prompts()
        print("📝 Available prompt templates:")
        for prompt in prompts:
            print(f"  • {prompt}")
        return

    if args.list_models:
        models = workflow.get_available_models()
        print("🤖 Available models:")
        for model in models:
            print(f"  • {model}")
        return

    # Load trial data for testing
    trials_data = []
    if args.trials_file:
        trials_path = Path(args.trials_file)
        if not trials_path.exists():
            print(f"❌ Trials file not found: {trials_path}")
            sys.exit(1)

        try:
            import json

            # Read NDJSON file (one JSON object per line)
            trials_data = []
            with open(trials_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        trial_data = json.loads(line)
                        trials_data.append(trial_data)

            print(f"📋 Loaded {len(trials_data)} trials for optimization testing")

        except Exception as e:
            print(f"❌ Failed to load trials data: {e}")
            sys.exit(1)
    else:
        print("⚠️  No trials file provided. Using sample data for optimization.")
        # Could load default sample trials here

    # Prepare workflow parameters
    workflow_kwargs = {"trials_data": trials_data}

    # Parse prompts and models
    if args.prompts:
        workflow_kwargs["prompts"] = [p.strip() for p in args.prompts.split(",")]

    if args.models:
        workflow_kwargs["models"] = [m.strip() for m in args.models.split(",")]

    if args.max_combinations:
        workflow_kwargs["max_combinations"] = args.max_combinations

    if args.save_config:
        workflow_kwargs["output_config"] = args.save_config

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

    # Initialize and execute workflow
    try:
        result = workflow.execute(**workflow_kwargs)

        if result.success:
            print("✅ Optimization completed successfully!")

            # Print summary
            metadata = result.metadata
            if metadata:
                combinations_tested = metadata.get("total_combinations_tested", 0)
                successful_tests = metadata.get("successful_tests", 0)
                best_score = metadata.get("best_score", 0)
                best_combo = metadata.get("best_combination")

                print(f"🧪 Combinations tested: {combinations_tested}")
                print(f"✅ Successful tests: {successful_tests}")
                print(f"🏆 Best score: {best_score:.3f}")

                if best_combo:
                    print(f"🤖 Best model: {best_combo.get('model', 'unknown')}")
                    print(f"📝 Best prompt: {best_combo.get('prompt', 'unknown')}")

                if args.save_config:
                    print(f"💾 Optimal config saved to: {args.save_config}")

            # Print detailed results
            if result.data:
                print("\n📊 Detailed Results:")
                for i, combo_result in enumerate(result.data):
                    if combo_result.get("success"):
                        combo = combo_result.get("combination", {})
                        score = combo_result.get("average_score", 0)
                        print(
                            f"  {i+1}. {combo.get('model')} + {combo.get('prompt')}: {score:.3f}"
                        )
                    else:
                        combo = combo_result.get("combination", {})
                        error = combo_result.get("error", "Unknown error")
                        print(
                            f"  {i+1}. {combo.get('model')} + {combo.get('prompt')}: FAILED - {error}"
                        )

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


if __name__ == "__main__":
    main()
