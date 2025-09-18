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

from src.shared.cli_utils import McodeCLI
from src.utils.config import Config
from src.workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow


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

    # Required arguments for file-based optimization
    parser.add_argument(
        "--trials-file",
        help="Path to NDJSON file containing trial data for testing"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        help="Number of cross validation folds"
    )

    # Additional arguments
    parser.add_argument(
        "--prompts", help="Comma-separated list of prompt templates to test"
    )

    parser.add_argument("--models", help="Comma-separated list of LLM models to test")

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

    # Handle list commands (no CORE Memory needed for these)
    workflow = TrialsOptimizerWorkflow(config, memory_storage=False)

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

    # Validate required arguments for optimization
    if not args.trials_file:
        print("❌ --trials-file is required for optimization")
        sys.exit(1)

    if args.cv_folds is None:
        print("❌ --cv-folds is required for optimization")
        sys.exit(1)

    # Load trial data from file (required)
    trials_path = Path(args.trials_file)
    if not trials_path.exists():
        print(f"❌ Trials file not found: {trials_path}")
        sys.exit(1)

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
            for line in file_content.split('\n'):
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
    workflow_kwargs = {
        "trials_data": trials_data,
        "cv_folds": args.cv_folds
    }

    # Parse prompts and models - use defaults if not specified
    if args.prompts:
        workflow_kwargs["prompts"] = [p.strip() for p in args.prompts.split(",")]
    else:
        workflow_kwargs["prompts"] = ["direct_mcode_evidence_based_concise"]

    if args.models:
        workflow_kwargs["models"] = [m.strip() for m in args.models.split(",")]
    else:
        workflow_kwargs["models"] = ["deepseek-coder"]

    if args.max_combinations:
        workflow_kwargs["max_combinations"] = args.max_combinations

    if args.save_config:
        workflow_kwargs["output_config"] = args.save_config

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

    # Initialize and execute workflow (disable CORE Memory - optimizer saves to local files only)
    try:
        workflow = TrialsOptimizerWorkflow(config, memory_storage=False)
        result = workflow.execute(**workflow_kwargs)

        if result.success:
            print("✅ Optimization completed successfully!")

            # Print summary
            metadata = result.metadata
            if metadata:
                combinations_tested = metadata.get("total_combinations_tested", 0)
                cv_folds = metadata.get("cv_folds", 3)
                successful_tests = metadata.get("successful_tests", 0)
                best_score = metadata.get("best_score", 0)
                best_combo = metadata.get("best_combination")

                print(f"🧪 Combinations tested: {combinations_tested}")
                print(f"🔀 Cross validation folds: {cv_folds}")
                print(f"✅ Successful tests: {successful_tests}")
                print(f"🏆 Best CV score: {best_score:.3f}")

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
                        cv_score = combo_result.get("cv_average_score", 0)
                        cv_std = combo_result.get("cv_std_score", 0)
                        folds = combo_result.get("cv_folds", 3)
                        print(
                            f"  {i+1}. {combo.get('model')} + {combo.get('prompt')}: {cv_score:.3f} ± {cv_std:.3f} ({folds}-fold CV)"
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
