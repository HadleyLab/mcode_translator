#!/usr/bin/env python3
"""
Script to run pairwise cross-validation optimization for mCODE prompts.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path so src imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.pairwise_cross_validation import PairwiseCrossValidator
from src.utils.logging_config import setup_logging


def main():
    """Run pairwise cross-validation optimization."""
    # Setup centralized logging
    setup_logging("INFO")
    print("🚀 Starting mCODE Prompt Optimization")

    # Initialize validator
    validator = PairwiseCrossValidator()
    validator.initialize()

    # Get available prompts and models
    prompts = validator.get_available_prompts()
    models = validator.get_available_models()

    print(f"📋 Available prompts: {len(prompts)}")
    for prompt in prompts:
        print(f"   - {prompt}")

    print(f"🤖 Available models: {len(models)}")
    for model in models:
        print(f"   - {model}")

    # Load trial data
    trials_file = Path(__file__).parent.parent / "examples" / "data" / "optimization_trials.json"
    trials = validator.load_trials(str(trials_file))

    print(f"📊 Loaded {len(trials)} trials for optimization")

    # Generate pairwise tasks (limit for testing)
    max_comparisons = 20  # Limit to avoid too many API calls
    tasks = validator.generate_pairwise_tasks(
        prompts=prompts,
        models=models,
        trials=trials,
        max_comparisons=max_comparisons
    )

    print(f"🔄 Generated {len(tasks)} pairwise comparison tasks")

    # Run validation
    validator.run_pairwise_validation(tasks)

    # Analyze results
    analysis = validator.analyze_pairwise_results()

    # Save results
    validator.save_results(detailed_report=True)

    # Print summary
    validator.print_summary()

    validator.shutdown()
    print("✅ Optimization completed successfully!")


if __name__ == "__main__":
    main()