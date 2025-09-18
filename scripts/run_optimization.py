#!/usr/bin/env python3
"""
Enhanced Script to run comprehensive pairwise cross-validation optimization for mCODE prompts.
Tests multiple models and prompts across extensive trial data.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path so src imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.pairwise_cross_validation import PairwiseCrossValidator
from src.utils.logging_config import setup_logging


def load_trials_from_ndjson(file_path: str, max_trials: Optional[int] = None) -> List[dict]:
    """Load trials from NDJSON file."""
    trials = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_trials and i >= max_trials:
                break
            line = line.strip()
            if line:
                try:
                    trial = json.loads(line)
                    trials.append(trial)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping malformed trial at line {i+1}: {e}")
    return trials


def main():
    """Run comprehensive pairwise cross-validation optimization."""
    parser = argparse.ArgumentParser(description="Run comprehensive mCODE optimization")
    parser.add_argument("--trials-file", default="raw_trials.ndjson",
                       help="Path to trials file (NDJSON format)")
    parser.add_argument("--max-trials", type=int, default=50,
                       help="Maximum number of trials to use")
    parser.add_argument("--max-comparisons", type=int, default=100,
                       help="Maximum number of pairwise comparisons")
    parser.add_argument("--models", nargs="*", default=None,
                       help="Specific models to test (default: all available)")
    parser.add_argument("--prompts", nargs="*", default=None,
                       help="Specific prompts to test (default: all available)")
    parser.add_argument("--cv-folds", type=int, default=3,
                       help="Number of cross-validation folds")
    parser.add_argument("--workers", type=int, default=5,
                       help="Number of concurrent workers")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Setup centralized logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    print("🚀 Starting Comprehensive mCODE Prompt Optimization")
    print(f"📊 Configuration: {args.max_trials} trials, {args.cv_folds} folds, {args.workers} workers")

    # Initialize validator
    validator = PairwiseCrossValidator()
    validator.initialize()

    # Get available prompts and models
    all_prompts = validator.get_available_prompts()
    all_models = validator.get_available_models()

    # Filter to specified models/prompts or use all
    prompts = args.prompts if args.prompts else all_prompts
    models = args.models if args.models else all_models

    print(f"📋 Testing prompts: {len(prompts)}")
    for prompt in prompts:
        print(f"   - {prompt}")

    print(f"🤖 Testing models: {len(models)}")
    for model in models:
        print(f"   - {model}")

    # Load trial data
    trials_file = Path(project_root) / args.trials_file
    if not trials_file.exists():
        print(f"❌ Trials file not found: {trials_file}")
        return

    print(f"📂 Loading trials from: {trials_file}")
    trials = load_trials_from_ndjson(str(trials_file), args.max_trials)
    print(f"📊 Loaded {len(trials)} trials for optimization")

    if len(trials) == 0:
        print("❌ No valid trials loaded!")
        return

    # Generate pairwise tasks
    print(f"🔄 Generating pairwise comparison tasks...")
    tasks = validator.generate_pairwise_tasks(
        prompts=prompts,
        models=models,
        trials=trials,
        max_comparisons=args.max_comparisons
    )

    print(f"✅ Generated {len(tasks)} pairwise comparison tasks")
    print(f"📊 Total combinations: {len(prompts)} × {len(models)} × {len(trials)} = {len(prompts) * len(models) * len(trials)}")
    print(f"🎯 Limited to: {args.max_comparisons} comparisons")

    # Run validation
    print("🏃 Running pairwise validation...")
    validator.run_pairwise_validation(tasks, max_workers=args.workers)

    # Analyze results
    print("📊 Analyzing pairwise results...")
    analysis = validator.analyze_pairwise_results()

    # Save results
    print("💾 Saving comprehensive results...")
    validator.save_results(detailed_report=True)

    # Print summary
    print("\n" + "="*60)
    print("📋 OPTIMIZATION SUMMARY")
    print("="*60)
    validator.print_summary()

    # Additional comprehensive analysis
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE ANALYSIS")
    print("="*60)

    if analysis:
        # Best performing combinations
        best_combinations = analysis.get('best_combinations', [])
        if best_combinations:
            print("🏆 TOP PERFORMING COMBINATIONS:")
            for i, combo in enumerate(best_combinations[:5], 1):
                score = combo.get('score', 0)
                model = combo.get('model', 'unknown')
                prompt = combo.get('prompt', 'unknown')
                print(f"   {i}. {model} + {prompt}: {score:.3f}")

        # Model comparison
        model_comparison = analysis.get('model_comparison', {})
        if model_comparison:
            print("\n🤖 MODEL PERFORMANCE COMPARISON:")
            for model, stats in model_comparison.items():
                avg_score = stats.get('average_score', 0)
                consistency = stats.get('consistency', 0)
                print(f"   • {model}: {avg_score:.3f} (consistency: {consistency:.3f})")

        # Prompt comparison
        prompt_comparison = analysis.get('prompt_comparison', {})
        if prompt_comparison:
            print("\n📝 PROMPT PERFORMANCE COMPARISON:")
            for prompt, stats in prompt_comparison.items():
                avg_score = stats.get('average_score', 0)
                print(f"   • {prompt}: {avg_score:.3f}")

    validator.shutdown()
    print("\n✅ Comprehensive optimization completed successfully!")
    print(f"📊 Results saved to optimization results directory")


if __name__ == "__main__":
    main()