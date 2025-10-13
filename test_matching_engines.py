#!/usr/bin/env python3
"""
Test script for patient-trial matching engines.
Tests both RegexRulesEngine and LLMMatchingEngine on small subsets.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from matching.regex_engine import RegexRulesEngine
from matching.llm_engine import LLMMatchingEngine
from matching.batch_matcher import BatchMatcher
from matching.evaluator import MatchingEvaluator
from matching.data_loader import get_sample_data
from config.patterns_config import load_regex_rules


async def test_engines():
    """Test both matching engines on small sample data."""

    print("🚀 Testing Patient-Trial Matching Engines")
    print("=" * 50)

    # Load sample data
    print("📥 Loading sample data...")
    sample_data = get_sample_data(
        patients_file="selected_patients_100.ndjson",
        trials_file="selected_trials_100.ndjson",
        sample_size=3
    )

    patients = sample_data['patients']
    trials = sample_data['trials']
    pairs = sample_data['pairs']

    print(f"✅ Loaded {len(patients)} patients, {len(trials)} trials, {len(pairs)} pairs")

    # Initialize engines
    print("\n🔧 Initializing engines...")

    # Regex engine
    regex_rules = load_regex_rules()
    regex_engine = RegexRulesEngine(rules=regex_rules, cache_enabled=False)
    print("✅ RegexRulesEngine initialized")

    # LLM engine
    try:
        llm_engine = LLMMatchingEngine(
            model_name="deepseek-coder",
            prompt_name="patient_matcher",
            cache_enabled=False
        )
        print("✅ LLMMatchingEngine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM engine: {e}")
        return

    # Test individual engines
    evaluator = MatchingEvaluator()

    engines_to_test = [
        ("RegexRulesEngine", regex_engine),
        ("LLMMatchingEngine", llm_engine)
    ]

    results_by_engine = {}

    for engine_name, engine in engines_to_test:
        print(f"\n🎯 Testing {engine_name}...")
        print("-" * 30)

        # Create batch matcher
        batch_matcher = BatchMatcher(engine, max_concurrent=2)

        # Run matching
        try:
            results = await batch_matcher.match_batch(pairs)
            results_by_engine[engine_name] = results

            # Calculate scores
            scores = [evaluator.calculate_match_score(r) for r in results]

            print(f"✅ {engine_name} completed {len(results)} matches")
            print(".3f")
            print(f"   High-confidence matches (≥0.7): {sum(1 for s in scores if s >= 0.7)}")
            print(f"   Failed matches: {sum(1 for r in results if r.error)}")

            # Show sample results
            print("\n   Sample Results:")
            for i, (pair, result) in enumerate(zip(pairs[:2], results[:2])):
                patient_id = pair['patient']['entry'][0]['resource'].get('id', 'unknown')
                trial_id = pair['trial'].get('nctId', pair['trial'].get('id', 'unknown'))
                score = evaluator.calculate_match_score(result)
                status = "ERROR" if result.error else f"Score: {score:.3f}"
                print(f"     Pair {i+1}: Patient {patient_id} × Trial {trial_id} → {status}")

        except Exception as e:
            print(f"❌ {engine_name} failed: {e}")
            results_by_engine[engine_name] = []

    # Compare engines if both succeeded
    if len(results_by_engine) == 2:
        print("\n🔍 Comparing Engines...")
        print("-" * 30)

        regex_results = results_by_engine["RegexRulesEngine"]
        llm_results = results_by_engine["LLMMatchingEngine"]

        comparison = evaluator.compare_engines(
            regex_results, llm_results,
            "RegexRulesEngine", "LLMMatchingEngine"
        )

        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        # Generate reports
        print("\n📊 Detailed Reports:")
        print("-" * 30)

        for engine_name, results in results_by_engine.items():
            report = evaluator.generate_report(results, engine_name)
            print(f"\n{engine_name} Report:")
            print(report)

    print("\n🎉 Testing completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_engines())