#!/usr/bin/env python3
"""
Test Validation Pipeline Parallelization

A test script to demonstrate and verify the validation pipeline parallelization:
- Parallel batch processing of multiple trials
- Parallel input validation
- Parallel section processing
- Parallel compliance score calculation
- Performance comparison between sequential and parallel processing
"""

import json
import time
from pathlib import Path

from src.pipeline.pipeline import McodePipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_sample_trials(count: int = 5) -> list:
    """Load sample trial data for testing."""
    sample_trials = []

    # Try to load from existing sample files
    sample_files = [
        "examples/data/complete_trial.json",
        "examples/data/dry_run_trials.json",
        "examples/data/fetched_trials.json"
    ]

    loaded_count = 0
    for sample_file in sample_files:
        if Path(sample_file).exists() and loaded_count < count:
            try:
                with open(sample_file, 'r') as f:
                    if sample_file.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            sample_trials.extend(data[:count - loaded_count])
                        else:
                            sample_trials.append(data)
                        loaded_count = len(sample_trials)
            except Exception as e:
                logger.warning(f"Failed to load {sample_file}: {e}")

    # If we don't have enough samples, create synthetic ones
    while len(sample_trials) < count:
        synthetic_trial = {
            "nct_id": f"NCT{10000000 + len(sample_trials)}",
            "title": f"Synthetic Trial {len(sample_trials) + 1}",
            "conditions": ["Breast Cancer", "Solid Tumor"],
            "interventions": [{"type": "Drug", "name": "Test Drug"}],
            "phases": ["Phase 2"],
            "study_type": "Interventional",
            "description": "A synthetic clinical trial for testing parallel validation pipeline.",
            "eligibility_criteria": "Patients with cancer diagnosis."
        }
        sample_trials.append(synthetic_trial)

    return sample_trials[:count]


def test_sequential_processing(trials_data: list) -> tuple:
    """Test sequential processing for baseline comparison."""
    print("🔄 Testing sequential processing...")

    pipeline = McodePipeline()

    start_time = time.time()
    results = pipeline.process_batch(trials_data)
    sequential_time = time.time() - start_time

    print(".2f"    print(f"  Results: {len(results)} trials processed")

    return results, sequential_time


def test_parallel_processing(trials_data: list, max_workers: int = 4) -> tuple:
    """Test parallel processing with validation pipeline."""
    print(f"\n🔄 Testing parallel processing with {max_workers} workers...")

    pipeline = McodePipeline()

    start_time = time.time()
    results = pipeline.process_batch_parallel(trials_data, max_workers=max_workers)
    parallel_time = time.time() - start_time

    print(f"  Sequential processing time: {sequential_time:.2f}s")
    print(f"  Results: {len(results)} trials processed")
    return results, parallel_time


def test_parallel_validation_only(trials_data: list, max_workers: int = 4) -> tuple:
    """Test parallel validation without full processing."""
    print(f"\n🔍 Testing parallel validation only with {max_workers} workers...")

    pipeline = McodePipeline()

    start_time = time.time()
    validation_results = pipeline.validate_batch_parallel(trials_data, max_workers=max_workers)
    validation_time = time.time() - start_time

    valid_count = sum(validation_results)
    print(".2f"    print(f"  Results: {valid_count}/{len(validation_results)} trials valid")

    return validation_results, validation_time


def compare_performance(sequential_time: float, parallel_time: float, validation_time: float, trial_count: int):
    """Compare and display performance metrics."""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 60)

    print("
⏱️  TIMING RESULTS:"    print(".2f"    print(".2f"    print(".2f"
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(".2f"
        efficiency = (speedup / 4) * 100  # Assuming 4 workers
        print(".1f"
    print("
📈 PER-TRIAL METRICS:"    print(".3f"    print(".3f"    print(".3f"
    print("
✅ VALIDATION RESULTS:"    print(f"  • Parallel validation: {validation_time:.3f}s per trial")
    print(".1f"
def analyze_results(results: list):
    """Analyze processing results."""
    print("\n" + "=" * 60)
    print("📋 RESULTS ANALYSIS")
    print("=" * 60)

    successful = sum(1 for r in results if r and not r.error)
    failed = sum(1 for r in results if r and r.error)
    total_elements = sum(len(r.mcode_mappings) if r else 0 for r in results)
    avg_compliance = sum(r.validation_results.compliance_score if r else 0 for r in results) / len(results) if results else 0

    print("
🔢 PROCESSING SUMMARY:"    print(f"  • Successful trials: {successful}")
    print(f"  • Failed trials: {failed}")
    print(f"  • Total mCODE elements: {total_elements}")
    print(".2f"
    if successful > 0:
        print(".1f"
    print("
📊 ELEMENT DISTRIBUTION:"    element_types = {}
    for result in results:
        if result and result.mcode_mappings:
            for elem in result.mcode_mappings:
                elem_type = elem.element_type
                element_types[elem_type] = element_types.get(elem_type, 0) + 1

    for elem_type, count in sorted(element_types.items()):
        print(f"  • {elem_type}: {count}")


def main():
    """Run all validation parallelization tests."""
    print("🚀 Validation Pipeline Parallelization Tests")
    print("=" * 60)

    # Load sample data
    trial_count = 8  # Good number for demonstrating parallelization
    print(f"📥 Loading {trial_count} sample trials...")

    trials_data = load_sample_trials(trial_count)
    print(f"✅ Loaded {len(trials_data)} trials")

    try:
        # Test sequential processing (baseline)
        seq_results, seq_time = test_sequential_processing(trials_data)

        # Test parallel processing
        par_results, par_time = test_parallel_processing(trials_data, max_workers=4)

        # Test parallel validation only
        val_results, val_time = test_parallel_validation_only(trials_data, max_workers=4)

        # Compare performance
        compare_performance(seq_time, par_time, val_time, trial_count)

        # Analyze results
        analyze_results(par_results)

        print("\n" + "=" * 60)
        print("✅ All validation parallelization tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())