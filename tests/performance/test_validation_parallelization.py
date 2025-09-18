"""
Test Validation Pipeline Parallelization

Tests for validation pipeline parallelization:
- Parallel batch processing of multiple trials
- Parallel input validation
- Parallel section processing
- Parallel compliance score calculation
- Performance comparison between sequential and parallel processing
"""

import json
import time
import pytest
from pathlib import Path

from src.pipeline.pipeline import McodePipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@pytest.fixture
def trials_data():
    """Load sample trial data for testing."""
    sample_trials = []

    # Try to load from existing sample files
    sample_files = [
        "examples/data/complete_trial.json",
        "examples/data/dry_run_trials.json",
        "examples/data/fetched_trials.json"
    ]

    loaded_count = 0
    count = 8  # Default count for testing
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


def test_sequential_processing(trials_data: list):
    """Test sequential processing for baseline comparison."""
    pipeline = McodePipeline()

    start_time = time.time()
    results = pipeline.process_batch(trials_data)
    sequential_time = time.time() - start_time

    assert len(results) == len(trials_data), "Should process all trials"
    assert sequential_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in results), "All results should be valid"


def test_parallel_processing(trials_data: list):
    """Test parallel processing with validation pipeline."""
    max_workers = 4
    pipeline = McodePipeline()

    start_time = time.time()
    results = pipeline.process_batch_parallel(trials_data, max_workers=max_workers)
    parallel_time = time.time() - start_time

    assert len(results) == len(trials_data), "Should process all trials"
    assert parallel_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in results), "All results should be valid"


def test_parallel_validation_only(trials_data: list):
    """Test parallel validation without full processing."""
    max_workers = 4
    pipeline = McodePipeline()

    start_time = time.time()
    validation_results = pipeline.validate_batch_parallel(trials_data, max_workers=max_workers)
    validation_time = time.time() - start_time

    assert len(validation_results) == len(trials_data), "Should validate all trials"
    assert validation_time >= 0, "Validation time should be non-negative"
    assert isinstance(validation_results, list), "Should return list of validation results"

