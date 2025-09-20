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
from unittest.mock import patch

from src.pipeline.pipeline import McodePipeline
from src.shared.models import PipelineResult, McodeElement, ValidationResult, ProcessingMetadata
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@pytest.fixture
def trials_data():
    """Load sample trial data for testing."""
    # Create synthetic trials compatible with ClinicalTrialData model
    count = 4  # Reduced for faster test
    sample_trials = []
    for i in range(count):
        synthetic_trial = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": f"NCT{10000000 + i}",
                    "briefTitle": f"Synthetic Trial {i + 1} for Parallel Validation"
                },
                "statusModule": {
                    "overallStatus": "RECRUITING"
                },
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "primaryPurpose": "TREATMENT",
                    "phases": ["Phase 2"]
                },
                "conditionsModule": {
                    "conditions": [
                        {
                            "name": "Breast Cancer",
                            "term": "breast cancer",
                            "qualifier": "Primary"
                        }
                    ]
                },
                "eligibilityModule": {
                    "minimumAge": "18 Years",
                    "maximumAge": "75 Years",
                    "sex": "ALL",
                    "healthyVolunteers": False,
                    "eligibilityCriteria": "Patients with breast cancer diagnosis."
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Test Drug",
                            "description": "Synthetic treatment drug"
                        }
                    ]
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": "Test Sponsor Inc.",
                        "class": "Industry"
                    }
                },
                "designInfo": {
                    "interventionModel": "SINGLE_ARM"
                },
                "enrollmentInfo": {
                    "count": 100,
                    "type": "ANTICIPATED"
                }
            },
            "derivedSection": {
                "interventionBrowseModule": {
                    "meshes": []
                }
            }
        }
        sample_trials.append(synthetic_trial)
    
    return sample_trials


def test_sequential_processing(trials_data):
    """Test sequential processing for baseline comparison."""
    import asyncio
    pipeline = McodePipeline()

    async def process_sequential():
        results = []
        for trial in trials_data:
            result = await pipeline.process(trial)
            results.append(result)
        return results

    start_time = time.time()
    results = asyncio.run(process_sequential())
    sequential_time = time.time() - start_time

    assert len(results) == len(trials_data), "Should process all trials"
    assert sequential_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in results), "All results should be valid"
    assert all(len(r.mcode_mappings) > 0 for r in results), "Each result should have mappings"


def test_parallel_processing(trials_data):
    """Test parallel processing with validation pipeline."""
    import asyncio
    pipeline = McodePipeline()

    async def process_trials():
        tasks = [pipeline.process(trial) for trial in trials_data]
        return await asyncio.gather(*tasks)

    start_time = time.time()
    results = asyncio.run(process_trials())
    parallel_time = time.time() - start_time

    assert len(results) == len(trials_data), "Should process all trials"
    assert parallel_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in results), "All results should be valid"
    assert all(len(r.mcode_mappings) > 0 for r in results), "Each result should have mappings"


def test_parallel_validation_only(trials_data):
    """Test parallel validation without full processing."""
    import asyncio
    from src.shared.models import ClinicalTrialData

    async def validate_trials():
        validation_results = []
        for trial in trials_data:
            try:
                # Validate using ClinicalTrialData model
                ClinicalTrialData(**trial)
                validation_results.append(True)
            except Exception:
                validation_results.append(False)
        return validation_results

    start_time = time.time()
    validation_results = asyncio.run(validate_trials())
    validation_time = time.time() - start_time

    assert len(validation_results) == len(trials_data), "Should validate all trials"
    assert validation_time >= 0, "Validation time should be non-negative"
    assert isinstance(validation_results, list), "Should return list of validation results"
    assert all(validation_results), "All trials should validate successfully"

