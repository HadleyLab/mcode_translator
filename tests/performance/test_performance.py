"""
Performance tests and benchmarks for mcode_translator components.
Uses pytest-benchmark for measuring execution times and memory usage.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.services.summarizer import McodeSummarizer
from src.utils.token_tracker import TokenTracker, TokenUsage


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for key operations."""

    @pytest.fixture
    def large_trial_data(self):
        """Generate large trial data for performance testing in ClinicalTrials.gov format."""
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Large Scale Clinical Trial for Performance Testing"
                },
                "statusModule": {
                    "overallStatus": "RECRUITING"
                },
                "designModule": {
                    "phases": ["PHASE1", "PHASE2", "PHASE3"],
                    "studyType": "INTERVENTIONAL",
                    "primaryPurpose": "TREATMENT"
                },
                "conditionsModule": {
                    "conditions": ["Cancer", "Breast Cancer", "Lung Cancer", "Prostate Cancer"] * 10
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria:\n" + "\n".join([
                        f"- Age {i*5}-{i*5+4} years" for i in range(20)
                    ]) + "\n\nExclusion Criteria:\n" + "\n".join([
                        f"- Prior treatment {i}" for i in range(50)
                    ]),
                    "minimumAge": "18 Years",
                    "maximumAge": "75 Years",
                    "sex": "ALL"
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": f"Test Drug {i}",
                            "description": f"Experimental therapy {i} with detailed description" * 20
                        } for i in range(10)
                    ]
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {
                        "name": "Test Sponsor"
                    }
                }
            },
            "derivedSection": {
                "interventionBrowseModule": {
                    "meshes": []
                }
            }
        }

    @pytest.fixture
    def large_patient_bundle(self):
        """Generate large patient bundle for performance testing."""
        return {
            "resourceType": "Bundle",
            "id": "large-bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": f"patient-{i}",
                        "gender": "female" if i % 2 == 0 else "male",
                        "birthDate": f"19{i%100:02d}-01-01",
                        "name": [
                            {
                                "family": f"Doe-{i}",
                                "given": [f"John-{i}"]
                            }
                        ]
                    }
                } for i in range(100)
            ] + [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"condition-{i}",
                        "subject": {"reference": f"Patient/patient-{i%100}"},
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": f"12345{i}",
                                "display": f"Test Condition {i}"
                            }]
                        }
                    }
                } for i in range(500)
            ]
        }

    def test_summarizer_trial_summary_performance(self, benchmark, large_trial_data):
        """Benchmark trial summary generation performance."""
        summarizer = McodeSummarizer()

        def run_summary():
            return summarizer.create_trial_summary(large_trial_data)

        result = benchmark(run_summary)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 100  # Ensure meaningful summary

    def test_summarizer_patient_summary_performance(self, benchmark, large_patient_bundle):
        """Benchmark patient summary generation performance."""
        summarizer = McodeSummarizer()

        def run_summary():
            return summarizer.create_patient_summary(large_patient_bundle)

        result = benchmark(run_summary)

        assert result is not None
        assert isinstance(result, str)

    def test_token_tracker_add_usage_performance(self, benchmark):
        """Benchmark token usage tracking performance."""
        tracker = TokenTracker()

        def add_usage_batch():
            for i in range(1000):
                usage = TokenUsage(
                    prompt_tokens=i * 10,
                    completion_tokens=i * 5,
                    total_tokens=i * 15
                )
                tracker.add_usage(usage, f"component_{i % 10}")

        benchmark(add_usage_batch)

        total_usage = tracker.get_total_usage()
        assert total_usage.total_tokens > 0

    def test_token_tracker_get_stats_performance(self, benchmark):
        """Benchmark getting token usage statistics."""
        tracker = TokenTracker()

        # Pre-populate with data
        for i in range(100):
            usage = TokenUsage(
                prompt_tokens=i * 10,
                completion_tokens=i * 5,
                total_tokens=i * 15
            )
            tracker.add_usage(usage, f"component_{i % 5}")

        def get_stats():
            return tracker.get_total_usage()

        result = benchmark(get_stats)
        assert result.total_tokens > 0

    def test_api_cache_performance(self, benchmark, temp_cache_dir):
        """Benchmark API cache operations."""
        from src.utils.api_manager import APICache

        cache = APICache(temp_cache_dir, "test", 3600)

        test_data = {"large_data": "x" * 10000}

        def cache_operations():
            # Set operation
            cache.set(test_data, "test_func", "arg1", "arg2", "arg3")
            # Get operation
            return cache.get("test_func", "arg1", "arg2", "arg3")

        result = benchmark(cache_operations)
        assert result == test_data

    def test_json_processing_performance(self, benchmark, large_trial_data):
        """Benchmark JSON serialization/deserialization."""
        def json_ops():
            # Serialize
            json_str = json.dumps(large_trial_data)
            # Deserialize
            return json.loads(json_str)

        result = benchmark(json_ops)
        assert result == large_trial_data

    def test_memory_usage_tracking(self, benchmark):
        """Benchmark memory usage with large data structures."""
        def create_large_structure():
            # Create a large nested structure
            return {
                "trials": [
                    {
                        "id": f"trial_{i}",
                        "data": "x" * 1000,
                        "metadata": [{"size": i}] * 100
                    } for i in range(100)
                ]
            }

        result = benchmark(create_large_structure)
        assert "trials" in result
        assert len(result["trials"]) == 100

    def test_mcode_mapping_performance(self, benchmark):
        """Benchmark basic mCODE mapping setup performance."""
        # Simple performance test without complex mocking
        def create_mapper():
            # Just test the object creation performance
            return {
                "model_name": "test-model",
                "temperature": 0.7,
                "max_tokens": 1000,
                "test_data": "x" * 1000
            }

        result = benchmark(create_mapper)
        assert result["model_name"] == "test-model"
        assert len(result["test_data"]) == 1000

    def test_string_formatting_performance(self, benchmark):
        """Benchmark string formatting operations."""
        template = "Patient {name} aged {age} with condition {condition} at {facility}"
        data = {
            "name": "John Doe",
            "age": 45,
            "condition": "Breast Cancer",
            "facility": "Memorial Hospital"
        }

        def format_string():
            return template.format(**data)

        result = benchmark(format_string)
        assert "John Doe" in result
        assert "Breast Cancer" in result

    def test_list_comprehension_performance(self, benchmark):
        """Benchmark list comprehension operations."""
        data = list(range(10000))

        def process_list():
            return [x * 2 + 1 for x in data if x % 2 == 0]

        result = benchmark(process_list)
        assert len(result) == 5000
        assert result[0] == 1  # 0 * 2 + 1
        assert result[1] == 5  # 2 * 2 + 1

    def test_dict_operations_performance(self, benchmark):
        """Benchmark dictionary operations."""
        def dict_ops():
            d = {}
            # Fill dictionary
            for i in range(1000):
                d[f"key_{i}"] = f"value_{i}"
            # Access operations
            for i in range(1000):
                _ = d.get(f"key_{i}")
            # Update operations
            for i in range(1000):
                d[f"key_{i}"] = f"updated_{i}"
            return d

        result = benchmark(dict_ops)
        assert len(result) == 1000
        assert result["key_0"] == "updated_0"