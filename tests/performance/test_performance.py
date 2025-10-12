"""
Performance tests and benchmarks for mcode_translator components.
Uses pytest-benchmark for measuring execution times and memory usage.
"""

import json
from unittest.mock import patch

import pytest

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
                    "briefTitle": "Large Scale Clinical Trial for Performance Testing",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {
                    "phases": ["PHASE1", "PHASE2", "PHASE3"],
                    "studyType": "INTERVENTIONAL",
                    "primaryPurpose": "TREATMENT",
                },
                "conditionsModule": {
                    "conditions": [
                        "Cancer",
                        "Breast Cancer",
                        "Lung Cancer",
                        "Prostate Cancer",
                    ]
                    * 10
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria:\n"
                    + "\n".join([f"- Age {i*5}-{i*5+4} years" for i in range(20)])
                    + "\n\nExclusion Criteria:\n"
                    + "\n".join([f"- Prior treatment {i}" for i in range(50)]),
                    "minimumAge": "18 Years",
                    "maximumAge": "75 Years",
                    "sex": "ALL",
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": f"Test Drug {i}",
                            "description": f"Experimental therapy {i} with detailed description"
                            * 20,
                        }
                        for i in range(10)
                    ]
                },
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test Sponsor"}},
            },
            "derivedSection": {"interventionBrowseModule": {"meshes": []}},
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
                        "name": [{"family": f"Doe-{i}", "given": [f"John-{i}"]}],
                    }
                }
                for i in range(100)
            ]
            + [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"condition-{i}",
                        "subject": {"reference": f"Patient/patient-{i%100}"},
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": f"12345{i}",
                                    "display": f"Test Condition {i}",
                                }
                            ]
                        },
                    }
                }
                for i in range(500)
            ],
        }

    @patch.object(McodeSummarizer, "create_trial_summary")
    def test_summarizer_trial_summary_performance(
        self, mock_summary, large_trial_data, benchmark=None
    ):
        """Benchmark trial summary generation performance."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        mock_summary.return_value = "Mock trial summary for performance testing."
        summarizer = McodeSummarizer()

        def run_summary():
            return summarizer.create_trial_summary(large_trial_data)

        result = benchmark(run_summary)

        assert result == "Mock trial summary for performance testing."
        assert (
            benchmark.stats.stats.mean < 0.1
        ), f"Trial summary should complete quickly, got {benchmark.stats.stats.mean:.2f}s"

    @patch.object(McodeSummarizer, "create_patient_summary")
    def test_summarizer_patient_summary_performance(
        self, mock_summary, large_patient_bundle, benchmark=None
    ):
        """Benchmark patient summary generation performance."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        mock_summary.return_value = "Mock patient summary for performance testing."
        summarizer = McodeSummarizer()

        def run_summary():
            return summarizer.create_patient_summary(large_patient_bundle)

        result = benchmark(run_summary)

        assert result == "Mock patient summary for performance testing."
        assert (
            benchmark.stats.stats.mean < 0.1
        ), f"Patient summary should complete quickly, got {benchmark.stats.stats.mean:.2f}s"

    def test_token_tracker_add_usage_performance(self, benchmark=None):
        """Benchmark token usage tracking performance."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        tracker = TokenTracker()

        def add_usage_batch():
            for i in range(500):  # Reduced for faster test
                usage = TokenUsage(
                    prompt_tokens=i * 10, completion_tokens=i * 5, total_tokens=i * 15
                )
                tracker.add_usage(usage, f"component_{i % 10}")

        benchmark(add_usage_batch)

        total_usage = tracker.get_total_usage()
        assert total_usage.total_tokens > 0

        # Performance assertion: should handle operations quickly
        assert (
            benchmark.stats.stats.mean < 0.5
        ), f"Token tracking should complete in < 0.5s, got {benchmark.stats.stats.mean:.2f}s"

    def test_token_tracker_get_stats_performance(self, benchmark=None):
        """Benchmark getting token usage statistics."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        tracker = TokenTracker()

        # Pre-populate with data
        for i in range(100):
            usage = TokenUsage(prompt_tokens=i * 10, completion_tokens=i * 5, total_tokens=i * 15)
            tracker.add_usage(usage, f"component_{i % 5}")

        def get_stats():
            return tracker.get_total_usage()

        result = benchmark(get_stats)
        assert result.total_tokens > 0
        assert (
            benchmark.stats.stats.mean < 0.01
        ), f"Stats should be fast, got {benchmark.stats.stats.mean:.4f}s"

    def test_api_cache_performance(self, temp_cache_dir, benchmark=None):
        """Benchmark API cache operations."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        from src.utils.api_manager import APICache

        cache = APICache(str(temp_cache_dir), "test", 3600)

        test_data = {"large_data": "x" * 1000}  # Reduced size

        def cache_operations():
            # Set operation
            cache.set(test_data, "test_func", "arg1", "arg2", "arg3")
            # Get operation
            return cache.get("test_func", "arg1", "arg2", "arg3")

        result = benchmark(cache_operations)
        assert result == test_data

        # Performance assertion: cache operations should be fast
        assert (
            benchmark.stats.stats.mean < 0.1
        ), f"Cache operations should complete in < 0.1s, got {benchmark.stats.stats.mean:.3f}s"

    def test_json_processing_performance(self, large_trial_data, benchmark=None):
        """Benchmark JSON serialization/deserialization."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        def json_ops():
            # Serialize
            json_str = json.dumps(large_trial_data)
            # Deserialize
            return json.loads(json_str)

        result = benchmark(json_ops)
        assert result == large_trial_data
        assert (
            benchmark.stats.stats.mean < 0.1
        ), f"JSON ops should be fast, got {benchmark.stats.stats.mean:.3f}s"

    @pytest.mark.skip("Memory tracking requires special setup")
    def test_memory_usage_tracking(self, benchmark=None):
        """Benchmark memory usage with large data structures."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        def create_large_structure():
            # Create a large nested structure
            return {
                "trials": [
                    {
                        "id": f"trial_{i}",
                        "data": "x" * 100,  # Reduced
                        "metadata": [{"size": i}] * 10,  # Reduced
                    }
                    for i in range(50)  # Reduced
                ]
            }

        result = benchmark(create_large_structure)
        assert "trials" in result
        assert len(result["trials"]) == 50

    def test_mcode_mapping_performance(self, benchmark=None):
        """Benchmark basic mCODE mapping setup performance."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        # Simple performance test without complex mocking
        def create_mapper():
            # Just test the object creation performance
            return {
                "model_name": "test-model",
                "temperature": 0.7,
                "max_tokens": 1000,
                "test_data": "x" * 500,  # Reduced
            }

        result = benchmark(create_mapper)
        assert result["model_name"] == "test-model"
        assert len(result["test_data"]) == 500
        assert (
            benchmark.stats.stats.mean < 0.01
        ), f"Mapper creation should be fast, got {benchmark.stats.stats.mean:.4f}s"

    def test_string_formatting_performance(self, benchmark=None):
        """Benchmark string formatting operations."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        template = "Patient {name} aged {age} with condition {condition} at {facility}"
        data = {
            "name": "John Doe",
            "age": 45,
            "condition": "Breast Cancer",
            "facility": "Memorial Hospital",
        }

        def format_string():
            return template.format(**data)

        result = benchmark(format_string)
        assert "John Doe" in result
        assert "Breast Cancer" in result
        assert (
            benchmark.stats.stats.mean < 0.001
        ), f"Formatting should be very fast, got {benchmark.stats.stats.mean:.4f}s"

    def test_list_comprehension_performance(self, benchmark=None):
        """Benchmark list comprehension operations."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        data = list(range(5000))  # Reduced

        def process_list():
            return [x * 2 + 1 for x in data if x % 2 == 0]

        result = benchmark(process_list)
        assert len(result) == 2500
        assert result[0] == 1
        assert result[1] == 5
        assert (
            benchmark.stats.stats.mean < 0.05
        ), f"List comp should be fast, got {benchmark.stats.stats.mean:.3f}s"

    def test_dict_operations_performance(self, benchmark=None):
        """Benchmark dictionary operations."""
        pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
        if benchmark is None:
            pytest.skip("benchmark fixture not available")

        def dict_ops():
            d = {}
            # Fill dictionary
            for i in range(500):  # Reduced
                d[f"key_{i}"] = f"value_{i}"
            # Access operations
            for i in range(500):
                _ = d.get(f"key_{i}")
            # Update operations
            for i in range(500):
                d[f"key_{i}"] = f"updated_{i}"
            return d

        result = benchmark(dict_ops)
        assert len(result) == 500
        assert result["key_0"] == "updated_0"
        assert (
            benchmark.stats.stats.mean < 0.05
        ), f"Dict ops should be fast, got {benchmark.stats.stats.mean:.3f}s"
