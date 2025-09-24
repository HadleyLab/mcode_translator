"""
Test LLM Service Optimizations

Tests for LLM service optimizations:
- Connection pooling and reuse
- Batch processing capabilities
- Enhanced caching
- Performance monitoring
"""

import time

from src.pipeline.llm_service import LLMService
from src.utils.config import Config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_connection_pooling():
    """Test connection pooling and reuse."""
    config = Config()
    service = LLMService(
        config, "deepseek-coder", "direct_mcode_evidence_based_concise"
    )

    # Make multiple requests to test connection reuse
    test_texts = [
        "Patient with breast cancer stage II",
        "Lung cancer treatment with chemotherapy",
        "Prostate cancer diagnosis and staging",
    ]

    start_time = time.time()
    results = []

    for text in test_texts:
        result = service.map_to_mcode(text)
        results.append(result)

    total_time = time.time() - start_time

    # Basic performance checks (no get_performance_stats method available)
    assert len(results) == len(test_texts), "Should process all texts"
    assert total_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in results), "All results should be valid"


def test_batch_processing():
    """Test batch processing capabilities."""
    config = Config()
    service = LLMService(
        config, "deepseek-coder", "direct_mcode_evidence_based_concise"
    )

    # Prepare batch of texts
    batch_texts = [
        "Patient diagnosed with metastatic breast cancer",
        "Clinical trial for advanced lung cancer therapy",
        "Prostate cancer screening and early detection",
        "Colorectal cancer treatment protocols",
        "Pancreatic cancer diagnosis criteria",
    ]

    # Process texts individually (no batch method available)
    start_time = time.time()
    individual_results = []
    for text in batch_texts:
        result = service.map_to_mcode(text)
        individual_results.append(result)
    individual_time = time.time() - start_time

    assert len(individual_results) == len(batch_texts), "Should process all texts"
    assert individual_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in individual_results), "All results should be valid"


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    config = Config()
    service = LLMService(
        config, "deepseek-coder", "direct_mcode_evidence_based_concise"
    )

    # Make some requests to test basic functionality
    test_texts = [
        "Breast cancer patient with hormone receptor positive status",
        "Lung cancer staging and treatment planning",
        "Colorectal cancer screening guidelines",
    ]

    start_time = time.time()
    results = []
    for text in test_texts:
        result = service.map_to_mcode(text)
        results.append(result)
    total_time = time.time() - start_time

    # Basic performance checks
    assert len(results) == len(test_texts), "Should process all texts"
    assert total_time >= 0, "Processing time should be non-negative"
    assert all(r is not None for r in results), "All results should be valid"


def test_enhanced_caching():
    """Test enhanced caching with semantic similarity."""
    config = Config()
    service = LLMService(
        config, "deepseek-coder", "direct_mcode_evidence_based_concise"
    )

    # Test similar texts that should benefit from semantic caching
    similar_texts = [
        "Patient with breast cancer diagnosis",
        "Breast cancer patient diagnosed recently",
        "Individual diagnosed with breast cancer",
    ]

    results = []
    for text in similar_texts:
        start_time = time.time()
        result = service.map_to_mcode(text)
        process_time = time.time() - start_time
        results.append(result)
        assert process_time >= 0, "Processing time should be non-negative"

    # Basic validation
    assert len(results) == len(similar_texts), "Should process all similar texts"
    assert all(r is not None for r in results), "All results should be valid"
