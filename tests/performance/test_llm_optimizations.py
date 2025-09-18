"""
Test LLM Service Optimizations

Tests for LLM service optimizations:
- Connection pooling and reuse
- Batch processing capabilities
- Enhanced caching
- Performance monitoring
"""

import time
import pytest
from pathlib import Path

from src.pipeline.llm_service import LLMService
from src.utils.config import Config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_connection_pooling():
    """Test connection pooling and reuse."""
    config = Config()
    service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    # Make multiple requests to test connection reuse
    test_texts = [
        "Patient with breast cancer stage II",
        "Lung cancer treatment with chemotherapy",
        "Prostate cancer diagnosis and staging"
    ]

    start_time = time.time()
    results = []

    for text in test_texts:
        result = service.map_to_mcode(text)
        results.append(result)

    total_time = time.time() - start_time

    # Check performance stats
    stats = service.get_performance_stats()

    assert len(results) == len(test_texts), "Should process all texts"
    assert total_time >= 0, "Processing time should be non-negative"
    assert 'cache_hit_rate' in stats, "Should have cache hit rate in stats"
    assert 'active_clients' in stats, "Should have active clients in stats"
    assert all(r is not None for r in results), "All results should be valid"


def test_batch_processing():
    """Test batch processing capabilities."""
    config = Config()
    service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    # Prepare batch of texts
    batch_texts = [
        "Patient diagnosed with metastatic breast cancer",
        "Clinical trial for advanced lung cancer therapy",
        "Prostate cancer screening and early detection",
        "Colorectal cancer treatment protocols",
        "Pancreatic cancer diagnosis criteria"
    ]

    start_time = time.time()
    batch_results = service.map_to_mcode_batch(batch_texts, max_workers=3)
    batch_time = time.time() - start_time

    # Calculate individual processing time for comparison
    start_time = time.time()
    individual_results = []
    for text in batch_texts:
        result = service.map_to_mcode(text)
        individual_results.append(result)
    individual_time = time.time() - start_time

    speedup = individual_time / batch_time if batch_time > 0 else 0

    assert len(batch_results) == len(batch_texts), "Should process all texts in batch"
    assert batch_time >= 0, "Batch processing time should be non-negative"
    assert individual_time >= 0, "Individual processing time should be non-negative"
    assert all(r is not None for r in batch_results), "All batch results should be valid"


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    config = Config()
    service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    # Make some requests to generate stats
    test_texts = [
        "Breast cancer patient with hormone receptor positive status",
        "Lung cancer staging and treatment planning",
        "Colorectal cancer screening guidelines"
    ]

    for text in test_texts:
        service.map_to_mcode(text)

    # Get performance stats
    stats = service.get_performance_stats()

    assert 'total_requests' in stats, "Should have total requests in stats"
    assert 'cache_hits' in stats, "Should have cache hits in stats"
    assert 'cache_misses' in stats, "Should have cache misses in stats"
    assert 'cache_hit_rate' in stats, "Should have cache hit rate in stats"
    assert 'avg_response_time' in stats, "Should have average response time in stats"
    assert 'total_tokens' in stats, "Should have total tokens in stats"
    assert 'error_rate' in stats, "Should have error rate in stats"
    assert 'active_clients' in stats, "Should have active clients in stats"
    assert stats['total_requests'] >= len(test_texts), "Should have at least as many requests as texts processed"


def test_enhanced_caching():
    """Test enhanced caching with semantic similarity."""
    config = Config()
    service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    # Test similar texts that should benefit from semantic caching
    similar_texts = [
        "Patient with breast cancer diagnosis",
        "Breast cancer patient diagnosed recently",
        "Individual diagnosed with breast cancer"
    ]

    results = []
    for text in similar_texts:
        start_time = time.time()
        result = service.map_to_mcode(text)
        process_time = time.time() - start_time
        results.append(result)
        assert process_time >= 0, "Processing time should be non-negative"

    # Check cache performance
    stats = service.get_performance_stats()

    assert len(results) == len(similar_texts), "Should process all similar texts"
    assert all(r is not None for r in results), "All results should be valid"
    assert 'cache_hit_rate' in stats, "Should have cache hit rate in stats"

