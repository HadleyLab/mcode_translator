#!/usr/bin/env python3
"""
Test LLM Service Optimizations

A test script to demonstrate and verify the LLM service optimizations:
- Connection pooling and reuse
- Batch processing capabilities
- Enhanced caching
- Performance monitoring
"""

import time
from pathlib import Path

from src.pipeline.llm_service import LLMService
from src.utils.config import Config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_connection_pooling():
    """Test connection pooling and reuse."""
    print("🔄 Testing connection pooling...")

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

    for i, text in enumerate(test_texts):
        print(f"  Processing text {i+1}/3...")
        result = service.map_to_mcode(text)
        results.append(result)

    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.2f}s")
    # Check performance stats
    stats = service.get_performance_stats()
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Active clients: {stats['active_clients']}")

    return results


def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n🔄 Testing batch processing...")

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

    print(f"  Processing {len(batch_texts)} texts in batch...")

    start_time = time.time()
    batch_results = service.map_to_mcode_batch(batch_texts, max_workers=3)
    batch_time = time.time() - start_time

    print(f"  Batch processing time: {batch_time:.2f}s")
    # Calculate individual processing time for comparison
    start_time = time.time()
    individual_results = []
    for text in batch_texts:
        result = service.map_to_mcode(text)
        individual_results.append(result)
    individual_time = time.time() - start_time

    print(f"  Individual processing time: {individual_time:.2f}s")
    speedup = individual_time / batch_time if batch_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    return batch_results


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n📊 Testing performance monitoring...")

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

    # Get and display performance stats
    stats = service.get_performance_stats()

    print("  Performance Statistics:")
    print(f"    Total requests: {stats['total_requests']}")
    print(f"    Cache hits: {stats['cache_hits']}")
    print(f"    Cache misses: {stats['cache_misses']}")
    print(f"    Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"    Average response time: {stats['avg_response_time']:.2f}s")
    print(f"    Total tokens used: {stats['total_tokens']}")
    print(f"    Error rate: {stats['error_rate']:.1%}")
    print(f"    Active cached clients: {stats['active_clients']}")
    print(f"    Oldest client age: {stats['oldest_client_age']:.1f}s")

    return stats


def test_enhanced_caching():
    """Test enhanced caching with semantic similarity."""
    print("\n💾 Testing enhanced caching...")

    config = Config()
    service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    # Test similar texts that should benefit from semantic caching
    similar_texts = [
        "Patient with breast cancer diagnosis",
        "Breast cancer patient diagnosed recently",
        "Individual diagnosed with breast cancer"
    ]

    print("  Processing similar texts to test semantic caching...")

    results = []
    for i, text in enumerate(similar_texts):
        print(f"  Text {i+1}: Processing...")
        start_time = time.time()
        result = service.map_to_mcode(text)
        process_time = time.time() - start_time
        results.append(result)
        print(f"    Processing time: {process_time:.2f}s")
    # Check cache performance
    stats = service.get_performance_stats()
    print(f"  Final cache hit rate: {stats['cache_hit_rate']:.1%}")

    return results


def main():
    """Run all LLM optimization tests."""
    print("🚀 LLM Service Optimization Tests")
    print("=" * 50)

    try:
        # Test connection pooling
        test_connection_pooling()

        # Test batch processing
        test_batch_processing()

        # Test performance monitoring
        test_performance_monitoring()

        # Test enhanced caching
        test_enhanced_caching()

        print("\n" + "=" * 50)
        print("✅ All LLM optimization tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())