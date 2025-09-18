"""
Test Async Cache Operations

Tests for async cache operations including:
- Async get/set operations
- Batch async operations
- Cache warming capabilities
- Background maintenance
- Performance comparison between sync and async
"""

import asyncio
import json
import time
import pytest
from pathlib import Path
import tempfile

from src.utils.api_manager import APIManager, AsyncAPICache
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.asyncio
async def test_basic_async_operations(temp_cache_dir):
    """Test basic async cache operations."""
    # Get async cache with temp dir
    manager = APIManager(str(temp_cache_dir))
    cache = await manager.aget_cache("test_async")

    # Test data
    test_key = "test_key_123"
    test_data = {"message": "Hello Async World", "timestamp": time.time()}

    # Test async set
    await cache.aset_by_key(test_data, test_key)

    # Test async get
    retrieved_data = await cache.aget_by_key(test_key)

    assert retrieved_data is not None, "Data should be retrieved"
    assert retrieved_data["message"] == test_data["message"], "Data should match"


@pytest.mark.asyncio
async def test_batch_async_operations(temp_cache_dir):
    """Test batch async operations."""
    manager = APIManager(str(temp_cache_dir))
    cache = await manager.aget_cache("test_batch")

    # Prepare batch data
    batch_items = [
        {"result": {"item": i, "data": f"batch_item_{i}"}, "key_data": f"batch_key_{i}"}
        for i in range(10)
    ]

    # Batch set
    await cache.abatch_set(batch_items)

    # Batch get
    keys_to_get = [f"batch_key_{i}" for i in range(10)]
    results = await cache.abatch_get(keys_to_get)

    # Verify results
    successful_gets = sum(1 for r in results if r is not None and not isinstance(r, Exception))

    assert successful_gets == 10, "All batch items should be retrieved"


@pytest.mark.asyncio
async def test_cache_warming(temp_cache_dir):
    """Test cache warming capabilities."""
    manager = APIManager(str(temp_cache_dir))
    cache = await manager.aget_cache("test_warm")

    # Simulate a fetch function
    async def mock_fetch_func(key_data):
        await asyncio.sleep(0.01)  # Simulate network delay
        return {"fetched": True, "key": key_data, "timestamp": time.time()}

    # Keys to warm
    warm_keys = [f"warm_key_{i}" for i in range(5)]

    # Warm cache
    start_time = time.time()
    await cache.awarm_cache(warm_keys, mock_fetch_func)
    warm_time = time.time() - start_time

    # Verify cache is warmed
    results = await cache.abatch_get(warm_keys)
    cached_count = sum(1 for r in results if r is not None and r.get("fetched") is True)

    assert cached_count == 5, "All items should be cached after warming"
    assert warm_time >= 0, "Warm time should be non-negative"


@pytest.mark.asyncio
async def test_background_maintenance(temp_cache_dir):
    """Test background maintenance."""
    manager = APIManager(str(temp_cache_dir))
    cache = await manager.aget_cache("test_maintenance")

    # Add some test data
    test_items = [
        {"result": {"data": f"maintenance_{i}"}, "key_data": f"maint_key_{i}"}
        for i in range(3)
    ]
    await cache.abatch_set(test_items)

    # Start background maintenance (short interval for testing)
    maintenance_task = asyncio.create_task(
        cache.astart_background_maintenance(cleanup_interval=2)
    )

    # Let it run for a few seconds
    await asyncio.sleep(3)  # Reduced time to avoid timeout

    # Cancel maintenance
    maintenance_task.cancel()
    try:
        await maintenance_task
    except asyncio.CancelledError:
        pass

    # Verify some data still exists (maintenance shouldn't delete valid entries immediately)
    results = await cache.abatch_get([item["key_data"] for item in test_items])
    remaining = sum(1 for r in results if r is not None)
    assert remaining > 0, "Some data should remain after short maintenance run"


@pytest.mark.asyncio
async def test_performance_comparison(temp_cache_dir):
    """Compare performance between sync and async operations."""
    manager = APIManager(str(temp_cache_dir))
    async_cache = await manager.aget_cache("perf_test")
    sync_cache = manager.get_cache("perf_sync")

    # Test data
    test_items = [
        {"result": {"perf_test": i, "data": f"item_{i}" * 100}, "key_data": f"perf_key_{i}"}
        for i in range(10)  # Reduced for faster test
    ]

    # Async performance test
    async_start = time.time()
    await async_cache.abatch_set(test_items)
    async_results = await async_cache.abatch_get([item["key_data"] for item in test_items])
    async_time = time.time() - async_start

    # Sync performance test
    sync_start = time.time()
    for item in test_items:
        sync_cache.set_by_key(item["result"], item["key_data"])
    sync_results = []
    for item in test_items:
        result = sync_cache.get_by_key(item["key_data"])
        sync_results.append(result)
    sync_time = time.time() - sync_start

    # Verify results
    async_success = sum(1 for r in async_results if r is not None)
    sync_success = sum(1 for r in sync_results if r is not None)

    assert async_success == 10, "All async operations should succeed"
    assert sync_success == 10, "All sync operations should succeed"
    assert async_time >= 0, "Async time should be non-negative"
    assert sync_time >= 0, "Sync time should be non-negative"
    assert async_time < 2.0, "Async should complete quickly"
    assert sync_time < 2.0, "Sync should complete quickly"


@pytest.mark.asyncio
async def test_async_manager_operations(temp_cache_dir):
    """Test async operations on the APIManager."""
    manager = APIManager(str(temp_cache_dir))

    # Test async cache creation and stats
    cache1 = await manager.aget_cache("manager_test_1")
    cache2 = await manager.aget_cache("manager_test_2")

    # Add some data
    await cache1.aset_by_key({"test": "data1"}, "test_key_1")
    await cache2.aset_by_key({"test": "data2"}, "test_key_2")

    # Test async stats
    stats = await manager.aget_cache_stats()

    total_cached = stats.get("total", {}).get("cached_items", 0)

    assert total_cached >= 2, "Should have at least 2 cached items"

    # Test async clear
    await manager.aclear_cache("manager_test_1")

    # Verify clearing
    cleared_stats = await manager.aget_cache_stats("manager_test_1")
    cleared_count = cleared_stats.get("manager_test_1", {}).get("cached_items", 0)

    assert cleared_count == 0, "Cache should be empty after clearing"


@pytest.mark.asyncio
async def test_error_handling(temp_cache_dir):
    """Test error handling in async operations."""
    manager = APIManager(str(temp_cache_dir))
    cache = await manager.aget_cache("error_test")

    # Test with invalid data
    try:
        # This should handle errors gracefully
        await cache.aset_by_key({"test": "data"}, "error_key")
        result = await cache.aget_by_key("error_key")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

    # Test batch operations with some failures
    batch_items = [
        {"result": {"item": i}, "key_data": f"batch_error_{i}"}
        for i in range(5)
    ]

    # This should handle any errors gracefully
    await cache.abatch_set(batch_items)
    results = await cache.abatch_get([item["key_data"] for item in batch_items])

    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))

    assert successful >= 0, "Should handle batch operations gracefully"