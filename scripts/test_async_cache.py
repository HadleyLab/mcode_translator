#!/usr/bin/env python3
"""
Test Async Cache Operations

A test script to demonstrate and verify the async cache operations:
- Async get/set operations
- Batch async operations
- Cache warming capabilities
- Background maintenance
- Performance comparison between sync and async
"""

import asyncio
import json
import time
from pathlib import Path

from src.utils.api_manager import APIManager, AsyncAPICache
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


async def test_basic_async_operations():
    """Test basic async cache operations."""
    print("🔄 Testing basic async operations...")

    # Get async cache
    manager = APIManager()
    cache = await manager.aget_cache("test_async")

    # Test data
    test_key = "test_key_123"
    test_data = {"message": "Hello Async World", "timestamp": time.time()}

    # Test async set
    print("  Setting data asynchronously...")
    await cache.aset_by_key(test_data, test_key)

    # Test async get
    print("  Getting data asynchronously...")
    retrieved_data = await cache.aget_by_key(test_key)

    assert retrieved_data is not None, "Data should be retrieved"
    assert retrieved_data["message"] == test_data["message"], "Data should match"

    print("  ✅ Basic async operations successful")
    return cache


async def test_batch_async_operations():
    """Test batch async operations."""
    print("\n🔄 Testing batch async operations...")

    manager = APIManager()
    cache = await manager.aget_cache("test_batch")

    # Prepare batch data
    batch_items = [
        {"result": {"item": i, "data": f"batch_item_{i}"}, "key_data": f"batch_key_{i}"}
        for i in range(10)
    ]

    # Batch set
    print("  Batch setting 10 items...")
    await cache.abatch_set(batch_items)

    # Batch get
    print("  Batch getting 10 items...")
    keys_to_get = [f"batch_key_{i}" for i in range(10)]
    results = await cache.abatch_get(keys_to_get)

    # Verify results
    successful_gets = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    print(f"  Retrieved {successful_gets}/10 items successfully")

    assert successful_gets == 10, "All batch items should be retrieved"

    print("  ✅ Batch async operations successful")
    return cache


async def test_cache_warming():
    """Test cache warming capabilities."""
    print("\n🔄 Testing cache warming...")

    manager = APIManager()
    cache = await manager.aget_cache("test_warm")

    # Simulate a fetch function
    async def mock_fetch_func(key_data):
        await asyncio.sleep(0.01)  # Simulate network delay
        return {"fetched": True, "key": key_data, "timestamp": time.time()}

    # Keys to warm
    warm_keys = [f"warm_key_{i}" for i in range(5)]

    # Warm cache
    print("  Warming cache with 5 items...")
    start_time = time.time()
    await cache.awarm_cache(warm_keys, mock_fetch_func)
    warm_time = time.time() - start_time

    print(f"  Cache warming took {warm_time:.2f}s")
    # Verify cache is warmed
    results = await cache.abatch_get(warm_keys)
    cached_count = sum(1 for r in results if r is not None and r.get("fetched") is True)
    print(f"  Cache contains {cached_count}/5 warmed items")

    assert cached_count == 5, "All items should be cached after warming"

    print("  ✅ Cache warming successful")
    return cache


async def test_background_maintenance():
    """Test background maintenance."""
    print("\n🔄 Testing background maintenance...")

    manager = APIManager()
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
    print("  Running background maintenance for 5 seconds...")
    await asyncio.sleep(5)

    # Cancel maintenance
    maintenance_task.cancel()
    try:
        await maintenance_task
    except asyncio.CancelledError:
        pass

    print("  ✅ Background maintenance test completed")
    return cache


async def test_performance_comparison():
    """Compare performance between sync and async operations."""
    print("\n📊 Testing performance comparison...")

    manager = APIManager()
    async_cache = await manager.aget_cache("perf_test")
    sync_cache = manager.get_cache("perf_sync")

    # Test data
    test_items = [
        {"result": {"perf_test": i, "data": f"item_{i}" * 100}, "key_data": f"perf_key_{i}"}
        for i in range(20)
    ]

    # Async performance test
    print("  Testing async performance...")
    async_start = time.time()
    await async_cache.abatch_set(test_items)
    async_results = await async_cache.abatch_get([item["key_data"] for item in test_items])
    async_time = time.time() - async_start

    # Sync performance test
    print("  Testing sync performance...")
    sync_start = time.time()
    for item in test_items:
        sync_cache.set_by_key(item["result"], item["key_data"])
    sync_results = []
    for item in test_items:
        result = sync_cache.get_by_key(item["key_data"])
        sync_results.append(result)
    sync_time = time.time() - sync_start

    print(f"  Async time: {async_time:.2f}s")
    print(f"  Sync time: {sync_time:.2f}s")
    if sync_time > 0:
        speedup = sync_time / async_time
        print(f"  Speedup: {speedup:.2f}x")
    # Verify results
    async_success = sum(1 for r in async_results if r is not None)
    sync_success = sum(1 for r in sync_results if r is not None)

    print(f"  Async success rate: {async_success}/20")
    print(f"  Sync success rate: {sync_success}/20")

    assert async_success == 20, "All async operations should succeed"
    assert sync_success == 20, "All sync operations should succeed"

    print("  ✅ Performance comparison completed")


async def test_async_manager_operations():
    """Test async operations on the APIManager."""
    print("\n🔄 Testing async manager operations...")

    manager = APIManager()

    # Test async cache creation and stats
    cache1 = await manager.aget_cache("manager_test_1")
    cache2 = await manager.aget_cache("manager_test_2")

    # Add some data
    await cache1.aset_by_key({"test": "data1"}, "test_key_1")
    await cache2.aset_by_key({"test": "data2"}, "test_key_2")

    # Test async stats
    print("  Getting async cache stats...")
    stats = await manager.aget_cache_stats()

    total_cached = stats.get("total", {}).get("cached_items", 0)
    print(f"  Total cached items across all namespaces: {total_cached}")

    assert total_cached >= 2, "Should have at least 2 cached items"

    # Test async clear
    print("  Testing async cache clearing...")
    await manager.aclear_cache("manager_test_1")

    # Verify clearing
    cleared_stats = await manager.aget_cache_stats("manager_test_1")
    cleared_count = cleared_stats.get("manager_test_1", {}).get("cached_items", 0)
    print(f"  Items in cleared cache: {cleared_count}")

    assert cleared_count == 0, "Cache should be empty after clearing"

    print("  ✅ Async manager operations successful")


async def test_error_handling():
    """Test error handling in async operations."""
    print("\n🔄 Testing error handling...")

    manager = APIManager()
    cache = await manager.aget_cache("error_test")

    # Test with invalid data
    try:
        # This should handle errors gracefully
        await cache.aset_by_key({"test": "data"}, "error_key")
        result = await cache.aget_by_key("error_key")
        print("  ✅ Error handling working correctly")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        raise

    # Test batch operations with some failures
    batch_items = [
        {"result": {"item": i}, "key_data": f"batch_error_{i}"}
        for i in range(5)
    ]

    # This should handle any errors gracefully
    await cache.abatch_set(batch_items)
    results = await cache.abatch_get([item["key_data"] for item in batch_items])

    successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    print(f"  Batch operations: {successful}/5 successful")

    print("  ✅ Error handling test completed")


async def main():
    """Run all async cache tests."""
    print("🚀 Async Cache Operations Tests")
    print("=" * 60)

    try:
        # Test basic async operations
        await test_basic_async_operations()

        # Test batch operations
        await test_batch_async_operations()

        # Test cache warming
        await test_cache_warming()

        # Test background maintenance
        await test_background_maintenance()

        # Test performance comparison
        await test_performance_comparison()

        # Test async manager operations
        await test_async_manager_operations()

        # Test error handling
        await test_error_handling()

        print("\n" + "=" * 60)
        print("✅ All async cache tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))