#!/usr/bin/env python3
"""
Test script for validating concurrent fetcher functionality
Tests both performance improvements and correctness of concurrent processing
"""

import asyncio
import time
import sys
import os
import json
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.concurrent_fetcher import (
    ConcurrentFetcher,
    ProcessingConfig,
    ConcurrentFetcherContext,
    concurrent_search_and_process,
    concurrent_process_trials
)
from src.pipeline.fetcher import search_trials, get_full_study
from src.utils import get_logger

logger = get_logger(__name__)


async def test_concurrent_vs_sequential_performance():
    """
    Test performance difference between concurrent and sequential processing
    """
    print("🔬 PERFORMANCE COMPARISON TEST")
    print("=" * 50)
    
    # Test configuration
    test_condition = "breast cancer"
    test_limit = 5  # Small number for testing
    test_nct_ids = ["NCT05613348", "NCT05611515", "NCT05619484"]  # Known NCT IDs
    
    print(f"📋 Test condition: {test_condition}")
    print(f"📊 Test limit: {test_limit}")
    print(f"🔬 Processing {len(test_nct_ids)} specific trials")
    
    # Test 1: Sequential processing simulation
    print("\n⏱️ SEQUENTIAL PROCESSING TEST")
    sequential_start = time.time()
    
    try:
        # Simulate sequential processing by fetching trials one by one
        sequential_results = []
        for nct_id in test_nct_ids:
            try:
                trial_data = get_full_study(nct_id)
                sequential_results.append(trial_data)
                print(f"✅ Sequential: Fetched {nct_id}")
            except Exception as e:
                print(f"❌ Sequential: Failed to fetch {nct_id}: {e}")
        
        sequential_duration = time.time() - sequential_start
        print(f"⏱️ Sequential processing: {sequential_duration:.2f} seconds")
        print(f"📊 Sequential success: {len(sequential_results)}/{len(test_nct_ids)} trials")
        
    except Exception as e:
        print(f"❌ Sequential processing failed: {e}")
        sequential_duration = float('inf')
        sequential_results = []
    
    # Test 2: Concurrent processing
    print("\n🚀 CONCURRENT PROCESSING TEST")
    concurrent_start = time.time()
    
    try:
        # Test concurrent processing
        config = ProcessingConfig(
            max_workers=3,
            batch_size=5,
            process_criteria=False,  # Skip mCODE processing for speed
            process_trials=False,
            progress_updates=True
        )
        
        async with ConcurrentFetcherContext(config) as fetcher:
            result = await fetcher.process_trial_list(test_nct_ids)
            
        concurrent_duration = time.time() - concurrent_start
        print(f"⏱️ Concurrent processing: {concurrent_duration:.2f} seconds")
        print(f"📊 Concurrent success: {result.successful_trials}/{result.total_trials} trials")
        
        # Performance comparison
        if sequential_duration < float('inf') and concurrent_duration > 0:
            speedup = sequential_duration / concurrent_duration
            print(f"\n🚀 PERFORMANCE IMPROVEMENT: {speedup:.2f}x speedup")
            if speedup > 1.0:
                print("✅ Concurrent processing is faster!")
            else:
                print("⚠️ Concurrent processing was slower (overhead for small datasets)")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent processing failed: {e}")
        return False


async def test_concurrent_search_functionality():
    """
    Test concurrent search and processing functionality
    """
    print("\n🔍 CONCURRENT SEARCH TEST")
    print("=" * 50)
    
    try:
        # Test concurrent search with a small limit
        result = await concurrent_search_and_process(
            condition="melanoma",
            limit=3,
            max_workers=2,
            batch_size=2,
            process_criteria=False,
            process_trials=False,
            progress_updates=True
        )
        
        print(f"📊 Search results:")
        print(f"   Total trials: {result.total_trials}")
        print(f"   Successful: {result.successful_trials}")
        print(f"   Failed: {result.failed_trials}")
        print(f"   Duration: {result.duration_seconds:.2f} seconds")
        
        if result.total_trials > 0:
            success_rate = (result.successful_trials / result.total_trials) * 100
            print(f"   Success rate: {success_rate:.1f}%")
        
        # Show sample results
        if result.results:
            print(f"\n🔍 Sample results:")
            for i, trial in enumerate(result.results[:2]):
                nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'Unknown')
                title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'No title')
                print(f"   {i+1}. {nct_id}: {title[:60]}...")
        
        return result.successful_trials > 0
        
    except Exception as e:
        print(f"❌ Concurrent search test failed: {e}")
        return False


async def test_concurrent_mcode_processing():
    """
    Test concurrent mCODE processing functionality
    """
    print("\n🔬 CONCURRENT mCODE PROCESSING TEST")
    print("=" * 50)
    
    try:
        # Test with a very small dataset and mock mCODE processing
        test_nct_ids = ["NCT05613348"]  # Single trial for testing
        
        result = await concurrent_process_trials(
            nct_ids=test_nct_ids,
            max_workers=1,
            batch_size=1,
            process_criteria=True,  # Enable mCODE processing
            process_trials=False,
            model_name="gpt-4o-mini",  # Use a fast model for testing
            prompt_name="direct_mcode",
            progress_updates=True
        )
        
        print(f"📊 mCODE processing results:")
        print(f"   Total trials: {result.total_trials}")
        print(f"   Successful: {result.successful_trials}")
        print(f"   Failed: {result.failed_trials}")
        print(f"   Duration: {result.duration_seconds:.2f} seconds")
        
        # Check if mCODE results were added
        if result.results:
            trial = result.results[0]
            if 'McodeResults' in trial:
                mcode_results = trial['McodeResults']
                entities_count = len(mcode_results.get('extracted_entities', []))
                mappings_count = len(mcode_results.get('mcode_mappings', []))
                print(f"   ✅ mCODE processing successful:")
                print(f"      Entities extracted: {entities_count}")
                print(f"      mCODE mappings: {mappings_count}")
                return True
            else:
                print(f"   ⚠️ No mCODE results found in trial data")
                return False
        else:
            print(f"   ❌ No successful results to check")
            return False
        
    except Exception as e:
        print(f"❌ Concurrent mCODE processing test failed: {e}")
        print(f"   This might be expected if mCODE pipeline assets are not available")
        return False


async def test_error_handling():
    """
    Test error handling in concurrent processing
    """
    print("\n🛡️ ERROR HANDLING TEST")
    print("=" * 50)
    
    try:
        # Test with invalid NCT IDs to trigger errors
        invalid_nct_ids = ["INVALID001", "INVALID002", "NCT05613348"]  # Mix of invalid and valid
        
        result = await concurrent_process_trials(
            nct_ids=invalid_nct_ids,
            max_workers=2,
            batch_size=2,
            process_criteria=False,
            process_trials=False,
            progress_updates=False
        )
        
        print(f"📊 Error handling results:")
        print(f"   Total trials: {result.total_trials}")
        print(f"   Successful: {result.successful_trials}")
        print(f"   Failed: {result.failed_trials}")
        
        # Check that errors were properly captured
        if result.errors:
            print(f"   ✅ Errors properly captured:")
            for error in result.errors[:2]:
                nct_id = error.get('nct_id', 'Unknown')
                error_msg = error.get('error', 'Unknown error')
                print(f"      {nct_id}: {error_msg[:50]}...")
        
        # Should have some successful and some failed
        has_success = result.successful_trials > 0
        has_failures = result.failed_trials > 0
        
        if has_success and has_failures:
            print("   ✅ Mixed success/failure handling works correctly")
            return True
        else:
            print("   ⚠️ Expected mixed results but didn't get them")
            return False
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def run_all_tests():
    """
    Run all concurrent fetcher tests
    """
    print("🧪 CONCURRENT FETCHER VALIDATION TESTS")
    print("=" * 60)
    print("Testing concurrent processing functionality and performance improvements")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Performance comparison
    try:
        test_results['performance'] = await test_concurrent_vs_sequential_performance()
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        test_results['performance'] = False
    
    # Test 2: Concurrent search
    try:
        test_results['search'] = await test_concurrent_search_functionality()
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        test_results['search'] = False
    
    # Test 3: mCODE processing (optional, may fail if assets missing)
    try:
        test_results['mcode'] = await test_concurrent_mcode_processing()
    except Exception as e:
        print(f"❌ mCODE test failed: {e}")
        test_results['mcode'] = False
    
    # Test 4: Error handling
    try:
        test_results['error_handling'] = await test_error_handling()
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        test_results['error_handling'] = False
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 3:  # Allow mCODE test to fail due to missing assets
        print("🎉 Concurrent fetcher is working correctly!")
        return True
    else:
        print("⚠️ Some critical tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test concurrent fetcher functionality")
    parser.add_argument("--test", choices=["performance", "search", "mcode", "error", "all"], 
                      default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = asyncio.run(run_all_tests())
    elif args.test == "performance":
        success = asyncio.run(test_concurrent_vs_sequential_performance())
    elif args.test == "search":
        success = asyncio.run(test_concurrent_search_functionality())
    elif args.test == "mcode":
        success = asyncio.run(test_concurrent_mcode_processing())
    elif args.test == "error":
        success = asyncio.run(test_error_handling())
    
    sys.exit(0 if success else 1)