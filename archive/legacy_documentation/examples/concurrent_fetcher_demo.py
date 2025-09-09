#!/usr/bin/env python3
"""
Example usage of the concurrent fetcher functionality
Demonstrates how to use the new concurrent processing features
"""

import asyncio
import sys
import os
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.concurrent_fetcher import (
    concurrent_search_and_process,
    concurrent_process_trials,
    ConcurrentFetcher,
    ProcessingConfig,
    ConcurrentFetcherContext
)
from src.utils import get_logger

logger = get_logger(__name__)


async def example_1_simple_concurrent_search():
    """
    Example 1: Simple concurrent search and processing
    """
    print("📋 Example 1: Simple Concurrent Search")
    print("-" * 40)
    
    # Search for breast cancer trials and process them concurrently
    result = await concurrent_search_and_process(
        condition="breast cancer",
        limit=5,  # Small number for demo
        max_workers=3,
        process_trials=False,  # Skip mCODE processing for speed
        progress_updates=True
    )
    
    print(f"✅ Processed {result.successful_trials}/{result.total_trials} trials")
    print(f"⏱️ Duration: {result.duration_seconds:.2f} seconds")
    print(f"🚀 Rate: {result.total_trials/result.duration_seconds:.1f} trials/sec")
    
    return result


async def example_2_specific_trials_with_mcode():
    """
    Example 2: Process specific trials with mCODE processing
    """
    print("\n🔬 Example 2: Specific Trials with mCODE Processing")
    print("-" * 50)
    
    # Process specific NCT IDs with mCODE processing
    nct_ids = ["NCT05613348", "NCT05611515"]
    
    try:
        result = await concurrent_process_trials(
            nct_ids=nct_ids,
            max_workers=2,
            process_criteria=True,  # Enable mCODE processing
            model_name="gpt-4o-mini",  # Fast model for demo
            prompt_name="direct_mcode",
            progress_updates=True
        )
        
        print(f"✅ Processed {result.successful_trials}/{result.total_trials} trials")
        print(f"⏱️ Duration: {result.duration_seconds:.2f} seconds")
        
        # Show mCODE results
        for trial in result.results:
            if 'McodeResults' in trial:
                nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'Unknown')
                mcode_results = trial['McodeResults']
                entities = len(mcode_results.get('extracted_entities', []))
                mappings = len(mcode_results.get('mcode_mappings', []))
                print(f"   {nct_id}: {entities} entities, {mappings} mappings")
        
        return result
        
    except Exception as e:
        print(f"❌ mCODE processing failed: {e}")
        print("   This might be expected if mCODE pipeline assets are not configured")
        return None


async def example_3_advanced_configuration():
    """
    Example 3: Advanced configuration with custom settings
    """
    print("\n⚙️ Example 3: Advanced Configuration")
    print("-" * 38)
    
    # Create custom configuration
    config = ProcessingConfig(
        max_workers=4,
        batch_size=8,
        process_criteria=False,
        process_trials=False,
        model_name="gpt-4o",
        prompt_name="direct_mcode_comprehensive",
        progress_updates=True
    )
    
    # Use context manager for automatic cleanup
    async with ConcurrentFetcherContext(config) as fetcher:
        # Search and process trials
        result = await fetcher.search_and_process_trials(
            condition="lung cancer",
            limit=4
        )
        
        print(f"✅ Processed {result.successful_trials}/{result.total_trials} trials")
        print(f"⏱️ Duration: {result.duration_seconds:.2f} seconds")
        
        # Get task queue statistics
        stats = result.task_stats
        print(f"📊 Queue stats:")
        print(f"   Workers: {stats.get('workers_running', 'N/A')}")
        print(f"   Completion rate: {stats.get('completion_rate', 0)*100:.1f}%")
        
        return result


async def example_4_performance_comparison():
    """
    Example 4: Performance comparison between sequential and concurrent
    """
    print("\n🏃 Example 4: Performance Comparison")
    print("-" * 38)
    
    # Test with the same dataset
    test_nct_ids = ["NCT05613348", "NCT05611515", "NCT05619484"]
    
    # Sequential simulation (fetch one by one)
    print("⏱️ Sequential simulation...")
    start_time = time.time()
    
    from src.pipeline.fetcher import get_full_study
    sequential_results = []
    for nct_id in test_nct_ids:
        try:
            trial = get_full_study(nct_id)
            sequential_results.append(trial)
        except Exception as e:
            print(f"   Failed to fetch {nct_id}: {e}")
    
    sequential_time = time.time() - start_time
    print(f"   Sequential: {len(sequential_results)} trials in {sequential_time:.2f}s")
    
    # Concurrent processing
    print("🚀 Concurrent processing...")
    concurrent_result = await concurrent_process_trials(
        nct_ids=test_nct_ids,
        max_workers=3,
        progress_updates=False
    )
    
    print(f"   Concurrent: {concurrent_result.successful_trials} trials in {concurrent_result.duration_seconds:.2f}s")
    
    # Compare performance
    if sequential_time > 0 and concurrent_result.duration_seconds > 0:
        speedup = sequential_time / concurrent_result.duration_seconds
        print(f"🚀 Speedup: {speedup:.2f}x faster with concurrent processing")
    
    return concurrent_result


async def run_all_examples():
    """
    Run all examples to demonstrate concurrent fetcher capabilities
    """
    print("🧪 CONCURRENT FETCHER EXAMPLES")
    print("=" * 50)
    print("Demonstrating concurrent processing capabilities")
    print("=" * 50)
    
    try:
        # Example 1: Simple search
        await example_1_simple_concurrent_search()
        
        # Example 2: mCODE processing (may fail if assets missing)
        await example_2_specific_trials_with_mcode()
        
        # Example 3: Advanced configuration
        await example_3_advanced_configuration()
        
        # Example 4: Performance comparison
        await example_4_performance_comparison()
        
        print("\n🎉 All examples completed successfully!")
        print("\nKey benefits of concurrent processing:")
        print("  ✅ Faster processing of multiple trials")
        print("  ✅ Efficient resource utilization")
        print("  ✅ Progress tracking and monitoring")
        print("  ✅ Robust error handling")
        print("  ✅ Backward compatibility with existing API")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run concurrent fetcher examples")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4], 
                      help="Run a specific example (1-4), or omit to run all")
    
    args = parser.parse_args()
    
    if args.example == 1:
        asyncio.run(example_1_simple_concurrent_search())
    elif args.example == 2:
        asyncio.run(example_2_specific_trials_with_mcode())
    elif args.example == 3:
        asyncio.run(example_3_advanced_configuration())
    elif args.example == 4:
        asyncio.run(example_4_performance_comparison())
    else:
        success = asyncio.run(run_all_examples())
        sys.exit(0 if success else 1)