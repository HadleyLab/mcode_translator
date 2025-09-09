#!/usr/bin/env python3
"""
Test script to verify prompt benchmarks work across models and caching system functions correctly.
This tests both API caching and LLM caching after cache directories were deleted.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline.fetcher import search_trials, get_full_study, calculate_total_studies
from pipeline.mcode_llm import McodeMapper
from src.utils.api_manager import UnifiedAPIManager
from src.optimization.benchmark_task_tracker import BenchmarkTaskTrackerUI

def test_api_caching():
    """Test API caching functionality"""
    print("🧪 Testing API Caching...")
    
    # Clear API cache to start fresh using UnifiedAPIManager
    api_cache = UnifiedAPIManager().get_cache("api")
    api_cache.clear_cache()
    
    # Test first call (should miss cache)
    start_time = time.time()
    trials_data_1 = search_trials("cancer", max_results=5)
    time_1 = time.time() - start_time
    
    # Test second call (should hit cache)
    start_time = time.time()
    trials_data_2 = search_trials("cancer", max_results=5)
    time_2 = time.time() - start_time
    
    # Verify cache hit
    stats = api_cache.get_stats()
    print(f"API Cache Stats: {stats}")
    print(f"First call time: {time_1:.3f}s")
    print(f"Second call time: {time_2:.3f}s")
    
    if stats['hits'] > 0 and time_2 < time_1:
        print("✅ API caching working correctly!")
        assert True, "API caching working correctly"
    else:
        print("❌ API caching not working as expected")
        assert False, f"API caching not working: hits={stats['hits']}, time_1={time_1:.3f}s, time_2={time_2:.3f}s"

def test_llm_caching():
    """Test LLM caching functionality"""
    print("\n🧪 Testing LLM Caching...")
    
    # Create LLM instance using concrete implementation
    llm = McodeMapper(model_name="gpt-3.5-turbo")
    
    # Clear LLM cache to start fresh using UnifiedAPIManager
    llm_cache = UnifiedAPIManager().get_cache("llm")
    llm_cache.clear_cache()
    
    # Test prompt
    test_prompt = "Translate this medical term: Hypertension"
    
    # First call (should miss cache)
    start_time = time.time()
    result_1 = llm.generate_response(test_prompt)
    time_1 = time.time() - start_time
    
    # Second call (should hit cache)
    start_time = time.time()
    result_2 = llm.generate_response(test_prompt)
    time_2 = time.time() - start_time
    
    # Verify cache hit
    stats = llm_cache.get_stats()
    print(f"LLM Cache Stats: {stats}")
    print(f"First call time: {time_1:.3f}s")
    print(f"Second call time: {time_2:.3f}s")
    
    if stats['hits'] > 0 and time_2 < time_1:
        print("✅ LLM caching working correctly!")
        assert True, "LLM caching working correctly"
    else:
        print("❌ LLM caching not working as expected")
        assert False, f"LLM caching not working: hits={stats['hits']}, time_1={time_1:.3f}s, time_2={time_2:.3f}s"

def test_across_models():
    """Test prompt benchmarks across different models"""
    print("\n🧪 Testing Across Models...")
    
    models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]
    prompts = [
        "Translate this medical term: Hypertension",
        "Explain the difference between Type 1 and Type 2 diabetes",
        "What are the common symptoms of asthma?"
    ]
    
    results = {}
    
    for model in models:
        print(f"\nTesting model: {model}")
        llm = McodeMapper(model_name=model)
        
        model_results = []
        for i, prompt in enumerate(prompts):
            try:
                start_time = time.time()
                result = llm.generate_response(prompt)
                response_time = time.time() - start_time
                
                model_results.append({
                    'prompt': prompt,
                    'response_time': response_time,
                    'result_length': len(result) if result else 0
                })
                
                print(f"  Prompt {i+1}: {response_time:.3f}s, {len(result)} chars")
                
            except Exception as e:
                print(f"  ❌ Error with prompt {i+1}: {e}")
                model_results.append({
                    'prompt': prompt,
                    'error': str(e),
                    'response_time': None
                })
        
        results[model] = model_results
    
    return results

def test_benchmark_tracker():
    """Test benchmark task tracker functionality"""
    print("\n🧪 Testing Benchmark Task Tracker...")
    
    tracker = BenchmarkTaskTrackerUI()
    
    # Test cache statistics display
    print("Cache Statistics:")
    tracker.display_cache_stats()
    
    # Test cache clearing
    print("\nClearing caches...")
    tracker.clear_caches()
    
    # Verify caches are cleared using UnifiedAPIManager
    api_cache = UnifiedAPIManager().get_cache("api")
    llm_cache = UnifiedAPIManager().get_cache("llm")
    
    api_stats = api_cache.get_stats()
    llm_stats = llm_cache.get_stats()
    
    print(f"API Cache after clear: {api_stats}")
    print(f"LLM Cache after clear: {llm_stats}")
    
    if api_stats['total_items'] == 0 and llm_stats['total_items'] == 0:
        print("✅ Caches cleared successfully!")
        assert True, "Caches cleared successfully"
    else:
        print("❌ Caches not cleared properly")
        assert False, f"Caches not cleared properly: API items={api_stats['total_items']}, LLM items={llm_stats['total_items']}"

def main():
    """Main test function"""
    print("🚀 Starting Prompt Benchmark Tests Across Models")
    print("=" * 60)
    
    # Check if cache directories exist
    api_cache_dir = Path(".api_cache")
    llm_cache_dir = Path(".llm_cache")
    
    print(f"API Cache directory exists: {api_cache_dir.exists()}")
    print(f"LLM Cache directory exists: {llm_cache_dir.exists()}")
    
    # Run tests
    api_success = test_api_caching()
    llm_success = test_llm_caching()
    benchmark_success = test_benchmark_tracker()
    
    # Test across models
    print("\n" + "=" * 60)
    print("🏃‍♂️ Running Cross-Model Benchmark Tests...")
    model_results = test_across_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print(f"API Caching: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"LLM Caching: {'✅ PASS' if llm_success else '❌ FAIL'}")
    print(f"Benchmark Tracker: {'✅ PASS' if benchmark_success else '❌ FAIL'}")
    
    print("\n📈 Model Performance Summary:")
    for model, results in model_results.items():
        successful_runs = sum(1 for r in results if 'error' not in r)
        avg_time = sum(r['response_time'] for r in results if r['response_time'] is not None) / max(1, successful_runs)
        print(f"  {model}: {successful_runs}/{len(results)} successful, avg time: {avg_time:.3f}s")
    
    # Save detailed results
    with open('benchmark_results.json', 'w') as f:
        json.dump(model_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to benchmark_results.json")
    
    # Final status
    overall_success = api_success and llm_success and benchmark_success
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)