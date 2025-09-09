#!/usr/bin/env python3
"""
Quick Test Experiment for Breast Cancer mCODE Translation
=========================================================

This script runs a smaller test experiment with a subset of models and prompts
to validate the approach before running the full comprehensive experiment.

Test Configuration:
- 5 selected breast cancer trials
- 3 models (fast, reliable subset)
- 3 prompts (diverse approaches)
- Total: 9 combinations

This allows rapid validation of the experimental setup.
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.concurrent_fetcher import concurrent_process_trials
from src.utils import get_logger

logger = get_logger(__name__)

# Quick test configuration
SELECTED_TRIALS = [
    "NCT01922921",  # HER2+ Stage IV
    "NCT01026116",  # Young patients
    "NCT06650748",  # HR+/HER2- 
    "NCT00109785",  # Locally advanced
    "NCT00616135"   # Reconstruction
]

# Subset for quick testing
TEST_MODELS = [
    "gpt-4o-mini",      # Fast and cost-effective
    "deepseek-coder",   # Code-focused
    "gpt-4o"            # Most reliable
]

TEST_PROMPTS = [
    "direct_mcode",           # Default
    "direct_mcode_simple",    # Minimal approach
    "direct_mcode_improved"   # Enhanced approach
]

async def quick_test_combination(model: str, prompt: str) -> dict:
    """Test a single model-prompt combination"""
    logger.info(f"🧪 Testing: {model} + {prompt}")
    
    try:
        result = await concurrent_process_trials(
            nct_ids=SELECTED_TRIALS,
            max_workers=3,
            batch_size=2,
            process_criteria=True,
            process_trials=False,
            model_name=model,
            prompt_name=prompt,
            progress_updates=False
        )
        
        # Extract key metrics
        metrics = {
            'model': model,
            'prompt': prompt,
            'success_rate': result.successful_trials / result.total_trials if result.total_trials > 0 else 0,
            'duration_seconds': result.duration_seconds,
            'successful_trials': result.successful_trials,
            'total_trials': result.total_trials,
            'processing_rate': result.successful_trials / result.duration_seconds if result.duration_seconds > 0 else 0
        }
        
        # Count entities and mappings
        total_entities = 0
        total_mappings = 0
        for trial_result in result.results:
            if 'McodeResults' in trial_result:
                mcode_results = trial_result['McodeResults']
                total_entities += len(mcode_results.get('extracted_entities', []))
                total_mappings += len(mcode_results.get('mcode_mappings', []))
        
        metrics['total_entities'] = total_entities
        metrics['total_mappings'] = total_mappings
        
        logger.info(f"✅ {model} + {prompt}: {result.successful_trials}/{result.total_trials} trials, "
                   f"{total_entities} entities, {total_mappings} mappings")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Failed {model} + {prompt}: {e}")
        return {
            'model': model,
            'prompt': prompt,
            'error': str(e)
        }

async def run_quick_test():
    """Run the quick test experiment"""
    logger.info("🚀 Starting Quick Test Experiment for Breast Cancer mCODE Translation")
    logger.info(f"📊 Testing {len(TEST_MODELS)} models x {len(TEST_PROMPTS)} prompts = {len(TEST_MODELS) * len(TEST_PROMPTS)} combinations")
    
    start_time = time.time()
    results = []
    
    total_combinations = len(TEST_MODELS) * len(TEST_PROMPTS)
    completed = 0
    
    for model in TEST_MODELS:
        for prompt in TEST_PROMPTS:
            completed += 1
            logger.info(f"🔄 Progress: {completed}/{total_combinations}")
            
            result = await quick_test_combination(model, prompt)
            results.append(result)
            
            # Brief pause between tests
            await asyncio.sleep(1)
    
    duration = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if 'error' not in r]
    
    print("\n" + "="*60)
    print("📊 QUICK TEST RESULTS SUMMARY")
    print("="*60)
    print(f"⏱️  Total duration: {duration:.1f} seconds")
    print(f"✅ Successful combinations: {len(successful_results)}/{len(results)}")
    
    if successful_results:
        # Sort by success rate, then by processing rate
        successful_results.sort(key=lambda x: (x['success_rate'], x['processing_rate']), reverse=True)
        
        print(f"\n🏆 TOP PERFORMING COMBINATIONS:")
        print("-" * 60)
        for i, result in enumerate(successful_results[:3], 1):
            print(f"{i}. {result['model']} + {result['prompt']}")
            print(f"   Success Rate: {result['success_rate']:.1%}")
            print(f"   Duration: {result['duration_seconds']:.1f}s")
            print(f"   Entities: {result['total_entities']}, Mappings: {result['total_mappings']}")
            print()
        
        # Model analysis
        model_performance = {}
        for model in TEST_MODELS:
            model_results = [r for r in successful_results if r['model'] == model]
            if model_results:
                avg_success = sum(r['success_rate'] for r in model_results) / len(model_results)
                avg_duration = sum(r['duration_seconds'] for r in model_results) / len(model_results)
                model_performance[model] = {'success_rate': avg_success, 'duration': avg_duration}
        
        print("📈 MODEL PERFORMANCE:")
        print("-" * 40)
        for model, perf in sorted(model_performance.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"{model}: {perf['success_rate']:.1%} success, {perf['duration']:.1f}s avg")
        
        # Prompt analysis
        prompt_performance = {}
        for prompt in TEST_PROMPTS:
            prompt_results = [r for r in successful_results if r['prompt'] == prompt]
            if prompt_results:
                avg_success = sum(r['success_rate'] for r in prompt_results) / len(prompt_results)
                avg_entities = sum(r['total_entities'] for r in prompt_results) / len(prompt_results)
                prompt_performance[prompt] = {'success_rate': avg_success, 'entities': avg_entities}
        
        print(f"\n💬 PROMPT PERFORMANCE:")
        print("-" * 40)
        for prompt, perf in sorted(prompt_performance.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            print(f"{prompt}: {perf['success_rate']:.1%} success, {perf['entities']:.1f} entities avg")
    
    # Save results
    results_file = f"quick_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'models_tested': TEST_MODELS,
            'prompts_tested': TEST_PROMPTS,
            'trials_tested': SELECTED_TRIALS,
            'results': results
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    print("="*60)
    
    # Recommendation for full experiment
    if successful_results:
        best_combo = successful_results[0]
        print(f"\n🎯 RECOMMENDATION FOR FULL EXPERIMENT:")
        print(f"   Based on quick test, consider prioritizing:")
        print(f"   Model: {best_combo['model']}")
        print(f"   Prompt: {best_combo['prompt']}")
        print(f"   This combination showed {best_combo['success_rate']:.1%} success rate")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_quick_test())