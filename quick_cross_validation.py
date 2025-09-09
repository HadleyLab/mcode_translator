#!/usr/bin/env python3
"""
Quick Optimization Test - Fast subset testing with caching

Tests a representative sample of prompt × model combinations for rapid optimization.

Author: mCODE Translation Team
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.pipeline import McodePipeline
from src.utils.logging_config import get_logger, setup_logging


async def quick_cross_validation():
    """Run a quick optimization test with caching."""
    logger = get_logger(__name__)
    setup_logging(level="INFO")
    
    logger.info("🚀 Starting Quick Optimization Test")
    
    # Representative sample of combinations (including Claude via OpenAI-compatible API)
    test_combinations = [
        ("direct_mcode_evidence_based_concise", "deepseek-coder"),
        ("direct_mcode_simple", "gpt-4o"),
        ("direct_mcode_structured", "claude-sonnet-4"),
        ("direct_mcode_comprehensive", "gpt-4o"),
        ("direct_mcode_evidence_based", "deepseek-coder"),
        ("direct_mcode_improved", "claude-3-5-haiku")
    ]
    
    # Load trials
    trials_file = "data/selected_breast_cancer_trials.json"
    if not Path(trials_file).exists():
        logger.error(f"❌ Trials file not found: {trials_file}")
        return
    
    with open(trials_file, 'r') as f:
        data = json.load(f)
    
    # Get first 3 trials for quick test
    if isinstance(data, list):
        trials = data[:3]
    elif 'successful_trials' in data:
        trials = data['successful_trials'][:3]
    else:
        trials = [data]
    
    logger.info(f"📥 Testing with {len(trials)} trials")
    
    results = []
    start_time = time.time()
    
    for prompt_name, model_name in test_combinations:
        logger.info(f"🔬 Testing: {prompt_name} × {model_name}")
        
        combo_start = time.time()
        successful = 0
        total_mappings = 0
        
        for i, trial in enumerate(trials):
            try:
                pipeline = McodePipeline(
                    prompt_name=prompt_name,
                    model_name=model_name
                )
                
                result = pipeline.process_clinical_trial(trial)
                mappings = len(result.mcode_mappings)
                quality = result.validation_results.get('compliance_score', 0.0)
                
                successful += 1
                total_mappings += mappings
                
                logger.info(f"  ✅ Trial {i+1}: {mappings} mappings, quality {quality:.3f}")
                
            except Exception as e:
                logger.error(f"  ❌ Trial {i+1} failed: {str(e)}")
        
        combo_duration = time.time() - combo_start
        
        result = {
            'prompt': prompt_name,
            'model': model_name,
            'successful_trials': successful,
            'total_trials': len(trials),
            'success_rate': successful / len(trials),
            'total_mappings': total_mappings,
            'avg_mappings': total_mappings / len(trials),
            'duration': combo_duration
        }
        
        results.append(result)
        logger.info(f"📊 Result: {successful}/{len(trials)} trials, "
                   f"{total_mappings} mappings, {combo_duration:.1f}s")
    
    total_duration = time.time() - start_time
    
    # Save quick results
    output_file = f"quick_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'duration': total_duration,
            'combinations_tested': len(test_combinations),
            'trials_per_combination': len(trials),
            'results': results
        }, f, indent=2)
    
    # Print summary
    logger.info("🎉 Quick optimization complete!")
    logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
    logger.info(f"📊 Tested {len(test_combinations)} combinations")
    
    # Show best performers
    best_success = max(results, key=lambda x: x['success_rate'])
    best_mappings = max(results, key=lambda x: x['total_mappings'])
    
    logger.info(f"🏆 Best success rate: {best_success['prompt']} × {best_success['model']} "
               f"({best_success['success_rate']:.1%})")
    logger.info(f"📋 Most mappings: {best_mappings['prompt']} × {best_mappings['model']} "
               f"({best_mappings['total_mappings']} total)")
    
    logger.info(f"💾 Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(quick_cross_validation())