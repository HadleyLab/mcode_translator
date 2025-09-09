#!/usr/bin/env python3
"""
Extended Cross-Validation - Comprehensive OpenAI & DeepSeek Optimization

Tests all combinations of working models and prompt strategies for full optimization.
Focuses on proven performers: OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5) and DeepSeek (coder, chat, reasoner).

Author: mCODE Translation Team
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.pipeline import McodePipeline
from src.utils.logging_config import get_logger, setup_logging


async def extended_cross_validation():
    """Run comprehensive optimization test with working models only."""
    logger = get_logger(__name__)
    setup_logging(level="INFO")
    
    logger.info("🚀 Starting Extended Cross-Validation - OpenAI & DeepSeek Focus")
    
    # Comprehensive combinations - proven performers only
    test_combinations = [
        # DeepSeek Models (Our Champions)
        ("direct_mcode_evidence_based", "deepseek-coder"),
        ("direct_mcode_evidence_based_concise", "deepseek-coder"),
        ("direct_mcode_simple", "deepseek-coder"),
        ("direct_mcode_comprehensive", "deepseek-coder"),
        ("direct_mcode_improved", "deepseek-coder"),
        ("direct_mcode_structured", "deepseek-coder"),
        
        ("direct_mcode_evidence_based", "deepseek-chat"),
        ("direct_mcode_evidence_based_concise", "deepseek-chat"),
        ("direct_mcode_simple", "deepseek-chat"),
        ("direct_mcode_comprehensive", "deepseek-chat"),
        
        ("direct_mcode_evidence_based", "deepseek-reasoner"),
        ("direct_mcode_evidence_based_concise", "deepseek-reasoner"),
        ("direct_mcode_simple", "deepseek-reasoner"),
        
        # OpenAI Models (Reliable Performers)
        ("direct_mcode_evidence_based", "gpt-4o"),
        ("direct_mcode_evidence_based_concise", "gpt-4o"),
        ("direct_mcode_simple", "gpt-4o"),
        ("direct_mcode_comprehensive", "gpt-4o"),
        ("direct_mcode_improved", "gpt-4o"),
        ("direct_mcode_structured", "gpt-4o"),
        
        ("direct_mcode_evidence_based", "gpt-4-turbo"),
        ("direct_mcode_evidence_based_concise", "gpt-4-turbo"),
        ("direct_mcode_simple", "gpt-4-turbo"),
        ("direct_mcode_comprehensive", "gpt-4-turbo"),
        
        ("direct_mcode_evidence_based", "gpt-4"),
        ("direct_mcode_simple", "gpt-4"),
        ("direct_mcode_comprehensive", "gpt-4"),
        
        ("direct_mcode_simple", "gpt-3.5-turbo"),
        ("direct_mcode_comprehensive", "gpt-3.5-turbo"),
    ]
    
    # Load trials
    trials_file = "data/selected_breast_cancer_trials.json"
    if not Path(trials_file).exists():
        logger.error(f"❌ Trials file not found: {trials_file}")
        return
    
    with open(trials_file, 'r') as f:
        data = json.load(f)
    
    # Use all available trials for comprehensive testing
    if isinstance(data, list):
        trials = data
    elif 'successful_trials' in data:
        trials = data['successful_trials']
    else:
        trials = [data]
    
    logger.info(f"📥 Testing with {len(trials)} trials across {len(test_combinations)} combinations")
    logger.info(f"🎯 Total test cases: {len(trials) * len(test_combinations)}")
    
    results = []
    start_time = time.time()
    
    for i, (prompt_name, model_name) in enumerate(test_combinations, 1):
        logger.info(f"🔬 Testing {i}/{len(test_combinations)}: {prompt_name} × {model_name}")
        
        combo_start = time.time()
        successful = 0
        total_mappings = 0
        errors = []
        
        for trial_idx, trial in enumerate(trials):
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
                
                logger.info(f"  ✅ Trial {trial_idx+1}: {mappings} mappings, quality {quality:.3f}")
                
            except Exception as e:
                error_msg = str(e)[:100]
                errors.append(error_msg)
                logger.error(f"  ❌ Trial {trial_idx+1} failed: {error_msg}")
        
        combo_duration = time.time() - combo_start
        
        result = {
            'prompt': prompt_name,
            'model': model_name,
            'successful_trials': successful,
            'total_trials': len(trials),
            'success_rate': successful / len(trials),
            'total_mappings': total_mappings,
            'avg_mappings': total_mappings / len(trials) if len(trials) > 0 else 0,
            'avg_mappings_per_success': total_mappings / successful if successful > 0 else 0,
            'duration': combo_duration,
            'errors': errors[:3]  # Keep first 3 errors for debugging
        }
        
        results.append(result)
        
        # Progress update
        success_pct = (successful / len(trials)) * 100
        logger.info(f"📊 Result: {successful}/{len(trials)} trials ({success_pct:.1f}%), "
                   f"{total_mappings} mappings, {combo_duration:.1f}s")
        
        # Brief pause to prevent rate limiting
        await asyncio.sleep(0.5)
    
    total_duration = time.time() - start_time
    
    # Save extended results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"extended_optimization_{timestamp}.json"
    
    summary_stats = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': total_duration,
        'duration_minutes': total_duration / 60,
        'combinations_tested': len(test_combinations),
        'trials_per_combination': len(trials),
        'total_test_cases': len(trials) * len(test_combinations),
        'working_models': ['deepseek-coder', 'deepseek-chat', 'deepseek-reasoner', 
                          'gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
        'prompt_strategies': list(set([r['prompt'] for r in results]))
    }
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary_stats,
            'results': results
        }, f, indent=2)
    
    # Comprehensive analysis
    logger.info("🎉 Extended optimization complete!")
    logger.info(f"⏱️  Total duration: {total_duration/60:.1f} minutes")
    logger.info(f"📊 Tested {len(test_combinations)} combinations × {len(trials)} trials")
    
    # Top performers analysis
    successful_results = [r for r in results if r['success_rate'] > 0]
    
    if successful_results:
        # Best by total mappings
        best_mappings = max(successful_results, key=lambda x: x['total_mappings'])
        logger.info(f"🏆 Most mappings: {best_mappings['prompt']} × {best_mappings['model']} "
                   f"({best_mappings['total_mappings']} total)")
        
        # Best by success rate
        best_success = max(successful_results, key=lambda x: x['success_rate'])
        logger.info(f"🎯 Best success rate: {best_success['prompt']} × {best_success['model']} "
                   f"({best_success['success_rate']:.1%})")
        
        # Best by efficiency (mappings per second)
        for r in successful_results:
            r['efficiency'] = r['total_mappings'] / r['duration'] if r['duration'] > 0 else 0
        best_efficiency = max(successful_results, key=lambda x: x['efficiency'])
        logger.info(f"⚡ Most efficient: {best_efficiency['prompt']} × {best_efficiency['model']} "
                   f"({best_efficiency['efficiency']:.1f} mappings/sec)")
        
        # Provider comparison
        deepseek_results = [r for r in successful_results if 'deepseek' in r['model']]
        openai_results = [r for r in successful_results if 'gpt' in r['model']]
        
        if deepseek_results:
            deepseek_avg = sum(r['total_mappings'] for r in deepseek_results) / len(deepseek_results)
            logger.info(f"🤖 DeepSeek average: {deepseek_avg:.1f} mappings/combination")
        
        if openai_results:
            openai_avg = sum(r['total_mappings'] for r in openai_results) / len(openai_results)
            logger.info(f"🧠 OpenAI average: {openai_avg:.1f} mappings/combination")
    
    logger.info(f"💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(extended_cross_validation())