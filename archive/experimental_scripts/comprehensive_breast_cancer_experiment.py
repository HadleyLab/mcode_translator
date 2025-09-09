#!/usr/bin/env python3
"""
Comprehensive Breast Cancer mCODE Translation Experiment
========================================================

This script conducts a large-scale experiment to find the optimal combination of models 
and prompts for breast cancer clinical trial mCODE extraction using the concurrent fetcher.

Selected Test Trials:
1. NCT01922921 - HER2+ Stage IV with vaccine therapy (metastatic, targeted therapy)
2. NCT01026116 - Young patients (<40) with adjuvant therapy (early stage, age-specific)  
3. NCT06650748 - HR+/HER2- with CDK4/6 inhibitors (hormone receptor positive, newer targeted therapy)
4. NCT00109785 - Locally advanced breast cancer with PET imaging (locally advanced, diagnostic)
5. NCT00616135 - Breast reconstruction post-lumpectomy (surgical/reconstruction focus)

Models to Test:
- deepseek-coder
- deepseek-chat
- deepseek-reasoner
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo

Prompts to Test:
- direct_mcode (default)
- direct_mcode_simple
- direct_mcode_comprehensive
- direct_mcode_minimal
- direct_mcode_structured
- direct_mcode_optimization
- direct_mcode_improved
"""

import asyncio
import sys
import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.concurrent_fetcher import (
    concurrent_process_trials,
    ConcurrentFetcher,
    ProcessingConfig,
    ConcurrentFetcherContext
)
from src.utils import get_logger, PromptLoader, ModelLoader

logger = get_logger(__name__)

# Experiment Configuration
SELECTED_TRIALS = [
    "NCT01922921",  # HER2+ Stage IV with vaccine therapy
    "NCT01026116",  # Young patients with adjuvant therapy  
    "NCT06650748",  # HR+/HER2- with CDK4/6 inhibitors
    "NCT00109785",  # Locally advanced with PET imaging
    "NCT00616135"   # Breast reconstruction post-lumpectomy
]

MODELS_TO_TEST = [
    "deepseek-coder",
    "deepseek-chat", 
    "deepseek-reasoner",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo"
]

PROMPTS_TO_TEST = [
    "direct_mcode",              # default
    "direct_mcode_simple",       
    "direct_mcode_comprehensive",
    "direct_mcode_minimal",
    "direct_mcode_structured",
    "direct_mcode_optimization",
    "direct_mcode_improved"
]

# Results storage
RESULTS_DIR = Path("experiment_results")
GOLD_STANDARD_FILE = "gold_standard_breast_cancer.json"
EXPERIMENT_RESULTS_FILE = "comprehensive_experiment_results.json"
ANALYSIS_REPORT_FILE = "experiment_analysis_report.md"

class BreastCancerExperiment:
    """Main experiment controller for comprehensive breast cancer mCODE evaluation"""
    
    def __init__(self):
        self.results_dir = Path(RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
        self.prompt_loader = PromptLoader()
        self.model_loader = ModelLoader()
        self.experiment_results = {}
        self.gold_standard = {}
        
    async def generate_gold_standard(self) -> Dict[str, Any]:
        """
        Generate gold standard mCODE translations using the best current approach
        Uses the default prompt and most reliable model for reference
        """
        logger.info("🏆 Generating gold standard mCODE translations...")
        
        # Use the most reliable model and prompt for gold standard
        gold_standard_model = "gpt-4o"  # Most reliable for consistency
        gold_standard_prompt = "direct_mcode_comprehensive"  # Most complete prompt
        
        try:
            result = await concurrent_process_trials(
                nct_ids=SELECTED_TRIALS,
                max_workers=2,  # Conservative for quality
                batch_size=1,   # Process one at a time for consistency
                process_criteria=True,
                process_trials=False,
                model_name=gold_standard_model,
                prompt_name=gold_standard_prompt,
                progress_updates=True
            )
            
            # Extract and structure gold standard data
            gold_standard = {}
            for trial_result in result.results:
                nct_id = trial_result.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
                if nct_id and 'McodeResults' in trial_result:
                    mcode_results = trial_result['McodeResults']
                    gold_standard[nct_id] = {
                        'trial_title': trial_result.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', ''),
                        'model_used': gold_standard_model,
                        'prompt_used': gold_standard_prompt,
                        'extracted_entities': mcode_results.get('extracted_entities', []),
                        'mcode_mappings': mcode_results.get('mcode_mappings', []),
                        'entity_count': len(mcode_results.get('extracted_entities', [])),
                        'mapping_count': len(mcode_results.get('mcode_mappings', [])),
                        'processing_metadata': mcode_results.get('metadata', {}),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Save gold standard
            gold_standard_path = self.results_dir / GOLD_STANDARD_FILE
            with open(gold_standard_path, 'w') as f:
                json.dump(gold_standard, f, indent=2)
            
            logger.info(f"✅ Gold standard generated for {len(gold_standard)} trials")
            logger.info(f"📁 Saved to: {gold_standard_path}")
            
            self.gold_standard = gold_standard
            return gold_standard
            
        except Exception as e:
            logger.error(f"❌ Failed to generate gold standard: {e}")
            raise
    
    async def run_model_prompt_combination(self, model: str, prompt: str) -> Dict[str, Any]:
        """
        Run a specific model-prompt combination on all test trials
        """
        logger.info(f"🔬 Testing combination: {model} + {prompt}")
        
        try:
            result = await concurrent_process_trials(
                nct_ids=SELECTED_TRIALS,
                max_workers=3,  # Moderate parallelism
                batch_size=2,
                process_criteria=True,
                process_trials=False,
                model_name=model,
                prompt_name=prompt,
                progress_updates=False  # Reduce log noise
            )
            
            # Extract results for analysis
            combination_results = {
                'model': model,
                'prompt': prompt,
                'total_trials': result.total_trials,
                'successful_trials': result.successful_trials,
                'failed_trials': result.failed_trials,
                'success_rate': result.successful_trials / result.total_trials if result.total_trials > 0 else 0,
                'duration_seconds': result.duration_seconds,
                'processing_rate': result.successful_trials / result.duration_seconds if result.duration_seconds > 0 else 0,
                'trial_results': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Process each trial result
            for trial_result in result.results:
                nct_id = trial_result.get('protocolSection', {}).get('identificationModule', {}).get('nctId')
                if nct_id and 'McodeResults' in trial_result:
                    mcode_results = trial_result['McodeResults']
                    combination_results['trial_results'][nct_id] = {
                        'extracted_entities': mcode_results.get('extracted_entities', []),
                        'mcode_mappings': mcode_results.get('mcode_mappings', []),
                        'entity_count': len(mcode_results.get('extracted_entities', [])),
                        'mapping_count': len(mcode_results.get('mcode_mappings', [])),
                        'processing_metadata': mcode_results.get('metadata', {})
                    }
            
            # Add errors if any
            if result.errors:
                combination_results['errors'] = result.errors
            
            logger.info(f"✅ Completed {model} + {prompt}: {result.successful_trials}/{result.total_trials} trials successful")
            return combination_results
            
        except Exception as e:
            logger.error(f"❌ Failed combination {model} + {prompt}: {e}")
            return {
                'model': model,
                'prompt': prompt,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experiment across all model-prompt combinations
        """
        logger.info("🚀 Starting comprehensive breast cancer mCODE experiment...")
        logger.info(f"📊 Testing {len(MODELS_TO_TEST)} models x {len(PROMPTS_TO_TEST)} prompts = {len(MODELS_TO_TEST) * len(PROMPTS_TO_TEST)} combinations")
        
        experiment_start_time = time.time()
        all_results = []
        
        total_combinations = len(MODELS_TO_TEST) * len(PROMPTS_TO_TEST)
        completed_combinations = 0
        
        # Run all combinations
        for model in MODELS_TO_TEST:
            for prompt in PROMPTS_TO_TEST:
                completed_combinations += 1
                logger.info(f"🔄 Progress: {completed_combinations}/{total_combinations} - Testing {model} + {prompt}")
                
                try:
                    combination_result = await self.run_model_prompt_combination(model, prompt)
                    all_results.append(combination_result)
                    
                    # Brief pause between combinations to avoid overwhelming APIs
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Failed combination {model} + {prompt}: {e}")
                    all_results.append({
                        'model': model,
                        'prompt': prompt,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        experiment_duration = time.time() - experiment_start_time
        
        # Compile final results
        experiment_summary = {
            'experiment_metadata': {
                'start_time': datetime.fromtimestamp(experiment_start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': experiment_duration,
                'total_combinations_tested': len(all_results),
                'models_tested': MODELS_TO_TEST,
                'prompts_tested': PROMPTS_TO_TEST,
                'trials_tested': SELECTED_TRIALS
            },
            'results': all_results
        }
        
        # Save results
        results_path = self.results_dir / EXPERIMENT_RESULTS_FILE
        with open(results_path, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        
        logger.info(f"✅ Experiment completed in {experiment_duration:.2f} seconds")
        logger.info(f"📁 Results saved to: {results_path}")
        
        self.experiment_results = experiment_summary
        return experiment_summary
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of experiment results
        """
        logger.info("📊 Analyzing experiment results...")
        
        if not self.experiment_results or not self.gold_standard:
            raise ValueError("Must run experiment and generate gold standard first")
        
        analysis = {
            'summary_statistics': {},
            'model_performance': {},
            'prompt_performance': {},
            'trial_difficulty': {},
            'optimal_combinations': {},
            'recommendations': {}
        }
        
        results = self.experiment_results['results']
        
        # Calculate summary statistics
        successful_combinations = [r for r in results if 'error' not in r and r.get('success_rate', 0) > 0]
        analysis['summary_statistics'] = {
            'total_combinations': len(results),
            'successful_combinations': len(successful_combinations),
            'success_rate': len(successful_combinations) / len(results) if results else 0,
            'average_trial_success_rate': sum(r.get('success_rate', 0) for r in successful_combinations) / len(successful_combinations) if successful_combinations else 0,
            'average_processing_time': sum(r.get('duration_seconds', 0) for r in successful_combinations) / len(successful_combinations) if successful_combinations else 0
        }
        
        # Analyze model performance
        model_stats = {}
        for model in MODELS_TO_TEST:
            model_results = [r for r in results if r.get('model') == model and 'error' not in r]
            if model_results:
                model_stats[model] = {
                    'combinations_tested': len(model_results),
                    'average_success_rate': sum(r.get('success_rate', 0) for r in model_results) / len(model_results),
                    'average_duration': sum(r.get('duration_seconds', 0) for r in model_results) / len(model_results),
                    'total_entities_extracted': sum(
                        sum(trial.get('entity_count', 0) for trial in r.get('trial_results', {}).values())
                        for r in model_results
                    ),
                    'total_mappings_generated': sum(
                        sum(trial.get('mapping_count', 0) for trial in r.get('trial_results', {}).values())
                        for r in model_results
                    )
                }
        analysis['model_performance'] = model_stats
        
        # Analyze prompt performance
        prompt_stats = {}
        for prompt in PROMPTS_TO_TEST:
            prompt_results = [r for r in results if r.get('prompt') == prompt and 'error' not in r]
            if prompt_results:
                prompt_stats[prompt] = {
                    'combinations_tested': len(prompt_results),
                    'average_success_rate': sum(r.get('success_rate', 0) for r in prompt_results) / len(prompt_results),
                    'average_duration': sum(r.get('duration_seconds', 0) for r in prompt_results) / len(prompt_results),
                    'total_entities_extracted': sum(
                        sum(trial.get('entity_count', 0) for trial in r.get('trial_results', {}).values())
                        for r in prompt_results
                    ),
                    'total_mappings_generated': sum(
                        sum(trial.get('mapping_count', 0) for trial in r.get('trial_results', {}).values())
                        for r in prompt_results
                    )
                }
        analysis['prompt_performance'] = prompt_stats
        
        # Find optimal combinations
        successful_combinations_sorted = sorted(
            successful_combinations,
            key=lambda x: (x.get('success_rate', 0), -x.get('duration_seconds', float('inf'))),
            reverse=True
        )
        
        analysis['optimal_combinations'] = {
            'top_5_by_success_rate': [
                {
                    'model': r['model'],
                    'prompt': r['prompt'],
                    'success_rate': r.get('success_rate', 0),
                    'duration_seconds': r.get('duration_seconds', 0),
                    'processing_rate': r.get('processing_rate', 0)
                }
                for r in successful_combinations_sorted[:5]
            ]
        }
        
        # Generate recommendations
        best_model = max(model_stats.items(), key=lambda x: x[1]['average_success_rate'])[0] if model_stats else None
        best_prompt = max(prompt_stats.items(), key=lambda x: x[1]['average_success_rate'])[0] if prompt_stats else None
        
        analysis['recommendations'] = {
            'best_overall_model': best_model,
            'best_overall_prompt': best_prompt,
            'recommended_combination': f"{best_model} + {best_prompt}" if best_model and best_prompt else None,
            'cost_effectiveness_notes': "Analysis based on success rate and processing time",
            'scalability_considerations': "Consider API rate limits and costs for large-scale deployment"
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """
        Generate comprehensive markdown report
        """
        logger.info("📝 Generating comprehensive report...")
        
        report = f"""# Comprehensive Breast Cancer mCODE Translation Experiment Report

Generated: {datetime.now().isoformat()}

## Executive Summary

This experiment evaluated {len(MODELS_TO_TEST)} language models and {len(PROMPTS_TO_TEST)} prompt strategies across {len(SELECTED_TRIALS)} diverse breast cancer clinical trials to determine the optimal configuration for large-scale mCODE translation.

### Key Findings

- **Total Combinations Tested**: {analysis['summary_statistics']['total_combinations']}
- **Successful Combinations**: {analysis['summary_statistics']['successful_combinations']}
- **Overall Success Rate**: {analysis['summary_statistics']['success_rate']:.1%}
- **Recommended Configuration**: {analysis['recommendations']['recommended_combination']}

## Test Configuration

### Selected Trials
{chr(10).join([f"- **{trial}**: {self.gold_standard.get(trial, {}).get('trial_title', 'N/A')}" for trial in SELECTED_TRIALS])}

### Models Tested
{chr(10).join([f"- {model}" for model in MODELS_TO_TEST])}

### Prompts Tested
{chr(10).join([f"- {prompt}" for prompt in PROMPTS_TO_TEST])}

## Results Analysis

### Model Performance Ranking

| Model | Avg Success Rate | Avg Duration (s) | Total Entities | Total Mappings |
|-------|------------------|------------------|----------------|----------------|
"""
        
        # Add model performance table
        for model, stats in sorted(analysis['model_performance'].items(), 
                                 key=lambda x: x[1]['average_success_rate'], reverse=True):
            report += f"| {model} | {stats['average_success_rate']:.1%} | {stats['average_duration']:.1f} | {stats['total_entities_extracted']} | {stats['total_mappings_generated']} |\n"
        
        report += f"""
### Prompt Performance Ranking

| Prompt | Avg Success Rate | Avg Duration (s) | Total Entities | Total Mappings |
|--------|------------------|------------------|----------------|----------------|
"""
        
        # Add prompt performance table
        for prompt, stats in sorted(analysis['prompt_performance'].items(), 
                                  key=lambda x: x[1]['average_success_rate'], reverse=True):
            report += f"| {prompt} | {stats['average_success_rate']:.1%} | {stats['average_duration']:.1f} | {stats['total_entities_extracted']} | {stats['total_mappings_generated']} |\n"
        
        report += f"""
### Top 5 Model-Prompt Combinations

| Rank | Model | Prompt | Success Rate | Duration (s) | Processing Rate |
|------|-------|---------|--------------|--------------|-----------------|
"""
        
        # Add top combinations table
        for i, combo in enumerate(analysis['optimal_combinations']['top_5_by_success_rate'], 1):
            report += f"| {i} | {combo['model']} | {combo['prompt']} | {combo['success_rate']:.1%} | {combo['duration_seconds']:.1f} | {combo['processing_rate']:.3f} trials/s |\n"
        
        report += f"""
## Recommendations for Large-Scale Deployment

### Optimal Configuration
**Model**: {analysis['recommendations']['best_overall_model']}
**Prompt**: {analysis['recommendations']['best_overall_prompt']}

### Rationale
- Highest success rate across diverse breast cancer trial types
- Reasonable processing time for scalability
- Consistent performance across different trial characteristics

### Implementation Considerations
1. **Cost Management**: Monitor API usage costs for chosen model
2. **Rate Limiting**: Implement appropriate delays for large-scale processing
3. **Quality Assurance**: Regular validation against gold standard
4. **Error Handling**: Robust fallback mechanisms for failed translations

### Scalability Notes
- {analysis['recommendations']['cost_effectiveness_notes']}
- {analysis['recommendations']['scalability_considerations']}

## Conclusion

Based on comprehensive testing across {len(MODELS_TO_TEST) * len(PROMPTS_TO_TEST)} model-prompt combinations, the recommended configuration provides the best balance of accuracy, speed, and reliability for large-scale breast cancer clinical trial mCODE translation.

---
*Report generated by the mCODE Translator Comprehensive Evaluation System*
"""
        
        # Save report
        report_path = self.results_dir / ANALYSIS_REPORT_FILE
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to: {report_path}")
        return report

async def main():
    """
    Main execution function for the comprehensive experiment
    """
    logger.info("🧪 Starting Comprehensive Breast Cancer mCODE Translation Experiment")
    logger.info("=" * 80)
    
    experiment = BreastCancerExperiment()
    
    try:
        # Step 1: Generate gold standard
        logger.info("📋 Step 1: Generating gold standard translations...")
        await experiment.generate_gold_standard()
        
        # Step 2: Run comprehensive experiment
        logger.info("🔬 Step 2: Running comprehensive model-prompt experiment...")
        await experiment.run_comprehensive_experiment()
        
        # Step 3: Analyze results
        logger.info("📊 Step 3: Analyzing results...")
        analysis = experiment.analyze_results()
        
        # Step 4: Generate report
        logger.info("📝 Step 4: Generating comprehensive report...")
        report = experiment.generate_report(analysis)
        
        logger.info("🎉 Experiment completed successfully!")
        logger.info(f"📁 All results saved in: {experiment.results_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"✅ Total combinations tested: {analysis['summary_statistics']['total_combinations']}")
        print(f"🏆 Recommended configuration: {analysis['recommendations']['recommended_combination']}")
        print(f"📈 Overall success rate: {analysis['summary_statistics']['success_rate']:.1%}")
        print(f"📁 Results directory: {experiment.results_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("⚠️ Experiment interrupted by user")
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())