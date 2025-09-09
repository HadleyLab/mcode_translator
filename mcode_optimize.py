#!/usr/bin/env python3
"""
mCODE Optimization Suite - Comprehensive Prompt × Model Testing

Optimizes mCODE translation by testing all available prompt and model combinations
across clinical trials for comprehensive validation with caching for speed.

Author: mCODE Translation Team
Version: 2.0.0
License: MIT
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse

from src.pipeline import McodePipeline
from src.utils.config import Config
from src.utils.logging_config import get_logger, setup_logging
from src.utils.prompt_loader import PromptLoader
from src.utils.model_loader import ModelLoader
from src.pipeline.task_queue import (
    PipelineTaskQueue, 
    BenchmarkTask, 
    initialize_task_queue,
    shutdown_task_queue
)
from src.shared.types import TaskStatus


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for mCODE optimization suite."""
    parser = argparse.ArgumentParser(
        description="mCODE Optimization Suite - Test all prompt × model combinations to find optimal configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full optimization
  python mcode_optimize.py
  
  # Optimize with specific models only
  python mcode_optimize.py --models deepseek-coder,gpt-4o
  
  # Optimize with specific prompts only  
  python mcode_optimize.py --prompts direct_mcode_evidence_based_concise,direct_mcode_simple
  
  # Fast optimization (limit combinations)
  python mcode_optimize.py --max-combinations 10
  
  # Use more workers for faster processing
  python mcode_optimize.py --workers 10
  
  # Generate detailed optimization report
  python mcode_optimize.py --detailed-report
  python mcode_optimize.py --detailed-report
        """
    )
    
    parser.add_argument(
        "--models",
        help="Comma-separated list of models to test (default: all available)"
    )
    
    parser.add_argument(
        "--prompts", 
        help="Comma-separated list of prompts to test (default: all available)"
    )
    
    parser.add_argument(
        "--trials-file",
        default="data/selected_breast_cancer_trials.json",
        help="Path to trials data file (default: data/selected_breast_cancer_trials.json)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="optimization_results",
        help="Output directory for results (default: optimization_results)"
    )
    
    parser.add_argument(
        "--max-combinations",
        type=int,
        help="Maximum number of prompt×model combinations to test"
    )
    
    parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generate detailed analysis report"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent workers (default: 3)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


class CrossValidationSuite:
    """
    Comprehensive optimization testing suite for mCODE translation.
    Tests all prompt × model combinations to find optimal configurations using async task queue.
    """
    
    def __init__(self, output_dir: str = "optimization_results"):
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize loaders
        self.config = Config()
        self.prompt_loader = PromptLoader()
        self.model_loader = ModelLoader()
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.summary_stats: Dict[str, Any] = {}
        
        # Task queue for async processing
        self.task_queue: Optional[PipelineTaskQueue] = None
        self.task_results: Dict[str, Any] = {}
        
    async def initialize(self, max_workers: int = 3) -> None:
        """Initialize the async task queue and workers."""
        self.logger.info(f"🚀 Initializing optimization suite with {max_workers} workers")
        self.task_queue = await initialize_task_queue(max_workers=max_workers)
        
    async def shutdown(self) -> None:
        """Shutdown the task queue and workers."""
        if self.task_queue:
            self.logger.info("🛑 Shutting down optimization suite")
            await shutdown_task_queue()
            self.task_queue = None
        
    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt names."""
        prompts_config = self.prompt_loader.list_available_prompts()
        prompt_names = list(prompts_config.keys())
        
        self.logger.info(f"📋 Found {len(prompt_names)} available prompts: {prompt_names}")
        return prompt_names
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        models_config = self.model_loader.list_available_models()
        model_names = list(models_config.keys())
        
        self.logger.info(f"🧠 Found {len(model_names)} available models: {model_names}")
        return model_names
    
    def load_trials(self, trials_file: str) -> List[Dict[str, Any]]:
        """Load trial data from file."""
        trials_path = Path(trials_file)
        
        if not trials_path.exists():
            raise FileNotFoundError(f"Trials file not found: {trials_path}")
        
        with open(trials_path, 'r') as f:
            data = json.load(f)
        
        # Handle different file formats
        if isinstance(data, list):
            trials = data
        elif isinstance(data, dict):
            if 'successful_trials' in data:
                trials = data['successful_trials']
            elif 'trials' in data:
                trials = data['trials']
            else:
                trials = [data]  # Single trial
        else:
            raise ValueError("Invalid trials file format")
        
        # Limit to 5 trials for optimization
        trials = trials[:5]
        
        self.logger.info(f"📥 Loaded {len(trials)} trials for optimization")
        return trials
    
    def generate_combinations(
        self, 
        prompts: List[str], 
        models: List[str], 
        max_combinations: int = None
    ) -> List[Tuple[str, str]]:
        """Generate prompt × model combinations."""
        combinations = [(prompt, model) for prompt in prompts for model in models]
        
        if max_combinations and len(combinations) > max_combinations:
            # Take a representative sample
            step = len(combinations) // max_combinations
            combinations = combinations[::step][:max_combinations]
            self.logger.info(f"🎯 Limited to {len(combinations)} combinations (sampling)")
        
        self.logger.info(f"🔄 Generated {len(combinations)} prompt × model combinations")
        return combinations
    
    async def run_optimization(
        self, 
        combinations: List[Tuple[str, str]], 
        trials: List[Dict[str, Any]], 
        max_workers: int = 3
    ) -> None:
        """Run optimization using async task queue with workers."""
        if not self.task_queue:
            await self.initialize(max_workers)
            
        self.logger.info(f"� Running optimization with {max_workers} workers")
        
        start_time = time.time()
        
        # Create benchmark tasks for each combination/trial pair
        tasks_created = []
        task_callbacks = {}
        
        for prompt_name, model_name in combinations:
            for i, trial in enumerate(trials):
                # Create a benchmark task for this combination/trial pair
                trial_id = self._extract_trial_id(trial, i)
                task = BenchmarkTask(
                    task_id=str(uuid.uuid4()),
                    prompt_name=prompt_name,
                    model_name=model_name,
                    trial_id=trial_id,
                    trial_data=trial,
                    prompt_type="DIRECT_MCODE",
                    pipeline_type="DIRECT_MCODE"
                )
                
                # Add callback to track results
                def create_callback(combo_info):
                    def callback(completed_task: BenchmarkTask):
                        self.task_results[completed_task.task_id] = {
                            'combination': combo_info,
                            'task': completed_task
                        }
                    return callback
                
                combo_info = (prompt_name, model_name, i, trial)
                task_callbacks[task.task_id] = create_callback(combo_info)
                
                # Submit task to queue
                task_id = await self.task_queue.add_task(task, task_callbacks[task.task_id])
                tasks_created.append((task_id, prompt_name, model_name, i))
        
        self.logger.info(f"📤 Submitted {len(tasks_created)} tasks to queue")
        
        # Wait for all tasks to complete with progress updates
        await self._wait_for_completion_with_progress(len(tasks_created), len(combinations))
        
        # Process results by combination
        self._process_combination_results(combinations, trials, len(tasks_created))
        
        duration = time.time() - start_time
        self.logger.info(f"🏁 Optimization completed in {duration:.2f} seconds")
        
    async def _wait_for_completion_with_progress(self, total_tasks: int, total_combinations: int) -> None:
        """Wait for all tasks to complete with periodic progress updates."""
        self.logger.info("📊 Starting progress monitoring...")
        
        completed_last_check = 0
        start_time = time.time()
        
        while True:
            await asyncio.sleep(2.0)  # Check every 2 seconds
            
            stats = self.task_queue.get_task_stats()
            completed = stats['completed_tasks']
            
            if completed >= total_tasks:
                break
            
            # Calculate progress metrics
            elapsed = time.time() - start_time
            progress_pct = (completed / total_tasks) * 100
            
            # Calculate processing rate
            new_completions = completed - completed_last_check
            rate = new_completions / 2.0  # tasks per second (checked every 2 seconds)
            
            # Estimate time remaining
            if rate > 0:
                remaining_tasks = total_tasks - completed
                eta_seconds = remaining_tasks / rate
                eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "calculating..."
            
            # Only log progress every 10 seconds to reduce noise
            if elapsed % 10 < 2:  # Log roughly every 10 seconds
                self.logger.info(f"📈 Progress: {completed}/{total_tasks} tasks ({progress_pct:.1f}%) "
                               f"- Rate: {rate:.1f} tasks/sec - ETA: {eta_str} "
                               f"- Workers: {stats['workers_running']}")
            
            completed_last_check = completed
        
        # Final progress message
        final_elapsed = time.time() - start_time
        final_rate = total_tasks / final_elapsed if final_elapsed > 0 else 0
        self.logger.info(f"🏁 All tasks completed in {final_elapsed:.2f}s "
                       f"(avg rate: {final_rate:.1f} tasks/sec)")
    
    def _process_combination_results(self, combinations: List[Tuple[str, str]], trials: List[Dict[str, Any]], total_tasks: int) -> None:
        """Process task results and organize by combination."""
        combination_results = {}
        
        # Group results by combination
        for task_id, result_data in self.task_results.items():
            task = result_data['task']
            combo_info = result_data['combination']
            prompt_name, model_name, trial_index, trial_data = combo_info
            
            combo_key = f"{prompt_name}_{model_name}"
            
            if combo_key not in combination_results:
                combination_results[combo_key] = {
                    'prompt_name': prompt_name,
                    'model_name': model_name,
                    'trial_results': [],
                    'successful_trials': 0,
                    'total_mappings': 0,
                    'total_entities': 0,
                    'quality_scores': []
                }
            
            # Process individual trial result
            trial_id = self._extract_trial_id(trial_data, trial_index)
            
            if task.status == TaskStatus.SUCCESS and task.result:
                num_mappings = len(task.result.mcode_mappings)
                num_entities = len(task.result.extracted_entities)
                quality_score = task.result.validation_results.get('compliance_score', 0.0)
                
                trial_result = {
                    'trial_id': trial_id,
                    'trial_index': trial_index,
                    'success': True,
                    'num_mappings': num_mappings,
                    'num_entities': num_entities,
                    'quality_score': quality_score,
                    'token_usage': task.result.token_usage,
                    'error': None
                }
                
                combination_results[combo_key]['successful_trials'] += 1
                combination_results[combo_key]['total_mappings'] += num_mappings
                combination_results[combo_key]['total_entities'] += num_entities
                combination_results[combo_key]['quality_scores'].append(quality_score)
                
                self.logger.info(f"  ✅ {prompt_name} × {model_name} - Trial {trial_index+1}: {num_mappings} mappings, quality {quality_score:.3f}")
            else:
                trial_result = {
                    'trial_id': trial_id,
                    'trial_index': trial_index,
                    'success': False,
                    'error': task.error_message,
                    'num_mappings': 0,
                    'num_entities': 0,
                    'quality_score': 0.0
                }
                
                self.logger.error(f"  ❌ {prompt_name} × {model_name} - Trial {trial_index+1} failed: {task.error_message}")
            
            combination_results[combo_key]['trial_results'].append(trial_result)
        
        # Calculate final combination metrics and add to results
        for combo_key, combo_data in combination_results.items():
            num_trials = len(combo_data['trial_results'])
            successful_trials = combo_data['successful_trials']
            quality_scores = combo_data['quality_scores']
            
            combination_result = {
                'combination_id': combo_key,
                'prompt_name': combo_data['prompt_name'],
                'model_name': combo_data['model_name'],
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': 0.0,  # Individual combination duration not tracked in async mode
                'total_trials': num_trials,
                'successful_trials': successful_trials,
                'success_rate': successful_trials / num_trials if num_trials > 0 else 0.0,
                'total_mappings': combo_data['total_mappings'],
                'total_entities': combo_data['total_entities'],
                'avg_mappings_per_trial': combo_data['total_mappings'] / num_trials if num_trials > 0 else 0,
                'avg_entities_per_trial': combo_data['total_entities'] / num_trials if num_trials > 0 else 0,
                'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                'trial_results': combo_data['trial_results']
            }
            
            self.results.append(combination_result)
            
            self.logger.info(f"📊 Combination complete: {combo_data['prompt_name']} × {combo_data['model_name']} - "
                           f"{successful_trials}/{num_trials} trials successful, "
                           f"avg quality {combination_result['avg_quality_score']:.3f}, "
                           f"{combo_data['total_mappings']} total mappings")
    
    def _extract_trial_id(self, trial_data: Dict[str, Any], index: int) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial_data['protocolSection']['identificationModule']['nctId']
        except (KeyError, TypeError):
            return f"trial_{index}"
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results."""
        if not self.results:
            return {}
        
        # Overall statistics
        total_combinations = len(self.results)
        total_trials = sum(r['total_trials'] for r in self.results)
        total_successful = sum(r['successful_trials'] for r in self.results)
        overall_success_rate = total_successful / total_trials if total_trials > 0 else 0
        
        # Quality metrics
        quality_scores = [r['avg_quality_score'] for r in self.results if r['avg_quality_score'] > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Mapping statistics
        total_mappings = sum(r['total_mappings'] for r in self.results)
        avg_mappings_per_combo = total_mappings / total_combinations if total_combinations > 0 else 0
        
        # Best performers
        best_quality = max(self.results, key=lambda x: x['avg_quality_score'])
        best_mappings = max(self.results, key=lambda x: x['total_mappings'])
        most_reliable = max(self.results, key=lambda x: x['success_rate'])
        
        # Prompt analysis
        prompt_stats = {}
        for result in self.results:
            prompt = result['prompt_name']
            if prompt not in prompt_stats:
                prompt_stats[prompt] = {'quality_scores': [], 'mappings': [], 'success_rates': []}
            prompt_stats[prompt]['quality_scores'].append(result['avg_quality_score'])
            prompt_stats[prompt]['mappings'].append(result['total_mappings'])
            prompt_stats[prompt]['success_rates'].append(result['success_rate'])
        
        # Model analysis
        model_stats = {}
        for result in self.results:
            model = result['model_name']
            if model not in model_stats:
                model_stats[model] = {'quality_scores': [], 'mappings': [], 'success_rates': []}
            model_stats[model]['quality_scores'].append(result['avg_quality_score'])
            model_stats[model]['mappings'].append(result['total_mappings'])
            model_stats[model]['success_rates'].append(result['success_rate'])
        
        analysis = {
            'summary': {
                'total_combinations': total_combinations,
                'total_trials_tested': total_trials,
                'overall_success_rate': overall_success_rate,
                'average_quality_score': avg_quality,
                'total_mappings_generated': total_mappings,
                'avg_mappings_per_combination': avg_mappings_per_combo
            },
            'best_performers': {
                'highest_quality': {
                    'combination': f"{best_quality['prompt_name']} × {best_quality['model_name']}",
                    'quality_score': best_quality['avg_quality_score']
                },
                'most_mappings': {
                    'combination': f"{best_mappings['prompt_name']} × {best_mappings['model_name']}",
                    'total_mappings': best_mappings['total_mappings']
                },
                'most_reliable': {
                    'combination': f"{most_reliable['prompt_name']} × {most_reliable['model_name']}",
                    'success_rate': most_reliable['success_rate']
                }
            },
            'prompt_analysis': {
                prompt: {
                    'avg_quality': sum(stats['quality_scores']) / len(stats['quality_scores']),
                    'avg_mappings': sum(stats['mappings']) / len(stats['mappings']),
                    'avg_success_rate': sum(stats['success_rates']) / len(stats['success_rates'])
                }
                for prompt, stats in prompt_stats.items()
            },
            'model_analysis': {
                model: {
                    'avg_quality': sum(stats['quality_scores']) / len(stats['quality_scores']),
                    'avg_mappings': sum(stats['mappings']) / len(stats['mappings']),
                    'avg_success_rate': sum(stats['success_rates']) / len(stats['success_rates'])
                }
                for model, stats in model_stats.items()
            }
        }
        
        self.summary_stats = analysis
        return analysis
    
    def save_results(self, detailed_report: bool = False) -> None:
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_combinations': len(self.results),
                    'total_duration': sum(r['duration_seconds'] for r in self.results)
                },
                'results': self.results,
                'analysis': self.summary_stats
            }, f, indent=2)
        
        self.logger.info(f"💾 Results saved to {results_file}")
        
        # Save summary report
        summary_file = self.output_dir / f"optimization_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
        
        self.logger.info(f"📊 Summary saved to {summary_file}")
        
        # Generate detailed report if requested
        if detailed_report:
            self.generate_detailed_report(timestamp)
    
    def generate_detailed_report(self, timestamp: str) -> None:
        """Generate a detailed markdown report."""
        report_file = self.output_dir / f"optimization_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# mCODE Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary section
            summary = self.summary_stats.get('summary', {})
            f.write("## Summary\n\n")
            f.write(f"- **Total Combinations Tested**: {summary.get('total_combinations', 0)}\n")
            f.write(f"- **Overall Success Rate**: {summary.get('overall_success_rate', 0):.1%}\n")
            f.write(f"- **Average Quality Score**: {summary.get('average_quality_score', 0):.3f}\n")
            f.write(f"- **Total mCODE Mappings**: {summary.get('total_mappings_generated', 0)}\n\n")
            
            # Best performers
            best = self.summary_stats.get('best_performers', {})
            f.write("## Best Performers\n\n")
            f.write(f"- **Highest Quality**: {best.get('highest_quality', {}).get('combination', 'N/A')} "
                   f"({best.get('highest_quality', {}).get('quality_score', 0):.3f})\n")
            f.write(f"- **Most Mappings**: {best.get('most_mappings', {}).get('combination', 'N/A')} "
                   f"({best.get('most_mappings', {}).get('total_mappings', 0)} mappings)\n")
            f.write(f"- **Most Reliable**: {best.get('most_reliable', {}).get('combination', 'N/A')} "
                   f"({best.get('most_reliable', {}).get('success_rate', 0):.1%} success)\n\n")
            
            # Prompt analysis
            prompt_analysis = self.summary_stats.get('prompt_analysis', {})
            f.write("## Prompt Analysis\n\n")
            f.write("| Prompt | Avg Quality | Avg Mappings | Success Rate |\n")
            f.write("|--------|-------------|--------------|---------------|\n")
            for prompt, stats in prompt_analysis.items():
                f.write(f"| {prompt} | {stats['avg_quality']:.3f} | {stats['avg_mappings']:.1f} | {stats['avg_success_rate']:.1%} |\n")
            f.write("\n")
            
            # Model analysis
            model_analysis = self.summary_stats.get('model_analysis', {})
            f.write("## Model Analysis\n\n")
            f.write("| Model | Avg Quality | Avg Mappings | Success Rate |\n")
            f.write("|-------|-------------|--------------|---------------|\n")
            for model, stats in model_analysis.items():
                f.write(f"| {model} | {stats['avg_quality']:.3f} | {stats['avg_mappings']:.1f} | {stats['avg_success_rate']:.1%} |\n")
        
        self.logger.info(f"📄 Detailed report saved to {report_file}")


async def main():
    """Main entry point for optimization suite."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting mCODE Optimization Suite")
    
    suite = None
    try:
        # Initialize suite
        suite = CrossValidationSuite(args.output_dir)
        
        # Load trials
        trials = suite.load_trials(args.trials_file)
        
        # Get available prompts and models
        if args.prompts:
            prompts = [p.strip() for p in args.prompts.split(',')]
        else:
            prompts = suite.get_available_prompts()
        
        if args.models:
            models = [m.strip() for m in args.models.split(',')]
        else:
            models = suite.get_available_models()
        
        # Generate combinations
        combinations = suite.generate_combinations(prompts, models, args.max_combinations)
        
        logger.info(f"🎯 Testing {len(combinations)} combinations across {len(trials)} trials")
        logger.info(f"📊 Total tests: {len(combinations) * len(trials)}")
        
        # Run optimization
        start_time = time.time()
        
        await suite.run_optimization(combinations, trials, args.workers)
        
        duration = time.time() - start_time
        
        # Analyze results
        logger.info("📈 Analyzing results...")
        analysis = suite.analyze_results()
        
        # Save results
        suite.save_results(args.detailed_report)
        
        # Print summary
        logger.info("🎉 Optimization complete!")
        logger.info(f"⏱️  Total duration: {duration:.2f} seconds")
        logger.info(f"📊 Tested {len(combinations)} combinations")
        logger.info(f"✅ Overall success rate: {analysis['summary']['overall_success_rate']:.1%}")
        logger.info(f"🎯 Average quality score: {analysis['summary']['average_quality_score']:.3f}")
        logger.info(f"📋 Total mCODE mappings: {analysis['summary']['total_mappings_generated']}")
        
        # Show best performer
        best = analysis['best_performers']['highest_quality']
        logger.info(f"🏆 Best combination: {best['combination']} (quality: {best['quality_score']:.3f})")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise
    finally:
        # Ensure proper cleanup
        if suite:
            await suite.shutdown()


if __name__ == "__main__":
    asyncio.run(main())