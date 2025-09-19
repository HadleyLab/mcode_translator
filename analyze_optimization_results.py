#!/usr/bin/env python3
"""
Analyze Optimization Results - Generate comprehensive performance analysis of mCODE models.

This script analyzes optimization runs and generates detailed performance metrics including
time, token usage, cost analysis, and mCODE generation quality using the new PerformanceMetrics system.
"""

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class OptimizationResultsAnalyzer:
    """Analyze optimization results and generate performance metrics."""

    def __init__(self, optimization_runs_dir: str = "optimization_runs"):
        self.runs_dir = Path(optimization_runs_dir)
        self.results = []
        self.model_stats = defaultdict(list)
        self.combination_stats = defaultdict(list)

    def load_all_results(self) -> None:
        """Load all optimization run results."""
        if not self.runs_dir.exists():
            print(f"❌ Optimization runs directory not found: {self.runs_dir}")
            return

        run_files = list(self.runs_dir.glob("run_*.json"))
        if not run_files:
            print(f"❌ No optimization run files found in {self.runs_dir}")
            return

        print(f"📂 Found {len(run_files)} optimization run files")

        for run_file in run_files:
            try:
                with open(run_file, 'r') as f:
                    result = json.load(f)
                    self.results.append(result)

                    # Extract model and prompt info
                    combination = result.get('combination', {})
                    model = combination.get('model', 'unknown')
                    prompt = combination.get('prompt', 'unknown')

                    # Store by model
                    self.model_stats[model].append(result)

                    # Store by combination
                    combo_key = f"{model}_{prompt}"
                    self.combination_stats[combo_key].append(result)

                print(f"✅ Loaded: {run_file.name}")

            except Exception as e:
                print(f"❌ Failed to load {run_file.name}: {e}")

        print(f"📊 Loaded {len(self.results)} total results")

    def analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics for each model."""
        analysis = {
            'summary': {
                'total_runs': len(self.results),
                'models_tested': len(self.model_stats),
                'combinations_tested': len(self.combination_stats),
                'timestamp': datetime.now().isoformat()
            },
            'model_performance': {},
            'combination_performance': {},
            'overall_metrics': {}
        }

        # Analyze each model
        for model, results in self.model_stats.items():
            model_analysis = self._analyze_model_results(model, results)
            analysis['model_performance'][model] = model_analysis

        # Analyze each combination
        for combo_key, results in self.combination_stats.items():
            combo_analysis = self._analyze_combination_results(combo_key, results)
            analysis['combination_performance'][combo_key] = combo_analysis

        # Calculate overall metrics
        analysis['overall_metrics'] = self._calculate_overall_metrics()

        return analysis

    def _analyze_model_results(self, model: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for a specific model using new PerformanceMetrics."""
        if not results:
            return {'error': 'No results available'}

        # Extract metrics using new PerformanceMetrics system
        cv_scores = []
        element_counts = []
        performance_data = []
        mcode_elements = []

        for result in results:
            cv_scores.append(result.get('cv_average_score', 0))
            element_counts.append(result.get('total_elements', 0))

            # Extract performance metrics
            perf_metrics = result.get('performance_metrics', {})
            if perf_metrics:
                performance_data.append(perf_metrics)

            # Extract mCODE elements for detailed analysis
            predicted_mcode = result.get('predicted_mcode', [])
            mcode_elements.extend(predicted_mcode)

        # Calculate performance statistics
        perf_stats = self._calculate_performance_stats(performance_data)

        # Calculate statistics
        analysis = {
            'runs_count': len(results),
            'cv_score': {
                'mean': statistics.mean(cv_scores) if cv_scores else 0,
                'median': statistics.median(cv_scores) if cv_scores else 0,
                'std': statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0,
                'min': min(cv_scores) if cv_scores else 0,
                'max': max(cv_scores) if cv_scores else 0
            },
            'elements_generated': {
                'mean': statistics.mean(element_counts) if element_counts else 0,
                'median': statistics.median(element_counts) if element_counts else 0,
                'total': sum(element_counts)
            },
            'performance_rating': self._rate_performance(cv_scores),
            'performance_metrics': perf_stats,
            'mcode_analysis': self._analyze_mcode_elements(mcode_elements)
        }

        return analysis

    def _analyze_combination_results(self, combo_key: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for a specific model-prompt combination."""
        if not results:
            return {'error': 'No results available'}

        # Extract metrics
        cv_scores = [r.get('cv_average_score', 0) for r in results]
        element_counts = [r.get('total_elements', 0) for r in results]

        # Extract performance metrics
        performance_data = []
        for r in results:
            perf_metrics = r.get('performance_metrics', {})
            if perf_metrics:
                performance_data.append(perf_metrics)

        perf_stats = self._calculate_performance_stats(performance_data)

        return {
            'runs_count': len(results),
            'cv_score': {
                'mean': statistics.mean(cv_scores) if cv_scores else 0,
                'std': statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0
            },
            'elements_generated': {
                'mean': statistics.mean(element_counts) if element_counts else 0,
                'total': sum(element_counts)
            },
            'performance_metrics': perf_stats
        }

    def _calculate_performance_stats(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Calculate performance statistics from PerformanceMetrics data."""
        if not performance_data:
            return {}

        # Extract metrics
        processing_times = [p.get('processing_time_seconds', 0) for p in performance_data]
        tokens_used = [p.get('tokens_used', 0) for p in performance_data]
        costs = [p.get('estimated_cost_usd', 0) for p in performance_data]
        elements_per_sec = [p.get('elements_per_second', 0) for p in performance_data]
        tokens_per_sec = [p.get('tokens_per_second', 0) for p in performance_data]

        return {
            'processing_time': {
                'mean': statistics.mean(processing_times) if processing_times else 0,
                'std': statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                'total': sum(processing_times)
            },
            'token_usage': {
                'mean': statistics.mean(tokens_used) if tokens_used else 0,
                'std': statistics.stdev(tokens_used) if len(tokens_used) > 1 else 0,
                'total': sum(tokens_used)
            },
            'cost_analysis': {
                'mean_usd': statistics.mean(costs) if costs else 0,
                'std_usd': statistics.stdev(costs) if len(costs) > 1 else 0,
                'total_usd': sum(costs)
            },
            'throughput': {
                'elements_per_second': statistics.mean(elements_per_sec) if elements_per_sec else 0,
                'tokens_per_second': statistics.mean(tokens_per_sec) if tokens_per_sec else 0
            }
        }

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics across all results."""
        if not self.results:
            return {}

        all_cv_scores = [r.get('cv_average_score', 0) for r in self.results]
        all_elements = [r.get('total_elements', 0) for r in self.results]

        # Aggregate performance metrics
        all_performance_data = []
        for r in self.results:
            perf_metrics = r.get('performance_metrics', {})
            if perf_metrics:
                all_performance_data.append(perf_metrics)

        perf_stats = self._calculate_performance_stats(all_performance_data)

        return {
            'total_cv_scores': len(all_cv_scores),
            'average_cv_score': statistics.mean(all_cv_scores) if all_cv_scores else 0,
            'total_elements_generated': sum(all_elements),
            'average_elements_per_run': statistics.mean(all_elements) if all_elements else 0,
            'performance_summary': perf_stats
        }

    def _analyze_mcode_elements(self, mcode_elements: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive analysis of mCODE elements."""
        if not mcode_elements:
            return {'error': 'No mCODE elements to analyze'}

        # Element type distribution
        element_types = defaultdict(int)
        code_systems = defaultdict(int)
        confidence_scores = []
        elements_with_codes = 0
        elements_with_confidence = 0

        for element in mcode_elements:
            # Count element types
            element_type = element.get('element_type', 'unknown')
            element_types[element_type] += 1

            # Count code systems
            code = element.get('code')
            system = element.get('system')
            if code and system:
                code_systems[system] += 1
                elements_with_codes += 1

            # Collect confidence scores
            confidence = element.get('confidence_score')
            if confidence is not None:
                confidence_scores.append(confidence)
                elements_with_confidence += 1

        # Calculate statistics
        analysis = {
            'total_elements': len(mcode_elements),
            'element_types': dict(element_types),
            'code_systems': dict(code_systems),
            'quality_metrics': {
                'elements_with_codes': elements_with_codes,
                'elements_with_confidence': elements_with_confidence,
                'code_coverage': elements_with_codes / len(mcode_elements) if mcode_elements else 0,
                'confidence_coverage': elements_with_confidence / len(mcode_elements) if mcode_elements else 0
            }
        }

        # Confidence score statistics
        if confidence_scores:
            analysis['confidence_stats'] = {
                'mean': statistics.mean(confidence_scores),
                'median': statistics.median(confidence_scores),
                'std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
                'min': min(confidence_scores),
                'max': max(confidence_scores)
            }

        # Element type diversity
        analysis['diversity_metrics'] = {
            'unique_element_types': len(element_types),
            'most_common_type': max(element_types.items(), key=lambda x: x[1]) if element_types else None,
            'element_type_distribution': {k: v/len(mcode_elements) for k, v in element_types.items()}
        }

        return analysis

    def _rate_performance(self, cv_scores: List[float]) -> str:
        """Rate model performance based on CV scores."""
        if not cv_scores:
            return 'unknown'

        avg_score = statistics.mean(cv_scores)

        if avg_score >= 0.8:
            return 'excellent'
        elif avg_score >= 0.6:
            return 'good'
        elif avg_score >= 0.4:
            return 'fair'
        elif avg_score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'

    def generate_report(self, output_file: str = "optimization_analysis_report.json") -> None:
        """Generate comprehensive analysis report."""
        analysis = self.analyze_model_performance()

        # Save detailed report
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"💾 Saved analysis report to: {output_file}")

        # Print summary to console
        self._print_summary_report(analysis)

    def _print_summary_report(self, analysis: Dict[str, Any]) -> None:
        """Print summary report to console."""
        print("\n" + "="*60)
        print("🎯 MCODE OPTIMIZATION ANALYSIS REPORT")
        print("="*60)

        summary = analysis.get('summary', {})
        print(f"📊 Total Runs: {summary.get('total_runs', 0)}")
        print(f"🤖 Models Tested: {summary.get('models_tested', 0)}")
        print(f"🔄 Combinations Tested: {summary.get('combinations_tested', 0)}")

        print("\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 40)

        # Sort models by average CV score
        model_perf = analysis.get('model_performance', {})
        sorted_models = sorted(
            model_perf.items(),
            key=lambda x: x[1].get('cv_score', {}).get('mean', 0),
            reverse=True
        )

        for i, (model, perf) in enumerate(sorted_models, 1):
            cv_score = perf.get('cv_score', {})
            rating = perf.get('performance_rating', 'unknown')
            runs = perf.get('runs_count', 0)
            perf_metrics = perf.get('performance_metrics', {})

            print(f"{i}. {model}")
            print(f"   📈 CV Score: {cv_score.get('mean', 0):.3f} ± {cv_score.get('std', 0):.3f}")
            print(f"   📊 Rating: {rating.upper()}")
            print(f"   🎯 Runs: {runs}")

            # Show performance metrics if available
            if perf_metrics:
                processing = perf_metrics.get('processing_time', {})
                tokens = perf_metrics.get('token_usage', {})
                cost = perf_metrics.get('cost_analysis', {})

                if processing.get('mean', 0) > 0:
                    print(f"   ⏱️  Time: {processing.get('mean', 0):.2f}s")
                if tokens.get('mean', 0) > 0:
                    print(f"   🎫 Tokens: {tokens.get('mean', 0):.0f}")
                if cost.get('mean_usd', 0) > 0:
                    print(f"   💰 Cost: ${cost.get('mean_usd', 0):.4f}")
            print()

        # Overall metrics
        overall = analysis.get('overall_metrics', {})
        perf_summary = overall.get('performance_summary', {})

        print("📈 OVERALL METRICS:")
        print("-" * 20)
        print(f"📈 Average CV Score: {overall.get('average_cv_score', 0):.3f}")
        print(f"🧬 Total Elements Generated: {overall.get('total_elements_generated', 0)}")
        print(f"📊 Average Elements per Run: {overall.get('average_elements_per_run', 0):.1f}")

        # Performance metrics
        if perf_summary:
            processing = perf_summary.get('processing_time', {})
            tokens = perf_summary.get('token_usage', {})
            cost = perf_summary.get('cost_analysis', {})
            throughput = perf_summary.get('throughput', {})

            print("\n⚡ PERFORMANCE ANALYSIS:")
            print("-" * 25)
            print(f"⏱️  Avg Processing Time: {processing.get('mean', 0):.2f}s ± {processing.get('std', 0):.2f}s")
            print(f"🎫 Avg Token Usage: {tokens.get('mean', 0):.0f} ± {tokens.get('std', 0):.0f}")
            print(f"💰 Avg Cost: ${cost.get('mean_usd', 0):.4f} ± ${cost.get('std_usd', 0):.4f}")
            print(f"🚀 Throughput: {throughput.get('elements_per_second', 0):.1f} elem/s, {throughput.get('tokens_per_second', 0):.0f} tok/s")

        # mCODE Analysis Summary
        print("\n🧬 MCODE ELEMENT ANALYSIS:")
        print("-" * 30)

        # Show best model mCODE analysis
        model_perf = analysis.get('model_performance', {})
        if model_perf:
            best_model = max(model_perf.items(), key=lambda x: x[1].get('cv_score', {}).get('mean', 0))
            mcode_analysis = best_model[1].get('mcode_analysis', {})

            if mcode_analysis and 'error' not in mcode_analysis:
                quality = mcode_analysis.get('quality_metrics', {})
                print("\n🧬 BEST MODEL MCODE ANALYSIS:")
                print("-" * 30)
                print(f"📊 Total Elements: {mcode_analysis.get('total_elements', 0)}")
                print(f"🎯 Unique Types: {mcode_analysis.get('diversity_metrics', {}).get('unique_element_types', 0)}")
                print(f"🔢 Code Coverage: {quality.get('code_coverage', 0):.1%}")
                print(f"🎯 Confidence Coverage: {quality.get('confidence_coverage', 0):.1%}")

        print("\n✅ Comprehensive Analysis Complete!")


def main():
    """Main entry point for analysis script."""
    print("🔬 Starting mCODE Optimization Results Analysis")

    analyzer = OptimizationResultsAnalyzer()
    analyzer.load_all_results()

    if not analyzer.results:
        print("❌ No results to analyze. Run optimization first.")
        return

    analyzer.generate_report()


if __name__ == "__main__":
    main()