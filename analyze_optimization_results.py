#!/usr/bin/env python3
"""
Analyze Optimization Results - Generate comprehensive performance analysis of mCODE models.

This script analyzes the optimization runs saved in the optimization_runs/ directory
and generates detailed performance metrics for each model with respect to mCODE generation.
"""

import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind


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
        """Analyze results for a specific model."""
        if not results:
            return {'error': 'No results available'}

        # Extract metrics
        cv_scores = []
        element_counts = []
        execution_times = []
        mcode_elements = []

        for result in results:
            cv_scores.append(result.get('cv_average_score', 0))
            element_counts.append(result.get('total_elements', 0))

            # Extract mCODE elements for detailed analysis
            predicted_mcode = result.get('predicted_mcode', [])
            mcode_elements.extend(predicted_mcode)

            # Extract execution time if available
            timestamp = result.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    execution_times.append(dt.timestamp())
                except:
                    pass

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
            'mcode_analysis': self._analyze_mcode_elements(mcode_elements)
        }

        return analysis

    def _analyze_combination_results(self, combo_key: str, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for a specific model-prompt combination."""
        if not results:
            return {'error': 'No results available'}

        # Similar analysis as model results but for combinations
        cv_scores = [r.get('cv_average_score', 0) for r in results]
        element_counts = [r.get('total_elements', 0) for r in results]

        return {
            'runs_count': len(results),
            'cv_score': {
                'mean': statistics.mean(cv_scores) if cv_scores else 0,
                'std': statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0
            },
            'elements_generated': {
                'mean': statistics.mean(element_counts) if element_counts else 0,
                'total': sum(element_counts)
            }
        }

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics across all results."""
        if not self.results:
            return {}

        all_cv_scores = [r.get('cv_average_score', 0) for r in self.results]
        all_elements = [r.get('total_elements', 0) for r in self.results]

        return {
            'total_cv_scores': len(all_cv_scores),
            'average_cv_score': statistics.mean(all_cv_scores) if all_cv_scores else 0,
            'total_elements_generated': sum(all_elements),
            'average_elements_per_run': statistics.mean(all_elements) if all_elements else 0
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

            print(f"{i}. {model}")
            print(f"   📈 CV Score: {cv_score.get('mean', 0):.3f} ± {cv_score.get('std', 0):.3f}")
            print(f"   📊 Rating: {rating.upper()}")
            print(f"   🎯 Runs: {runs}")
            print()

        # Overall metrics
        overall = analysis.get('overall_metrics', {})
        print("📈 OVERALL METRICS:")
        print("-" * 20)
        print(f"📈 Average CV Score: {overall.get('average_cv_score', 0):.3f}")
        print(f"🧬 Total Elements Generated: {overall.get('total_elements_generated', 0)}")
        print(f"📊 Average Elements per Run: {overall.get('average_elements_per_run', 0):.1f}")

        # mCODE Analysis Summary
        print("\n🧬 MCODE ELEMENT ANALYSIS:")
        print("-" * 30)

        # Get mCODE analysis from the best performing model
        model_perf = analysis.get('model_performance', {})
        if model_perf:
            best_model = max(model_perf.items(), key=lambda x: x[1].get('cv_score', {}).get('mean', 0))
            mcode_analysis = best_model[1].get('mcode_analysis', {})

            if mcode_analysis and 'error' not in mcode_analysis:
                print(f"📊 Total mCODE Elements: {mcode_analysis.get('total_elements', 0)}")
                print(f"🎯 Unique Element Types: {mcode_analysis.get('diversity_metrics', {}).get('unique_element_types', 0)}")

                # Element type distribution
                element_types = mcode_analysis.get('element_types', {})
                if element_types:
                    print("🔍 Element Type Distribution:")
                    sorted_types = sorted(element_types.items(), key=lambda x: x[1], reverse=True)
                    for elem_type, count in sorted_types[:5]:  # Top 5
                        percentage = (count / mcode_analysis['total_elements']) * 100
                        print(f"   • {elem_type}: {count} ({percentage:.1f}%)")

                # Quality metrics
                quality = mcode_analysis.get('quality_metrics', {})
                if quality:
                    print("✨ Quality Metrics:")
                    print(f"   • Code Coverage: {quality.get('code_coverage', 0):.1%}")
                    print(f"   • Confidence Coverage: {quality.get('confidence_coverage', 0):.1%}")

                # Confidence statistics
                confidence_stats = mcode_analysis.get('confidence_stats', {})
                if confidence_stats:
                    print("🎯 Confidence Statistics:")
                    print(f"   • Mean: {confidence_stats.get('mean', 0):.3f}")
                    print(f"   • Range: {confidence_stats.get('min', 0):.3f} - {confidence_stats.get('max', 0):.3f}")

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