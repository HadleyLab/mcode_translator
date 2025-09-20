#!/usr/bin/env python3
"""
Test script for inter-rater reliability analysis of mCODE elements.

This script demonstrates how to run inter-rater reliability analysis
on clinical trial data using different LLM models and prompts.
"""

import asyncio
import json
from pathlib import Path

from src.optimization.inter_rater_reliability import InterRaterReliabilityAnalyzer


async def test_inter_rater_reliability():
    """Test the inter-rater reliability analysis."""

    # Load trial data
    trials_file = "data/select_breast_cancer_trials.ndjson"
    trials_data = []

    print(f"Loading trials from {trials_file}...")
    with open(trials_file, "r") as f:
        for line in f:
            if line.strip():
                trials_data.append(json.loads(line))

    # Limit to first 3 trials for testing
    trials_data = trials_data[:3]
    print(f"Loaded {len(trials_data)} trials for testing")

    # Define rater configurations (successful combinations from previous runs)
    rater_configs = [
        {"model": "deepseek-coder", "prompt": "direct_mcode_evidence_based_concise"},
        {"model": "deepseek-chat", "prompt": "direct_mcode_evidence_based_concise"},
        {"model": "deepseek-coder", "prompt": "direct_mcode_evidence_based"},
    ]

    print(f"Testing {len(rater_configs)} rater configurations:")
    for config in rater_configs:
        print(f"  - {config['model']} + {config['prompt']}")

    # Initialize analyzer
    analyzer = InterRaterReliabilityAnalyzer()
    analyzer.initialize()

    # Run analysis
    print("\n🔬 Running inter-rater reliability analysis...")
    analysis = await analyzer.run_analysis(trials_data, rater_configs, max_concurrent=2)

    # Save results
    print("💾 Saving results...")
    analyzer.save_results()

    # Generate and save report
    print("📊 Generating report...")
    report = analyzer.generate_report()
    report_path = Path("optimization_runs") / "test_inter_rater_reliability_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"✅ Analysis complete! Report saved to: {report_path}")

    # Print summary
    print("\n📈 Summary:")
    print(f"   Trials analyzed: {analysis.num_trials}")
    print(f"   Raters compared: {analysis.num_raters}")

    if analysis.overall_metrics:
        for metric_name, metrics in analysis.overall_metrics.items():
            print(f"   {metric_name}: {metrics.percentage_agreement:.3f} agreement")
            if hasattr(metrics, 'fleiss_kappa') and metrics.fleiss_kappa != 0:
                print(f"      Fleiss' Kappa: {metrics.fleiss_kappa:.3f}")

    print(f"\nDetailed results saved to optimization_runs/inter_rater_analysis_results/")


if __name__ == "__main__":
    asyncio.run(test_inter_rater_reliability())