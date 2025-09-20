#!/usr/bin/env python3
"""
Test script for inter-rater reliability analysis using cached optimization results.

This script demonstrates how to run inter-rater reliability analysis
on existing optimization data without making new API calls.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from src.optimization.inter_rater_reliability import InterRaterReliabilityAnalyzer, RaterResult
from src.shared.models import McodeElement


def load_cached_optimization_results() -> Dict[str, Dict[str, RaterResult]]:
    """Load cached optimization results and convert to RaterResult format."""
    optimization_dir = Path("optimization_runs")
    rater_results = {}

    # Load successful optimization runs
    successful_runs = [
        "run_20250919_203657_deepseek-coder_direct_mcode_evidence_based_concise.json",
        "run_20250919_203657_deepseek-chat_direct_mcode_evidence_based_concise.json",
        "run_20250919_203657_deepseek-coder_direct_mcode_evidence_based.json"
    ]

    for run_file in successful_runs:
        run_path = optimization_dir / run_file
        if run_path.exists():
            with open(run_path, "r") as f:
                run_data = json.load(f)

            if run_data.get("success"):
                # Extract rater info from filename
                parts = run_file.replace("run_20250919_203657_", "").replace(".json", "").split("_")
                model = parts[0]
                prompt = "_".join(parts[1:])

                # Get predicted mCODE elements and convert to McodeElement objects
                predicted_mcode_dicts = run_data.get("predicted_mcode", [])
                predicted_mcode = []

                for elem_dict in predicted_mcode_dicts:
                    try:
                        elem = McodeElement(**elem_dict)
                        predicted_mcode.append(elem)
                    except Exception as e:
                        print(f"Warning: Failed to convert element {elem_dict}: {e}")

                # Create RaterResult
                rater_result = RaterResult(
                    rater_id=f"{model}_{prompt}",
                    trial_id="cached_trial",  # Use a generic trial ID since we don't have individual trials
                    mcode_elements=predicted_mcode,
                    success=True,
                    execution_time_seconds=0.0
                )

                # Store in results
                trial_id = "cached_trial"
                if trial_id not in rater_results:
                    rater_results[trial_id] = {}
                rater_results[trial_id][f"{model}_{prompt}"] = rater_result

    return rater_results


def test_inter_rater_cached():
    """Test the inter-rater reliability analysis using cached data."""

    print("Loading cached optimization results...")
    cached_results = load_cached_optimization_results()

    if not cached_results:
        print("❌ No cached results found")
        return

    print(f"Loaded results for {len(cached_results)} trials")

    # Initialize analyzer
    analyzer = InterRaterReliabilityAnalyzer()
    analyzer.initialize()

    # Manually set the cached results
    analyzer.rater_results = cached_results

    # Run analysis
    print("\n🔬 Running inter-rater reliability analysis on cached data...")
    analysis = analyzer._analyze_agreement()

    # Set the analysis results
    analyzer.analysis_results = analysis

    # Save results
    print("💾 Saving results...")
    analyzer.save_results()

    # Generate and save report
    print("📊 Generating report...")
    report = analyzer.generate_report()
    report_path = Path("optimization_runs") / "cached_inter_rater_reliability_report.md"
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
            if metrics.fleiss_kappa != 0:
                print(f"      Fleiss' Kappa: {metrics.fleiss_kappa:.3f}")
            elif metrics.cohens_kappa != 0:
                print(f"      Cohen's Kappa: {metrics.cohens_kappa:.3f}")

    print(f"\nDetailed results saved to inter_rater_analysis_results/")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    test_inter_rater_cached()