#!/usr/bin/env python3
"""
Demo: Trial Summary with Detailed Codes

This script demonstrates the improved trial summary generation
that includes detailed codes (SNOMED, RxNorm, LOINC, etc.) inline
with mCODE elements for better NLP processing.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.services.summarizer import McodeSummarizer


def main():
    """Demonstrate trial summary with detailed codes."""
    print("🧬 TRIAL SUMMARY WITH DETAILED CODES DEMO")
    print("=" * 60)

    # Load trial data
    trial_file = PROJECT_ROOT / "complete_trial.json"
    if not trial_file.exists():
        print(f"❌ Trial data file not found: {trial_file}")
        return

    print(f"📄 Loading trial data from: {trial_file}")

    with open(trial_file, 'r') as f:
        trial_data = json.load(f)[0]  # Get first trial

    print("✅ Trial data loaded successfully")
    # Create summarizer
    print("\n🔧 Initializing McodeSummarizer...")
    summarizer = McodeSummarizer(include_dates=True)
    print("✅ Summarizer initialized")

    # Generate summary
    print("\n📝 Generating trial summary with detailed codes...")
    summary = summarizer.create_trial_summary(trial_data)

    print("\n" + "=" * 60)
    print("📋 TRIAL SUMMARY OUTPUT")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("🔍 ANALYSIS OF DETAILED CODES")
    print("=" * 60)

    # Analyze the codes in the summary
    code_patterns = [
        ("SNOMED", "SNOMED:"),
        ("MeSH", "MeSH:"),
        ("ClinicalTrials.gov", "ClinicalTrials.gov"),
        ("mCODE", "(mCODE:")
    ]

    for name, pattern in code_patterns:
        count = summary.count(pattern)
        if count > 0:
            print(f"✅ {name}: {count} occurrences")

    print("\n🎯 Key Features Demonstrated:")
    print("   • Active sentence structure with clinical features as subjects")
    print("   • Detailed medical codes inline with mCODE elements")
    print("   • Comprehensive coverage of conditions, interventions, and outcomes")
    print("   • NLP-optimized formatting for better entity extraction")

    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    main()