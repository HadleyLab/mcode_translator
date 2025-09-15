#!/usr/bin/env python3
"""
Demo: Patient Summary with Detailed Codes

This script demonstrates the improved patient summary generation
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
    """Demonstrate patient summary with detailed codes."""
    print("👤 PATIENT SUMMARY WITH DETAILED CODES DEMO")
    print("=" * 60)

    # Load patient data
    patient_file = PROJECT_ROOT / "breast_cancer_patients_demo.json"
    if not patient_file.exists():
        print(f"❌ Patient data file not found: {patient_file}")
        print("   Run the full demo first to generate patient data")
        return

    print(f"📄 Loading patient data from: {patient_file}")

    with open(patient_file, 'r') as f:
        patient_data = json.load(f)[0]  # Get first patient

    print("✅ Patient data loaded successfully")

    # Create summarizer
    print("\n🔧 Initializing McodeSummarizer...")
    summarizer = McodeSummarizer(include_dates=True)
    print("✅ Summarizer initialized")

    # Generate summary
    print("\n📝 Generating patient summary with detailed codes...")
    summary = summarizer.create_patient_summary(patient_data)

    print("\n" + "=" * 60)
    print("📋 PATIENT SUMMARY OUTPUT")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("🔍 ANALYSIS OF DETAILED CODES")
    print("=" * 60)

    # Analyze the codes in the summary
    code_patterns = [
        ("SNOMED", "SNOMED:"),
        ("RxNorm", "RxNorm:"),
        ("LOINC", "LOINC:"),
        ("ICD", "ICD:"),
        ("mCODE", "(mCODE:")
    ]

    for name, pattern in code_patterns:
        count = summary.count(pattern)
        if count > 0:
            print(f"✅ {name}: {count} occurrences")

    print("\n🎯 Key Features Demonstrated:")
    print("   • Active sentence structure with clinical features as subjects")
    print("   • Detailed medical codes inline with mCODE elements")
    print("   • Comprehensive coverage of diagnoses, procedures, medications")
    print("   • NLP-optimized formatting for better entity extraction")

    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    main()