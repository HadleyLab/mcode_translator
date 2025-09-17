#!/usr/bin/env python3
"""
Test script for the new ultra-lean McodePipeline
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline import McodePipeline
from src.utils.logging_config import setup_logging

def main():
    """Test the new pipeline with sample data"""
    setup_logging()

    # Load sample trial data
    with open("tests/data/sample_trial.json", "r") as f:
        trial_data = json.load(f)

    print("🚀 Testing new ultra-lean McodePipeline...")
    print(f"📋 Sample trial: {trial_data['protocolSection']['identificationModule']['nctId']}")

    # Initialize pipeline
    pipeline = McodePipeline()

    # Process trial
    print("⚙️ Processing trial...")
    result = pipeline.process(trial_data)

    # Display results
    print("\n✅ Processing completed!")
    print(f"📊 Mapped {len(result.mcode_mappings)} mCODE elements")
    print(f"🎯 Compliance score: {result.validation_results.compliance_score:.2f}")
    print(f"🤖 Model used: {result.metadata.model_used}")
    print(f"📝 Prompt used: {result.metadata.prompt_used}")

    if result.mcode_mappings:
        print("\n📋 Sample mCODE mappings:")
        for i, elem in enumerate(result.mcode_mappings[:3]):  # Show first 3
            print(f"  {i+1}. {elem.element_type}: {elem.display or elem.code}")

    if result.error:
        print(f"❌ Error: {result.error}")
    else:
        print("✅ No errors - pipeline working correctly!")

if __name__ == "__main__":
    main()