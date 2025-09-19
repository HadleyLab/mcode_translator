#!/usr/bin/env python3
"""
Debug script to test different types of trials and see what mCODE elements are extracted.
"""

import json
from src.pipeline.pipeline import McodePipeline

def test_trial_types():
    """Test different types of trials to see mCODE extraction differences."""

    # Load trials from raw_trials.ndjson
    trials = []
    with open('raw_trials.ndjson', 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Test first 5 trials
                break
            line = line.strip()
            if line:
                try:
                    trial = json.loads(line)
                    trials.append(trial)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed trial at line {i+1}: {e}")

    print(f"Testing {len(trials)} trials from raw_trials.ndjson")
    print()

    pipeline = McodePipeline(
        model_name="deepseek-coder",
        prompt_name="direct_mcode_evidence_based_concise"
    )

    for i, trial in enumerate(trials):
        nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')
        brief_title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'unknown')
        conditions = trial.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])

        print(f"=== TRIAL {i+1}: {nct_id} ===")
        print(f"Title: {brief_title}")
        print(f"Conditions: {conditions}")
        print()

        try:
            result = pipeline.process(trial)
            print(f"mCODE elements extracted: {len(result.mcode_mappings)}")
            print(f"Compliance score: {result.validation_results.compliance_score}")

            if result.mcode_mappings:
                print("Sample elements:")
                for j, element in enumerate(result.mcode_mappings[:3]):
                    print(f"  {j+1}. {element.element_type}: {element.display}")
            else:
                print("❌ No mCODE elements extracted!")

        except Exception as e:
            print(f"❌ Error processing trial: {e}")

        print("-" * 50)
        print()

if __name__ == "__main__":
    test_trial_types()