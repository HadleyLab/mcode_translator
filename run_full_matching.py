#!/usr/bin/env python3
"""
Run full matching experiment on the curated gold standard dataset (gold_standard_matches.ndjson).
Processes all 10,000 patient-trial pairs using both regex and LLM matching engines.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from matching.regex_engine import RegexRulesEngine
from matching.llm_engine import LLMMatchingEngine
from matching.batch_matcher import BatchMatcher
from config.patterns_config import load_regex_rules


def load_gold_standard_pairs(gold_standard_file: str) -> List[Dict[str, Any]]:
    """Load patient-trial pairs from gold standard dataset."""
    pairs = []
    with open(gold_standard_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                pairs.append(record)
    return pairs


def load_patient_data(patient_id: str, patients_file: str = "selected_patients.ndjson") -> Dict[str, Any]:
    """Load patient data for a given patient ID."""
    with open(patients_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                patient = json.loads(line.strip())
                # Extract patient ID from FHIR bundle
                bundle = patient.get("entry", [])
                for entry in bundle:
                    if entry["resource"]["resourceType"] == "Patient":
                        if entry["resource"]["id"] == patient_id:
                            return patient
    raise ValueError(f"Patient {patient_id} not found in {patients_file}")


def load_trial_data(trial_id: str, trials_file: str = "selected_trials.ndjson") -> Dict[str, Any]:
    """Load trial data for a given trial ID."""
    with open(trials_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trial = json.loads(line.strip())
                # Extract trial ID from ClinicalTrials.gov format
                current_trial_id = trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                if current_trial_id == trial_id:
                    return trial
    raise ValueError(f"Trial {trial_id} not found in {trials_file}")


async def run_full_matching():
    """Run full matching experiment on gold standard dataset."""

    print("🚀 Running Full Patient-Trial Matching Experiment")
    print("=" * 60)
    print("Processing 10,000 patient-trial pairs from gold standard dataset")
    print("=" * 60)

    # Load gold standard pairs
    print("📂 Loading gold standard dataset...")
    try:
        gold_standard_pairs = load_gold_standard_pairs("gold_standard_matches.ndjson")
        print(f"✅ Loaded {len(gold_standard_pairs)} gold standard pairs")
    except Exception as e:
        print(f"❌ Failed to load gold standard dataset: {e}")
        return

    # Initialize engines
    print("🔧 Initializing matching engines...")

    # Regex engine
    try:
        regex_rules = load_regex_rules()
        regex_engine = RegexRulesEngine(rules=regex_rules, cache_enabled=True)
        print("✅ RegexRulesEngine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize regex engine: {e}")
        return

    # LLM engine
    try:
        llm_engine = LLMMatchingEngine(
            model_name="deepseek-coder",
            prompt_name="patient_matcher",
            cache_enabled=True
        )
        print("✅ LLMMatchingEngine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM engine: {e}")
        return

    # Prepare patient-trial pairs for matching
    print("🔄 Preparing patient-trial pairs for matching...")
    matching_pairs = []
    for i, record in enumerate(gold_standard_pairs):
        if i % 1000 == 0:
            print(f"   Processed {i}/{len(gold_standard_pairs)} pairs...")

        try:
            patient_data = load_patient_data(record["patient_id"])
            trial_data = load_trial_data(record["trial_id"])
            matching_pairs.append({
                'patient': patient_data,
                'trial': trial_data,
                'gold_standard': record
            })
        except Exception as e:
            print(f"⚠️  Warning: Failed to load data for pair {record['patient_id']}-{record['trial_id']}: {e}")
            continue

    print(f"✅ Prepared {len(matching_pairs)} matching pairs")

    # Run matching for both engines
    engines_to_run = [
        ("RegexRulesEngine", regex_engine, "regex_matching_results.ndjson"),
        ("LLMMatchingEngine", llm_engine, "llm_matching_results.ndjson")
    ]

    for engine_name, engine, output_file in engines_to_run:
        print(f"\n🎯 Running {engine_name} on {len(matching_pairs)} pairs...")
        print("-" * 50)

        try:
            # Create batch matcher with optimized concurrency
            batch_matcher = BatchMatcher(engine, max_concurrent=10)

            # Process pairs in batches and track progress
            results = []
            batch_size = 100
            total_batches = (len(matching_pairs) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(matching_pairs))
                batch_pairs = matching_pairs[start_idx:end_idx]

                print(f"   Processing batch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx-1})...")

                batch_results = await batch_matcher.match_batch(batch_pairs)
                results.extend(batch_results)

                # Progress update
                processed = len(results)
                success_rate = sum(1 for r in results if not r.error) / len(results) if results else 0
                print(".1f")

            # Calculate statistics
            total_pairs = len(results)
            successful_matches = sum(1 for r in results if not r.error)
            error_matches = total_pairs - successful_matches

            # Calculate confidence scores
            confidence_scores = []
            total_elements = 0
            for result in results:
                if not result.error and result.elements:
                    total_elements += len(result.elements)
                    for element in result.elements:
                        confidence_scores.append(element.confidence_score)

            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            avg_elements_per_match = total_elements / successful_matches if successful_matches > 0 else 0

            # Save results
            print(f"💾 Saving results to {output_file}...")
            batch_matcher.save_results(results, output_file)

            # Save statistics
            stats_file = output_file.replace('.ndjson', '_stats.json')
            stats = {
                'total_pairs': total_pairs,
                'successful_matches': successful_matches,
                'error_matches': error_matches,
                'success_rate': successful_matches / total_pairs if total_pairs > 0 else 0,
                'average_confidence_score': avg_confidence,
                'average_elements_per_match': avg_elements_per_match,
                'total_matched_elements': total_elements,
                'engine_name': engine_name
            }

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            print("✅ Matching completed!")
            print(f"   Results saved to: {output_file}")
            print(f"   Statistics saved to: {stats_file}")
            print(f"   Total pairs processed: {stats['total_pairs']}")
            print(f"   Successful matches: {stats['successful_matches']}")
            print(".3f")
            print(".3f")
            print(".3f")

        except Exception as e:
            print(f"❌ {engine_name} failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    print("\n🎉 Full matching experiment completed for both engines!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_full_matching())