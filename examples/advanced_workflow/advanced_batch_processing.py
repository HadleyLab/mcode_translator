#!/usr/bin/env python3
"""
🚀 mCODE Translator - Advanced Batch Processing Example

This example demonstrates advanced workflow features including:
- Batch processing of multiple clinical trials
- Engine comparison (Regex vs LLM)
- Performance benchmarking
- Result aggregation and analysis
- Error handling and recovery
"""

import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_flow_coordinator import process_clinical_trials_flow


def advanced_batch_processing() -> bool:
    """Demonstrate advanced batch processing workflow."""
    print("🚀 mCODE Translator - Advanced Batch Processing")
    print("=" * 60)

    # Multiple trial IDs for batch processing
    trial_ids = [
        "NCT02364999",  # PALOMA-2 (Breast Cancer)
        "NCT02735178",  # KEYNOTE-042 (Lung Cancer)
        "NCT03470922",  # CheckMate 238 (Melanoma)
    ]

    print(f"🎯 Processing {len(trial_ids)} Clinical Trials")
    print("-" * 40)
    for i, trial_id in enumerate(trial_ids, 1):
        print(f"   {i}. {trial_id}")
    print()

    # Test both engines
    engines = ["regex", "llm"]
    results = {}

    for engine in engines:
        print(f"🔧 Testing {engine.upper()} Engine")
        print("-" * 30)

        config = {
            "validate_data": True,
            "store_results": False,
            "enable_logging": True,
            "processing_engine": engine
        }

        start_time = time.time()

        try:
            result = process_clinical_trials_flow(
                trial_ids=trial_ids,
                config=config
            )

            processing_time = time.time() - start_time
            results[engine] = {
                "result": result,
                "time": processing_time,
                "success": result.success if hasattr(result, 'success') else True
            }

            print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            print(f"   ✅ Success: {results[engine]['success']}")
            print()

        except Exception as e:
            processing_time = time.time() - start_time
            results[engine] = {
                "result": None,
                "time": processing_time,
                "success": False,
                "error": str(e)
            }
            print(f"   ⏱️  Processing Time: {processing_time:.2f}s")
            print(f"   ❌ Error: {e}")
            print()

    # Compare results
    print("📊 ENGINE COMPARISON RESULTS")
    print("=" * 60)

    print("Performance Metrics:")
    print("Engine      | Time (s) | Success | Trials Processed")
    print("------------|----------|---------|------------------")

    for engine in engines:
        data = results[engine]
        time_str = f"{data['time']:.2f}"
        success_str = "✅" if data['success'] else "❌"
        trials_count = len(trial_ids) if data['success'] else 0
        print("12")

    print()

    # Detailed analysis for successful engines
    for engine in engines:
        data = results[engine]
        if not data['success']:
            continue

        print(f"🔍 {engine.upper()} Engine Detailed Results")
        print("-" * 40)

        result = data['result']
        if result and hasattr(result, 'data') and result.data:
            total_elements = 0
            total_confidence = 0.0
            element_count = 0

            for trial_data in result.data:
                if 'mcode_mappings' in trial_data and trial_data['mcode_mappings']:
                    mappings = trial_data['mcode_mappings']
                    total_elements += len(mappings)

                    for mapping in mappings:
                        if 'confidence_score' in mapping and mapping['confidence_score']:
                            total_confidence += mapping['confidence_score']
                            element_count += 1

            avg_confidence = total_confidence / element_count if element_count > 0 else 0

            print(f"   • Total mCODE Elements: {total_elements}")
            print(f"   • Average Confidence: {avg_confidence:.1%}")
            print(f"   • Elements per Trial: {total_elements / len(trial_ids):.1f}")
        print()

    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("=" * 60)

    regex_time = results.get('regex', {}).get('time', float('inf'))
    llm_time = results.get('llm', {}).get('time', float('inf'))

    if regex_time < llm_time:
        speedup = llm_time / regex_time if regex_time > 0 else 1
        print(f"⚡ Regex engine is {speedup:.1f}x faster for this workload")
        print("   → Use RegexEngine for speed-critical batch processing")
    else:
        print("🧠 LLM engine provides intelligent processing")
        print("   → Use LLMEngine when accuracy is more important than speed")

    print()
    print("🎯 Use Case Guidance:")
    print("   • Large datasets (100+ trials): RegexEngine")
    print("   • Complex eligibility criteria: LLMEngine")
    print("   • Research analysis: Both engines for comparison")
    print("   • Production pipelines: RegexEngine with periodic LLM validation")

    print()
    print("🎉 Advanced batch processing example completed!")
    print()
    print("💡 Next Steps:")
    print("   • Try different trial sets")
    print("   • Experiment with batch sizes")
    print("   • Enable CORE Memory storage")
    print("   • Add custom validation rules")

    return all(results[engine]['success'] for engine in engines)


if __name__ == "__main__":
    success = advanced_batch_processing()
    sys.exit(0 if success else 1)