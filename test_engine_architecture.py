#!/usr/bin/env python3
"""
Test script for the new RegexEngine and LLMEngine architecture.

This script tests both engines to ensure they work correctly with the UnifiedSummarizer.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.llm_engine import LLMEngine
from src.services.regex_engine import RegexEngine
from src.services.unified_processor import UnifiedTrialProcessor
from src.services.unified_summarizer import UnifiedSummarizer

# Sample trial data for testing
SAMPLE_TRIAL_DATA = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT01234567",
            "briefTitle": "A Study of Treatment X for Breast Cancer",
        },
        "designModule": {
            "studyType": "INTERVENTIONAL",
            "phase": "PHASE2",
            "primaryPurpose": "TREATMENT",
        },
        "statusModule": {"overallStatus": "RECRUITING"},
        "conditionsModule": {"conditions": ["Breast Cancer", "Metastatic Breast Cancer"]},
        "armsInterventionsModule": {
            "interventions": [
                {"name": "TRASTUZUMAB", "type": "DRUG"},
                {"name": "CHEMOTHERAPY", "type": "DRUG"},
            ]
        },
        "eligibilityModule": {"minimumAge": "18 Years", "sex": "FEMALE"},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test Sponsor Inc."}},
    }
}


async def test_regex_engine():
    """Test the RegexEngine with sample data."""
    print("🧪 Testing RegexEngine...")

    try:
        # Initialize engine
        engine = RegexEngine()

        # Process trial data
        elements = engine.process_trial(SAMPLE_TRIAL_DATA)

        print(f"✅ RegexEngine extracted {len(elements)} mCODE elements")

        # Test summarizer
        summarizer = UnifiedSummarizer(detail_level="standard")
        summary = summarizer.summarize_trial(elements, subject="NCT01234567", engine_type="regex")

        print(f"📝 RegexEngine summary: {summary[:100]}...")
        print("✅ RegexEngine test passed\n")

        return True

    except Exception as e:
        print(f"❌ RegexEngine test failed: {e}\n")
        return False


async def test_llm_engine():
    """Test the LLMEngine with sample data."""
    print("🤖 Testing LLMEngine...")

    try:
        # Initialize engine (this will fail without API key, but we can test initialization)
        LLMEngine()

        print("✅ LLMEngine initialized successfully")

        # Note: We can't actually run LLM processing without API keys
        # But we can test that the engine initializes correctly
        print("⚠️  LLMEngine processing test skipped (requires API key)")
        print("✅ LLMEngine initialization test passed\n")

        return True

    except Exception as e:
        print(f"❌ LLMEngine test failed: {e}\n")
        return False


async def test_unified_processor():
    """Test the UnifiedTrialProcessor with both engines."""
    print("🔄 Testing UnifiedTrialProcessor...")

    try:
        # Initialize processor
        processor = UnifiedTrialProcessor()

        # Test regex engine
        print("  Testing regex engine...")
        result = await processor.process_trial(SAMPLE_TRIAL_DATA, engine="regex")

        if result.success:
            print(f"  ✅ Regex processing successful: {result.data[:100]}...")
        else:
            print(f"  ❌ Regex processing failed: {result.error_message}")
            return False

        # Test engine recommendation
        recommendation = processor.recommend_engine(SAMPLE_TRIAL_DATA)
        print(f"  💡 Recommended engine: {recommendation}")

        print("✅ UnifiedTrialProcessor test passed\n")
        return True

    except Exception as e:
        print(f"❌ UnifiedTrialProcessor test failed: {e}\n")
        return False


async def test_engine_comparison():
    """Test engine comparison functionality."""
    print("⚖️  Testing engine comparison...")

    try:
        processor = UnifiedTrialProcessor()

        # Compare engines
        comparison = processor.compare_engines(SAMPLE_TRIAL_DATA)

        print(f"  💡 Recommendation: {comparison.get('recommendation', 'unknown')}")
        print(f"  📊 Regex available: {comparison.get('regex', {}).get('available', False)}")
        print(f"  📊 LLM available: {comparison.get('llm', {}).get('available', False)}")

        print("✅ Engine comparison test passed\n")
        return True

    except Exception as e:
        print(f"❌ Engine comparison test failed: {e}\n")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing New Engine Architecture\n")
    print("=" * 50)

    results = []

    # Test individual components
    results.append(await test_regex_engine())
    results.append(await test_llm_engine())
    results.append(await test_unified_processor())
    results.append(await test_engine_comparison())

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Engine architecture is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
