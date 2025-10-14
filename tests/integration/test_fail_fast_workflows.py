#!/usr/bin/env python3
"""
Test script to verify fail-fast behavior in updated workflows.

Tests immediate failure on invalid input parameters for:
- TrialsFetcherWorkflow
- TrialsProcessorWorkflow
- TrialsOptimizerWorkflow
- TrialsSummarizerWorkflow

All workflows should fail immediately without graceful degradation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.workflows.trials_fetcher import TrialsFetcherWorkflow
from src.workflows.trials_processor import TrialsProcessor
from src.workflows.trials_optimizer import TrialsOptimizerWorkflow
from src.workflows.trials_summarizer import TrialsSummarizerWorkflow
from src.config.heysol_config import get_config
from src.storage.mcode_memory_storage import OncoCoreMemory


def test_trials_fetcher_fail_fast() -> bool:
    """Test TrialsFetcherWorkflow fails immediately on invalid parameters."""
    print("🧪 Testing TrialsFetcherWorkflow fail-fast behavior...")

    memory_storage = OncoCoreMemory()
    # Create a mock Config object since workflows expect Config type
    from src.utils.config import Config
    mock_config = Config()
    workflow = TrialsFetcherWorkflow(mock_config, memory_storage)

    # Test 1: Missing required parameters
    try:
        workflow.execute()
        print("❌ FAIL: Should have failed on missing parameters")
        return False
    except KeyError as e:
        if "condition" in str(e):
            print("✅ PASS: Failed immediately on missing parameters")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 2: Invalid parameter combination (multiple fetch types)
    try:
        workflow.execute(condition="cancer", nct_id="NCT123456", nct_ids=["NCT123456"], limit=10, output_path=None)
        print("❌ FAIL: Should have failed on multiple fetch parameters")
        return False
    except ValueError as e:
        if "Invalid fetch parameters" in str(e):
            print("✅ PASS: Failed immediately on multiple fetch types")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 3: Empty condition
    try:
        workflow.execute(condition="", limit=10, output_path=None, nct_id=None, nct_ids=None)
        print("❌ FAIL: Should have failed on empty condition")
        return False
    except ValueError as e:
        if "No valid fetch parameters provided" in str(e):
            print("✅ PASS: Failed immediately on empty condition")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    return True


def test_trials_processor_fail_fast() -> bool:
    """Test TrialsProcessorWorkflow fails immediately on invalid parameters."""
    print("🧪 Testing TrialsProcessorWorkflow fail-fast behavior...")

    config = get_config()
    memory_storage = OncoCoreMemory()
    workflow = TrialsProcessor(config, memory_storage)

    # Test 1: Missing trials_data
    try:
        workflow.execute(engine="llm", model="gpt-4", prompt="test", workers=1, store_in_memory=False)
        print("❌ FAIL: Should have failed on missing trials_data")
        return False
    except ValueError as e:
        if "trials_data is required" in str(e):
            print("✅ PASS: Failed immediately on missing trials_data")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 2: Missing engine
    try:
        workflow.execute(trials_data=[], model="gpt-4", prompt="test", workers=1, store_in_memory=False)
        print("❌ FAIL: Should have failed on missing engine")
        return False
    except ValueError as e:
        if "engine is required" in str(e):
            print("✅ PASS: Failed immediately on missing engine")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 3: Missing model
    try:
        workflow.execute(trials_data=[], engine="llm", prompt="test", workers=1, store_in_memory=False)
        print("❌ FAIL: Should have failed on missing model")
        return False
    except ValueError as e:
        if "model is required" in str(e):
            print("✅ PASS: Failed immediately on missing model")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 4: Missing prompt
    try:
        workflow.execute(trials_data=[], engine="llm", model="gpt-4", workers=1, store_in_memory=False)
        print("❌ FAIL: Should have failed on missing prompt")
        return False
    except ValueError as e:
        if "prompt is required" in str(e):
            print("✅ PASS: Failed immediately on missing prompt")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 5: Missing workers
    try:
        workflow.execute(trials_data=[], engine="llm", model="gpt-4", prompt="test", store_in_memory=False)
        print("❌ FAIL: Should have failed on missing workers")
        return False
    except ValueError as e:
        if "workers is required" in str(e):
            print("✅ PASS: Failed immediately on missing workers")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 6: Missing store_in_memory
    try:
        workflow.execute(trials_data=[], engine="llm", model="gpt-4", prompt="test", workers=1)
        print("❌ FAIL: Should have failed on missing store_in_memory")
        return False
    except ValueError as e:
        if "store_in_memory is required" in str(e):
            print("✅ PASS: Failed immediately on missing store_in_memory")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 7: Invalid trials_data type - this should fail during validation
    try:
        workflow.execute(trials_data="not_a_list", engine="llm", model="gpt-4", prompt="test", workers=1, store_in_memory=False)
        print("❌ FAIL: Should have failed on invalid trials_data type")
        return False
    except TypeError as e:
        if "trials_data must be a list" in str(e):
            print("✅ PASS: Failed immediately on invalid trials_data type")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    return True


def test_trials_optimizer_fail_fast() -> bool:
    """Test TrialsOptimizerWorkflow fails immediately on invalid parameters."""
    print("🧪 Testing TrialsOptimizerWorkflow fail-fast behavior...")

    from src.utils.config import Config
    mock_config = Config()
    memory_storage = OncoCoreMemory()
    workflow = TrialsOptimizerWorkflow(mock_config, memory_storage)

    # Test 1: Missing trials_data
    try:
        workflow.execute(prompts=["test"], models=["gpt-4"], max_combinations=1, cv_folds=3)
        print("❌ FAIL: Should have failed on missing trials_data")
        return False
    except KeyError as e:
        if "trials_data" in str(e):
            print("✅ PASS: Failed immediately on missing trials_data")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 2: Empty trials_data
    try:
        workflow.execute(trials_data=[], prompts=["test"], models=["gpt-4"], max_combinations=1, cv_folds=3)
        print("❌ FAIL: Should have failed on empty trials_data")
        return False
    except ValueError as e:
        if "No trial data provided" in str(e):
            print("✅ PASS: Failed immediately on empty trials_data")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    return True


def test_trials_summarizer_fail_fast() -> bool:
    """Test TrialsSummarizerWorkflow fails immediately on invalid parameters."""
    print("🧪 Testing TrialsSummarizerWorkflow fail-fast behavior...")

    config = get_config()
    memory_storage = OncoCoreMemory()
    workflow = TrialsSummarizerWorkflow(config, memory_storage)

    # Test 1: Missing trials_data
    try:
        workflow.execute(store_in_memory=False)
        print("❌ FAIL: Should have failed on missing trials_data")
        return False
    except ValueError as e:
        if "trials_data is required" in str(e):
            print("✅ PASS: Failed immediately on missing trials_data")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 2: Missing store_in_memory
    try:
        workflow.execute(trials_data=[])
        print("❌ FAIL: Should have failed on missing store_in_memory")
        return False
    except ValueError as e:
        if "store_in_memory is required" in str(e):
            print("✅ PASS: Failed immediately on missing store_in_memory")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 3: Invalid trials_data type
    try:
        workflow.execute(trials_data="not_a_list", store_in_memory=False)
        print("❌ FAIL: Should have failed on invalid trials_data type")
        return False
    except TypeError as e:
        if "trials_data must be a list" in str(e):
            print("✅ PASS: Failed immediately on invalid trials_data type")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 4: Invalid store_in_memory type
    try:
        workflow.execute(trials_data=[], store_in_memory="not_a_bool")
        print("❌ FAIL: Should have failed on invalid store_in_memory type")
        return False
    except TypeError as e:
        if "store_in_memory must be a bool" in str(e):
            print("✅ PASS: Failed immediately on invalid store_in_memory type")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    # Test 5: Empty trials_data
    try:
        workflow.execute(trials_data=[], store_in_memory=False)
        print("❌ FAIL: Should have failed on empty trials_data")
        return False
    except ValueError as e:
        if "trials_data cannot be empty" in str(e):
            print("✅ PASS: Failed immediately on empty trials_data")
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False

    return True


def main() -> None:
    """Run all fail-fast tests."""
    print("🚀 Starting fail-fast behavior tests for workflows...\n")

    tests = [
        test_trials_fetcher_fail_fast,
        test_trials_processor_fail_fast,
        test_trials_optimizer_fail_fast,
        test_trials_summarizer_fail_fast,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ FAIL: Test {test_func.__name__} crashed: {e}\n")

    print(f"📊 Test Results: {passed}/{total} workflows passed fail-fast tests")

    if passed == total:
        print("🎉 All workflows demonstrate proper fail-fast behavior!")
        sys.exit(0)
    else:
        print("💥 Some workflows failed to demonstrate fail-fast behavior!")
        sys.exit(1)


if __name__ == "__main__":
    main()