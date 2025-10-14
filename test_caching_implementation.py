#!/usr/bin/env python3
"""
Test script for LLM response caching implementation in expert panel system.

This script tests the caching functionality to ensure it provides cost reduction
and performance improvements while maintaining accuracy.
"""

import asyncio
import json
import time
from pathlib import Path

from src.matching.clinical_expert_agent import ClinicalExpertAgent
from src.matching.expert_panel_manager import ExpertPanelManager
from src.utils.api_manager import APIManager
from src.utils.config import Config


async def test_clinical_expert_caching():
    """Test caching functionality in ClinicalExpertAgent."""
    print("🧪 Testing ClinicalExpertAgent caching...")

    # Sample patient and trial data
    patient_data = {
        "id": "patient_001",
        "age": 45,
        "gender": "female",
        "conditions": ["breast cancer"],
        "stage": "II",
        "biomarkers": {"ER": "positive", "PR": "positive", "HER2": "negative"},
        "performance_status": 0,
        "comorbidities": ["hypertension"],
        "current_medications": ["lisinopril"]
    }

    trial_criteria = {
        "trial_id": "NCT00123456",
        "conditions": ["breast cancer"],
        "eligibilityCriteria": "Postmenopausal women with ER-positive breast cancer",
        "age_min": 40,
        "age_max": 70,
        "biomarkers": {"ER": "positive"},
        "performance_status_max": 1
    }

    # Initialize expert agent with caching enabled
    config = Config()
    expert = ClinicalExpertAgent(
        model_name="deepseek-coder",
        expert_type="clinical_reasoning",
        config=config
    )

    print(f"💾 Caching enabled: {expert.enable_caching}")
    print(f"📊 Initial cache stats: {expert._get_cache_stats()}")

    # First assessment (cache miss)
    print("\n🔬 First assessment (should be cache miss)...")
    start_time = time.time()
    result1 = await expert.assess_match(patient_data, trial_criteria)
    first_assessment_time = time.time() - start_time

    print(f"⏱️  First assessment time: {first_assessment_time:.2f}s")
    print(f"📊 Cache stats after first: {expert._get_cache_stats()}")

    # Second assessment with same data (cache hit)
    print("\n🔬 Second assessment (should be cache hit)...")
    start_time = time.time()
    result2 = await expert.assess_match(patient_data, trial_criteria)
    second_assessment_time = time.time() - start_time

    print(f"⏱️  Second assessment time: {second_assessment_time:.2f}s")
    print(f"📊 Cache stats after second: {expert._get_cache_stats()}")

    # Verify results are identical
    print(f"\n✅ Results identical: {result1 == result2}")

    # Test with different data (cache miss)
    print("\n🔬 Third assessment with different data (should be cache miss)...")
    patient_data_modified = patient_data.copy()
    patient_data_modified["age"] = 50

    start_time = time.time()
    result3 = await expert.assess_match(patient_data_modified, trial_criteria)
    third_assessment_time = time.time() - start_time

    print(f"⏱️  Third assessment time: {third_assessment_time:.2f}s")
    print(f"📊 Final cache stats: {expert._get_cache_stats()}")

    # Calculate performance improvement
    if second_assessment_time < first_assessment_time * 0.5:
        print(f"🚀 Cache performance improvement: {((first_assessment_time - second_assessment_time) / first_assessment_time * 100):.1f}% faster")
    else:
        print("⚠️  Cache performance improvement not significant")

    return expert._get_cache_stats()


async def test_expert_panel_caching():
    """Test caching functionality in ExpertPanelManager."""
    print("\n🧪 Testing ExpertPanelManager caching...")

    # Sample patient and trial data
    patient_data = {
        "id": "patient_002",
        "age": 55,
        "gender": "male",
        "conditions": ["lung cancer"],
        "stage": "III",
        "biomarkers": {"EGFR": "positive"},
        "performance_status": 1,
        "comorbidities": ["COPD", "diabetes"],
        "current_medications": ["metformin", "insulin"]
    }

    trial_criteria = {
        "trial_id": "NCT00234567",
        "conditions": ["lung cancer"],
        "eligibilityCriteria": "Patients with EGFR-positive NSCLC, age 18-75",
        "age_min": 18,
        "age_max": 75,
        "biomarkers": {"EGFR": "positive"},
        "performance_status_max": 2
    }

    # Initialize expert panel with caching
    config = Config()
    panel = ExpertPanelManager(
        model_name="deepseek-coder",
        config=config,
        max_concurrent_experts=2
    )

    print(f"💾 Panel caching enabled: {panel.enable_caching}")
    print(f"📊 Initial panel cache stats: {panel._get_panel_cache_stats()}")

    # First panel assessment (cache miss for all experts)
    print("\n🔬 First panel assessment (should be cache misses)...")
    start_time = time.time()
    result1 = await panel.assess_with_expert_panel(patient_data, trial_criteria)
    first_panel_time = time.time() - start_time

    print(f"⏱️  First panel assessment time: {first_panel_time:.2f}s")
    print(f"📊 Panel cache stats after first: {panel._get_panel_cache_stats()}")

    # Second panel assessment (cache hits for all experts)
    print("\n🔬 Second panel assessment (should be cache hits)...")
    start_time = time.time()
    result2 = await panel.assess_with_expert_panel(patient_data, trial_criteria)
    second_panel_time = time.time() - start_time

    print(f"⏱️  Second panel assessment time: {second_panel_time:.2f}s")
    print(f"📊 Panel cache stats after second: {panel._get_panel_cache_stats()}")

    # Verify results are identical
    print(f"\n✅ Panel results identical: {result1['is_match'] == result2['is_match']}")
    print(f"✅ Panel confidence scores identical: {result1['confidence_score'] == result2['confidence_score']}")

    # Calculate performance improvement
    if second_panel_time < first_panel_time * 0.5:
        print(f"🚀 Panel cache performance improvement: {((first_panel_time - second_panel_time) / first_panel_time * 100):.1f}% faster")
    else:
        print("⚠️  Panel cache performance improvement not significant")

    # Log comprehensive cache performance
    panel.log_cache_performance()

    # Get optimization recommendations
    recommendations = panel.get_cache_optimization_recommendations()
    print("\n💡 Cache Optimization Recommendations:")
    for rec in recommendations:
        print(f"  • {rec}")

    return panel._get_panel_cache_stats()


async def test_cache_persistence():
    """Test that cache persists between different agent instances."""
    print("\n🧪 Testing cache persistence...")

    patient_data = {"id": "patient_003", "age": 60, "conditions": ["prostate cancer"]}
    trial_criteria = {"trial_id": "NCT00345678", "conditions": ["prostate cancer"]}

    config = Config()

    # First agent instance
    expert1 = ClinicalExpertAgent(
        model_name="deepseek-coder",
        expert_type="pattern_recognition",
        config=config
    )

    print("Creating first assessment...")
    await expert1.assess_match(patient_data, trial_criteria)
    stats1 = expert1._get_cache_stats()
    print(f"📊 First agent cache stats: {stats1}")

    # Second agent instance (should share cache)
    expert2 = ClinicalExpertAgent(
        model_name="deepseek-coder",
        expert_type="pattern_recognition",
        config=config
    )

    print("Creating second assessment (should be cache hit)...")
    await expert2.assess_match(patient_data, trial_criteria)
    stats2 = expert2._get_cache_stats()
    print(f"📊 Second agent cache stats: {stats2}")

    # Verify cache was shared
    if stats2['cache_hits'] > stats1['cache_hits']:
        print("✅ Cache persistence working correctly")
    else:
        print("⚠️  Cache persistence may not be working")


async def main():
    """Run all caching tests."""
    print("🚀 Starting LLM Response Caching Tests")
    print("=" * 50)

    try:
        # Test individual expert caching
        expert_stats = await test_clinical_expert_caching()

        # Test expert panel caching
        panel_stats = await test_expert_panel_caching()

        # Test cache persistence
        await test_cache_persistence()

        print("\n" + "=" * 50)
        print("🎉 CACHING TESTS COMPLETED")
        print("\n📊 SUMMARY:")
        print(f"💡 Expert cache hit rate: {expert_stats['hit_rate']:.2%}")
        print(f"💡 Panel cache hit rate: {panel_stats['panel_hit_rate']:.2%}")
        print(f"💰 Estimated API calls saved: {panel_stats['panel_total_time_saved_seconds'] / 5:.1f}")

        if expert_stats['hit_rate'] > 0.3 and panel_stats['panel_hit_rate'] > 0.3:
            print("✅ CACHING IMPLEMENTATION SUCCESSFUL!")
        else:
            print("⚠️  Caching working but hit rates could be improved")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())