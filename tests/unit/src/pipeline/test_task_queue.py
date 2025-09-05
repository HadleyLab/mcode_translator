#!/usr/bin/env python3
"""
Test script to verify the task queue functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import asyncio
import json
from src.pipeline.task_queue import PipelineTaskQueue, BenchmarkTask
from src.shared.types import TaskStatus


async def test_task_queue():
    """Test the task queue functionality"""
    print("Testing Task Queue Functionality")
    print("=" * 40)
    
    # Create task queue
    task_queue = PipelineTaskQueue(max_workers=2)
    
    # Load gold standard data for testing
    gold_standard_path = Path('examples/breast_cancer_data/breast_cancer_her2_positive.gold.json')
    with open(gold_standard_path, 'r') as f:
        gold_data = json.load(f)
    
    # Extract the expected entities and Mcode mappings from the gold standard
    expected_entities = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_extraction']['entities']
    expected_mappings = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    # Create sample trial data
    trial_data = {
        "briefTitle": "A Trial for HER2-Positive Breast Cancer",
        "detailedDescription": "This is a clinical trial for patients with HER2-positive metastatic breast cancer.",
        "eligibility": {
            "criteria": "Patients must have measurable disease and ECOG performance status 0-1."
        }
    }
    
    # Create sample tasks
    task1 = BenchmarkTask(
        prompt_name="generic_extraction",
        model_name="deepseek-coder",
        trial_id="test_trial_1",
        trial_data=trial_data,
        expected_entities=expected_entities,
        expected_mappings=expected_mappings
    )
    
    task2 = BenchmarkTask(
        prompt_name="generic_mapping",
        model_name="deepseek-coder",
        trial_id="test_trial_2",
        trial_data=trial_data,
        expected_entities=expected_entities,
        expected_mappings=expected_mappings
    )
    
    # Start workers
    await task_queue.start_workers()
    
    # Add tasks to queue
    await task_queue.add_task(task1)
    await task_queue.add_task(task2)
    
    # Wait for tasks to complete
    await asyncio.sleep(5)  # Give tasks time to process
    
    # Stop workers
    await task_queue.stop_workers()
    
    # Check results
    tasks = task_queue.get_all_tasks()
    print(f"Total tasks: {len(tasks)}")
    
    for task in tasks:
        print(f"Task {task.task_id}: {task.prompt_name} - Status: {task.status}")
        if task.status == TaskStatus.SUCCESS:
            print(f"  Precision: {task.precision:.3f}")
            print(f"  Recall: {task.recall:.3f}")
            print(f"  F1 Score: {task.f1_score:.3f}")
            print(f"  Compliance Score: {task.compliance_score:.3f}")
        else:
            print(f"  Error: {task.error_message}")
    
    print("\nTask queue test completed!")


if __name__ == "__main__":
    asyncio.run(test_task_queue())