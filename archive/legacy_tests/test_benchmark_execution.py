#!/usr/bin/env python3
"""
Test script to verify benchmark task execution functionality.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.task_queue import (BenchmarkTask, PipelineTaskQueue,
                                     TaskStatus, initialize_task_queue,
                                     shutdown_task_queue)
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

import pytest


@pytest.mark.asyncio
async def test_benchmark_execution():
    """Test benchmark task execution"""
    logger.info("Starting benchmark execution test...")
    
    try:
        # Initialize task queue
        task_queue = await initialize_task_queue(max_workers=2)
        logger.info("Task queue initialized successfully")
        
        # Create a simple test task
        test_task = BenchmarkTask(
            prompt_name="direct_mcode_simple",
            model_name="deepseek-coder",
            trial_id="breast_cancer_her2_positive",
            trial_data={
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT12345678",
                        "briefTitle": "Test Clinical Trial"
                    },
                    "conditionsModule": {
                        "conditions": ["HER2-Positive Breast Cancer", "Metastatic Breast Cancer"]
                    }
                }
            },
            prompt_type="DIRECT_MCODE",
            expected_entities=[],
            expected_mappings=[],
            pipeline_type="DIRECT_MCODE",
            optimization_parameters={'metric': 'f1_score'},
            prompt_info={'prompt_key': 'direct_mcode_simple'}
        )
        
        logger.info(f"Created test task: {test_task.task_id}")
        logger.info(f"Prompt: {test_task.prompt_name}")
        logger.info(f"Model: {test_task.model_name}")
        logger.info(f"Trial ID: {test_task.trial_id}")
        
        # Add task to queue
        await task_queue.add_task(test_task)
        logger.info("Task added to queue")
        
        # Start workers
        await task_queue.start_workers()
        logger.info("Workers started")
        
        # Wait for task completion
        while task_queue.completed_tasks < 1:
            import asyncio
            await asyncio.sleep(0.1)
        
        # Check task result
        completed_task = task_queue.get_task(test_task.task_id)
        if completed_task:
            logger.info(f"Task completed with status: {completed_task.status}")
            if completed_task.status == TaskStatus.SUCCESS:
                logger.info(f"Task successful!")
                # Safely convert metrics to strings to avoid formatting issues
                f1_score_str = f"{completed_task.f1_score:.3f}" if completed_task.f1_score is not None else "N/A"
                precision_str = f"{completed_task.precision:.3f}" if completed_task.precision is not None else "N/A"
                recall_str = f"{completed_task.recall:.3f}" if completed_task.recall is not None else "N/A"
                logger.info(f"F1 Score: {f1_score_str}")
                logger.info(f"Precision: {precision_str}")
                logger.info(f"Recall: {recall_str}")
                logger.info(f"Duration: {completed_task.duration_ms}ms")
                logger.info(f"Token usage: {completed_task.token_usage}")
                return True
            else:
                logger.error(f"Task failed: {completed_task.error_message}")
                return False
        else:
            logger.error("Task not found in queue")
            return False
            
    except Exception as e:
        logger.error(f"Error in benchmark execution test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Shutdown task queue
        await shutdown_task_queue()
        logger.info("Task queue shutdown completed")

def main():
    """Main function"""
    import asyncio
    result = asyncio.run(test_benchmark_execution())
    
    if result:
        logger.info("Benchmark execution test completed successfully")
        return True
    else:
        logger.error("Benchmark execution test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)