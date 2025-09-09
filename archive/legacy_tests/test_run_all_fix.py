#!/usr/bin/env python3
"""
Test script to verify Run All functionality is working correctly.
This script tests that the _create_and_queue_task method works properly
and that tasks are being added to the queue.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, patch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.optimization.pipeline_task_tracker import PipelineTaskTrackerUI, TaskStatus

def test_create_and_queue_task():
    """Test that _create_and_queue_task creates and queues tasks correctly"""
    print("Testing _create_and_queue_task method...")
    
    # Create a mock UI instance
    ui = PipelineTaskTrackerUI()
    
    # Mock the sample data
    ui.sample_trial_data = {
        "test_cases": {
            "test_case_1": {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT123456",
                        "briefTitle": "Test Clinical Trial"
                    }
                }
            }
        }
    }
    
    # Mock the background_tasks.create method
    with patch('src.optimization.pipeline_task_tracker.background_tasks') as mock_bg_tasks:
        mock_bg_tasks.create = Mock()
        
        # Test creating a NLP to mCODE task
        task_config = {
            'pipeline_type': 'NLP to mCODE',
            'extraction_prompt': 'generic_extraction',
            'mapping_prompt': 'generic_mapping'
        }
        
        ui._create_and_queue_task(task_config)
        
        # Verify task was created and added to tasks list
        assert len(ui.tasks) == 1, f"Expected 1 task, got {len(ui.tasks)}"
        task = ui.tasks[0]
        assert task.pipeline_type == 'NLP to mCODE'
        assert task.prompt_info['extraction_prompt'] == 'generic_extraction'
        assert task.prompt_info['mapping_prompt'] == 'generic_mapping'
        assert task.status == TaskStatus.PENDING
        
        # Verify background task was created
        assert mock_bg_tasks.create.called, "background_tasks.create should have been called"
        
        print("✓ NLP to mCODE task creation test passed")
        
        # Test creating a Direct to mCODE task
        task_config2 = {
            'pipeline_type': 'Direct to mCODE',
            'direct_prompt': 'direct_text_to_mcode_mapping'
        }
        
        ui._create_and_queue_task(task_config2)
        
        # Verify second task was created
        assert len(ui.tasks) == 2, f"Expected 2 tasks, got {len(ui.tasks)}"
        task2 = ui.tasks[1]
        assert task2.pipeline_type == 'Direct to mCODE'
        assert task2.prompt_info['direct_prompt'] == 'direct_text_to_mcode_mapping'
        
        print("✓ Direct to mCODE task creation test passed")
    
    print("All _create_and_queue_task tests passed!")

def test_run_selected_tasks():
    """Test that _run_selected_tasks calls _create_and_queue_task for selected tasks"""
    print("\nTesting _run_selected_tasks method...")
    
    # Create a mock UI instance
    ui = PipelineTaskTrackerUI()
    
    # Mock the sample data
    ui.sample_trial_data = {
        "test_cases": {
            "test_case_1": {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT123456",
                        "briefTitle": "Test Clinical Trial"
                    }
                }
            }
        }
    }
    
    # Mock the checkbox values
    ui.nlp_extraction_checkboxes = {
        'generic_extraction': Mock(value=True),
        'comprehensive_extraction': Mock(value=False)
    }
    ui.mcode_mapping_checkboxes = {
        'generic_mapping': Mock(value=True),
        'comprehensive_mapping': Mock(value=False)
    }
    ui.direct_mcode_checkboxes = {
        'direct_text_to_mcode_mapping': Mock(value=True)
    }
    
    # Mock the _create_and_queue_task method
    with patch.object(ui, '_create_and_queue_task') as mock_create_task:
        with patch.object(ui, '_add_notification'):
            with patch.object(ui.batch_status_label, 'set_text'):
                
                # Call the method
                ui._run_selected_tasks()
                
                # Verify _create_and_queue_task was called for the selected combinations
                assert mock_create_task.call_count == 2, f"Expected 2 calls, got {mock_create_task.call_count}"
                
                # Check the first call (NLP to mCODE)
                first_call_args = mock_create_task.call_args_list[0][0][0]
                assert first_call_args['pipeline_type'] == 'NLP to mCODE'
                assert first_call_args['extraction_prompt'] == 'generic_extraction'
                assert first_call_args['mapping_prompt'] == 'generic_mapping'
                
                # Check the second call (Direct to mCODE)
                second_call_args = mock_create_task.call_args_list[1][0][0]
                assert second_call_args['pipeline_type'] == 'Direct to mCODE'
                assert second_call_args['direct_prompt'] == 'direct_text_to_mcode_mapping'
                
                print("✓ _run_selected_tasks test passed")

def test_error_handling():
    """Test error handling in Run All mode"""
    print("\nTesting error handling...")
    
    ui = PipelineTaskTrackerUI()
    
    # Test with no sample data
    ui.sample_trial_data = None
    with patch('src.optimization.pipeline_task_tracker.ui.notify') as mock_notify:
        ui._run_selected_tasks()
        mock_notify.assert_called_with("No sample data available", type='warning')
        print("✓ No sample data error handling test passed")
    
    # Test with no tasks selected
    ui.sample_trial_data = {"test_cases": {"test1": {}}}
    ui.nlp_extraction_checkboxes = {'generic_extraction': Mock(value=False)}
    ui.mcode_mapping_checkboxes = {'generic_mapping': Mock(value=False)}
    ui.direct_mcode_checkboxes = {'direct_text_to_mcode_mapping': Mock(value=False)}
    
    with patch('src.optimization.pipeline_task_tracker.ui.notify') as mock_notify:
        ui._run_selected_tasks()
        mock_notify.assert_called_with("No tasks selected", type='warning')
        print("✓ No tasks selected error handling test passed")

if __name__ == "__main__":
    print("Running Run All functionality tests...")
    print("=" * 50)
    
    try:
        test_create_and_queue_task()
        test_run_selected_tasks()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Run All functionality is working correctly.")
        print("The _create_and_queue_task method is properly placed within the class")
        print("and should now work in the UI.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)