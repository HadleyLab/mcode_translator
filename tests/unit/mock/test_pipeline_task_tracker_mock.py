"""
Unit tests for the PipelineTaskTrackerUI component using mocks
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.pipeline_task_tracker import PipelineTaskTrackerUI


class TestPipelineTaskTrackerUIMock(unittest.TestCase):
    """Test cases for PipelineTaskTrackerUI using mocks"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the NiceGUI components to avoid UI initialization
        with patch('src.optimization.pipeline_task_tracker.ui'), \
             patch('src.optimization.pipeline_task_tracker.background_tasks'), \
             patch('src.optimization.pipeline_task_tracker.run'):
            self.tracker = PipelineTaskTrackerUI()

    def test_initialization(self):
        """Test that PipelineTaskTrackerUI initializes correctly"""
        self.assertIsNotNone(self.tracker.framework)
        self.assertIsInstance(self.tracker.available_prompts, dict)
        self.assertIsInstance(self.tracker.available_models, dict)
        self.assertIsInstance(self.tracker.trial_data, dict)
        self.assertIsInstance(self.tracker.gold_standard_data, dict)
        self.assertFalse(self.tracker.is_benchmark_running)
        self.assertFalse(self.tracker.benchmark_cancelled)
        self.assertIsInstance(self.tracker.live_logs, list)

    def test_load_libraries(self):
        """Test that prompt and model libraries are loaded correctly"""
        # This should not raise an exception
        self.tracker._load_libraries()
        # Check that we have some prompts and models (even if mocked)
        self.assertIsInstance(self.tracker.available_prompts, dict)
        self.assertIsInstance(self.tracker.available_models, dict)

    def test_load_test_data(self):
        """Test that test data is loaded correctly"""
        # This should not raise an exception
        self.tracker._load_test_data()
        # Check that we have dictionaries for trial and gold standard data
        self.assertIsInstance(self.tracker.trial_data, dict)
        self.assertIsInstance(self.tracker.gold_standard_data, dict)

    def test_create_pipeline_callback(self):
        """Test that pipeline callback is created correctly"""
        callback = self.tracker._create_pipeline_callback()
        self.assertTrue(callable(callback))

    def test_update_preloaded_validation_status(self):
        """Test updating preloaded validation status"""
        # Add a sample validation
        self.tracker.preloaded_validations = [{
            'prompt': 'test_prompt',
            'model': 'test_model',
            'trial': 'test_trial',
            'status': 'Pending',
            'details': 'Waiting to run',
            'status_icon': '🔵',
            'precision': '-',
            'recall': '-',
            'f1_score': '-',
            'duration_ms': '-',
            'token_usage': '-'
        }]
        
        # Update the validation status
        self.tracker._update_preloaded_validation_status(
            'test_prompt', 'test_model', 'test_trial',
            'Success', 'Test completed', '✅',
            '0.95', '0.90', '0.92', '100.5', '500'
        )
        
        # Check that the validation was updated
        updated_validation = self.tracker.preloaded_validations[0]
        self.assertEqual(updated_validation['status'], 'Success')
        self.assertEqual(updated_validation['details'], 'Test completed')
        self.assertEqual(updated_validation['status_icon'], '✅')
        self.assertEqual(updated_validation['precision'], '0.95')
        self.assertEqual(updated_validation['f1_score'], '0.92')

    def test_log_message_handling(self):
        """Test that log messages are handled correctly"""
        # Test adding a log message
        test_log = "Test log message"
        self.tracker._handle_log_message(test_log)
        
        # Check that log was added to live_logs
        self.assertEqual(len(self.tracker.live_logs), 1)
        self.assertEqual(self.tracker.live_logs[0], test_log)
        
        # Test adding multiple logs
        additional_logs = ["Log 2", "Log 3", "Log 4"]
        for log in additional_logs:
            self.tracker._handle_log_message(log)
        
        # Check that all logs are present
        self.assertEqual(len(self.tracker.live_logs), 4)
        self.assertEqual(self.tracker.live_logs[-1], "Log 4")

    def test_log_message_truncation(self):
        """Test that log messages are truncated when exceeding limit"""
        # Set a small log limit for testing
        original_limit = self.tracker.MAX_LOG_LINES
        self.tracker.MAX_LOG_LINES = 3
        
        try:
            # Add more logs than the limit
            logs = ["Log 1", "Log 2", "Log 3", "Log 4", "Log 5"]
            for log in logs:
                self.tracker._handle_log_message(log)
            
            # Check that only the most recent logs are kept
            self.assertEqual(len(self.tracker.live_logs), 3)
            self.assertEqual(self.tracker.live_logs, ["Log 3", "Log 4", "Log 5"])
        finally:
            # Restore original limit
            self.tracker.MAX_LOG_LINES = original_limit

    def test_stop_benchmark_when_not_running(self):
        """Test stopping benchmark when not running"""
        with patch('src.optimization.pipeline_task_tracker.ui.notify') as mock_notify:
            self.tracker._stop_benchmark()
            mock_notify.assert_called_with("No benchmark is currently running", type='info')

    def test_reset_interface(self):
        """Test resetting the interface"""
        # Set some state
        self.tracker.is_benchmark_running = False  # Not actually running
        self.tracker.benchmark_cancelled = True  # But was cancelled previously
        self.tracker.benchmark_results = [{'test': 'result'}]
        self.tracker.validation_results = [{'test': 'validation'}]
        self.tracker.live_logs = ["log1", "log2", "log3"]
        
        with patch('src.optimization.pipeline_task_tracker.ui.notify'):
            self.tracker._reset_interface()
        
        # Check that state was reset
        self.assertFalse(self.tracker.is_benchmark_running)
        self.assertFalse(self.tracker.benchmark_cancelled)  # Should be reset to False
        self.assertEqual(self.tracker.benchmark_results, [])
        self.assertEqual(self.tracker.validation_results, [])
        self.assertEqual(self.tracker.live_logs, [])
        # Note: benchmark_progress.value is a MagicMock due to UI mocking, so we can't directly test its value


if __name__ == '__main__':
    unittest.main()