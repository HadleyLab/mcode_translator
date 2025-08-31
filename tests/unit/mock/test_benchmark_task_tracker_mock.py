"""
Unit tests for the BenchmarkTaskTrackerUI component using mocks
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.benchmark_task_tracker import BenchmarkTaskTrackerUI


class TestBenchmarkTaskTrackerUIMock(unittest.TestCase):
    """Test cases for BenchmarkTaskTrackerUI using mocks"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the NiceGUI components to avoid UI initialization
        with patch('src.optimization.benchmark_task_tracker.ui'), \
             patch('src.optimization.benchmark_task_tracker.background_tasks'), \
             patch('src.optimization.benchmark_task_tracker.run'):
            self.tracker = BenchmarkTaskTrackerUI()

    def test_initialization(self):
        """Test that BenchmarkTaskTrackerUI initializes correctly"""
        self.assertIsNotNone(self.tracker.framework)
        self.assertIsInstance(self.tracker.available_prompts, dict)
        self.assertIsInstance(self.tracker.available_models, dict)
        self.assertIsInstance(self.tracker.trial_data, dict)
        self.assertIsInstance(self.tracker.gold_standard_data, dict)
        self.assertFalse(self.tracker.is_benchmark_running)
        self.assertFalse(self.tracker.benchmark_cancelled)

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

    def test_stop_benchmark_when_not_running(self):
        """Test stopping benchmark when not running"""
        with patch('src.optimization.benchmark_task_tracker.ui.notify') as mock_notify:
            self.tracker._stop_benchmark()
            mock_notify.assert_called_with("No benchmark is currently running", type='info')

    def test_reset_interface(self):
        """Test resetting the interface"""
        # Set some state
        self.tracker.is_benchmark_running = False  # Not actually running
        self.tracker.benchmark_cancelled = True  # But was cancelled previously
        self.tracker.benchmark_results = [{'test': 'result'}]
        self.tracker.validation_results = [{'test': 'validation'}]
        
        with patch('src.optimization.benchmark_task_tracker.ui.notify'):
            self.tracker._reset_interface()
        
        # Check that state was reset
        self.assertFalse(self.tracker.is_benchmark_running)
        self.assertFalse(self.tracker.benchmark_cancelled)  # Should be reset to False
        self.assertEqual(self.tracker.benchmark_results, [])
        self.assertEqual(self.tracker.validation_results, [])
        # Note: benchmark_progress.value is a MagicMock due to UI mocking, so we can't directly test its value


if __name__ == '__main__':
    unittest.main()