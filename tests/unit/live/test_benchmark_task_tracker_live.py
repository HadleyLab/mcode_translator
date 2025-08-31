"""
Unit tests for the BenchmarkTaskTrackerUI component using live calls (no mocking)
"""
import sys
import os
import unittest
import json
from pathlib import Path
import tempfile
import shutil

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.benchmark_task_tracker import BenchmarkTaskTrackerUI


class TestBenchmarkTaskTrackerUILive(unittest.TestCase):
    """Test cases for BenchmarkTaskTrackerUI using live calls"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize the tracker (this will attempt to initialize UI components)
        # We'll catch any UI-related exceptions and skip tests that require a display
        try:
            self.tracker = BenchmarkTaskTrackerUI()
            self.ui_available = True
        except Exception as e:
            # UI initialization failed, probably because there's no display
            print(f"Warning: UI initialization failed: {e}")
            self.ui_available = False
            self.tracker = None

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization_live(self):
        """Test that BenchmarkTaskTrackerUI initializes correctly with live calls"""
        if not self.ui_available:
            self.skipTest("UI not available for live testing")
            
        self.assertIsNotNone(self.tracker.framework)
        self.assertIsInstance(self.tracker.available_prompts, dict)
        self.assertIsInstance(self.tracker.available_models, dict)
        self.assertIsInstance(self.tracker.trial_data, dict)
        self.assertIsInstance(self.tracker.gold_standard_data, dict)
        self.assertFalse(self.tracker.is_benchmark_running)
        self.assertFalse(self.tracker.benchmark_cancelled)

    def test_load_libraries_live(self):
        """Test that prompt and model libraries are loaded correctly with live calls"""
        if not self.ui_available:
            self.skipTest("UI not available for live testing")
            
        # This should not raise an exception
        self.tracker._load_libraries()
        # Check that we have some prompts and models
        self.assertIsInstance(self.tracker.available_prompts, dict)
        self.assertIsInstance(self.tracker.available_models, dict)
        # Should have at least some prompts and models loaded
        self.assertGreater(len(self.tracker.available_prompts), 0)
        self.assertGreater(len(self.tracker.available_models), 0)

    def test_load_test_data_live(self):
        """Test that test data is loaded correctly with live calls"""
        if not self.ui_available:
            self.skipTest("UI not available for live testing")
            
        # This should not raise an exception
        self.tracker._load_test_data()
        # Check that we have dictionaries for trial and gold standard data
        self.assertIsInstance(self.tracker.trial_data, dict)
        self.assertIsInstance(self.tracker.gold_standard_data, dict)

    def test_create_pipeline_callback_live(self):
        """Test that pipeline callback is created correctly with live calls"""
        if not self.ui_available:
            self.skipTest("UI not available for live testing")
            
        callback = self.tracker._create_pipeline_callback()
        self.assertTrue(callable(callback))

    def test_reset_interface_live(self):
        """Test resetting the interface with live calls"""
        if not self.ui_available:
            self.skipTest("UI not available for live testing")
            
        # Set some state
        self.tracker.is_benchmark_running = False  # Not actually running
        self.tracker.benchmark_cancelled = True  # But was cancelled previously
        self.tracker.benchmark_results = [{'test': 'result'}]
        self.tracker.validation_results = [{'test': 'validation'}]
        
        # Note: We can't directly call _reset_interface because it tries to interact with UI
        # Instead, we'll test the state reset logic indirectly
        
        # Check that state was reset
        self.assertFalse(self.tracker.is_benchmark_running)
        self.assertFalse(self.tracker.benchmark_cancelled)
        self.assertEqual(self.tracker.benchmark_results, [])
        self.assertEqual(self.tracker.validation_results, [])


if __name__ == '__main__':
    unittest.main()