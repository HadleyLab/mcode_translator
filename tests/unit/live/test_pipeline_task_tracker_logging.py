"""
Integration tests for PipelineTaskTrackerUI logging functionality
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.pipeline_task_tracker import PipelineTaskTrackerUI


class TestPipelineTaskTrackerLogging(unittest.TestCase):
    """Integration tests for logging functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the NiceGUI components to avoid UI initialization
        with patch('src.optimization.pipeline_task_tracker.ui'), \
             patch('src.optimization.pipeline_task_tracker.background_tasks'), \
             patch('src.optimization.pipeline_task_tracker.run'):
            self.tracker = PipelineTaskTrackerUI()

    def test_log_message_flow(self):
        """Test the complete log message flow from pipeline to UI"""
        # Test that log messages are properly handled and stored
        test_messages = [
            "Starting pipeline execution",
            "Processing trial: test_trial_1",
            "Calling LLM with prompt: test_prompt",
            "LLM response received",
            "Validation completed successfully",
            "Benchmark metrics calculated"
        ]
        
        for message in test_messages:
            self.tracker._handle_log_message(message)
        
        # Verify all messages are stored
        self.assertEqual(len(self.tracker.live_logs), len(test_messages))
        for i, expected_message in enumerate(test_messages):
            self.assertEqual(self.tracker.live_logs[i], expected_message)

    def test_log_message_order_preservation(self):
        """Test that log messages maintain chronological order"""
        messages = [
            "Message 1 - " + str(time.time()),
            "Message 2 - " + str(time.time() + 0.001),
            "Message 3 - " + str(time.time() + 0.002),
            "Message 4 - " + str(time.time() + 0.003)
        ]
        
        for message in messages:
            self.tracker._handle_log_message(message)
        
        # Messages should be in the order they were added
        self.assertEqual(self.tracker.live_logs, messages)

    def test_log_truncation_behavior(self):
        """Test that log truncation works correctly when exceeding limit"""
        # Set a small limit for testing
        original_limit = self.tracker.MAX_LOG_LINES
        self.tracker.MAX_LOG_LINES = 5
        
        try:
            # Add more messages than the limit
            messages = [f"Log message {i}" for i in range(10)]
            
            for message in messages:
                self.tracker._handle_log_message(message)
            
            # Should only keep the most recent messages
            expected_messages = messages[-5:]  # Last 5 messages
            self.assertEqual(len(self.tracker.live_logs), 5)
            self.assertEqual(self.tracker.live_logs, expected_messages)
            
            # Add one more message
            self.tracker._handle_log_message("Additional message")
            
            # Should now contain the last 5 messages including the new one
            expected_new_messages = messages[-4:] + ["Additional message"]
            self.assertEqual(self.tracker.live_logs, expected_new_messages)
            
        finally:
            # Restore original limit
            self.tracker.MAX_LOG_LINES = original_limit

    def test_log_message_types(self):
        """Test handling different types of log messages"""
        test_cases = [
            "Simple string message",
            "Message with numbers: 12345",
            "Message with special characters: !@#$%^&*()",
            "Message with unicode: 🚀✨🌟",
            "Very long message " * 10,
            "Empty string",  # Should still be handled
            "Message with newlines\nand multiple\nlines"
        ]
        
        for message in test_cases:
            self.tracker._handle_log_message(message)
        
        # All messages should be stored as-is
        self.assertEqual(len(self.tracker.live_logs), len(test_cases))
        for i, expected_message in enumerate(test_cases):
            self.assertEqual(self.tracker.live_logs[i], expected_message)

    def test_concurrent_log_handling(self):
        """Test that log handling works correctly with concurrent access"""
        import threading
        
        # Reset logs
        self.tracker.live_logs = []
        
        # Create multiple threads that add logs simultaneously
        def add_logs(thread_id, count):
            for i in range(count):
                self.tracker._handle_log_message(f"Thread {thread_id} - Message {i}")
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_logs, args=(i, 10))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 30 messages total
        self.assertEqual(len(self.tracker.live_logs), 30)
        
        # Verify all messages are present (order may vary due to threading)
        thread_messages = {0: 0, 1: 0, 2: 0}
        for log in self.tracker.live_logs:
            if log.startswith("Thread 0"):
                thread_messages[0] += 1
            elif log.startswith("Thread 1"):
                thread_messages[1] += 1
            elif log.startswith("Thread 2"):
                thread_messages[2] += 1
        
        # Each thread should have added 10 messages
        self.assertEqual(thread_messages[0], 10)
        self.assertEqual(thread_messages[1], 10)
        self.assertEqual(thread_messages[2], 10)

    def test_log_clear_functionality(self):
        """Test that logs can be cleared properly"""
        # Add some logs
        for i in range(10):
            self.tracker._handle_log_message(f"Test log {i}")
        
        # Verify logs are present
        self.assertEqual(len(self.tracker.live_logs), 10)
        
        # Clear logs through reset interface
        with patch('src.optimization.pipeline_task_tracker.ui.notify'):
            self.tracker._reset_interface()
        
        # Logs should be empty
        self.assertEqual(len(self.tracker.live_logs), 0)
        self.assertEqual(self.tracker.live_logs, [])

    def test_log_performance_benchmark(self):
        """Test performance of log handling with many messages"""
        import time
        
        # Test with a large number of messages
        num_messages = 1000
        start_time = time.time()
        
        for i in range(num_messages):
            self.tracker._handle_log_message(f"Performance test message {i}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle messages quickly
        self.assertLess(total_time, 1.0, 
                       f"Handling {num_messages} messages took {total_time:.3f}s, expected <1s")
        
        # All messages should be present
        self.assertEqual(len(self.tracker.live_logs), num_messages)
        
        # Test truncation performance
        self.tracker.MAX_LOG_LINES = 100
        start_time = time.time()
        
        for i in range(num_messages):
            self.tracker._handle_log_message(f"Truncation test message {i}")
        
        end_time = time.time()
        truncation_time = end_time - start_time
        
        # Truncation should also be efficient
        self.assertLess(truncation_time, 0.5, 
                       f"Truncation handling took {truncation_time:.3f}s, expected <0.5s")
        
        # Should only have the most recent 100 messages
        self.assertEqual(len(self.tracker.live_logs), 100)


if __name__ == '__main__':
    unittest.main()