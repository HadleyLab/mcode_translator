import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.task_queue import BenchmarkTask

class TestBenchmarkTask(unittest.TestCase):

    def test_initialization(self):
        task = BenchmarkTask(
            prompt_name="p1",
            model_name="c1",
            trial_id="t1",
            expected_entities=[],
            expected_mappings=[]
        )
        self.assertIsNotNone(task)
        self.assertEqual(task.prompt_name, "p1")

if __name__ == '__main__':
    unittest.main()