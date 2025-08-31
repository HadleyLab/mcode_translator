import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.optimization.ui_utils.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):

    def test_initialization(self):
        # This class has file I/O, which can be tested.
        pass

if __name__ == '__main__':
    unittest.main()