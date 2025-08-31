import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.cache_manager import CacheManager

class TestCacheManager(unittest.TestCase):

    def test_initialization(self):
        cache = CacheManager()
        self.assertIsNotNone(cache)
        self.assertIsNotNone(cache.cache)

if __name__ == '__main__':
    unittest.main()