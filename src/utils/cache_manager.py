"""
Cache Manager for mCODE Translator
Provides utilities for managing and debugging disk-based caches
"""
import os
import diskcache
import json
from typing import Any, Dict, List

class CacheManager:
    """Manages disk-based caches for debugging and inspection"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Base directory for all caches
        """
        self.cache_dir = cache_dir
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize caches
        self.fetcher_cache = diskcache.Cache(os.path.join(cache_dir, "fetcher_cache"))
        self.code_extraction_cache = diskcache.Cache(os.path.join(cache_dir, "code_extraction_cache"))
        self.llm_cache = diskcache.Cache(os.path.join(cache_dir, "llm_cache"))
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {}
        
        # Fetcher cache stats
        fetcher_stats = self.fetcher_cache.stats()
        stats["fetcher_cache"] = {
            "size": len(self.fetcher_cache),
            "directory": self.fetcher_cache.directory,
            "stats": fetcher_stats
        }
        
        # Code extraction cache stats
        code_stats = self.code_extraction_cache.stats()
        stats["code_extraction_cache"] = {
            "size": len(self.code_extraction_cache),
            "directory": self.code_extraction_cache.directory,
            "stats": code_stats
        }
        
        # LLM cache stats
        llm_stats = self.llm_cache.stats()
        stats["llm_cache"] = {
            "size": len(self.llm_cache),
            "directory": self.llm_cache.directory,
            "stats": llm_stats
        }
        
        return stats
    
    def clear_cache(self, cache_name: str = None) -> bool:
        """
        Clear a specific cache or all caches
        
        Args:
            cache_name: Name of cache to clear (fetcher_cache, code_extraction_cache) or None for all
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_name is None:
                # Clear all caches
                self.fetcher_cache.clear()
                self.code_extraction_cache.clear()
                self.llm_cache.clear()
                return True
            elif cache_name == "fetcher_cache":
                self.fetcher_cache.clear()
                return True
            elif cache_name == "code_extraction_cache":
                self.code_extraction_cache.clear()
                return True
            elif cache_name == "llm_cache":
                self.llm_cache.clear()
                return True
            else:
                return False
        except Exception as e:
            print(f"Error clearing cache {cache_name}: {e}")
            return False
    
    def list_cache_keys(self, cache_name: str) -> List[str]:
        """
        List all keys in a specific cache
        
        Args:
            cache_name: Name of cache to inspect (fetcher_cache, code_extraction_cache)
            
        Returns:
            List of cache keys
        """
        keys = []
        try:
            if cache_name == "fetcher_cache":
                keys = list(self.fetcher_cache.iterkeys())
            elif cache_name == "code_extraction_cache":
                keys = list(self.code_extraction_cache.iterkeys())
            elif cache_name == "llm_cache":
                keys = list(self.llm_cache.iterkeys())
        except Exception as e:
            print(f"Error listing cache keys for {cache_name}: {e}")
        
        return keys
    
    def get_cache_entry(self, cache_name: str, key: str) -> Any:
        """
        Get a specific entry from a cache
        
        Args:
            cache_name: Name of cache to inspect
            key: Key of the entry to retrieve
            
        Returns:
            Cache entry value or None if not found
        """
        try:
            if cache_name == "fetcher_cache":
                return self.fetcher_cache.get(key)
            elif cache_name == "code_extraction_cache":
                return self.code_extraction_cache.get(key)
            elif cache_name == "llm_cache":
                return self.llm_cache.get(key)
        except Exception as e:
            print(f"Error retrieving cache entry {key} from {cache_name}: {e}")
        
        return None
    
    def delete_cache_entry(self, cache_name: str, key: str) -> bool:
        """
        Delete a specific entry from a cache
        
        Args:
            cache_name: Name of cache to modify
            key: Key of the entry to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if cache_name == "fetcher_cache":
                return self.fetcher_cache.delete(key)
            elif cache_name == "code_extraction_cache":
                return self.code_extraction_cache.delete(key)
            elif cache_name == "llm_cache":
                return self.llm_cache.delete(key)
        except Exception as e:
            print(f"Error deleting cache entry {key} from {cache_name}: {e}")
        
        return False

# Global cache manager instance
cache_manager = CacheManager()

def debug_cache_info():
    """
    Print debug information about all caches
    """
    stats = cache_manager.get_cache_stats()
    print("=== Cache Debug Information ===")
    for cache_name, cache_info in stats.items():
        print(f"\n{cache_name}:")
        print(f"  Directory: {cache_info['directory']}")
        print(f"  Size: {cache_info['size']} entries")
        print(f"  Stats: {cache_info['stats']}")

if __name__ == "__main__":
    debug_cache_info()