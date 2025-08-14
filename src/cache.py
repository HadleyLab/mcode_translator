import os
import json
import time
from typing import Any, Optional
from .config import Config


class CacheManager:
    """
    Manages caching of API responses to reduce API calls
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = config.cache_dir
        
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key
        
        Args:
            key: Cache key
            
        Returns:
            File path for the cache entry
        """
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")
        
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.config.is_cache_enabled():
            return None
            
        cache_path = self._get_cache_path(key)
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            return None
            
        # Check if cache is expired
        if time.time() - os.path.getmtime(cache_path) > self.config.cache_ttl:
            # Remove expired cache file
            os.remove(cache_path)
            return None
            
        # Read cached data
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If there's an error reading the cache, remove it
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.config.is_cache_enabled():
            return
            
        cache_path = self._get_cache_path(key)
        
        # Write cache data
        try:
            with open(cache_path, 'w') as f:
                json.dump(value, f)
        except IOError:
            # If we can't write to cache, just ignore it
            pass
    
    def clear(self) -> None:
        """
        Clear all cache entries
        """
        if not self.config.is_cache_enabled() or not self.cache_dir:
            return
            
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)