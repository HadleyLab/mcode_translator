"""
Cache decorator for API responses with disk-based caching
"""
import json
import hashlib
import os
import time
from functools import wraps
from typing import Any, Dict, Optional, Callable
import logging

# Set up logging
logger = logging.getLogger(__name__)

class APICache:
    """Simple disk-based cache for API responses"""
    
    def __init__(self, cache_dir: str = ".api_cache"):
        """
        Initialize the cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized API cache at {cache_dir}")
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """
        Generate a cache key based on function name and arguments
        
        Args:
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key as a string
        """
        # Create a hashable representation of the arguments
        key_data = {
            "function": func_name,
            "args": args,
            "kwargs": kwargs
        }
        
        # Convert to JSON string and hash it
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key
        
        Args:
            cache_key: The cache key
            
        Returns:
            Path to the cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, func_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Get cached result if available
        
        Args:
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        return self._get_by_key(cache_key)
    
    def _get_by_key(self, cache_key_data: Any) -> Optional[Any]:
        """
        Get cached result by cache key data
        
        Args:
            cache_key_data: Data to generate cache key from
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._generate_cache_key(cache_key_data)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache has expired (24 hours by default)
            if time.time() - cached_data.get('timestamp', 0) > cached_data.get('ttl', 86400):
                os.remove(cache_path)
                return None
            
            logger.info(f"Cache HIT with key {cache_key[:8]}...")
            return cached_data['result']
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None
    
    def get_by_key(self, cache_key_data: Any) -> Optional[Any]:
        """
        Get cached result by cache key data (public method)
        
        Args:
            cache_key_data: Data to generate cache key from
            
        Returns:
            Cached result or None if not found/expired
        """
        return self._get_by_key(cache_key_data)
    
    def set(self, result: Any, func_name: str, *args, **kwargs) -> None:
        """
        Store result in cache
        
        Args:
            result: Result to cache
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        self._set_by_key(result, cache_key, func_name, args, kwargs)
    
    def _set_by_key(self, result: Any, cache_key: str, func_name: str = "unknown", args: tuple = (), kwargs: dict = {}, ttl: int = 86400) -> None:
        """
        Store result in cache by cache key
        
        Args:
            result: Result to cache
            cache_key: The cache key
            func_name: Name of the function (for logging)
            args: Positional arguments (for logging)
            kwargs: Keyword arguments (for logging)
            ttl: Time to live in seconds (default: 24 hours)
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                'result': result,
                'timestamp': time.time(),
                'ttl': ttl,
                'function': func_name,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f)
            
            logger.info(f"Cache STORED with key {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
    
    def _generate_cache_key(self, key_data: Any) -> str:
        """
        Generate a cache key based on the provided data
        
        Args:
            key_data: Data to generate cache key from
            
        Returns:
            Cache key as a string
        """
        # If key_data is already a string, use it directly
        if isinstance(key_data, str):
            return key_data
            
        # Otherwise, create a hashable representation
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def set_by_key(self, result: Any, cache_key_data: Any, ttl: int = 86400) -> None:
        """
        Store result in cache by cache key data (public method)
        
        Args:
            result: Result to cache
            cache_key_data: Data to generate cache key from
            ttl: Time to live in seconds (default: 24 hours)
        """
        cache_key = self._generate_cache_key(cache_key_data)
        self._set_by_key(result, cache_key, ttl=ttl)
    
    def clear(self) -> None:
        """Clear all cached data"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def clear_cache(self) -> None:
        """Alias for clear() method"""
        self.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = os.listdir(self.cache_dir)
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f))
                            for f in cache_files if os.path.isfile(os.path.join(self.cache_dir, f)))
            return {
                'cache_dir': self.cache_dir,
                'cached_items': len(cache_files),
                'total_items': len(cache_files),  # Alias for compatibility
                'total_size_bytes': total_size
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {
                'cache_dir': self.cache_dir,
                'cached_items': 0,
                'total_items': 0,
                'total_size_bytes': 0,
                'error': str(e)
            }

# Global cache instance
api_cache = APICache()

def cache_api_response(ttl: int = 86400):
    """
    Decorator to cache API responses
    
    Args:
        ttl: Time to live in seconds (default: 24 hours)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get cached result
            cached_result = api_cache.get(func.__name__, *args, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # If not cached, call the function
            logger.info(f"Cache MISS for {func.__name__}, calling function...")
            result = func(*args, **kwargs)
            
            # Store result in cache with specified TTL
            api_cache.set(result, func.__name__, *args, **kwargs)
            
            return result
        return wrapper
    return decorator

# Convenience functions for manual cache operations
def clear_api_cache():
    """Clear all API cache"""
    api_cache.clear()

def get_cache_stats(cache_dir: str = None) -> Dict[str, Any]:
    """Get cache statistics"""
    try:
        target_dir = cache_dir if cache_dir else api_cache.cache_dir
        cache_files = os.listdir(target_dir)
        total_size = sum(os.path.getsize(os.path.join(target_dir, f))
                        for f in cache_files if os.path.isfile(os.path.join(target_dir, f)))
        return {
            'cache_dir': target_dir,
            'cached_items': len(cache_files),
            'total_size_bytes': total_size
        }
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {e}")
        target_dir = cache_dir if cache_dir else api_cache.cache_dir
        return {
            'cache_dir': target_dir,
            'cached_items': 0,
            'total_size_bytes': 0,
            'error': str(e)
        }