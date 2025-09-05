"""
Unified API Manager - Centralized cache management for all API calls in the system
"""
import json
import hashlib
import os
import time
from typing import Any, Dict, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)


class APICache:
    """Generic disk-based cache for API responses"""
    
    def __init__(self, cache_dir: str = ".api_cache", namespace: str = "default"):
        """
        Initialize the cache
        
        Args:
            cache_dir: Base directory for cache files
            namespace: Namespace for this cache instance
        """
        self.cache_dir = cache_dir
        self.namespace = namespace
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized API cache for namespace '{namespace}' at {cache_dir}")
    
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
            "namespace": self.namespace,
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
            
            # Check if cache has expired (never expire by default)
            ttl = cached_data.get('ttl', None)
            if ttl is not None and time.time() - cached_data.get('timestamp', 0) > ttl:
                os.remove(cache_path)
                return None
            
            logger.info(f"Cache HIT with key {cache_key[:8]}... in namespace '{self.namespace}'")
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
        Store result in cache with default TTL
        
        Args:
            result: Result to cache
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        self._set_by_key(result, cache_key, func_name, args, kwargs)
    
    def _set_by_key(self, result: Any, cache_key: str, func_name: str = "unknown", args: tuple = (), kwargs: dict = {}, ttl: int = None) -> None:
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
                'ttl': ttl,  # None means never expire
                'namespace': self.namespace,
                'function': func_name,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f)
            
            logger.info(f"Cache STORED with key {cache_key[:8]}... in namespace '{self.namespace}'")
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
    
    def set_by_key(self, result: Any, cache_key_data: Any, ttl: int = None) -> None:
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
        """Clear all cached data for this namespace"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"Cache cleared for namespace '{self.namespace}'")
        except Exception as e:
            logger.warning(f"Failed to clear cache for namespace '{self.namespace}': {e}")
    
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
                'namespace': self.namespace,
                'cached_items': len(cache_files),
                'total_items': len(cache_files),  # Alias for compatibility
                'total_size_bytes': total_size
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats for namespace '{self.namespace}': {e}")
            return {
                'cache_dir': self.cache_dir,
                'namespace': self.namespace,
                'cached_items': 0,
                'total_items': 0,
                'total_size_bytes': 0,
                'error': str(e)
            }


class UnifiedAPIManager:
    """Unified API manager handling caching for all API calls in the system"""
    
    def __init__(self, cache_dir: str = ".api_cache", default_ttl: int = None):
        """
        Initialize the unified API manager
        
        Args:
            cache_dir: Base directory for all cache files
            default_ttl: Default time-to-live for cache entries in seconds
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.caches = {}
        # Create base cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache(self, cache_namespace: str = "default") -> APICache:
        """
        Get a cache instance for a specific namespace
        
        Args:
            cache_namespace: Namespace for the cache (e.g., "llm", "clinical_trials")
            
        Returns:
            APICache instance for the specified namespace
        """
        if cache_namespace not in self.caches:
            # Create namespace-specific cache directory
            namespace_cache_dir = os.path.join(self.cache_dir, cache_namespace)
            self.caches[cache_namespace] = APICache(
                cache_dir=namespace_cache_dir,
                namespace=cache_namespace
            )
        return self.caches[cache_namespace]
    
    def clear_cache(self, cache_namespace: str = None) -> None:
        """
        Clear cache for a specific namespace or all caches
        
        Args:
            cache_namespace: Namespace to clear, or None to clear all caches
        """
        if cache_namespace is None:
            # Clear all caches
            for cache in self.caches.values():
                cache.clear()
            # Also clear the base directory of any files
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to clear base cache directory: {e}")
        else:
            # Clear specific namespace
            if cache_namespace in self.caches:
                self.caches[cache_namespace].clear()
            # Also clear the namespace directory
            namespace_cache_dir = os.path.join(self.cache_dir, cache_namespace)
            try:
                if os.path.exists(namespace_cache_dir):
                    for filename in os.listdir(namespace_cache_dir):
                        file_path = os.path.join(namespace_cache_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to clear namespace cache directory {namespace_cache_dir}: {e}")
    
    def get_cache_stats(self, cache_namespace: str = None) -> Dict[str, Any]:
        """
        Get cache statistics for a specific namespace or all caches
        
        Args:
            cache_namespace: Namespace to get stats for, or None for all caches
            
        Returns:
            Dictionary with cache statistics
        """
        if cache_namespace is None:
            # Get stats for all caches
            stats = {}
            total_items = 0
            total_size = 0
            
            for namespace, cache in self.caches.items():
                namespace_stats = cache.get_stats()
                stats[namespace] = namespace_stats
                total_items += namespace_stats.get('cached_items', 0)
                total_size += namespace_stats.get('total_size_bytes', 0)
            
            stats['total'] = {
                'cached_items': total_items,
                'total_size_bytes': total_size,
                'namespaces': list(self.caches.keys())
            }
            return stats
        else:
            # Get stats for specific namespace
            cache = self.get_cache(cache_namespace)
            return {cache_namespace: cache.get_stats()}