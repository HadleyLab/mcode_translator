"""
API Cache - Synchronous disk-based cache for API responses
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)


class APICache:
    """Generic disk-based cache for API responses"""

    def __init__(self, cache_dir: str, namespace: str, default_ttl: int):
        """
        Initialize the cache with strict TTL requirement

        Args:
            cache_dir: Base directory for cache files
            namespace: Namespace for this cache instance
            default_ttl: Required TTL for this cache from config
        """
        self.cache_dir = cache_dir
        self.namespace = namespace
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(
            f"Initialized API cache for namespace '{namespace}' at {cache_dir} with TTL {default_ttl}"
        )

    def _get_cache_key(self, func_name: str, *args: Any, **kwargs: Any) -> str:
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
            "kwargs": kwargs,
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

    def get(self, func_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
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
            with open(cache_path, "r") as f:
                cached_data = json.load(f)

            # Check if cache has expired (ttl=0 means never expire)
            ttl = cached_data.get("ttl", None)
            if (
                ttl is not None
                and ttl > 0
                and time.time() - cached_data.get("timestamp", 0) > ttl
            ):
                os.remove(cache_path)
                return None

            # Disable verbose cache logging to avoid confusion
            pass
            return cached_data["result"]
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

    def set(self, result: Any, func_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Store result in cache with configured TTL

        Args:
            result: Result to cache
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        self._set_by_key(result, cache_key, func_name, args, kwargs)

    def _set_by_key(
        self,
        result: Any,
        cache_key: str,
        func_name: str = "unknown",
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] = {},
    ) -> None:
        """
        Store result in cache by cache key using configured TTL

        Args:
            result: Result to cache
            cache_key: The cache key
            func_name: Name of the function (for logging)
            args: Positional arguments (for logging)
            kwargs: Keyword arguments (for logging)
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            # Handle Pydantic models by converting to dict
            serializable_result = self._make_serializable(result)

            cached_data = {
                "result": serializable_result,
                "timestamp": time.time(),
                "ttl": self.default_ttl,  # 0 means never expire
                "namespace": self.namespace,
                "function": func_name,
                "args": str(args),
                "kwargs": str(kwargs),
            }

            with open(cache_path, "w") as f:
                json.dump(cached_data, f, default=str)

            # Disable verbose cache logging to avoid confusion
            pass
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """
        Make an object serializable by converting Pydantic models to dicts

        Args:
            obj: Object to make serializable

        Returns:
            Serializable version of the object
        """
        if hasattr(obj, "model_dump"):
            # Pydantic model
            return obj.model_dump()
        elif isinstance(obj, dict):
            # Recursively handle dicts
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively handle lists
            return [self._make_serializable(item) for item in obj]
        else:
            # Return as-is for primitive types
            return obj

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

    def set_by_key(self, result: Any, cache_key_data: Any) -> None:
        """
        Store result in cache by cache key data using configured TTL

        Args:
            result: Result to cache
            cache_key_data: Data to generate cache key from
        """
        cache_key = self._generate_cache_key(cache_key_data)
        self._set_by_key(result, cache_key)

    def clear(self) -> None:
        """Clear all cached data for this namespace"""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"Cache cleared for namespace '{self.namespace}'")
        except Exception as e:
            logger.warning(
                f"Failed to clear cache for namespace '{self.namespace}': {e}"
            )

    def clear_cache(self) -> None:
        """Alias for clear() method"""
        self.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = os.listdir(self.cache_dir)
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in cache_files
                if os.path.isfile(os.path.join(self.cache_dir, f))
            )
            return {
                "cache_dir": self.cache_dir,
                "namespace": self.namespace,
                "cached_items": len(cache_files),
                "total_items": len(cache_files),  # Alias for compatibility
                "total_size_bytes": total_size,
            }
        except Exception as e:
            logger.warning(
                f"Failed to get cache stats for namespace '{self.namespace}': {e}"
            )
            return {
                "cache_dir": self.cache_dir,
                "namespace": self.namespace,
                "cached_items": 0,
                "total_items": 0,
                "total_size_bytes": 0,
                "error": str(e),
            }
