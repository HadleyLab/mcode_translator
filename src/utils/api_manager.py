"""
Unified API Manager - Centralized cache management for all API calls in the system
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from .api_cache import APICache
from .async_api_cache import AsyncAPICache

# Set up logging
logger = logging.getLogger(__name__)


class APIManager:
    """Unified API manager handling caching for all API calls in the system"""

    def __init__(self, cache_dir: str = ".api_cache"):
        """
        Initialize the unified API manager with strict config-based TTL

        Args:
            cache_dir: Base directory for all cache files
        """
        from src.utils.config import Config

        config = Config()

        self.cache_dir = cache_dir
        self.default_ttl = config.get_cache_ttl()
        self.caches = {}
        # Create base cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(
            f"APIManager initialized with config TTL: {self.default_ttl} seconds"
        )

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
                namespace=cache_namespace,
                default_ttl=self.default_ttl,
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
                logger.warning(
                    f"Failed to clear namespace cache directory {namespace_cache_dir}: {e}"
                )

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
                total_items += namespace_stats.get("cached_items", 0)
                total_size += namespace_stats.get("total_size_bytes", 0)

            stats["total"] = {
                "cached_items": total_items,
                "total_size_bytes": total_size,
                "namespaces": list(self.caches.keys()),
            }
            return stats
        else:
            # Get stats for specific namespace
            cache = self.get_cache(cache_namespace)
            return {cache_namespace: cache.get_stats()}

    # Async Cache Operations
    async def aget_cache(self, cache_namespace: str = "default") -> "AsyncAPICache":
        """
        Get an async cache instance for a specific namespace

        Args:
            cache_namespace: Namespace for the cache

        Returns:
            AsyncAPICache instance for the specified namespace
        """
        if cache_namespace not in self.caches:
            # Create namespace-specific cache directory
            namespace_cache_dir = os.path.join(self.cache_dir, cache_namespace)
            self.caches[cache_namespace] = AsyncAPICache(
                cache_dir=namespace_cache_dir,
                namespace=cache_namespace,
                default_ttl=self.default_ttl,
            )
        return self.caches[cache_namespace]

    async def aclear_cache(self, cache_namespace: str = None) -> None:
        """
        Async clear cache for a specific namespace or all caches

        Args:
            cache_namespace: Namespace to clear, or None to clear all caches
        """
        if cache_namespace is None:
            # Clear all async caches
            tasks = [cache.aclear() for cache in self.caches.values()]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Clear base directory
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
                await self.caches[cache_namespace].aclear()

            # Clear namespace directory
            namespace_cache_dir = os.path.join(self.cache_dir, cache_namespace)
            try:
                if os.path.exists(namespace_cache_dir):
                    for filename in os.listdir(namespace_cache_dir):
                        file_path = os.path.join(namespace_cache_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            except Exception as e:
                logger.warning(
                    f"Failed to clear namespace cache directory {namespace_cache_dir}: {e}"
                )

    async def aget_cache_stats(self, cache_namespace: str = None) -> Dict[str, Any]:
        """
        Async get cache statistics for a specific namespace or all caches

        Args:
            cache_namespace: Namespace to get stats for, or None for all caches

        Returns:
            Dictionary with cache statistics
        """
        if cache_namespace is None:
            # Get stats for all async caches
            tasks = [
                cache.aget_stats()
                for cache in self.caches.values()
                if hasattr(cache, "aget_stats")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            stats = {}
            total_items = 0
            total_size = 0

            for (namespace, cache), result in zip(self.caches.items(), results):
                if not isinstance(result, Exception):
                    stats[namespace] = result
                    total_items += result.get("cached_items", 0)
                    total_size += result.get("total_size_bytes", 0)

            stats["total"] = {
                "cached_items": total_items,
                "total_size_bytes": total_size,
                "namespaces": list(self.caches.keys()),
            }
            return stats
        else:
            # Get stats for specific namespace
            cache = await self.aget_cache(cache_namespace)
            stats = await cache.aget_stats()
            return {cache_namespace: stats}


