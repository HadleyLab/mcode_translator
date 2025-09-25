"""
Async API Cache - Asynchronous disk-based cache for API responses
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)


class AsyncAPICache:
    """Async version of disk-based cache for API responses"""

    def __init__(
        self, cache_dir: str, namespace: str, default_ttl: int, max_workers: int = 4
    ):
        """
        Initialize the async cache with thread pool for I/O operations

        Args:
            cache_dir: Base directory for cache files
            namespace: Namespace for this cache instance
            default_ttl: Required TTL for this cache from config
            max_workers: Maximum worker threads for I/O operations
        """
        self.cache_dir = cache_dir
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(
            f"Initialized Async API cache for namespace '{namespace}' at {cache_dir} with TTL {default_ttl}"
        )

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a cache key (same as sync version)"""
        key_data = {
            "function": func_name,
            "namespace": self.namespace,
            "args": args,
            "kwargs": kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    async def aget(self, func_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Async get cached result if available

        Args:
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        return await self._aget_by_key(cache_key)

    async def _aget_by_key(self, cache_key_data: Any) -> Optional[Any]:
        """
        Async get cached result by cache key data

        Args:
            cache_key_data: Data to generate cache key from

        Returns:
            Cached result or None if not found/expired
        """
        loop = asyncio.get_event_loop()

        def _sync_get():
            cache_key = self._generate_cache_key(cache_key_data)
            cache_path = self._get_cache_path(cache_key)

            if not os.path.exists(cache_path):
                return None

            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)

                # Check expiration
                ttl = cached_data.get("ttl", None)
                if (
                    ttl
                    and ttl > 0
                    and time.time() - cached_data.get("timestamp", 0) > ttl
                ):
                    os.remove(cache_path)
                    return None

                logger.debug(
                    f"Async Cache HIT {cache_key[:8]}... in '{self.namespace}'"
                )
                return cached_data["result"]
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
                return None

        return await loop.run_in_executor(self.executor, _sync_get)

    async def aset(self, result: Any, func_name: str, *args, **kwargs) -> None:
        """
        Async store result in cache

        Args:
            result: Result to cache
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._get_cache_key(func_name, *args, **kwargs)
        await self._aset_by_key(result, cache_key, func_name, args, kwargs)

    async def _aset_by_key(
        self,
        result: Any,
        cache_key: str,
        func_name: str = "unknown",
        args: tuple = (),
        kwargs: dict = {},
    ) -> None:
        """
        Async store result in cache by cache key

        Args:
            result: Result to cache
            cache_key: The cache key
            func_name: Name of the function (for logging)
            args: Positional arguments (for logging)
            kwargs: Keyword arguments (for logging)
        """
        loop = asyncio.get_event_loop()

        def _sync_set():
            cache_path = self._get_cache_path(cache_key)

            try:
                serializable_result = self._make_serializable(result)
                cached_data = {
                    "result": serializable_result,
                    "timestamp": time.time(),
                    "ttl": self.default_ttl,
                    "namespace": self.namespace,
                    "function": func_name,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }

                with open(cache_path, "w") as f:
                    json.dump(cached_data, f, default=str)

                logger.debug(
                    f"Async Cache STORED {cache_key[:8]}... in '{self.namespace}'"
                )
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

        await loop.run_in_executor(self.executor, _sync_set)

    async def aget_by_key(self, cache_key_data: Any) -> Optional[Any]:
        """
        Async get cached result by cache key data

        Args:
            cache_key_data: Data to generate cache key from

        Returns:
            Cached result or None if not found/expired
        """
        return await self._aget_by_key(cache_key_data)

    async def aset_by_key(self, result: Any, cache_key_data: Any) -> None:
        """
        Async store result in cache by cache key data

        Args:
            result: Result to cache
            cache_key_data: Data to generate cache key from
        """
        cache_key = self._generate_cache_key(cache_key_data)
        await self._aset_by_key(result, cache_key)

    async def aclear(self) -> None:
        """Async clear all cached data for this namespace"""
        loop = asyncio.get_event_loop()

        def _sync_clear():
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Async Cache cleared for namespace '{self.namespace}'")
            except Exception as e:
                logger.warning(
                    f"Failed to clear async cache for namespace '{self.namespace}': {e}"
                )

        await loop.run_in_executor(self.executor, _sync_clear)

    async def aget_stats(self) -> Dict[str, Any]:
        """Async get cache statistics"""
        loop = asyncio.get_event_loop()

        def _sync_stats():
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
                    "total_items": len(cache_files),
                    "total_size_bytes": total_size,
                }
            except Exception as e:
                logger.warning(
                    f"Failed to get async cache stats for namespace '{self.namespace}': {e}"
                )
                return {
                    "cache_dir": self.cache_dir,
                    "namespace": self.namespace,
                    "cached_items": 0,
                    "total_items": 0,
                    "total_size_bytes": 0,
                    "error": str(e),
                }

        return await loop.run_in_executor(self.executor, _sync_stats)

    def _make_serializable(self, obj: Any) -> Any:
        """Make an object serializable (same as sync version)"""
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
        """Generate a cache key (same as sync version)"""
        if isinstance(key_data, str):
            return key_data

        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    # Batch Operations
    async def abatch_get(self, key_data_list: List[Any]) -> List[Optional[Any]]:
        """
        Async batch get multiple cache entries

        Args:
            key_data_list: List of cache key data

        Returns:
            List of cached results (None for misses)
        """
        tasks = [self._aget_by_key(key_data) for key_data in key_data_list]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def abatch_set(self, items: List[Dict[str, Any]]) -> None:
        """
        Async batch set multiple cache entries

        Args:
            items: List of dicts with 'result' and 'key_data' keys
        """
        tasks = []
        for item in items:
            result = item["result"]
            key_data = item["key_data"]
            tasks.append(self.aset_by_key(result, key_data))

        await asyncio.gather(*tasks, return_exceptions=True)

    # Cache Warming
    async def awarm_cache(self, key_data_list: List[Any], fetch_func: callable) -> None:
        """
        Async warm cache by pre-fetching and caching results

        Args:
            key_data_list: List of cache key data to warm
            fetch_func: Function to fetch data if not cached
        """
        logger.info(
            f"Warming cache for {len(key_data_list)} items in namespace '{self.namespace}'"
        )

        # Check what's already cached
        cached_results = await self.abatch_get(key_data_list)

        # Identify cache misses
        to_fetch = []
        for i, (key_data, result) in enumerate(zip(key_data_list, cached_results)):
            if result is None or isinstance(result, Exception):
                to_fetch.append((i, key_data))

        if not to_fetch:
            logger.info("All items already cached - no warming needed")
            return

        logger.info(f"Fetching {len(to_fetch)} uncached items")

        # Fetch missing items (this could be parallelized depending on fetch_func)
        fetch_tasks = []
        for idx, key_data in to_fetch:
            task = self._fetch_and_cache(key_data, fetch_func)
            fetch_tasks.append(task)

        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        logger.info(f"Cache warming completed for namespace '{self.namespace}'")

    async def _fetch_and_cache(self, key_data: Any, fetch_func: callable) -> None:
        """Fetch data and cache it"""
        try:
            result = await fetch_func(key_data)
            await self.aset_by_key(result, key_data)
        except Exception as e:
            logger.warning(f"Failed to fetch and cache item: {e}")

    # Background Maintenance
    async def astart_background_maintenance(self, cleanup_interval: int = 300) -> None:
        """
        Start background maintenance tasks

        Args:
            cleanup_interval: Interval in seconds for cleanup operations
        """
        logger.info(
            f"Starting background maintenance for cache namespace '{self.namespace}'"
        )

        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._aperform_maintenance()
            except asyncio.CancelledError:
                logger.info(
                    f"Background maintenance stopped for namespace '{self.namespace}'"
                )
                break
            except Exception as e:
                logger.warning(f"Background maintenance error: {e}")

    async def _aperform_maintenance(self) -> None:
        """Perform background maintenance tasks"""
        try:
            await self._acleanup_expired()
            stats = await self.aget_stats()
            logger.debug(f"Maintenance stats '{self.namespace}': {stats}")
        except Exception as e:
            logger.warning(f"Maintenance failed: {e}")

    async def _acleanup_expired(self) -> int:
        """Async cleanup expired cache entries"""
        loop = asyncio.get_event_loop()

        def _sync_cleanup():
            try:
                cleaned_count = 0
                current_time = time.time()

                for filename in os.listdir(self.cache_dir):
                    if not filename.endswith(".json"):
                        continue

                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            cached_data = json.load(f)

                        ttl = cached_data.get("ttl", None)
                        if (
                            ttl is not None
                            and ttl > 0
                            and current_time - cached_data.get("timestamp", 0) > ttl
                        ):
                            os.remove(file_path)
                            cleaned_count += 1
                    except Exception:
                        # If we can't read the file, remove it
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except:
                            pass

                if cleaned_count > 0:
                    logger.info(
                        f"Cleaned up {cleaned_count} expired cache entries in '{self.namespace}'"
                    )

                return cleaned_count

            except Exception as e:
                logger.warning(f"Failed to cleanup expired cache entries: {e}")
                return 0

        return await loop.run_in_executor(self.executor, _sync_cleanup)
