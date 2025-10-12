"""
Async API Cache - Pure asynchronous disk-based cache for API responses
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

# Set up logging
logger = logging.getLogger(__name__)


class AsyncAPICache:
    """Async version of disk-based cache for API responses"""

    def __init__(self, cache_dir: str, namespace: str, default_ttl: int):
        """
        Initialize the pure async cache without thread pools

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
            f"Initialized Pure Async API cache for namespace '{namespace}' at {cache_dir} with TTL {default_ttl}"
        )

    def _get_cache_key(self, func_name: str, *args: Any, **kwargs: Any) -> str:
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

    async def aget(self, func_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
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
        Pure async get cached result by cache key data

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
            # Use aiofiles for async file I/O if available, otherwise use asyncio.to_thread
            try:
                import aiofiles
                async with aiofiles.open(cache_path, 'r') as f:
                    content = await f.read()
                    cached_data = json.loads(content)
            except ImportError:
                # Fallback to asyncio.to_thread for Python 3.9+
                cached_data = await asyncio.to_thread(self._sync_load_json, cache_path)

            # Check expiration
            ttl = cached_data.get("ttl", None)
            if ttl and ttl > 0 and time.time() - cached_data.get("timestamp", 0) > ttl:
                await asyncio.to_thread(os.remove, cache_path)
                return None

            logger.debug(f"Async Cache HIT {cache_key[:8]}... in '{self.namespace}'")
            return cached_data["result"]
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    def _sync_load_json(self, cache_path: str) -> Dict[str, Any]:
        """Synchronous JSON loading helper"""
        with open(cache_path, 'r') as f:
            return json.load(f)

    async def aset(self, result: Any, func_name: str, *args: Any, **kwargs: Any) -> None:
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
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Pure async store result in cache by cache key

        Args:
            result: Result to cache
            cache_key: The cache key
            func_name: Name of the function (for logging)
            args: Positional arguments (for logging)
            kwargs: Keyword arguments (for logging)
        """
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

            # Use aiofiles for async file I/O if available, otherwise use asyncio.to_thread
            try:
                import aiofiles
                async with aiofiles.open(cache_path, 'w') as f:
                    await f.write(json.dumps(cached_data, default=str))
            except ImportError:
                # Fallback to asyncio.to_thread for Python 3.9+
                await asyncio.to_thread(self._sync_write_json, cache_path, cached_data)

            logger.debug(f"Async Cache STORED {cache_key[:8]}... in '{self.namespace}'")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def _sync_write_json(self, cache_path: str, data: Dict[str, Any]) -> None:
        """Synchronous JSON writing helper"""
        with open(cache_path, 'w') as f:
            json.dump(data, f, default=str)

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
        """Pure async clear all cached data for this namespace"""
        try:
            # Use asyncio.to_thread for directory operations
            await asyncio.to_thread(self._sync_clear_cache)
            logger.info(f"Async Cache cleared for namespace '{self.namespace}'")
        except Exception as e:
            logger.warning(f"Failed to clear async cache for namespace '{self.namespace}': {e}")

    def _sync_clear_cache(self) -> None:
        """Synchronous cache clearing helper"""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    async def aget_stats(self) -> Dict[str, Any]:
        """Pure async get cache statistics"""
        try:
            # Use asyncio.to_thread for directory operations
            return await asyncio.to_thread(self._sync_get_stats)
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

    def _sync_get_stats(self) -> Dict[str, Any]:
        """Synchronous stats gathering helper"""
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
    async def abatch_get(self, key_data_list: List[Any]) -> List[Any]:
        """
        Async batch get multiple cache entries

        Args:
            key_data_list: List of cache key data

        Returns:
            List of cached results (None for misses, exceptions for failures)
        """
        tasks = [self._aget_by_key(key_data) for key_data in key_data_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(results)

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
    async def awarm_cache(self, key_data_list: List[Any], fetch_func: Callable[..., Any]) -> None:
        """
        Async warm cache by pre-fetching and caching results

        Args:
            key_data_list: List of cache key data to warm
            fetch_func: Function to fetch data if not cached
        """
        logger.info(f"Warming cache for {len(key_data_list)} items in namespace '{self.namespace}'")

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

    async def _fetch_and_cache(self, key_data: Any, fetch_func: Callable[..., Any]) -> None:
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
        logger.info(f"Starting background maintenance for cache namespace '{self.namespace}'")

        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._aperform_maintenance()
            except asyncio.CancelledError:
                logger.info(f"Background maintenance stopped for namespace '{self.namespace}'")
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
        """Pure async cleanup expired cache entries"""
        try:
            # Use asyncio.to_thread for cleanup operations
            return await asyncio.to_thread(self._sync_cleanup_expired)
        except Exception as e:
            logger.warning(f"Failed to cleanup expired cache entries: {e}")
            return 0

    def _sync_cleanup_expired(self) -> int:
        """Synchronous cleanup helper"""
        cleaned_count = 0
        current_time = time.time()

        for filename in os.listdir(self.cache_dir):
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(self.cache_dir, filename)
            try:
                with open(file_path) as f:
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
                except Exception:
                    pass

        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} expired cache entries in '{self.namespace}'"
            )

        return cleaned_count
