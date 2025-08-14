"""
Cache Layer module for improving query performance through result caching.
Implements LRU cache with TTL for search results and filtered data.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
import dataclasses
from datetime import datetime, timedelta
import json
import hashlib
from collections import OrderedDict
from .search_index import IndexedCode
from .query_parser import QueryExpression, QueryTerm, QueryOperator


@dataclass
class CacheEntry:
    """Represents a cached result with metadata"""
    data: Any
    expiry: datetime
    access_count: int = 0


class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle complex objects"""
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, QueryOperator):
            return obj.value
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class CacheLayer:
    """Manages caching of search results and filtered data"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache with size limit and TTL
        
        Args:
            max_size: Maximum number of entries in cache
            default_ttl: Default time-to-live in seconds
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._json_encoder = EnhancedJSONEncoder()

    def set(self, *args, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store value in cache
        
        Args:
            *args: Components to generate cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
        """
        key = self._generate_key(*args)
        expiry = datetime.now() + timedelta(seconds=ttl or self._default_ttl)
        
        # If cache is full, remove least recently used item
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            
        self._cache[key] = CacheEntry(data=value, expiry=expiry)
        self._cache.move_to_end(key)  # Move to end (most recently used)

    def get(self, args: Tuple) -> Optional[Any]:
        """
        Retrieve value from cache if it exists and is not expired
        
        Args:
            args: Tuple of components that make up the cache key
            
        Returns:
            Cached value if found and valid, None otherwise
        """
        key = self._generate_key(*args)
        
        if key not in self._cache:
            return None
            
        entry = self._cache[key]
        
        # Check if entry has expired
        if datetime.now() > entry.expiry:
            del self._cache[key]
            return None
            
        # Update access count and move to end (most recently used)
        entry.access_count += 1
        self._cache.move_to_end(key)
        
        return entry.data

    def _generate_key(self, *args: Any) -> str:
        """
        Generate cache key from arguments
        
        Args:
            *args: Variable arguments to include in key generation
            
        Returns:
            SHA-256 hash of the serialized arguments
        """
        # Convert all arguments to JSON-serializable format
        try:
            key_string = self._json_encoder.encode(args)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Unable to serialize cache key components: {e}")
        
        return hashlib.sha256(key_string.encode()).hexdigest()

    def invalidate(self, *args: Any) -> None:
        """
        Remove specific entry from cache
        
        Args:
            *args: Components of the cache key to invalidate
        """
        key = self._generate_key(*args)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from cache"""
        self._cache.clear()

    def clear_expired(self) -> None:
        """Remove all expired entries from cache"""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry.expiry
        ]
        for key in expired_keys:
            del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'utilization': len(self._cache) / self._max_size,
            'entry_count_by_type': self._get_entry_type_counts(),
            'most_accessed': self._get_most_accessed(limit=5)
        }

    def _get_entry_type_counts(self) -> Dict[str, int]:
        """Get counts of different types of cached entries"""
        type_counts: Dict[str, int] = {}
        for entry in self._cache.values():
            entry_type = type(entry.data).__name__
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        return type_counts

    def _get_most_accessed(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most frequently accessed cache entries"""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        return [(key, entry.access_count) for key, entry in sorted_entries[:limit]]