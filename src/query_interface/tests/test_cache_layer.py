"""
Unit tests for CacheLayer component.
Tests caching functionality, expiration, and eviction policies.
"""
import pytest
from datetime import datetime, timedelta
import time
from ..cache_layer import CacheLayer, CacheEntry
from ..query_parser import QueryExpression, QueryOperator, QueryTerm
from ..filter_engine import FilterCriteria


@pytest.fixture
def cache():
    """Create CacheLayer instance for tests"""
    return CacheLayer(max_size=3, default_ttl=1)


@pytest.fixture
def sample_data():
    """Sample data for testing cache operations"""
    return {
        'query': QueryExpression(
            operator=QueryOperator.AND,
            terms=[QueryTerm(field='code', value='J45.50')]
        ),
        'filter_criteria': FilterCriteria(
            types=['ICD-10'],
            categories=['respiratory']
        ),
        'results': ['result1', 'result2']
    }


class TestCacheLayer:
    """Test cases for CacheLayer"""

    def test_set_and_get(self, cache):
        """Test basic set and get operations"""
        cache.set(value='test_value', key_components=('key1', 'key2'), ttl=10)
        result = cache.get(('key1', 'key2'))
        
        assert result == 'test_value'

    def test_expiration(self, cache):
        """Test cache entry expiration"""
        cache.set(value='test_value', key_components=('key',), ttl=1)  # 1 second TTL
        
        # Value should be available immediately
        assert cache.get(('key',)) == 'test_value'
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Value should be expired
        assert cache.get(('key',)) is None

    def test_max_size_enforcement(self, cache):
        """Test cache size limit and LRU eviction"""
        # Add more items than max_size
        cache.set(value='value1', key_components=('key1',), ttl=10)
        cache.set(value='value2', key_components=('key2',), ttl=10)
        cache.set(value='value3', key_components=('key3',), ttl=10)
        cache.set(value='value4', key_components=('key4',), ttl=10)  # Should evict 'value1'
        
        assert cache.get(('key1',)) is None  # Evicted
        assert cache.get(('key2',)) == 'value2'
        assert cache.get(('key3',)) == 'value3'
        assert cache.get(('key4',)) == 'value4'

    def test_complex_keys(self, cache, sample_data):
        """Test caching with complex key components"""
        key_components = (sample_data['query'], sample_data['filter_criteria'])
        cache.set(
            value=sample_data['results'],
            key_components=key_components,
            ttl=10
        )
        
        result = cache.get((
            sample_data['query'],
            sample_data['filter_criteria']
        ))
        
        assert result == sample_data['results']

    def test_clear(self, cache):
        """Test clearing all cache entries"""
        cache.set(value='value1', key_components=('key1',), ttl=10)
        cache.set(value='value2', key_components=('key2',), ttl=10)
        
        cache.clear()
        
        assert cache.get(('key1',)) is None
        assert cache.get(('key2',)) is None
        assert len(cache._cache) == 0

    def test_clear_expired(self, cache):
        """Test clearing only expired entries"""
        # Add mix of expired and non-expired entries
        cache.set(value='value1', key_components=('key1',), ttl=1)  # Will expire
        cache.set(value='value2', key_components=('key2',), ttl=10)  # Won't expire
        
        # Wait for first entry to expire
        time.sleep(1.1)
        
        cache.clear_expired()
        
        assert cache.get(('key1',)) is None  # Should be cleared
        assert cache.get(('key2',)) == 'value2'  # Should remain

    def test_invalidate(self, cache):
        """Test invalidating specific cache entry"""
        cache.set(value='value1', key_components=('key1',), ttl=10)
        cache.set(value='value2', key_components=('key2',), ttl=10)
        
        cache.invalidate('key1')
        
        assert cache.get(('key1',)) is None
        assert cache.get(('key2',)) == 'value2'

    def test_access_count_tracking(self, cache):
        """Test tracking of cache access counts"""
        cache.set(value='value', key_components=('key',), ttl=10)
        
        # Access the value multiple times
        for _ in range(3):
            cache.get(('key',))
        
        entry = cache._cache[cache._generate_key('key')]
        assert entry.access_count == 3

    def test_cache_stats(self, cache):
        """Test cache statistics generation"""
        cache.set(value='value1', key_components=('key1',), ttl=10)
        cache.set(value='value2', key_components=('key2',), ttl=10)
        
        # Access values different numbers of times
        cache.get(('key1',))
        for _ in range(3):
            cache.get(('key2',))
        
        stats = cache.get_stats()
        
        assert stats['size'] == 2
        assert stats['max_size'] == 3
        assert stats['utilization'] == 2/3
        assert isinstance(stats['entry_count_by_type'], dict)
        assert isinstance(stats['most_accessed'], list)
        
        # Check most accessed entries
        most_accessed = stats['most_accessed']
        assert len(most_accessed) > 0
        # key2 should have higher access count
        assert most_accessed[0][1] > 1

    def test_key_generation_consistency(self, cache, sample_data):
        """Test consistency of cache key generation"""
        # Generate keys for same data multiple times
        key1 = cache._generate_key(
            sample_data['query'],
            sample_data['filter_criteria']
        )
        key2 = cache._generate_key(
            sample_data['query'],
            sample_data['filter_criteria']
        )
        
        assert key1 == key2

    def test_serialization(self, cache, sample_data):
        """Test serialization of complex objects for caching"""
        serialized = cache._serialize_object(sample_data['query'])
        
        assert isinstance(serialized, dict)
        assert 'operator' in serialized
        assert 'terms' in serialized

    def test_ttl_override(self, cache):
        """Test overriding default TTL"""
        # Set with custom TTL
        cache.set(value='value', key_components=('key',), ttl=2)  # 2 second TTL
        
        # Check after 1 second (default TTL)
        time.sleep(1.1)
        assert cache.get(('key',)) == 'value'  # Should still exist
        
        # Check after 2 seconds
        time.sleep(1)
        assert cache.get(('key',)) is None  # Should be expired

    def test_lru_behavior(self, cache):
        """Test Least Recently Used eviction behavior"""
        cache.set(value='value1', key_components=('key1',), ttl=10)
        cache.set(value='value2', key_components=('key2',), ttl=10)
        cache.set(value='value3', key_components=('key3',), ttl=10)
        
        # Access key1 to make it most recently used
        cache.get(('key1',))
        
        # Add new value, should evict key2 (least recently used)
        cache.set(value='value4', key_components=('key4',), ttl=10)
        
        assert cache.get(('key1',)) == 'value1'  # Should exist
        assert cache.get(('key2',)) is None      # Should be evicted
        assert cache.get(('key3',)) == 'value3'  # Should exist
        assert cache.get(('key4',)) == 'value4'  # Should exist