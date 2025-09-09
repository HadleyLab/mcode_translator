# Fetcher System Caching Update

## Overview

This document describes how the fetcher system will be updated to use the new unified caching approach.

## Current State

The fetcher system currently uses the `@cache_api_response` decorator from `src/utils/cache_decorator.py` to cache API responses:

```python
@cache_api_response(ttl=3600)  # Cache for 1 hour
def search_trials(search_expr: str, fields=None, max_results: int = 100, page_token: str = None):
    # Implementation
    pass

@cache_api_response(ttl=86400)  # Cache for 24 hours
def get_full_study(nct_id: str):
    # Implementation
    pass

@cache_api_response(ttl=3600)  # Cache for 1 hour
def calculate_total_studies(search_expr: str, fields=None, page_size: int = 100):
    # Implementation
    pass
```

This uses a global `api_cache` instance with a default cache directory of `.api_cache`.

## Proposed Changes

### 1. Replace Decorator Usage

The `@cache_api_response` decorator usage will be replaced with direct usage of the UnifiedAPIManager:

```python
# Instead of:
@cache_api_response(ttl=3600)
def search_trials(search_expr: str, fields=None, max_results: int = 100, page_token: str = None):

# Use:
from src.utils.api_manager import UnifiedAPIManager
api_manager = UnifiedAPIManager()
clinical_trials_cache = api_manager.get_cache("clinical_trials")

def search_trials(search_expr: str, fields=None, max_results: int = 100, page_token: str = None):
    # Generate cache key
    cache_key_data = {
        "function": "search_trials",
        "search_expr": search_expr,
        "fields": fields,
        "max_results": max_results,
        "page_token": page_token
    }
    
    # Try to get cached result
    cached_result = clinical_trials_cache.get_by_key(cache_key_data)
    if cached_result is not None:
        logger.info("Cache HIT for search_trials")
        return cached_result
    
    # If not cached, call the original function
    logger.info("Cache MISS for search_trials, calling function...")
    result = _search_trials(search_expr, fields, max_results, page_token)
    
    # Store result in cache with specified TTL
    clinical_trials_cache.set_by_key(result, cache_key_data, ttl=3600)
    
    return result
```

### 2. Update Cache Storage

The cache files will now be stored in a namespace-specific directory structure:

```
.api_cache/
  clinical_trials/
    [cache files]
```

Instead of the previous flat structure:

```
.api_cache/
  [cache files]
```

### 3. Preserve TTL Values

The existing TTL values will be preserved:
- `search_trials`: 1 hour (3600 seconds)
- `get_full_study`: 24 hours (86400 seconds)
- `calculate_total_studies`: 1 hour (3600 seconds)

## Implementation Steps

1. Import the UnifiedAPIManager in `src/pipeline/fetcher.py`
2. Create a cache instance for the "clinical_trials" namespace
3. Replace decorator usage with direct cache calls
4. Preserve existing TTL values for each function
5. Update cache key generation to match the new system
6. Maintain existing logging and error handling

## Benefits

1. **Consistent Cache Management**: All caching goes through the same unified system
2. **Namespace Separation**: Clinical trials cache files are separated from other cache namespaces
3. **Centralized Configuration**: Cache settings can be managed in one place
4. **Improved Maintainability**: Single point of control for cache-related functionality
5. **Better Organization**: Cache files are organized by namespace