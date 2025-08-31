# Cache Debugging Guide

This guide explains how to use the disk-based caching system and debugging tools for the mCODE Translator project.

## Overview

The project now uses `diskcache` instead of `@lru_cache` for persistent caching. This allows cache entries to persist between program runs and provides better debugging capabilities.

## Cache Locations

- **Fetcher Cache**: `./cache/fetcher_cache/` - Caches ClinicalTrials.gov API responses
- **Code Extraction Cache**: `./cache/code_extraction_cache/` - Caches code extraction results
- **LLM Cache**: `./cache/llm_cache/` - Caches LLM API responses

## Debugging Tool

The `cache_debug.py` script provides utilities for inspecting and managing caches during development.

### Usage

```bash
python cache_debug.py [command] [args...]
```

### Commands

1. **Show cache statistics**
   ```bash
   python cache_debug.py stats
   ```

2. **List cache keys**
   ```bash
   python cache_debug.py list fetcher_cache
   python cache_debug.py list code_extraction_cache
   python cache_debug.py list llm_cache
   ```

3. **Clear caches**
   ```bash
   python cache_debug.py clear                    # Clear all caches
   python cache_debug.py clear fetcher_cache      # Clear specific cache
   python cache_debug.py clear code_extraction_cache
   python cache_debug.py clear llm_cache
   ```

4. **Get cache entry**
   ```bash
   python cache_debug.py get fetcher_cache "key_name"
   python cache_debug.py get code_extraction_cache "key_name"
   python cache_debug.py get llm_cache "key_name"
   ```

5. **Delete cache entry**
   ```bash
   python cache_debug.py delete fetcher_cache "key_name"
   python cache_debug.py delete code_extraction_cache "key_name"
   python cache_debug.py delete llm_cache "key_name"
   ```

## Manual Cache Inspection

You can also inspect cache files directly in the file system:

- Navigate to `./cache/fetcher_cache/` or `./cache/code_extraction_cache/`
- Cache entries are stored as files with names based on their keys
- File contents are the cached data in serialized format

## Cache Management in Code

The `src/utils/cache_manager.py` module provides a programmatic interface for cache management:

```python
from src.utils.cache_manager import cache_manager

# Get cache statistics
stats = cache_manager.get_cache_stats()

# Clear a specific cache
cache_manager.clear_cache("fetcher_cache")

# List keys in a cache
keys = cache_manager.list_cache_keys("code_extraction_cache")

# Get a specific cache entry
entry = cache_manager.get_cache_entry("fetcher_cache", "key_name")

# Delete a specific cache entry
cache_manager.delete_cache_entry("fetcher_cache", "key_name")

# Work with LLM cache
llm_keys = cache_manager.list_cache_keys("llm_cache")
llm_entry = cache_manager.get_cache_entry("llm_cache", "key_name")
cache_manager.clear_cache("llm_cache")
```

## Development Workflow

1. **During development**, use the debugging tools to inspect cache behavior
2. **When testing**, clear caches to ensure fresh data
3. **For production**, caches will persist to improve performance

## Troubleshooting

If you encounter cache-related issues:

1. **Clear all caches**: `python cache_debug.py clear`
2. **Check cache statistics**: `python cache_debug.py stats`
3. **Verify cache directory permissions**
4. **Ensure sufficient disk space**

## Backend Swapping

To switch between different cache backends, modify the cache initialization in:
- `src/pipeline/fetcher.py`
- `archive/pipeline/code_extraction.py`

Example for in-memory cache (useful for testing):
```python
cache = diskcache.Cache(':memory:')
```

Example for different disk locations:
```python
cache = diskcache.Cache('/tmp/my_cache')
```

For LLM caching, the cache is automatically initialized in `./cache/llm_cache/` directory.