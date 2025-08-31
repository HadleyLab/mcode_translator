#!/usr/bin/env python3
"""
Cache Debugging Tool for mCODE Translator
Utility script to inspect and manage disk-based caches during development
"""
import sys
import os
import json
from src.utils.cache_manager import cache_manager

def main():
    """Main function for cache debugging"""
    if len(sys.argv) < 2:
        print("Usage: python cache_debug.py [command] [args...]")
        print("Commands:")
        print("  stats          - Show cache statistics")
        print("  list [cache]   - List keys in a cache (fetcher_cache, code_extraction_cache, or llm_cache)")
        print("  clear [cache]  - Clear a cache or all caches")
        print("  get [cache] [key] - Get a specific entry from a cache")
        print("  delete [cache] [key] - Delete a specific entry from a cache")
        return
    
    command = sys.argv[1]
    
    if command == "stats":
        # Show cache statistics
        stats = cache_manager.get_cache_stats()
        print("=== Cache Statistics ===")
        for cache_name, cache_info in stats.items():
            print(f"\n{cache_name}:")
            print(f"  Directory: {cache_info['directory']}")
            print(f"  Size: {cache_info['size']} entries")
            if cache_info['stats']:
                print(f"  Stats: {cache_info['stats']}")
    
    elif command == "list":
        if len(sys.argv) < 3:
            print("Usage: python cache_debug.py list [cache_name]")
            return
        
        cache_name = sys.argv[2]
        if cache_name not in ["fetcher_cache", "code_extraction_cache", "llm_cache"]:
            print(f"Invalid cache name: {cache_name}")
            print("Valid cache names: fetcher_cache, code_extraction_cache, llm_cache")
            return
        
        keys = cache_manager.list_cache_keys(cache_name)
        print(f"=== Keys in {cache_name} ===")
        for key in keys:
            print(f"  {key}")
        print(f"Total: {len(keys)} keys")
    
    elif command == "clear":
        cache_name = sys.argv[2] if len(sys.argv) > 2 else None
        if cache_name and cache_name not in ["fetcher_cache", "code_extraction_cache", "llm_cache"]:
            print(f"Invalid cache name: {cache_name}")
            print("Valid cache names: fetcher_cache, code_extraction_cache, llm_cache")
            return
        
        success = cache_manager.clear_cache(cache_name)
        if success:
            if cache_name:
                print(f"Cache {cache_name} cleared successfully")
            else:
                print("All caches cleared successfully")
        else:
            print("Failed to clear cache(s)")
    
    elif command == "get":
        if len(sys.argv) < 4:
            print("Usage: python cache_debug.py get [cache_name] [key]")
            return
        
        cache_name = sys.argv[2]
        key = sys.argv[3]
        
        if cache_name not in ["fetcher_cache", "code_extraction_cache", "llm_cache"]:
            print(f"Invalid cache name: {cache_name}")
            print("Valid cache names: fetcher_cache, code_extraction_cache, llm_cache")
            return
        
        entry = cache_manager.get_cache_entry(cache_name, key)
        if entry is not None:
            print(f"=== Entry from {cache_name} with key '{key}' ===")
            if isinstance(entry, (dict, list)):
                print(json.dumps(entry, indent=2, default=str))
            else:
                print(entry)
        else:
            print(f"No entry found in {cache_name} with key '{key}'")
    
    elif command == "delete":
        if len(sys.argv) < 4:
            print("Usage: python cache_debug.py delete [cache_name] [key]")
            return
        
        cache_name = sys.argv[2]
        key = sys.argv[3]
        
        if cache_name not in ["fetcher_cache", "code_extraction_cache", "llm_cache"]:
            print(f"Invalid cache name: {cache_name}")
            print("Valid cache names: fetcher_cache, code_extraction_cache, llm_cache")
            return
        
        success = cache_manager.delete_cache_entry(cache_name, key)
        if success:
            print(f"Entry with key '{key}' deleted from {cache_name}")
        else:
            print(f"Failed to delete entry with key '{key}' from {cache_name}")
    
    else:
        print(f"Unknown command: {command}")
        print("Usage: python cache_debug.py [command] [args...]")

if __name__ == "__main__":
    main()