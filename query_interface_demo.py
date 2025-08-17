"""
Query Interface Demo
Demonstrates how to use the query_interface package for searching medical codes
"""

# Import required components from the query_interface package
import sys
import os

# Add src to path so we can import query_interface
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from query_interface.query_parser import QueryParser
from query_interface.search_index import SearchIndex
from query_interface.filter_engine import FilterEngine
from query_interface.cache_layer import CacheLayer

def main():
    print("=== QUERY INTERFACE DEMO ===\n")
    
    # 1. Initialize the core components
    query_parser = QueryParser()
    search_index = SearchIndex()
    filter_engine = FilterEngine()
    cache = CacheLayer(max_size=100, default_ttl=3600)

    # 2. Add sample medical codes to the search index
    sample_codes = [
        {
            'code': 'C50.911',
            'type': 'ICD-10-CM',
            'category': 'oncology',
            'description': 'Malignant neoplasm of breast',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        },
        {
            'code': '12345',
            'type': 'CPT',
            'category': 'chemotherapy',
            'description': 'Chemotherapy administration',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        },
        {
            'code': 'ER+',
            'type': 'LOINC',
            'category': 'biomarker',
            'description': 'Estrogen receptor positive',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        },
        {
            'code': 'C34.90',
            'type': 'ICD-10-CM',
            'category': 'oncology',
            'description': 'Malignant neoplasm of lung',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        }
    ]

    # Add codes to the search index
    for code_data in sample_codes:
        search_index.add_code(code_data)

    print(f"Added {len(sample_codes)} medical codes to the search index\n")

    # 3. Demonstrate search functionality
    print("=== SEARCH EXAMPLES ===")

    # Simple search
    query_string = "breast"
    print(f"Searching for: '{query_string}'")
    parsed_query = query_parser.parse(query_string)
    results = search_index.search(parsed_query)
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  - {result.code}: {result.description}")

    print()

    # Field-specific search
    query_string = "type:ICD-10-CM AND category:oncology"
    print(f"Searching for: '{query_string}'")
    parsed_query = query_parser.parse(query_string)
    results = search_index.search(parsed_query)
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  - {result.code}: {result.description}")

    print()

    # 4. Demonstrate filtering functionality
    print("=== FILTER EXAMPLES ===")

    # Create filter criteria
    filter_criteria = {
        'types': ['ICD-10-CM', 'CPT'],
        'categories': ['oncology', 'chemotherapy'],
        'status': ['active']
    }

    # Apply filters
    filtered_results = filter_engine.apply_filters(
        list(search_index._codes.values()),  # All indexed codes
        filter_engine.create_filter_criteria(filter_criteria)
    )

    print(f"Filtered results: Found {len(filtered_results)} results")
    for result in filtered_results:
        print(f"  - {result.code} ({result.type}): {result.description}")

    print()

    # 5. Demonstrate cache usage
    print("=== CACHE EXAMPLE ===")

    # Cache a search result
    cache_key = ("search", "breast", 1, 10)  # (operation, query, page, limit)
    cache.set(*cache_key, value=[r.__dict__ for r in results], ttl=3600)  # Cache for 1 hour

    # Retrieve from cache
    cached_results = cache.get(cache_key)
    print(f"Retrieved {len(cached_results) if cached_results else 0} results from cache")

    # Show cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats['size']}/{stats['max_size']} entries")

    print("\n=== DEMO COMPLETE ===")

if __name__ == "__main__":
    main()