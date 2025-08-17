"""
Query Interface GUI Demo using NiceGUI
Demonstrates how to use the query_interface with a graphical user interface
"""

import sys
import os
from typing import List, Dict, Any

# Add src to path so we can import query_interface
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nicegui import ui, run
from query_interface.query_parser import QueryParser
from query_interface.search_index import SearchIndex
from query_interface.filter_engine import FilterEngine
from query_interface.cache_layer import CacheLayer

# Initialize components
query_parser = QueryParser()
search_index = SearchIndex()
filter_engine = FilterEngine()
cache = CacheLayer(max_size=100, default_ttl=3600)

# Add sample medical codes to the search index
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
    },
    {
        'code': 'Z00.00',
        'type': 'ICD-10-CM',
        'category': 'preventive',
        'description': 'General medical examination',
        'status': 'active',
        'metadata': {'effective_date': '2023-01-01'}
    },
    {
        'code': '99213',
        'type': 'CPT',
        'category': 'evaluation',
        'description': 'Office visit, established patient',
        'status': 'active',
        'metadata': {'effective_date': '2023-01-01'}
    }
]

# Add codes to the search index
for code_data in sample_codes:
    search_index.add_code(code_data)

# Global variables for UI state
search_results = []
filter_results = []
stats = {}

# UI Functions
async def perform_search():
    """Perform search using the query interface"""
    global search_results
    
    query_text = search_input.value
    if not query_text:
        ui.notify('Please enter a search query')
        return
    
    try:
        # Check cache first
        cache_key = f"search:{query_text}"
        cached_result = cache.get((cache_key,))
        
        if cached_result:
            search_results = cached_result
            ui.notify(f'Retrieved {len(search_results)} results from cache')
        else:
            # Parse and execute search
            parsed_query = query_parser.parse(query_text)
            results = search_index.search(parsed_query)
            search_results = [result.__dict__ for result in results]
            
            # Cache the results
            cache.set(cache_key, value=search_results, ttl=3600)
            ui.notify(f'Found {len(search_results)} results')
        
        # Update results display
        update_search_results()
        update_stats()
    except Exception as e:
        ui.notify(f'Error: {str(e)}', type='negative')

def perform_filter():
    """Perform filtering using the query interface"""
    global filter_results
    
    # Get filter criteria from UI
    filter_criteria = {}
    
    if type_filter.value:
        filter_criteria['types'] = type_filter.value.split(',')
    
    if category_filter.value:
        filter_criteria['categories'] = category_filter.value.split(',')
    
    if status_filter.value:
        filter_criteria['status'] = status_filter.value.split(',')
    
    if not filter_criteria:
        ui.notify('Please select at least one filter criterion')
        return
    
    try:
        # Apply filters
        all_codes = list(search_index._codes.values())
        filtered_results = filter_engine.apply_filters(
            all_codes, 
            filter_engine.create_filter_criteria(filter_criteria)
        )
        filter_results = [result.__dict__ for result in filtered_results]
        
        ui.notify(f'Filtered to {len(filter_results)} results')
        
        # Update results display
        update_filter_results()
        update_stats()
    except Exception as e:
        ui.notify(f'Error: {str(e)}', type='negative')

def update_search_results():
    """Update the search results display"""
    search_results_container.clear()
    
    with search_results_container:
        if search_results:
            ui.label(f'Search Results ({len(search_results)} found)').classes('text-xl font-bold mb-4')
            
            for result in search_results:
                with ui.card().classes('w-full mb-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(result['code']).classes('text-lg font-bold')
                        ui.label(result['type']).classes('text-sm text-gray-500')
                    
                    ui.label(result['description']).classes('mt-1')
                    
                    with ui.row().classes('w-full justify-between mt-2'):
                        ui.label(f"Category: {result['category']}").classes('text-sm')
                        ui.label(f"Status: {result['status']}").classes('text-sm')
        else:
            ui.label('No search results').classes('text-gray-500')

def update_filter_results():
    """Update the filter results display"""
    filter_results_container.clear()
    
    with filter_results_container:
        if filter_results:
            ui.label(f'Filtered Results ({len(filter_results)} found)').classes('text-xl font-bold mb-4')
            
            for result in filter_results:
                with ui.card().classes('w-full mb-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(result['code']).classes('text-lg font-bold')
                        ui.label(result['type']).classes('text-sm text-gray-500')
                    
                    ui.label(result['description']).classes('mt-1')
                    
                    with ui.row().classes('w-full justify-between mt-2'):
                        ui.label(f"Category: {result['category']}").classes('text-sm')
                        ui.label(f"Status: {result['status']}").classes('text-sm')
        else:
            ui.label('No filtered results').classes('text-gray-500')

def update_stats():
    """Update the statistics display"""
    global stats
    
    stats = {
        'total_codes': len(search_index._codes),
        'cache_size': len(cache._cache),
        'search_results': len(search_results),
        'filter_results': len(filter_results)
    }
    
    stats_container.clear()
    
    with stats_container:
        ui.label('System Statistics').classes('text-lg font-bold mb-2')
        with ui.grid(columns=2).classes('w-full gap-2'):
            ui.label('Total Codes:').classes('font-medium')
            ui.label(str(stats['total_codes']))
            
            ui.label('Cache Size:').classes('font-medium')
            ui.label(str(stats['cache_size']))
            
            ui.label('Search Results:').classes('font-medium')
            ui.label(str(stats['search_results']))
            
            ui.label('Filter Results:').classes('font-medium')
            ui.label(str(stats['filter_results']))

# Create the UI
with ui.column().classes('p-8 max-w-4xl mx-auto'):
    ui.label('Medical Code Query Interface Demo').classes('text-3xl font-bold text-center mb-8')
    
    # Search Section
    with ui.card().classes('w-full mb-6'):
        ui.label('Search Medical Codes').classes('text-2xl font-bold mb-4')
        
        with ui.column().classes('w-full gap-4'):
            # Search input
            search_input = ui.input(label='Search Query', placeholder='e.g., "breast", "type:ICD-10-CM"') \
                .classes('w-full')
            
            with ui.row().classes('w-full justify-center'):
                ui.button('Search', on_click=perform_search) \
                    .classes('text-lg px-6 py-2')
        
        # Search results container
        search_results_container = ui.column().classes('w-full mt-4')
    
    # Filter Section
    with ui.card().classes('w-full mb-6'):
        ui.label('Filter Medical Codes').classes('text-2xl font-bold mb-4')
        
        with ui.column().classes('w-full gap-4'):
            # Filter inputs
            type_filter = ui.input(label='Types (comma-separated)', placeholder='e.g., ICD-10-CM,CPT') \
                .classes('w-full')
            
            category_filter = ui.input(label='Categories (comma-separated)', placeholder='e.g., oncology,chemotherapy') \
                .classes('w-full')
            
            status_filter = ui.input(label='Status (comma-separated)', placeholder='e.g., active,deprecated') \
                .classes('w-full')
            
            with ui.row().classes('w-full justify-center'):
                ui.button('Filter', on_click=perform_filter) \
                    .classes('text-lg px-6 py-2')
        
        # Filter results container
        filter_results_container = ui.column().classes('w-full mt-4')
    
    # Statistics Section
    with ui.card().classes('w-full'):
        ui.label('Statistics').classes('text-2xl font-bold mb-4')
        stats_container = ui.column().classes('w-full')
    
    # Initialize displays
    update_search_results()
    update_filter_results()
    update_stats()

# Add some example queries for users to try
with ui.expansion('Example Queries', icon='help').classes('w-full max-w-4xl mx-auto mb-6'):
    with ui.column().classes('p-4'):
        ui.label('Try these example queries:').classes('font-bold mb-2')
        ui.label('- "breast" (simple text search)').classes('ml-4')
        ui.label('- "type:ICD-10-CM" (field-specific search)').classes('ml-4')
        ui.label('- "oncology" (search by category)').classes('ml-4')
        ui.label('- Use the filter section to narrow down results by type, category, or status').classes('ml-4 mt-2')

ui.run(title='Query Interface Demo', port=8082)