"""
Query Interface API Demo
Demonstrates how to use the query_interface API endpoints
"""

import sys
import os
from flask import Flask, jsonify
import json

# Add src to path so we can import query_interface
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from query_interface.query_parser import QueryParser
from query_interface.search_index import SearchIndex
from query_interface.filter_engine import FilterEngine
from query_interface.cache_layer import CacheLayer

# Monkey patch the missing methods
def total_codes(self):
    return len(self._codes)

def size(self):
    return len(self._cache)

# Add the missing methods to the classes
SearchIndex.total_codes = total_codes
CacheLayer.size = size

def create_sample_app():
    """Create a Flask app with the query_interface API"""
    app = Flask(__name__)
    
    # Initialize components
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
        }
    ]
    
    # Add codes to the search index
    for code_data in sample_codes:
        search_index.add_code(code_data)
    
    # Create simple API routes manually
    @app.route('/api/v1/codes/search')
    def search():
        from flask import request
        query = request.args.get('q', '')
        if not query:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query_parser = QueryParser()
        try:
            parsed_query = query_parser.parse(query)
            results = search_index.search(parsed_query)
            return jsonify({
                'results': [result.__dict__ for result in results],
                'metadata': {
                    'total': len(results)
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/v1/codes/filter', methods=['POST'])
    def filter_codes():
        from flask import request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        try:
            filter_criteria = request.get_json()
            if not filter_criteria or 'filters' not in filter_criteria:
                return jsonify({'error': 'Filter criteria required'}), 400
            
            # Apply filters
            all_codes = list(search_index._codes.values())
            filtered_results = filter_engine.apply_filters(
                all_codes, 
                filter_engine.create_filter_criteria(filter_criteria['filters'])
            )
            
            return jsonify({
                'results': [result.__dict__ for result in filtered_results],
                'metadata': {
                    'total': len(filtered_results)
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/v1/codes/stats')
    def get_stats():
        try:
            stats = {
                'cache': {
                    'size': cache.size(),
                },
                'index': {
                    'total_codes': search_index.total_codes(),
                }
            }
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def main():
    print("=== QUERY INTERFACE API DEMO ===\n")
    
    # Create the Flask app
    app = create_sample_app()
    
    # Simulate API calls
    with app.test_client() as client:
        print("1. Testing search endpoint:")
        response = client.get('/api/v1/codes/search?q=breast')
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"   Search for 'breast': {data['metadata']['total']} results found")
            for result in data['results']:
                print(f"     - {result['code']}: {result['description']}")
        else:
            print(f"   Error: {response.status_code} - {response.data}")
        
        print("\n2. Testing filter endpoint:")
        filter_data = {
            "filters": {
                "types": ["ICD-10-CM"],
                "categories": ["oncology"]
            }
        }
        response = client.post('/api/v1/codes/filter', 
                              json=filter_data,
                              content_type='application/json')
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"   Filter by ICD-10-CM oncology codes: {data['metadata']['total']} results found")
            for result in data['results']:
                print(f"     - {result['code']}: {result['description']}")
        else:
            print(f"   Error: {response.status_code} - {response.data}")
        
        print("\n3. Testing stats endpoint:")
        response = client.get('/api/v1/codes/stats')
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"   Cache stats: {data['cache']['size']} entries")
            print(f"   Index stats: {data['index']['total_codes']} total codes")
        else:
            print(f"   Error: {response.status_code} - {response.data}")
        
        print("\n=== API DEMO COMPLETE ===")

if __name__ == "__main__":
    main()