"""
API module implementing REST endpoints for the query interface.
Provides search, filter and stats endpoints with proper error handling.
"""
from typing import Dict, Any
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest

from .query_parser import QueryParser
from .search_index import SearchIndex
from .filter_engine import FilterEngine
from .cache_layer import CacheLayer
from .error_handling import APIError

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

def create_api_routes(search_index: SearchIndex, filter_engine: FilterEngine, cache: CacheLayer) -> Blueprint:
    """
    Create and configure API routes
    
    Args:
        search_index: SearchIndex instance for code lookups
        filter_engine: FilterEngine instance for applying filters
        cache: CacheLayer instance for caching results
    
    Returns:
        Blueprint: Flask blueprint with configured routes
    """
    api = Blueprint('api', __name__)
    query_parser = QueryParser()

    @api.route('/api/v1/codes/search')
    def search():
        """
        Search endpoint handling code lookups with pagination
        """
        # Get and validate query parameter
        query = request.args.get('q')
        if not query:
            raise APIError('MISSING_QUERY', 'Search query parameter is required')

        # Parse pagination parameters
        try:
            page = int(request.args.get('page', 1))
            limit = min(int(request.args.get('limit', DEFAULT_PAGE_SIZE)), MAX_PAGE_SIZE)
            if page < 1:
                raise ValueError("Page must be >= 1")
        except ValueError as e:
            raise APIError('INVALID_PAGINATION', str(e))

        # Check cache first
        cache_key = f"search:{query}:{page}:{limit}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return jsonify(cached_result)

        # Parse query and execute search
        try:
            parsed_query = query_parser.parse(query)
            results = search_index.search(parsed_query)
            
            # Calculate pagination
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_results = results[start_idx:end_idx]
            
            total_results = len(results)
            total_pages = (total_results + limit - 1) // limit

            response = {
                'results': paginated_results,
                'metadata': {
                    'total': total_results,
                    'page': page,
                    'limit': limit,
                    'pages': total_pages
                }
            }

            # Cache the response
            cache.set(cache_key, response)
            
            return jsonify(response)
            
        except Exception as e:
            raise APIError('SEARCH_ERROR', str(e))

    @api.route('/api/v1/codes/filter', methods=['POST'])
    def filter_codes():
        """
        Filter endpoint applying complex filtering criteria
        """
        if not request.is_json:
            raise APIError('INVALID_REQUEST', 'Request must be JSON')

        try:
            filter_criteria = request.get_json()
        except Exception:
            raise APIError('INVALID_JSON', 'Invalid JSON in request body')

        if not filter_criteria or 'filters' not in filter_criteria:
            raise APIError('MISSING_FILTERS', 'Filter criteria required')

        try:
            # Apply filters and return results
            results = filter_engine.apply_filters(filter_criteria['filters'])
            return jsonify({
                'results': results,
                'metadata': {
                    'total': len(results)
                }
            })
        except Exception as e:
            raise APIError('FILTER_ERROR', str(e))

    @api.route('/api/v1/codes/stats')
    def get_stats():
        """
        Stats endpoint providing system statistics
        """
        try:
            stats = {
                'cache': {
                    'size': cache.size(),
                    'hits': cache.stats['hits'],
                    'misses': cache.stats['misses']
                },
                'index': {
                    'total_codes': search_index.total_codes(),
                    'last_updated': search_index.last_updated.isoformat() 
                        if search_index.last_updated else None
                }
            }
            return jsonify(stats)
        except Exception as e:
            raise APIError('STATS_ERROR', str(e))

    return api