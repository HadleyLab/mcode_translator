"""
Unit tests for API endpoints.
Tests API functionality, request handling, and error responses.
"""
import pytest
import json
from flask import Flask
from werkzeug.exceptions import BadRequest

from ..api import create_api_routes
from ..search_index import SearchIndex
from ..filter_engine import FilterEngine


class TestAPI:
    """Test cases for API endpoints"""

    def test_search_endpoint(self, client, app_context, mock_search_index, mock_cache):
        """Test basic search functionality"""
        response = client.get('/api/v1/codes/search?q=asthma')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'results' in data
        assert 'metadata' in data
        assert len(data['results']) == 1
        assert data['results'][0]['code'] == 'J45.50'

    def test_search_with_invalid_query(self, client, app_context):
        """Test search with invalid query syntax"""
        response = client.get('/api/v1/codes/search?q=:invalid:query')
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data
        assert 'message' in data

    def test_search_pagination(self, client, app_context, mock_search_index):
        """Test search result pagination"""
        # Search with page size of 1
        response = client.get('/api/v1/codes/search?q=type:ICD-10&page=1&limit=1')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert len(data['results']) == 1
        assert data['metadata']['total'] == 2
        assert data['metadata']['pages'] == 2

    def test_filter_endpoint(self, client, app_context, mock_search_index, mock_filter_engine, sample_filter_criteria):
        """Test filter functionality"""
        response = client.post(
            '/api/v1/codes/filter',
            json={'filters': sample_filter_criteria},
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert len(data['results']) == 1
        assert data['results'][0]['type'] == 'ICD-10'
        assert data['results'][0]['category'] == 'respiratory'

    def test_filter_with_invalid_criteria(self, client, app_context):
        """Test filter with invalid criteria"""
        invalid_filter = {
            'filters': {
                'date_range': {
                    'start': 'invalid-date'
                }
            }
        }
        
        response = client.post(
            '/api/v1/codes/filter',
            json=invalid_filter,
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data
        assert 'message' in data

    def test_missing_query(self, client, app_context):
        """Test search without query parameter"""
        response = client.get('/api/v1/codes/search')
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert data['error'] == 'MISSING_QUERY'

    def test_stats_endpoint(self, client, app_context, mock_search_index, mock_cache):
        """Test stats endpoint"""
        response = client.get('/api/v1/codes/stats')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'cache' in data
        assert 'index' in data
        assert data['index']['total_codes'] > 0

    def test_invalid_page_number(self, client, app_context):
        """Test search with invalid page number"""
        response = client.get('/api/v1/codes/search?q=test&page=invalid')
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data

    def test_large_page_size(self, client, app_context, mock_search_index):
        """Test search with page size exceeding limit"""
        response = client.get('/api/v1/codes/search?q=test&limit=1000')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert data['metadata']['limit'] == 100  # Should be capped at 100

    def test_filter_without_body(self, client, app_context):
        """Test filter endpoint without request body"""
        response = client.post('/api/v1/codes/filter')
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert data['error'] == 'MISSING_FILTERS'

    def test_cache_usage(self, client, app_context, mock_search_index, mock_cache):
        """Test that repeated queries use cache"""
        # Make first request
        query = '/api/v1/codes/search?q=asthma'
        response1 = client.get(query)
        
        # Make second request with same query
        response2 = client.get(query)
        
        # Both should succeed and return same data
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.data == response2.data

    def test_error_handling_middleware(self, client, app_context):
        """Test global error handling"""
        # Trigger an internal error by passing invalid JSON
        response = client.post(
            '/api/v1/codes/filter',
            data='invalid json',
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data
        assert 'message' in data

    def test_empty_result_set(self, client, app_context, mock_search_index):
        """Test handling of empty result sets"""
        response = client.get('/api/v1/codes/search?q=nonexistent')
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert len(data['results']) == 0
        assert data['metadata']['total'] == 0

    def test_malformed_json(self, client, app_context):
        """Test handling of malformed JSON in request body"""
        response = client.post(
            '/api/v1/codes/filter',
            data='{invalid:json}',
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        assert response.status_code == 400
        assert 'error' in data