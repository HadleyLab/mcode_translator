"""
Unit tests for error handling middleware.
Tests error handling, custom exceptions, and error responses.
"""
import pytest
from flask import Flask, jsonify
from ..error_handling import (
    QueryInterfaceError,
    InvalidQueryError,
    InvalidFilterError,
    ResourceNotFoundError,
    ValidationError,
    RateLimitError,
    setup_error_handlers,
    validate_request,
    create_error_response
)


@pytest.fixture
def app():
    """Create Flask application for testing"""
    app = Flask(__name__)
    setup_error_handlers(app)
    
    # Add test routes that raise different errors
    @app.route('/test/query-error')
    def test_query_error():
        raise InvalidQueryError('Invalid query syntax')

    @app.route('/test/filter-error')
    def test_filter_error():
        raise InvalidFilterError('Invalid filter criteria')

    @app.route('/test/not-found')
    def test_not_found():
        raise ResourceNotFoundError('Resource not found')

    @app.route('/test/validation')
    def test_validation():
        raise ValidationError('Validation failed')

    @app.route('/test/rate-limit')
    def test_rate_limit():
        raise RateLimitError('Rate limit exceeded')

    @app.route('/test/generic-error')
    def test_generic_error():
        raise Exception('Unexpected error')

    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


class TestErrorHandling:
    """Test cases for error handling middleware"""

    def test_query_interface_error_base(self):
        """Test base QueryInterfaceError"""
        error = QueryInterfaceError('Test error')
        error_dict = error.to_dict()
        
        assert error_dict['error'] == 'INTERNAL_ERROR'
        assert error_dict['message'] == 'Test error'
        assert error_dict['status_code'] == 500

    def test_invalid_query_error(self, client):
        """Test InvalidQueryError handling"""
        response = client.get('/test/query-error')
        data = response.get_json()
        
        assert response.status_code == 400
        assert data['error'] == 'INVALID_QUERY'
        assert 'message' in data

    def test_invalid_filter_error(self, client):
        """Test InvalidFilterError handling"""
        response = client.get('/test/filter-error')
        data = response.get_json()
        
        assert response.status_code == 400
        assert data['error'] == 'INVALID_FILTER'
        assert 'message' in data

    def test_resource_not_found(self, client):
        """Test ResourceNotFoundError handling"""
        response = client.get('/test/not-found')
        data = response.get_json()
        
        assert response.status_code == 404
        assert data['error'] == 'NOT_FOUND'
        assert 'message' in data

    def test_validation_error(self, client):
        """Test ValidationError handling"""
        response = client.get('/test/validation')
        data = response.get_json()
        
        assert response.status_code == 400
        assert data['error'] == 'VALIDATION_ERROR'
        assert 'message' in data

    def test_rate_limit_error(self, client):
        """Test RateLimitError handling"""
        response = client.get('/test/rate-limit')
        data = response.get_json()
        
        assert response.status_code == 429
        assert data['error'] == 'RATE_LIMIT_EXCEEDED'
        assert 'message' in data

    def test_generic_error_handling(self, client):
        """Test handling of generic exceptions"""
        response = client.get('/test/generic-error')
        data = response.get_json()
        
        assert response.status_code == 500
        assert data['error'] == 'INTERNAL_ERROR'
        assert 'unexpected error' in data['message'].lower()

    def test_404_not_found(self, client):
        """Test handling of 404 routes"""
        response = client.get('/nonexistent-route')
        data = response.get_json()
        
        assert response.status_code == 404
        assert 'error' in data
        assert 'message' in data

    def test_request_validation(self):
        """Test request validation function"""
        valid_data = {
            'name': 'test',
            'age': 25
        }
        required_fields = {
            'name': str,
            'age': int
        }
        
        # Should not raise error
        validate_request(valid_data, required_fields)
        
        # Test missing field
        invalid_data = {'name': 'test'}
        with pytest.raises(ValidationError) as exc:
            validate_request(invalid_data, required_fields)
        assert 'Missing required field' in str(exc.value)
        
        # Test invalid type
        invalid_type_data = {
            'name': 'test',
            'age': '25'  # string instead of int
        }
        with pytest.raises(ValidationError) as exc:
            validate_request(invalid_type_data, required_fields)
        assert 'Invalid type' in str(exc.value)

    def test_create_error_response(self):
        """Test error response creation"""
        response = create_error_response(
            message='Test error',
            status_code=400,
            error_code='TEST_ERROR'
        )
        data = response.get_json()
        
        assert response.status_code == 400
        assert data['error'] == 'TEST_ERROR'
        assert data['message'] == 'Test error'
        assert data['status_code'] == 400

    def test_custom_status_code(self):
        """Test custom status code in error"""
        error = QueryInterfaceError('Test error', status_code=418)
        error_dict = error.to_dict()
        
        assert error_dict['status_code'] == 418

    def test_error_logging(self, app, caplog):
        """Test error logging configuration"""
        with app.test_client() as client:
            client.get('/test/generic-error')
            
        # Check that error was logged
        assert len(caplog.records) > 0
        assert any('Unexpected error' in record.message 
                  for record in caplog.records)

    def test_error_response_structure(self, client):
        """Test consistent error response structure"""
        endpoints = [
            '/test/query-error',
            '/test/filter-error',
            '/test/not-found',
            '/test/validation',
            '/test/rate-limit'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            data = response.get_json()
            
            # Check that all error responses have consistent structure
            assert 'error' in data
            assert 'message' in data
            assert 'status_code' in data
            assert isinstance(data['error'], str)
            assert isinstance(data['message'], str)
            assert isinstance(data['status_code'], int)