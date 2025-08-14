"""
Error handling module for the query interface.
Defines custom exceptions and error handling middleware.
"""
from typing import Dict, Any
from werkzeug.exceptions import HTTPException

class APIError(HTTPException):
    """
    Custom API exception for handling application-specific errors
    
    Attributes:
        code (str): Error code identifier
        message (str): Human readable error message
        status_code (int): HTTP status code
        payload (dict): Additional error context
    """
    def __init__(self, code: str, message: str, status_code: int = 400, payload: Dict[str, Any] = None):
        super().__init__()
        self.code = code
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for JSON response"""
        error = {
            'error': self.code,
            'message': self.message
        }
        if self.payload:
            error['details'] = self.payload
        return error

def register_error_handlers(app):
    """
    Register error handlers for the Flask application
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """Handle custom API errors"""
        response = error.to_dict()
        return response, error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        """Handle standard HTTP errors"""
        response = {
            'error': error.__class__.__name__,
            'message': str(error)
        }
        return response, error.code

    @app.errorhandler(Exception)
    def handle_generic_error(error):
        """Handle unexpected errors"""
        response = {
            'error': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred'
        }
        return response, 500

class QueryParserError(APIError):
    """Exception for query parsing errors"""
    def __init__(self, message: str, payload: Dict[str, Any] = None):
        super().__init__('QUERY_PARSER_ERROR', message, 400, payload)

class SearchError(APIError):
    """Exception for search operation errors"""
    def __init__(self, message: str, payload: Dict[str, Any] = None):
        super().__init__('SEARCH_ERROR', message, 400, payload)

class FilterError(APIError):
    """Exception for filter operation errors"""
    def __init__(self, message: str, payload: Dict[str, Any] = None):
        super().__init__('FILTER_ERROR', message, 400, payload)

class CacheError(APIError):
    """Exception for cache operation errors"""
    def __init__(self, message: str, payload: Dict[str, Any] = None):
        super().__init__('CACHE_ERROR', message, 500, payload)

class ValidationError(APIError):
    """Exception for input validation errors"""
    def __init__(self, message: str, payload: Dict[str, Any] = None):
        super().__init__('VALIDATION_ERROR', message, 400, payload)