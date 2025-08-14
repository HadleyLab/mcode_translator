"""
Pytest configuration and shared fixtures.
"""
import pytest
import logging
import json
from flask import Flask
from ..api import create_api_routes
from ..search_index import SearchIndex
from ..cache_layer import CacheLayer
from ..filter_engine import FilterEngine


@pytest.fixture
def app():
    """Create Flask application for testing"""
    app = Flask(__name__)
    app.config['TESTING'] = True
    
    with app.app_context():
        create_api_routes(app)
        return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def search_index():
    """Create SearchIndex instance with sample data"""
    index = SearchIndex()
    sample_data = [
        {
            'code': 'J45.50',
            'type': 'ICD-10',
            'category': 'respiratory',
            'description': 'Severe persistent asthma',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        },
        {
            'code': 'E11.9',
            'type': 'ICD-10',
            'category': 'endocrine',
            'description': 'Type 2 diabetes without complications',
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
    for code_data in sample_data:
        index.add_code(code_data)
    return index


@pytest.fixture
def filter_engine():
    """Create FilterEngine instance"""
    return FilterEngine()


@pytest.fixture
def cache():
    """Create CacheLayer instance"""
    return CacheLayer(max_size=3, default_ttl=1)


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Configure logging for each test"""
    caplog.set_level(logging.INFO)


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        {
            'code': 'J45.50',
            'type': 'ICD-10',
            'category': 'respiratory',
            'description': 'Severe persistent asthma',
            'status': 'active',
            'metadata': {'effective_date': '2023-01-01'}
        }
    ]


@pytest.fixture
def sample_filter_criteria():
    """Sample filter criteria for testing"""
    return {
        'types': ['ICD-10'],
        'categories': ['respiratory'],
        'status': ['active'],
        'date_range': {
            'start': '2023-01-01',
            'end': '2023-12-31'
        }
    }


@pytest.fixture
def app_context(app):
    """Provide Flask application context"""
    with app.app_context():
        yield app


@pytest.fixture
def mock_search_index(monkeypatch, search_index):
    """Mock the search index in the API module"""
    from .. import api
    monkeypatch.setattr(api, 'search_index', search_index)
    return search_index


@pytest.fixture
def mock_cache(monkeypatch, cache):
    """Mock the cache in the API module"""
    from .. import api
    monkeypatch.setattr(api, 'cache_layer', cache)
    return cache


@pytest.fixture
def mock_filter_engine(monkeypatch, filter_engine):
    """Mock the filter engine in the API module"""
    from .. import api
    monkeypatch.setattr(api, 'filter_engine', filter_engine)
    return filter_engine