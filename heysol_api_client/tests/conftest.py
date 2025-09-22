#!/usr/bin/env python3
"""
Test Configuration for HeySol API Client

This file contains pytest configuration and fixtures for the test suite.
"""

import os
import pytest
import requests_mock
from typing import Dict, Any, Generator
from pathlib import Path
from dotenv import load_dotenv

from heysol.client import HeySolClient
from heysol.exceptions import ValidationError
from heysol.config import HeySolConfig

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


@pytest.fixture(scope="session")
def test_config() -> HeySolConfig:
    """Get test configuration from environment or defaults."""
    return HeySolConfig.from_env()


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get API key from environment variables."""
    api_key = os.getenv("HEYSOL_API_KEY") or os.getenv("COREAI_API_KEY")
    if not api_key:
        pytest.skip("API key not configured - set HEYSOL_API_KEY or COREAI_API_KEY")
    return api_key


@pytest.fixture(scope="session")
def oauth2_credentials() -> Dict[str, str]:
    """Get OAuth2 credentials from environment variables."""
    client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
    client_secret = os.getenv("HEYSOL_OAUTH2_CLIENT_SECRET")
    redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

    if not client_id or not client_secret:
        pytest.skip("OAuth2 credentials not configured")

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri
    }


@pytest.fixture
def client(api_key: str) -> HeySolClient:
    """Create a test client instance with API key."""
    return HeySolClient(api_key=api_key, skip_mcp_init=True)


@pytest.fixture
def oauth2_client(oauth2_credentials: Dict[str, str]) -> HeySolClient:
    """Create a test client instance with OAuth2 authentication."""
    from heysol.oauth2 import InteractiveOAuth2Authenticator

    auth = InteractiveOAuth2Authenticator(
        client_id=oauth2_credentials["client_id"],
        client_secret=oauth2_credentials["client_secret"],
        redirect_uri=oauth2_credentials["redirect_uri"]
    )
    return HeySolClient(oauth2_auth=auth, skip_mcp_init=True)


@pytest.fixture
def mock_api_setup(requests_mock: requests_mock.Mocker, test_config: HeySolConfig) -> None:
    """Set up comprehensive mocks for all API endpoints using centralized config."""
    base_url = test_config.base_url

    # User endpoints
    requests_mock.get(
        f"{base_url}/user/profile",
        json={
            "id": "user-123",
            "name": "Test User",
            "email": "test@example.com",
            "role": "admin",
            "created_at": "2024-01-01T00:00:00Z"
        }
    )

    # Memory endpoints
    requests_mock.post(
        f"{base_url}/memory/knowledge-graph/search",
        json={"nodes": [], "edges": []}
    )
    requests_mock.post(
        f"{base_url}/memory/ingestion/queue",
        json={"queue_id": "queue-123"}
    )
    requests_mock.get(
        f"{base_url}/episodes/episode-123/facts",
        json=[]
    )
    requests_mock.get(
        f"{base_url}/memory/logs",
        json=[]
    )
    requests_mock.get(
        f"{base_url}/memory/logs/log-123",
        json={"id": "log-123"}
    )

    # Spaces endpoints
    requests_mock.get(
        f"{base_url}/spaces",
        json=[]
    )
    requests_mock.post(
        f"{base_url}/spaces",
        json={"id": "space-123", "name": "Test Space"}
    )
    requests_mock.get(
        f"{base_url}/spaces/space-123",
        json={"id": "space-123", "name": "Test Space"}
    )
    requests_mock.put(
        f"{base_url}/spaces/space-123",
        json={"id": "space-123", "name": "Updated Space"}
    )
    requests_mock.delete(
        f"{base_url}/spaces/space-123",
        json={"deleted": True}
    )

    # OAuth2 endpoints
    requests_mock.get(
        f"{base_url}/oauth2/authorize",
        json={"authorization_url": "https://example.com/auth"}
    )
    requests_mock.post(
        f"{base_url}/oauth2/token",
        json={"access_token": "token-123"}
    )
    requests_mock.get(
        f"{base_url}/oauth2/userinfo",
        json={"sub": "user-123"}
    )

    # Webhook endpoints
    requests_mock.post(
        f"{base_url}/webhooks",
        json={"webhook_id": "webhook-123"}
    )
    requests_mock.get(
        f"{base_url}/webhooks",
        json=[]
    )
    requests_mock.get(
        f"{base_url}/webhooks/webhook-123",
        json={"id": "webhook-123"}
    )
    requests_mock.put(
        f"{base_url}/webhooks/webhook-123",
        json={"id": "webhook-123"}
    )
    requests_mock.delete(
        f"{base_url}/webhooks/webhook-123",
        json={"deleted": True}
    )


@pytest.fixture
def mock_error_responses(requests_mock: requests_mock.Mocker, test_config: HeySolConfig) -> None:
    """Set up mock error responses for testing error handling."""
    base_url = test_config.base_url

    # 400 Bad Request
    requests_mock.get(
        f"{base_url}/user/profile",
        status_code=400,
        json={"error": "Invalid request parameters", "details": "Missing required field: 'id'"}
    )

    # 401 Unauthorized
    requests_mock.get(
        f"{base_url}/user/profile",
        status_code=401,
        json={"error": "Invalid API key or authentication failed"}
    )

    # 404 Not Found
    requests_mock.get(
        f"{base_url}/user/profile",
        status_code=404,
        json={"error": "Requested resource not found"}
    )

    # 429 Rate Limit
    requests_mock.get(
        f"{base_url}/user/profile",
        status_code=429,
        json={"error": "Rate limit exceeded"},
        headers={"Retry-After": "60"}
    )

    # 500 Server Error
    requests_mock.get(
        f"{base_url}/user/profile",
        status_code=500,
        json={"error": "Internal Server Error"}
    )


@pytest.fixture
def mock_unicode_responses(requests_mock: requests_mock.Mocker, test_config: HeySolConfig) -> None:
    """Set up mock responses with Unicode characters."""
    base_url = test_config.base_url

    requests_mock.get(
        f"{base_url}/user/profile",
        json={
            "id": "user-123",
            "name": "测试用户",  # Chinese characters
            "description": "User with émojis 🚀 and spëcial characters",
            "tags": ["标签1", "标签2"]  # Chinese tags
        }
    )


@pytest.fixture
def mock_large_responses(requests_mock: requests_mock.Mocker, test_config: HeySolConfig) -> None:
    """Set up mock responses with large payloads."""
    base_url = test_config.base_url

    large_response = {
        "data": [{"id": i, "value": f"test_value_{i}"} for i in range(1000)],
        "metadata": {"total_count": 1000, "page": 1}
    }

    requests_mock.get(
        f"{base_url}/memory/logs",
        json=large_response
    )


# Test markers for better test organization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests with live API")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "error: Error handling tests")
    config.addinivalue_line("markers", "edge: Edge case and boundary condition tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security-related tests")


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-live",
        action="store_true",
        default=False,
        help="Skip live API integration tests"
    )
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow running tests"
    )
    parser.addoption(
        "--only-unit",
        action="store_true",
        default=False,
        help="Run only unit tests"
    )


# Skip conditions based on command line options
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    skip_live = config.getoption("--skip-live")
    skip_slow = config.getoption("--skip-slow")
    only_unit = config.getoption("--only-unit")

    for item in items:
        # Skip live API tests if requested
        if skip_live and "integration" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Live API tests skipped"))

        # Skip slow tests if requested
        if skip_slow and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Slow tests skipped"))

        # Skip non-unit tests if only unit tests requested
        if only_unit and "unit" not in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Only unit tests requested"))


# Coverage configuration
@pytest.fixture(scope="session", autouse=True)
def configure_coverage():
    """Configure coverage settings for better reporting."""
    import os
    os.environ.setdefault("COVERAGE_PROCESS_START", "pyproject.toml")
    os.environ.setdefault("COVERAGE_FILE", ".coverage")


# Test data fixtures
@pytest.fixture
def sample_user_data() -> Dict[str, Any]:
    """Sample user data for testing."""
    return {
        "id": "user-123",
        "name": "Test User",
        "email": "test@example.com",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_space_data() -> Dict[str, Any]:
    """Sample space data for testing."""
    return {
        "id": "space-123",
        "name": "Test Space",
        "description": "A test space for research",
        "created_at": "2024-01-01T00:00:00Z",
        "stats": {
            "total_episodes": 100,
            "total_facts": 500,
            "storage_used_mb": 25.5
        }
    }


@pytest.fixture
def sample_memory_data() -> Dict[str, Any]:
    """Sample memory data for testing."""
    return {
        "content": "Clinical trial shows promising results for new treatment",
        "type": "research",
        "priority": "high",
        "tags": ["clinical-trial", "research", "medical"]
    }


@pytest.fixture
def sample_webhook_data() -> Dict[str, Any]:
    """Sample webhook data for testing."""
    return {
        "url": "https://example.com/webhook",
        "events": ["memory.ingest", "memory.search", "space.create"],
        "active": True,
        "secret": "webhook-secret-123"
    }