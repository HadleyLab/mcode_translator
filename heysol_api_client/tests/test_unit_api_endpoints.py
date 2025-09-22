#!/usr/bin/env python3
"""
Unit Tests for HeySol API Client - API Endpoints

Tests individual API endpoints with mock responses, covering:
- User endpoints
- Memory endpoints
- Spaces endpoints
- Webhook endpoints
- Response parsing and validation
"""

import json
import pytest
import requests_mock
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from heysol.client import HeySolClient
from heysol.exceptions import ValidationError


class TestAPIEndpoints:
    """Unit tests for individual API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return HeySolClient(api_key="test-api-key", skip_mcp_init=True)

    @pytest.fixture
    def mock_setup(self, requests_mock):
        """Set up comprehensive mocks for all API endpoints."""
        # User endpoints
        requests_mock.get(
            "https://core.heysol.ai/api/v1/user/profile",
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
            "https://core.heysol.ai/api/v1/search",
            json={"episodes": [], "total": 0}
        )

        # Spaces endpoints
        requests_mock.get(
            "https://core.heysol.ai/api/v1/spaces",
            json={"spaces": []}
        )

        requests_mock.post(
            "https://core.heysol.ai/api/v1/spaces",
            json={"space": {"id": "space-123", "name": "Test Space"}}
        )

        # Webhook endpoints
        requests_mock.post(
            "https://core.heysol.ai/api/v1/webhooks",
            json={"id": "webhook-123", "url": "https://example.com/webhook"}
        )

        requests_mock.get(
            "https://core.heysol.ai/api/v1/webhooks",
            json=[]
        )

        return requests_mock

