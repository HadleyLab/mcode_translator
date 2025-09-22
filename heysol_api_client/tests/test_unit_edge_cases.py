#!/usr/bin/env python3
"""
Unit Tests for HeySol API Client - Edge Cases and Boundary Conditions

Tests edge cases, boundary conditions, and special scenarios including:
- Query parameter edge cases
- Response parsing edge cases
- Authentication edge cases
- Performance edge cases
- Data validation edge cases
- Network condition edge cases
"""

import json
import time
import pytest
import requests
import requests_mock
from unittest.mock import Mock, patch
from typing import Dict, Any

from heysol.client import HeySolClient
from heysol.exceptions import ValidationError


class TestEdgeCases:
    """Unit tests for edge cases and boundary conditions."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return HeySolClient(api_key="test-api-key", skip_mcp_init=True)

    # Query Parameter Edge Cases
    def test_query_parameters_with_special_characters(self, client):
        """Test query parameters with special characters."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/memory/search",
                json={"episodes": [], "facts": []},
                status_code=200
            )

            # Test with special characters in query
            test_queries = [
                "test with spaces",
                "test@with$special#chars",
                "test%20with%20encoded%20spaces",
                "test'with\"quotes",
                "test\nwith\nnewlines",
                "test\twith\ttabs",
                "test🚀with🧪emojis"
            ]

            for query in test_queries:
                result = client.search(query, limit=10)
                assert result == {"episodes": [], "facts": []}

                # Verify the request was made with correct query parameters
                request = m.request_history[-1]
                assert "query" in request.qs
                assert request.qs["query"] == [query]

    def test_query_parameters_with_unicode(self, client):
        """Test query parameters with Unicode characters."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/memory/search",
                json={"episodes": [], "facts": []},
                status_code=200
            )

            # Test with various Unicode characters
            unicode_queries = [
                "测试查询",  # Chinese
                "тестовый запрос",  # Russian
                "consulta de prueba",  # Spanish with accents
                "🚀🧪💻",  # Emojis only
                "café",  # French with accent
                "naïve",  # Word with diaeresis
                "Москва",  # Cyrillic
                "العربية",  # Arabic
                "日本語",  # Japanese
            ]

            for query in unicode_queries:
                result = client.search(query, limit=10)
                assert result == {"episodes": [], "facts": []}

                # Verify the request was made with correct query parameters
                request = m.request_history[-1]
                assert "query" in request.qs
                assert request.qs["query"] == [query]

    def test_complex_query_parameters_combinations(self, client):
        """Test complex combinations of query parameters."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/webhooks",
                json=[],
                status_code=200
            )

            # Test all possible parameter combinations
            test_cases = [
                {"active": True, "limit": 50, "offset": 10},
                {"active": False, "limit": 1, "offset": 0},
                {"active": True, "limit": 100, "offset": 999},
                {"active": False, "limit": 25, "offset": 50},
            ]

            for params in test_cases:
                result = client.list_webhooks(**params)
                assert result == []

                # Verify all parameters are present
                request = m.request_history[-1]
                for key, value in params.items():
                    assert key in request.qs
                    # Handle boolean conversion to string
                    expected_value = str(value).lower() if isinstance(value, bool) else str(value)
                    assert request.qs[key] == [expected_value]

    def test_maximum_parameter_limits(self, client):
        """Test maximum limits for various parameters."""
        # Test maximum search limit
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/knowledge-graph/search",
                json={"nodes": [], "edges": []},
                status_code=200
            )

            result = client.search_knowledge_graph("test", limit=100, depth=5)
            assert result == {"nodes": [], "edges": []}

        # Test maximum offset
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/memory/logs",
                json=[],
                status_code=200
            )

            result = client.get_ingestion_logs(limit=10, offset=10000)
            assert result == []

    # Response Parsing Edge Cases
    def test_minimal_json_responses(self, client):
        """Test parsing of minimal JSON responses."""
        minimal_responses = [
            {},
            {"status": "ok"},
            {"data": None},
            {"result": []},
            {"items": {}},
            {"count": 0},
            {"success": True},
            {"error": False},
        ]

        for expected_response in minimal_responses:
            with requests_mock.Mocker() as m:
                m.get(
                    "https://core.heysol.ai/api/v1/user/profile",
                    json=expected_response,
                    status_code=200
                )

                result = client.get_user_profile()
                assert result == expected_response

    def test_boolean_values_in_responses(self, client):
        """Test parsing of boolean values in responses."""
        boolean_responses = [
            {"active": True, "enabled": False, "success": True},
            {"flags": [True, False, True]},
            {"settings": {"debug": False, "production": True}},
            {"status": {"ready": True, "error": False, "loading": False}},
        ]

        for expected_response in boolean_responses:
            with requests_mock.Mocker() as m:
                m.get(
                    "https://core.heysol.ai/api/v1/user/profile",
                    json=expected_response,
                    status_code=200
                )

                result = client.get_user_profile()
                assert result == expected_response

    def test_null_values_in_responses(self, client):
        """Test parsing of null values in responses."""
        null_responses = [
            {"name": None, "description": "test"},
            {"data": None},
            {"metadata": {"created": None, "updated": "2024-01-01"}},
            {"items": [None, "item1", None]},
            {"config": {"value": None, "enabled": True}},
        ]

        for expected_response in null_responses:
            with requests_mock.Mocker() as m:
                m.get(
                    "https://core.heysol.ai/api/v1/user/profile",
                    json=expected_response,
                    status_code=200
                )

                result = client.get_user_profile()
                assert result == expected_response

    def test_deeply_nested_responses(self, client):
        """Test parsing of deeply nested JSON responses."""
        deep_response = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "data": "deep value",
                                "count": 42
                            }
                        }
                    }
                }
            }
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json=deep_response,
                status_code=200
            )

            result = client.get_user_profile()
            assert result == deep_response
            assert result["level1"]["level2"]["level3"]["level4"]["level5"]["data"] == "deep value"

    # Authentication Edge Cases
    def test_api_key_with_special_characters(self):
        """Test API key with special characters."""
        special_keys = [
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key with spaces",
            "key🚀with🧪emojis",
            "key\twith\ttabs",
        ]

        for api_key in special_keys:
            client = HeySolClient(api_key=api_key, skip_mcp_init=True)

            with requests_mock.Mocker() as m:
                m.get(
                    "https://core.heysol.ai/api/v1/user/profile",
                    json={"id": "user-123"},
                    status_code=200
                )

                result = client.get_user_profile()
                assert result == {"id": "user-123"}

    def test_empty_headers_handling(self, client):
        """Test handling of empty headers in responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200,
                headers={}  # Empty headers
            )

            result = client.get_user_profile()
            assert result == {"id": "user-123"}

    def test_case_sensitive_headers(self, client):
        """Test handling of case-sensitive headers."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-key",
                    "X-Custom-Header": "custom-value"
                }
            )

            result = client.get_user_profile()
            assert result == {"id": "user-123"}

    # Performance Edge Cases
    def test_rapid_successive_requests(self, client):
        """Test rapid successive requests."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200
            )

            start_time = time.time()

            # Make 10 rapid requests
            for _ in range(10):
                result = client.get_user_profile()
                assert result == {"id": "user-123"}

            end_time = time.time()
            total_time = end_time - start_time

            # Should complete quickly (less than 2 seconds for 10 requests)
            assert total_time < 2.0

    def test_concurrent_request_handling(self, client):
        """Test handling of concurrent-like requests."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200
            )

            # Simulate concurrent requests by making multiple requests quickly
            results = []
            for i in range(5):
                result = client.get_user_profile()
                results.append(result)
                assert result == {"id": "user-123"}

            assert len(results) == 5
            assert all(result == {"id": "user-123"} for result in results)

    # Data Validation Edge Cases
    def test_extremely_long_strings(self, client):
        """Test handling of extremely long strings."""
        long_string = "x" * 100000  # 100KB string

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/ingestion/queue",
                json={"queue_id": "queue-123"},
                status_code=200
            )

            # Test with long data
            result = client.add_data_to_ingestion_queue(
                data={"content": long_string},
                tags=["long", "test"]
            )
            assert result == {"queue_id": "queue-123"}

    def test_extremely_long_urls(self, client):
        """Test handling of extremely long URLs."""
        long_url = "https://example.com/" + "path/" * 100 + "endpoint"  # Very long URL

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/webhooks",
                json={"webhook_id": "webhook-123"},
                status_code=200
            )

            # This should work with the client's URL handling
            result = client.register_webhook(long_url, ["memory.ingest"])
            assert result == {"webhook_id": "webhook-123"}

    def test_empty_arrays_and_objects(self, client):
        """Test handling of empty arrays and objects."""
        empty_responses = [
            {"items": [], "count": 0},
            {"data": {}, "metadata": {}},
            {"results": [], "pagination": {}},
            {"users": [], "groups": {}},
            {"config": {}, "settings": []},
        ]

        for expected_response in empty_responses:
            with requests_mock.Mocker() as m:
                m.get(
                    "https://core.heysol.ai/api/v1/user/profile",
                    json=expected_response,
                    status_code=200
                )

                result = client.get_user_profile()
                assert result == expected_response

    def test_mixed_data_types(self, client):
        """Test handling of mixed data types in responses."""
        mixed_response = {
            "string": "text",
            "number": 42,
            "float": 3.14159,
            "boolean": True,
            "null": None,
            "array": [1, "two", 3.0, True, None],
            "object": {
                "nested_string": "value",
                "nested_number": 100,
                "nested_array": [1, 2, 3]
            }
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json=mixed_response,
                status_code=200
            )

            result = client.get_user_profile()
            assert result == mixed_response

    # Network Condition Edge Cases
    def test_very_slow_responses(self, client):
        """Test handling of very slow responses."""
        with requests_mock.Mocker() as m:
            # Simulate a slow response
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200
            )

            start_time = time.time()
            result = client.get_user_profile()
            end_time = time.time()

            # Should still work but may take longer
            assert result == {"id": "user-123"}
            assert end_time - start_time < 5.0  # Should complete within 5 seconds

    def test_intermittent_failures(self, client):
        """Test handling of intermittent failures."""
        with requests_mock.Mocker() as m:
            # First request fails
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"error": "Temporary failure"},
                status_code=500
            )

            # The client should raise an HTTPError
            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    # Error Response Edge Cases
    def test_error_response_with_unicode(self, client):
        """Test handling of Unicode characters in error responses."""
        unicode_error = {
            "error": "测试错误",  # Chinese error message
            "details": "错误详情包含🚀和🧪",  # Chinese with emojis
            "code": "UNICODE_ERROR"
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json=unicode_error,
                status_code=400
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    def test_error_response_with_large_payload(self, client):
        """Test handling of large error response payloads."""
        large_error = {
            "error": "Large error response",
            "details": "x" * 10000,  # 10KB error message
            "trace": [{"file": f"file_{i}.py", "line": i} for i in range(100)]
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json=large_error,
                status_code=500
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()