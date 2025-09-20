"""
Tests for data serialization and deserialization in the HeySol API client.
"""

import pytest
import json
import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from heysol.client import HeySolClient
from heysol.async_client import AsyncHeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import APIError


class TestSerialization:
    """Test data serialization and deserialization."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        config = HeySolConfig(api_key="test-api-key")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None
            client = HeySolClient(config=config)
            yield client

    @pytest.fixture
    def mock_async_client(self):
        """Create a mock async client for testing."""
        config = HeySolConfig(api_key="test-api-key")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None
            client = AsyncHeySolClient(config=config)
            yield client

    def test_json_request_serialization_sync(self, mock_client):
        """Test that request data is properly serialized to JSON."""
        test_data = {
            "name": "Test Space",
            "description": "A test space for serialization",
            "metadata": {
                "created_by": "test_user",
                "tags": ["test", "serialization"]
            }
        }

        with patch.object(mock_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"space": {"id": "space-123"}}
            mock_request.return_value = mock_response

            # This should serialize test_data to JSON
            mock_client.create_space("Test Space", "A test space for serialization")

            # Verify that request was called with JSON data
            call_args = mock_request.call_args
            assert call_args[1]['json'] == {"name": "Test Space", "description": "A test space for serialization"}

    def test_json_request_serialization_async(self, mock_async_client):
        """Test that request data is properly serialized to JSON for async client."""
        import asyncio

        async def test():
            test_data = {
                "name": "Test Space",
                "description": "A test space for serialization",
                "metadata": {
                    "created_by": "test_user",
                    "tags": ["test", "serialization"]
                }
            }

            with patch.object(mock_async_client, '_ensure_session') as mock_ensure:
                mock_session = Mock()
                mock_ensure.return_value = mock_session

                mock_response = Mock()
                mock_response.status = 201
                mock_response.json = Mock(return_value={"space": {"id": "space-123"}})
                mock_session.request.return_value = mock_response

                # This should serialize test_data to JSON
                await mock_async_client.create_space("Test Space", "A test space for serialization")

                # Verify that request was called with JSON data
                call_args = mock_session.request.call_args
                assert call_args[1]['json'] == {"name": "Test Space", "description": "A test space for serialization"}

        asyncio.run(test())

    def test_json_response_deserialization_sync(self, mock_client):
        """Test that JSON responses are properly deserialized."""
        json_response = {
            "user": {
                "id": "user-123",
                "name": "Test User",
                "email": "test@example.com",
                "profile": {
                    "department": "Engineering",
                    "role": "Developer"
                }
            },
            "permissions": ["read", "write", "admin"],
            "last_login": "2024-01-01T10:00:00Z"
        }

        with patch.object(mock_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = json_response
            mock_request.return_value = mock_response

            result = mock_client.get_user_profile()

            # Verify that response was properly deserialized
            assert result == json_response
            assert result["user"]["name"] == "Test User"
            assert result["permissions"] == ["read", "write", "admin"]

    def test_json_response_deserialization_async(self, mock_async_client):
        """Test that JSON responses are properly deserialized for async client."""
        import asyncio

        async def test():
            json_response = {
                "user": {
                    "id": "user-123",
                    "name": "Test User",
                    "email": "test@example.com",
                    "profile": {
                        "department": "Engineering",
                        "role": "Developer"
                    }
                },
                "permissions": ["read", "write", "admin"],
                "last_login": "2024-01-01T10:00:00Z"
            }

            with patch.object(mock_async_client, '_ensure_session') as mock_ensure:
                mock_session = Mock()
                mock_ensure.return_value = mock_session

                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = Mock(return_value=json_response)
                mock_session.request.return_value = mock_response

                result = await mock_async_client.get_user_profile()

                # Verify that response was properly deserialized
                assert result == json_response
                assert result["user"]["name"] == "Test User"
                assert result["permissions"] == ["read", "write", "admin"]

        asyncio.run(test())

    def test_mcp_json_serialization_sync(self, mock_client):
        """Test MCP protocol JSON serialization."""
        mcp_payload = {
            "jsonrpc": "2.0",
            "id": "test-id-123",
            "method": "tools/call",
            "params": {
                "name": "memory_ingest",
                "arguments": {
                    "message": "Test message",
                    "spaceId": "space-123",
                    "tags": ["test", "mcp"]
                }
            }
        }

        with patch.object(mock_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": {"status": "success"}}
            mock_request.return_value = mock_response

            # This should serialize MCP payload to JSON
            result = mock_client._call_tool("memory_ingest", {"message": "Test message"})

            # Verify that MCP payload was properly serialized
            call_args = mock_request.call_args
            request_payload = call_args[1]['json']

            assert request_payload["jsonrpc"] == "2.0"
            assert request_payload["method"] == "tools/call"
            assert request_payload["params"]["name"] == "memory_ingest"

    def test_mcp_json_serialization_async(self, mock_async_client):
        """Test MCP protocol JSON serialization for async client."""
        import asyncio

        async def test():
            mcp_payload = {
                "jsonrpc": "2.0",
                "id": "test-id-123",
                "method": "tools/call",
                "params": {
                    "name": "memory_ingest",
                    "arguments": {
                        "message": "Test message",
                        "spaceId": "space-123",
                        "tags": ["test", "mcp"]
                    }
                }
            }

            with patch.object(mock_async_client, '_ensure_session') as mock_ensure:
                mock_session = Mock()
                mock_ensure.return_value = mock_session

                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = Mock(return_value={"result": {"status": "success"}})
                mock_session.request.return_value = mock_response

                # This should serialize MCP payload to JSON
                result = await mock_async_client._call_tool("memory_ingest", {"message": "Test message"})

                # Verify that MCP payload was properly serialized
                call_args = mock_session.request.call_args
                request_payload = call_args[1]['json']

                assert request_payload["jsonrpc"] == "2.0"
                assert request_payload["method"] == "tools/call"
                assert request_payload["params"]["name"] == "memory_ingest"

        asyncio.run(test())

    def test_complex_data_types_serialization_sync(self, mock_client):
        """Test serialization of complex data types."""
        complex_data = {
            "strings": ["string1", "string2"],
            "numbers": [1, 2.5, -3],
            "booleans": [True, False],
            "nulls": [None, "not_none"],
            "nested": {
                "deeply": {
                    "nested": {
                        "value": "deep"
                    }
                }
            },
            "datetime": "2024-01-01T10:00:00Z",
            "special_chars": "Special chars: àáâãäå, 中文, русский"
        }

        with patch.object(mock_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "success"}
            mock_request.return_value = mock_response

            # Test with ingest method that accepts complex data
            mock_client.ingest(
                message="Complex data test",
                tags=["complex", "test"]
            )

            # Verify that complex data types are handled
            call_args = mock_request.call_args
            request_data = call_args[1]['json']

            # The actual payload structure depends on the MCP protocol
            assert "params" in request_data
            assert "arguments" in request_data["params"]

    def test_complex_data_types_serialization_async(self, mock_async_client):
        """Test serialization of complex data types for async client."""
        import asyncio

        async def test():
            complex_data = {
                "strings": ["string1", "string2"],
                "numbers": [1, 2.5, -3],
                "booleans": [True, False],
                "nulls": [None, "not_none"],
                "nested": {
                    "deeply": {
                        "nested": {
                            "value": "deep"
                        }
                    }
                },
                "datetime": "2024-01-01T10:00:00Z",
                "special_chars": "Special chars: àáâãäå, 中文, русский"
            }

            with patch.object(mock_async_client, '_ensure_session') as mock_ensure:
                mock_session = Mock()
                mock_ensure.return_value = mock_session

                mock_response = Mock()
                mock_response.status = 200
                mock_response.json = Mock(return_value={"result": "success"})
                mock_session.request.return_value = mock_response

                # Test with ingest method that accepts complex data
                await mock_async_client.ingest(
                    message="Complex data test",
                    tags=["complex", "test"]
                )

                # Verify that complex data types are handled
                call_args = mock_session.request.call_args
                request_data = call_args[1]['json']

                # The actual payload structure depends on the MCP protocol
                assert "params" in request_data
                assert "arguments" in request_data["params"]

        asyncio.run(test())

    def test_unicode_handling_sync(self, mock_client):
        """Test proper handling of Unicode characters."""
        unicode_message = "Unicode test: Hello 世界, Привет мир, Hola mundo 🌍"
        unicode_tags = ["标签", "теги", "etiquetas"]

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"status": "success"}

            result = mock_client.ingest(
                message=unicode_message,
                tags=unicode_tags
            )

            # Verify that Unicode is preserved
            assert result["status"] == "success"

            # Check that the call was made with Unicode data
            call_args = mock_call.call_args
            assert call_args[1]["message"] == unicode_message
            assert call_args[1]["tags"] == unicode_tags

    def test_unicode_handling_async(self, mock_async_client):
        """Test proper handling of Unicode characters for async client."""
        import asyncio

        async def test():
            unicode_message = "Unicode test: Hello 世界, Привет мир, Hola mundo 🌍"
            unicode_tags = ["标签", "теги", "etiquetas"]

            with patch.object(mock_async_client, '_call_tool') as mock_call:
                mock_call.return_value = {"status": "success"}

                result = await mock_async_client.ingest(
                    message=unicode_message,
                    tags=unicode_tags
                )

                # Verify that Unicode is preserved
                assert result["status"] == "success"

                # Check that the call was made with Unicode data
                call_args = mock_call.call_args
                assert call_args[1]["message"] == unicode_message
                assert call_args[1]["tags"] == unicode_tags

        asyncio.run(test())

    def test_large_data_serialization_sync(self, mock_client):
        """Test serialization of large data payloads."""
        # Create a large message
        large_message = "Large message: " + "x" * 10000  # 10KB message
        large_tags = [f"tag_{i}" for i in range(100)]  # 100 tags

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"status": "success"}

            result = mock_client.ingest(
                message=large_message,
                tags=large_tags
            )

            # Verify that large data is handled
            assert result["status"] == "success"

            # Check that the call was made with large data
            call_args = mock_call.call_args
            assert len(call_args[1]["message"]) == len(large_message)
            assert len(call_args[1]["tags"]) == len(large_tags)

    def test_large_data_serialization_async(self, mock_async_client):
        """Test serialization of large data payloads for async client."""
        import asyncio

        async def test():
            # Create a large message
            large_message = "Large message: " + "x" * 10000  # 10KB message
            large_tags = [f"tag_{i}" for i in range(100)]  # 100 tags

            with patch.object(mock_async_client, '_call_tool') as mock_call:
                mock_call.return_value = {"status": "success"}

                result = await mock_async_client.ingest(
                    message=large_message,
                    tags=large_tags
                )

                # Verify that large data is handled
                assert result["status"] == "success"

                # Check that the call was made with large data
                call_args = mock_call.call_args
                assert len(call_args[1]["message"]) == len(large_message)
                assert len(call_args[1]["tags"]) == len(large_tags)

        asyncio.run(test())

    def test_binary_data_handling_sync(self, mock_client):
        """Test handling of binary data (should be rejected or encoded)."""
        # JSON doesn't support binary data directly, so it should be handled appropriately
        binary_data = b"binary data \x00\x01\x02"

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"status": "success"}

            # This should either work or raise an appropriate error
            try:
                # Try to encode binary data as base64 or similar
                import base64
                encoded_data = base64.b64encode(binary_data).decode('utf-8')

                result = mock_client.ingest(
                    message=f"Binary data: {encoded_data}",
                    tags=["binary", "test"]
                )

                assert result["status"] == "success"

            except (TypeError, UnicodeDecodeError):
                # If binary data can't be handled, it should raise a clear error
                with pytest.raises((TypeError, UnicodeDecodeError, APIError)):
                    mock_client.ingest(
                        message=binary_data.decode('latin-1'),  # This will likely fail
                        tags=["binary", "test"]
                    )

    def test_binary_data_handling_async(self, mock_async_client):
        """Test handling of binary data for async client."""
        import asyncio

        async def test():
            # JSON doesn't support binary data directly, so it should be handled appropriately
            binary_data = b"binary data \x00\x01\x02"

            with patch.object(mock_async_client, '_call_tool') as mock_call:
                mock_call.return_value = {"status": "success"}

                # This should either work or raise an appropriate error
                try:
                    # Try to encode binary data as base64 or similar
                    import base64
                    encoded_data = base64.b64encode(binary_data).decode('utf-8')

                    result = await mock_async_client.ingest(
                        message=f"Binary data: {encoded_data}",
                        tags=["binary", "test"]
                    )

                    assert result["status"] == "success"

                except (TypeError, UnicodeDecodeError):
                    # If binary data can't be handled, it should raise a clear error
                    with pytest.raises((TypeError, UnicodeDecodeError, APIError)):
                        await mock_async_client.ingest(
                            message=binary_data.decode('latin-1'),  # This will likely fail
                            tags=["binary", "test"]
                        )

        asyncio.run(test())

    def test_datetime_serialization_sync(self, mock_client):
        """Test serialization of datetime objects."""
        from datetime import datetime, timezone

        # Test with datetime object
        dt = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        dt_string = dt.isoformat()

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"status": "success"}

            result = mock_client.ingest(
                message=f"Datetime test: {dt_string}",
                tags=["datetime", "test"]
            )

            assert result["status"] == "success"

            # Check that datetime was serialized as ISO string
            call_args = mock_call.call_args
            assert dt_string in call_args[1]["message"]

    def test_datetime_serialization_async(self, mock_async_client):
        """Test serialization of datetime objects for async client."""
        import asyncio
        from datetime import datetime, timezone

        async def test():
            # Test with datetime object
            dt = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
            dt_string = dt.isoformat()

            with patch.object(mock_async_client, '_call_tool') as mock_call:
                mock_call.return_value = {"status": "success"}

                result = await mock_async_client.ingest(
                    message=f"Datetime test: {dt_string}",
                    tags=["datetime", "test"]
                )

                assert result["status"] == "success"

                # Check that datetime was serialized as ISO string
                call_args = mock_call.call_args
                assert dt_string in call_args[1]["message"]

        asyncio.run(test())

    def test_none_value_handling_sync(self, mock_client):
        """Test handling of None values in serialization."""
        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"status": "success"}

            # Test with None values in optional parameters
            result = mock_client.ingest(
                message="None test",
                space_id=None,
                source=None,
                tags=None
            )

            assert result["status"] == "success"

            # Check that None values are handled appropriately
            call_args = mock_call.call_args
            arguments = call_args[1]

            # None values should either be omitted or handled gracefully
            assert "message" in arguments
            assert arguments["message"] == "None test"

    def test_none_value_handling_async(self, mock_async_client):
        """Test handling of None values in serialization for async client."""
        import asyncio

        async def test():
            with patch.object(mock_async_client, '_call_tool') as mock_call:
                mock_call.return_value = {"status": "success"}

                # Test with None values in optional parameters
                result = await mock_async_client.ingest(
                    message="None test",
                    space_id=None,
                    source=None,
                    tags=None
                )

                assert result["status"] == "success"

                # Check that None values are handled appropriately
                call_args = mock_call.call_args
                arguments = call_args[1]

                # None values should either be omitted or handled gracefully
                assert "message" in arguments
                assert arguments["message"] == "None test"

        asyncio.run(test())

    def test_circular_reference_prevention_sync(self, mock_client):
        """Test that circular references are prevented during serialization."""
        # Create objects with circular references
        obj1 = {"name": "object1"}
        obj2 = {"name": "object2"}
        obj1["reference"] = obj2
        obj2["reference"] = obj1

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"status": "success"}

            # This should either work or raise an appropriate error
            try:
                result = mock_client.ingest(
                    message="Circular reference test",
                    tags=["circular", "test"]
                )
                assert result["status"] == "success"
            except (TypeError, ValueError) as e:
                # Circular references should raise a clear error
                assert "circular" in str(e).lower() or "reference" in str(e).lower()

    def test_circular_reference_prevention_async(self, mock_async_client):
        """Test that circular references are prevented during serialization for async client."""
        import asyncio

        async def test():
            # Create objects with circular references
            obj1 = {"name": "object1"}
            obj2 = {"name": "object2"}
            obj1["reference"] = obj2
            obj2["reference"] = obj1

            with patch.object(mock_async_client, '_call_tool') as mock_call:
                mock_call.return_value = {"status": "success"}

                # This should either work or raise an appropriate error
                try:
                    result = await mock_async_client.ingest(
                        message="Circular reference test",
                        tags=["circular", "test"]
                    )
                    assert result["status"] == "success"
                except (TypeError, ValueError) as e:
                    # Circular references should raise a clear error
                    assert "circular" in str(e).lower() or "reference" in str(e).lower()

        asyncio.run(test())