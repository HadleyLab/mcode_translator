"""
Minimal HeySol API client implementation for mCODE Translator CLI.
"""

from typing import Any, Dict, List, Optional


class HeySolAPIClient:
    """Minimal API client implementation."""

    def __init__(self, api_key: str, base_url: str, config=None):
        self.api_key = api_key
        self.base_url = base_url
        self.config = config

    def get_spaces(self) -> List[Dict[str, Any]]:
        """Get available spaces."""
        return []

    def is_mcp_available(self) -> bool:
        """Check if MCP is available."""
        return False

    def close(self) -> None:
        """Close the client."""
        pass


class HeySolMCPClient:
    """Minimal MCP client implementation."""

    def __init__(self, api_key: str, mcp_url: str, config=None):
        self.api_key = api_key
        self.mcp_url = mcp_url
        self.config = config
        self.tools = {}
        self.session_id = None

    def is_mcp_available(self) -> bool:
        """Check if MCP is available."""
        return False

    def close(self) -> None:
        """Close the client."""
        pass


class HeySolClient:
    """Minimal HeySol client for CLI operations."""

    def __init__(self, api_key: str, base_url: str = "https://core.heysol.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.api_client = HeySolAPIClient(api_key, base_url)
        self.mcp_client = None

    def get_spaces(self) -> List[Dict[str, Any]]:
        """Get available memory spaces."""
        return self.api_client.get_spaces()

    def is_mcp_available(self) -> bool:
        """Check if MCP is available."""
        return False

    def close(self) -> None:
        """Close the client."""
        if self.api_client:
            self.api_client.close()
        if self.mcp_client:
            self.mcp_client.close()