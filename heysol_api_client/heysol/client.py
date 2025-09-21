"""
Minimal HeySol API client implementation.
"""

import json
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from .config import HeySolConfig
from .exceptions import HeySolError, ValidationError


class HeySolClient:
    """
    Core client for interacting with the HeySol API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the HeySol API client.

        Args:
            api_key: HeySol API key
        """
        if not api_key:
            raise ValidationError("API key is required")

        self.api_key = api_key
        self.base_url = "https://core.heysol.ai/api/v1/mcp"

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an HTTP request."""
        url = urljoin(self.base_url, endpoint.lstrip("/"))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.request(
            method=method,
            url=url,
            json=data,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()
        return response.json()

    def ingest(self, message: str, space_id: Optional[str] = None) -> Dict[str, Any]:
        """Ingest data into CORE Memory."""
        if not message:
            raise ValidationError("Message is required for ingestion")

        payload = {"message": message}
        if space_id:
            payload["spaceId"] = space_id

        return self._make_request("POST", "memory/ingest", data=payload)

    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for memories in CORE Memory."""
        if not query:
            raise ValidationError("Search query is required")

        return self._make_request("GET", "memory/search", params={"query": query, "limit": limit})

    def get_spaces(self) -> Dict[str, Any]:
        """Get available memory spaces."""
        return self._make_request("GET", "memory/spaces")

    def create_space(self, name: str, description: str = "") -> str:
        """Create a new memory space."""
        if not name:
            raise ValidationError("Space name is required")

        payload = {"name": name, "description": description}
        data = self._make_request("POST", "memory/spaces", data=payload)
        return data.get("id") or data.get("space_id")

    def delete_log_entry(self, log_id: str) -> bool:
        """Delete a log entry from CORE Memory."""
        if not log_id:
            raise ValidationError("Log ID is required for deletion")

        self._make_request("DELETE", f"memory/logs/{log_id}")
        return True