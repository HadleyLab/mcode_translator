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

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            params=params,
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

    # User endpoints
    def get_user_profile(self) -> Dict[str, Any]:
        """Get the current user's profile."""
        return self._make_request("GET", "user/profile")

    # Memory endpoints
    def search_knowledge_graph(self, query: str, space_id: Optional[str] = None, limit: int = 10, depth: int = 2) -> Dict[str, Any]:
        """Search the knowledge graph for related concepts and entities."""
        if not query:
            raise ValidationError("Search query is required")

        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")

        if depth < 1 or depth > 5:
            raise ValidationError("Depth must be between 1 and 5")

        params = {"q": query, "limit": limit, "depth": depth}
        if space_id:
            params["space_id"] = space_id

        return self._make_request("POST", "memory/knowledge-graph/search", params=params)

    def add_data_to_ingestion_queue(self, data: Any, space_id: Optional[str] = None, priority: str = "normal", tags: Optional[list] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add data to the ingestion queue for processing."""
        payload = {"data": data, "priority": priority}
        if space_id:
            payload["space_id"] = space_id
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        return self._make_request("POST", "memory/ingestion/queue", data=payload)

    def get_episode_facts(self, episode_id: Optional[str] = None, space_id: Optional[str] = None, limit: int = 100, offset: int = 0, include_metadata: bool = True) -> list:
        """Get episode facts from CORE Memory."""
        params = {"limit": limit, "offset": offset, "include_metadata": include_metadata}
        if episode_id:
            params["episode_id"] = episode_id
        if space_id:
            params["space_id"] = space_id

        return self._make_request("GET", "memory/episodes/facts", params=params)

    def get_ingestion_logs(self, space_id: Optional[str] = None, limit: int = 100, offset: int = 0, status: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> list:
        """Get ingestion logs from CORE Memory."""
        params = {"limit": limit, "offset": offset}
        if space_id:
            params["space_id"] = space_id
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._make_request("GET", "memory/logs", params=params)

    def get_specific_log(self, log_id: str) -> Dict[str, Any]:
        """Get a specific ingestion log by ID."""
        if not log_id:
            raise ValidationError("Log ID is required")

        return self._make_request("GET", f"memory/logs/{log_id}")

    # Spaces endpoints
    def bulk_space_operations(self, operations: list) -> list:
        """Perform bulk operations on spaces."""
        return self._make_request("PUT", "spaces/bulk", data={"operations": operations})

    def get_space_details(self, space_id: str, include_stats: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """Get detailed information about a specific space."""
        if not space_id:
            raise ValidationError("Space ID is required")

        params = {"include_stats": include_stats, "include_metadata": include_metadata}
        return self._make_request("GET", f"spaces/{space_id}/details", params=params)

    def update_space(self, space_id: str, name: Optional[str] = None, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, is_public: Optional[bool] = None) -> Dict[str, Any]:
        """Update properties of an existing space."""
        if not space_id:
            raise ValidationError("Space ID is required")

        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if metadata is not None:
            payload["metadata"] = metadata
        if is_public is not None:
            payload["is_public"] = is_public

        return self._make_request("PUT", f"spaces/{space_id}", data=payload)

    def delete_space(self, space_id: str, confirm: bool = False, cascade: bool = False) -> Dict[str, Any]:
        """Delete a space and optionally all its contents."""
        if not space_id:
            raise ValidationError("Space ID is required")

        if not confirm:
            raise ValidationError("Space deletion requires confirmation (confirm=True)")

        params = {"cascade": cascade}
        return self._make_request("DELETE", f"spaces/{space_id}", params=params)

    # OAuth2 endpoints
    def get_oauth2_authorization_url(self, scope: str = "openid profile email") -> str:
        """Get OAuth2 authorization URL."""
        params = {"scope": scope}
        return self._make_request("GET", "oauth2/authorize", params=params)

    def oauth2_authorization_decision(self, decision: str, request_id: str) -> Dict[str, Any]:
        """Make OAuth2 authorization decision."""
        if decision not in ["allow", "deny"]:
            raise ValidationError("Decision must be 'allow' or 'deny'")

        if not request_id:
            raise ValidationError("Request ID is required")

        payload = {"decision": decision}
        return self._make_request("POST", f"oauth2/authorize/{request_id}", data=payload)

    def oauth2_token_exchange(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange OAuth2 authorization code for tokens."""
        if not code:
            raise ValidationError("Authorization code is required")

        if not redirect_uri:
            raise ValidationError("Redirect URI is required")

        payload = {"code": code, "redirect_uri": redirect_uri}
        return self._make_request("POST", "oauth2/token", data=payload)

    def get_oauth2_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get OAuth2 user information."""
        if not access_token:
            raise ValidationError("Access token is required")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        response = requests.get(
            f"{self.base_url}/oauth2/userinfo",
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()
        return response.json()

    def oauth2_token_introspection(self, token: str) -> Dict[str, Any]:
        """Introspect OAuth2 token."""
        if not token:
            raise ValidationError("Token is required")

        payload = {"token": token}
        return self._make_request("GET", "oauth2/introspect", data=payload)

    # Webhook endpoints
    def create_webhook(self, url: str, events: list, space_id: Optional[str] = None, secret: Optional[str] = None) -> Dict[str, Any]:
        """Create a new webhook."""
        if not url:
            raise ValidationError("Webhook URL is required")

        if not events:
            raise ValidationError("Events list is required")

        payload = {"url": url, "events": events}
        if space_id:
            payload["space_id"] = space_id
        if secret:
            payload["secret"] = secret

        return self._make_request("POST", "webhooks", data=payload)

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Get webhook details."""
        if not webhook_id:
            raise ValidationError("Webhook ID is required")

        return self._make_request("GET", f"webhooks/{webhook_id}")

    def update_webhook(self, webhook_id: str, url: Optional[str] = None, events: Optional[list] = None, space_id: Optional[str] = None, secret: Optional[str] = None, active: Optional[bool] = None) -> Dict[str, Any]:
        """Update webhook properties."""
        if not webhook_id:
            raise ValidationError("Webhook ID is required")

        payload = {}
        if url is not None:
            payload["url"] = url
        if events is not None:
            payload["events"] = events
        if space_id is not None:
            payload["space_id"] = space_id
        if secret is not None:
            payload["secret"] = secret
        if active is not None:
            payload["active"] = active

        return self._make_request("PUT", f"webhooks/{webhook_id}", data=payload)