"""
HeySol API client implementation with MCP (Model Context Protocol) support.
"""

import json
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

from .config import HeySolConfig
from .exceptions import HeySolError, ValidationError
from .oauth2 import InteractiveOAuth2Authenticator, OAuth2Tokens


class HeySolClient:
    """
    Core client for interacting with the HeySol API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, oauth2_auth: Optional[InteractiveOAuth2Authenticator] = None, skip_mcp_init: bool = False):
        """
        Initialize the HeySol API client.

        Args:
            api_key: HeySol API key (for API key authentication)
            base_url: Base URL for the API (optional, uses config default if not provided)
            oauth2_auth: OAuth2 authenticator instance (for OAuth2 authentication)
            skip_mcp_init: Skip MCP session initialization (useful for testing)
        """
        # Load configuration
        config = HeySolConfig.from_env()

        # Use provided values or fall back to config
        # Only load API key from config if not using OAuth2 and api_key is None (not empty string)
        if api_key is None and not oauth2_auth:
            api_key = config.api_key
        if not base_url:
            base_url = config.base_url

        # Validate authentication method
        if not api_key and not oauth2_auth:
            raise ValidationError("Either API key or OAuth2 authenticator is required")

        if api_key and oauth2_auth:
            raise ValidationError("Cannot use both API key and OAuth2 authentication simultaneously")

        self.api_key = api_key
        self.base_url = base_url
        self.source = config.source
        self.mcp_url = config.mcp_url
        self.session_id: Optional[str] = None
        self.oauth2_auth = oauth2_auth
        self.oauth2_tokens: Optional[OAuth2Tokens] = None
        self.tools: Dict[str, Any] = {}
        self.timeout = 60  # Default timeout

        # Initialize MCP session (skip for testing)
        if not skip_mcp_init:
            self._initialize_session()

    def _get_authorization_header(self) -> str:
        """Get the appropriate authorization header based on authentication method."""
        if self.oauth2_auth:
            # Use OAuth2 authentication
            try:
                return self.oauth2_auth.get_authorization_header()
            except HeySolError as e:
                raise HeySolError(f"OAuth2 authentication failed: {e}")
        else:
            # Use API key authentication
            if not self.api_key:
                raise HeySolError("No API key available for authentication")
            return f"Bearer {self.api_key}"

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an HTTP request using MCP JSON-RPC protocol."""
        url = self.base_url.rstrip("/") + "/" + endpoint.lstrip("/")

        # Get authorization header based on authentication method
        auth_header = self._get_authorization_header()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "Authorization": auth_header,
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        response = requests.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def _mcp_request(self, method: str, params: Optional[Dict[str, Any]] = None, stream: bool = False) -> Dict[str, Any]:
        """Make an MCP JSON-RPC request."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        }

        response = requests.post(
            self.mcp_url,
            json=payload,
            headers=self._get_mcp_headers(),
            timeout=self.timeout,
            stream=stream
        )

        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise HeySolError(f"HTTP error: {response.status_code} - {response.text}")

        return self._parse_mcp_response(response)

    def _get_mcp_headers(self) -> Dict[str, str]:
        """Get MCP-specific headers."""
        headers = {
            "Authorization": self._get_authorization_header(),
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _parse_mcp_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parse MCP JSON-RPC response."""
        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()

        if content_type == "application/json":
            msg = response.json()
        elif content_type == "text/event-stream":
            msg = None
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    break
            if msg is None:
                raise HeySolError("No JSON in SSE stream")
        else:
            raise HeySolError(f"Unexpected Content-Type: {content_type}")

        if "error" in msg:
            raise HeySolError(f"MCP error: {msg['error']}")

        return msg["result"]

    def _initialize_session(self) -> None:
        """Initialize MCP session and discover available tools."""
        # Initialize session
        init_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": True},
                "clientInfo": {"name": "heysol-python-client", "version": "1.0.0"},
            },
        }

        try:
            response = requests.post(
                self.mcp_url,
                json=init_payload,
                headers=self._get_mcp_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            result = self._parse_mcp_response(response)
            self.session_id = response.headers.get("Mcp-Session-Id") or self.session_id
        except Exception as e:
            raise HeySolError(f"Failed to initialize MCP session: {e}")

        # List available tools
        tools_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }

        try:
            response = requests.post(
                self.mcp_url,
                json=tools_payload,
                headers=self._get_mcp_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            result = self._parse_mcp_response(response)
            self.tools = {t["name"]: t for t in result.get("tools", [])}
        except Exception as e:
            raise HeySolError(f"Failed to list MCP tools: {e}")

    def _unwrap_tool_result(self, result: Any) -> Any:
        """Unwrap tool result from MCP response."""
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            for item in result["content"]:
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    try:
                        return json.loads(txt)
                    except json.JSONDecodeError:
                        return txt
        return result

    def ingest(self, message: str, space_id: Optional[str] = None) -> Dict[str, Any]:
        """Ingest data into CORE Memory using MCP protocol."""
        if not message:
            raise ValidationError("Message is required for ingestion")

        if "memory_ingest" not in self.tools:
            # Fallback to direct API call if MCP tool not available
            payload = {"message": message}
            if space_id:
                payload["spaceId"] = space_id
            return self._make_request("POST", "memory/ingest", data=payload)

        # Use MCP tool
        ingest_args = {"message": message, "source": "heysol-python-client"}
        if space_id:
            ingest_args["spaceId"] = space_id

        result = self._mcp_request("tools/call", {
            "name": "memory_ingest",
            "arguments": ingest_args
        }, stream=True)

        return self._unwrap_tool_result(result)

    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for memories in CORE Memory using MCP protocol."""
        if not query:
            raise ValidationError("Search query is required")

        if "memory_search" not in self.tools:
            # Fallback to direct API call if MCP tool not available
            return self._make_request("GET", "memory/search", params={"query": query, "limit": limit})

        # Use MCP tool
        search_args = {"query": query, "limit": limit}

        result = self._mcp_request("tools/call", {
            "name": "memory_search",
            "arguments": search_args
        })

        unwrapped_result = self._unwrap_tool_result(result)

        # Handle the actual search result format
        if isinstance(unwrapped_result, dict):
            return unwrapped_result
        elif isinstance(unwrapped_result, list):
            # Fallback: if we get a list, wrap it in a dict structure
            return {"episodes": unwrapped_result, "facts": []}
        else:
            return {"episodes": [], "facts": []}

    def get_spaces(self) -> Dict[str, Any]:
        """Get available memory spaces using MCP protocol."""
        if "memory_get_spaces" not in self.tools:
            # Fallback to direct API call if MCP tool not available
            return self._make_request("GET", "spaces")

        result = self._mcp_request("tools/call", {
            "name": "memory_get_spaces",
            "arguments": {}
        })

        unwrapped_result = self._unwrap_tool_result(result)

        # Extract spaces from the result
        if isinstance(unwrapped_result, dict) and "spaces" in unwrapped_result:
            return unwrapped_result["spaces"]
        elif isinstance(unwrapped_result, list):
            return unwrapped_result
        else:
            return []

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

    def get_episode_facts(self, episode_id: str, limit: int = 100, offset: int = 0, include_metadata: bool = True) -> list:
        """Get episode facts from CORE Memory."""
        if not episode_id:
            raise ValidationError("Episode ID is required")

        params = {"limit": limit, "offset": offset, "include_metadata": include_metadata}
        return self._make_request("GET", f"episodes/{episode_id}/facts", params=params)

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

        # Store original api_key temporarily
        original_api_key = self.api_key
        try:
            # Use the access token as the api_key for this request
            self.api_key = access_token
            return self._make_request("GET", "oauth2/userinfo")
        finally:
            # Restore original api_key
            self.api_key = original_api_key

    def oauth2_refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh OAuth2 access token."""
        if not refresh_token:
            raise ValidationError("Refresh token is required")

        payload = {"refresh_token": refresh_token}
        return self._make_request("POST", "oauth2/refresh", data=payload)

    def oauth2_revoke_token(self, token: str, token_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """Revoke OAuth2 token."""
        if not token:
            raise ValidationError("Token is required")

        payload = {"token": token}
        if token_type_hint:
            payload["token_type_hint"] = token_type_hint

        return self._make_request("POST", "oauth2/revoke", data=payload)

    def oauth2_token_introspection(self, token: str) -> Dict[str, Any]:
        """Introspect OAuth2 token."""
        if not token:
            raise ValidationError("Token is required")

        payload = {"token": token}
        return self._make_request("GET", "oauth2/introspect", data=payload)

    # Webhook endpoints
    def register_webhook(self, url: str, events: list, space_id: Optional[str] = None, secret: Optional[str] = None) -> Dict[str, Any]:
        """Register a new webhook."""
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

    def list_webhooks(self, space_id: Optional[str] = None, active: Optional[bool] = None, limit: int = 100, offset: int = 0) -> list:
        """List all webhooks."""
        params = {"limit": limit, "offset": offset}
        if space_id:
            params["space_id"] = space_id
        if active is not None:
            params["active"] = active

        return self._make_request("GET", "webhooks", params=params)

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

    def delete_webhook(self, webhook_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a webhook."""
        if not webhook_id:
            raise ValidationError("Webhook ID is required")

        if not confirm:
            raise ValidationError("Webhook deletion requires confirmation (confirm=True)")

        return self._make_request("DELETE", f"webhooks/{webhook_id}")