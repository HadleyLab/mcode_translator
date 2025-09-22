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


class HeySolClient:
    """
    Core client for interacting with the HeySol API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, skip_mcp_init: bool = False):
        """
        Initialize the HeySol API client.

        Args:
            api_key: HeySol API key (required for authentication)
            base_url: Base URL for the API (optional, uses config default if not provided)
            skip_mcp_init: Skip MCP session initialization (useful for testing)
        """
        # Load configuration
        config = HeySolConfig.from_env()

        # Use provided values or fall back to config
        if api_key is None:
            api_key = config.api_key
        if not base_url:
            base_url = config.base_url

        # Validate authentication
        if not api_key:
            raise ValidationError("API key is required")

        self.api_key = api_key
        self.base_url = base_url
        self.source = config.source
        self.mcp_url = config.mcp_url
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Any] = {}
        self.timeout = 60  # Default timeout

        # Initialize MCP session (skip for testing)
        if not skip_mcp_init:
            self._initialize_session()

    def _get_authorization_header(self) -> str:
        """Get the authorization header using API key."""
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

        return msg.get("result", msg)

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
            # If tools/list fails, try to initialize with known tools
            self.tools = {
                "memory_ingest": {"name": "memory_ingest", "description": "Ingest data into CORE Memory"},
                "memory_search": {"name": "memory_search", "description": "Search for memories in CORE Memory"},
                "memory_get_spaces": {"name": "memory_get_spaces", "description": "Get available memory spaces"},
                "get_user_profile": {"name": "get_user_profile", "description": "Get the current user's profile"}
            }

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

    def is_mcp_available(self) -> bool:
        """Check if MCP is available and initialized."""
        return bool(self.session_id and self.tools)

    def get_preferred_access_method(self, mcp_tool_name: Optional[str] = None) -> str:
        """Determine the preferred access method for a given operation."""
        if mcp_tool_name and mcp_tool_name in self.tools:
            return "mcp"
        elif self.is_mcp_available():
            return "mcp"
        else:
            return "direct_api"

    def ensure_mcp_available(self, tool_name: Optional[str] = None) -> None:
        """Ensure MCP is available and optionally check for specific tool."""
        if not self.is_mcp_available():
            raise HeySolError("MCP is not available. Please check your MCP configuration.")

        if tool_name and tool_name not in self.tools:
            raise HeySolError(f"MCP tool '{tool_name}' is not available. Available tools: {list(self.tools.keys())}")

    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available MCP tools."""
        return self.tools.copy()

    def ingest(self, message: str, space_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Ingest data into CORE Memory using direct API."""
        if not message:
            raise ValidationError("Message is required for ingestion")

        # Use direct API call with the correct format
        payload = {
            "episodeBody": message,
            "referenceTime": "2023-11-07T05:31:56Z",  # Use current timestamp
            "metadata": {},
            "source": self.source or "heysol-python-client",
            "sessionId": session_id or self.session_id or ""
        }
        if space_id:
            payload["spaceId"] = space_id

        return self._make_request("POST", "add", data=payload)

    def search(self, query: str, space_ids: Optional[list] = None, limit: int = 10, include_invalidated: bool = False) -> Dict[str, Any]:
        """Search for memories in CORE Memory using direct API."""
        if not query:
            raise ValidationError("Search query is required")

        # Use direct API call with correct format
        payload = {
            "query": query,
            "spaceIds": space_ids or [],
            "includeInvalidated": include_invalidated
        }

        params = {"limit": limit}

        return self._make_request("POST", "search", data=payload, params=params)

    def get_spaces(self) -> list:
        """Get available memory spaces using direct API."""
        # Use direct API call to GET /api/v1/spaces
        result = self._make_request("GET", "spaces")

        # Handle the expected response format
        if isinstance(result, dict) and "spaces" in result:
            return result["spaces"]
        elif isinstance(result, list):
            return result
        else:
            return []

    def create_space(self, name: str, description: str = "") -> str:
        """Create a new memory space."""
        if not name:
            raise ValidationError("Space name is required")

        payload = {"name": name, "description": description}
        data = self._make_request("POST", "spaces", data=payload)
        return data.get("id") or data.get("space_id")


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

        # Use the same search endpoint but with knowledge graph parameters
        payload = {
            "query": query,
            "spaceIds": [space_id] if space_id else [],
            "includeInvalidated": False
        }

        params = {"limit": limit, "depth": depth, "type": "knowledge_graph"}

        result = self._make_request("POST", "search", data=payload, params=params)

        # Handle knowledge graph response format
        if "results" in result:
            return result
        else:
            # Fallback: convert episodes format to results format if needed
            return {"results": result.get("episodes", [])}

    def add_data_to_ingestion_queue(self, data: Any, space_id: Optional[str] = None, priority: str = "normal", tags: Optional[list] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add data to the ingestion queue for processing."""
        # Use the same /add endpoint as ingest but with queue-specific payload
        payload = {
            "episodeBody": data if isinstance(data, str) else json.dumps(data),
            "referenceTime": "2023-11-07T05:31:56Z",  # Use current timestamp
            "metadata": metadata or {},
            "source": self.source or "heysol-python-client",
            "sessionId": ""
        }
        if space_id:
            payload["spaceId"] = space_id

        # Add queue-specific fields to metadata
        queue_metadata = payload["metadata"]
        queue_metadata["priority"] = priority
        if tags:
            queue_metadata["tags"] = tags

        result = self._make_request("POST", "add", data=payload)

        # Handle queue-specific response format
        if "success" in result and "id" in result:
            # Convert id to queueId for compatibility
            result["queueId"] = result.pop("id")
        elif "success" in result and "queueId" not in result:
            # If success but no queueId, generate one
            result["queueId"] = result.get("id", "unknown")

        return result

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
            params["spaceId"] = space_id
        if status:
            params["status"] = status
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        result = self._make_request("GET", "logs", params=params)

        # Handle the expected response format
        if isinstance(result, dict) and "logs" in result:
            return result["logs"]
        elif isinstance(result, list):
            return result
        else:
            return []

    def get_specific_log(self, log_id: str) -> Dict[str, Any]:
        """Get a specific ingestion log by ID."""
        if not log_id:
            raise ValidationError("Log ID is required")

        result = self._make_request("GET", f"logs/{log_id}")

        # Handle the expected response format
        if isinstance(result, dict) and "log" in result:
            return result["log"]
        else:
            return result

    # Spaces endpoints
    def bulk_space_operations(self, intent: str, space_id: Optional[str] = None, statement_ids: Optional[list] = None, space_ids: Optional[list] = None) -> Dict[str, Any]:
        """Perform bulk operations on spaces."""
        if not intent:
            raise ValidationError("Intent is required for bulk operations")

        payload = {"intent": intent}
        if space_id:
            payload["spaceId"] = space_id
        if statement_ids:
            payload["statementIds"] = statement_ids
        if space_ids:
            payload["spaceIds"] = space_ids

        return self._make_request("PUT", "spaces", data=payload)

    def get_space_details(self, space_id: str, include_stats: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """Get detailed information about a specific space."""
        if not space_id:
            raise ValidationError("Space ID is required")

        params = {"include_stats": include_stats, "include_metadata": include_metadata}
        return self._make_request("GET", f"spaces/{space_id}/details", params=params)

    def update_space(self, space_id: str, name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
        """Update properties of an existing space."""
        if not space_id:
            raise ValidationError("Space ID is required")

        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description

        return self._make_request("PUT", f"spaces/{space_id}", data=payload)

    def delete_space(self, space_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a space."""
        if not space_id:
            raise ValidationError("Space ID is required")

        if not confirm:
            raise ValidationError("Space deletion requires confirmation (confirm=True)")

        return self._make_request("DELETE", f"spaces/{space_id}")


    # Webhook endpoints
    def register_webhook(self, url: str, events: list, secret: str) -> Dict[str, Any]:
        """Register a new webhook."""
        if not url:
            raise ValidationError("Webhook URL is required")

        if not events:
            raise ValidationError("Webhook events are required")

        if secret == "":
            raise ValidationError("Webhook secret is required")

        # Use form data format as specified in API
        data = {"url": url, "events": ",".join(events), "secret": secret}

        # Create a custom request for form data
        request_url = self.base_url.rstrip("/") + "/" + "webhooks".lstrip("/")

        headers = {
            "Authorization": self._get_authorization_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json, text/event-stream, */*",
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        response = requests.post(
            url=request_url,
            data=data,  # This will automatically encode as form data
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

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

    def update_webhook(self, webhook_id: str, url: str, events: list, secret: str = "", active: bool = True) -> Dict[str, Any]:
        """Update webhook properties."""
        if not webhook_id:
            raise ValidationError("Webhook ID is required")

        if not url:
            raise ValidationError("Webhook URL is required")

        if not events:
            raise ValidationError("Webhook events are required")

        if not secret:
            raise ValidationError("Webhook secret is required")

        # Use form data format as specified in API
        data = {"url": url, "events": ",".join(events), "secret": secret, "active": str(active).lower()}

        # Create a custom request for form data
        request_url = self.base_url.rstrip("/") + "/" + f"webhooks/{webhook_id}".lstrip("/")

        headers = {
            "Authorization": self._get_authorization_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json, text/event-stream, */*",
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        response = requests.put(
            url=request_url,
            data=data,  # This will automatically encode as form data
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def delete_webhook(self, webhook_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a webhook."""
        if not webhook_id:
            raise ValidationError("Webhook ID is required")

        if not confirm:
            raise ValidationError("Webhook deletion requires confirmation (confirm=True)")

        return self._make_request("DELETE", f"webhooks/{webhook_id}")

    def delete_log_entry(self, log_id: str) -> Dict[str, Any]:
        """Delete a log entry from CORE Memory."""
        if not log_id:
            raise ValidationError("Log ID is required")

        # Use the DELETE endpoint with log ID in payload
        payload = {"id": log_id}
        return self._make_request("DELETE", f"logs/{log_id}", data=payload)