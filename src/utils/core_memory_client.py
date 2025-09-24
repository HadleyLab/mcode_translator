import json
import uuid
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from src.utils.config import Config

JSON = Dict[str, Any]


class CoreMemoryError(Exception):
    """Custom exception for CORE Memory errors."""

    pass


class CoreMemoryClient:
    """A lean client for interacting with the CORE Memory API using MCP protocol."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        source: Optional[str] = None,
    ):
        """
        Initializes the CoreMemoryClient with MCP protocol support.

        Args:
            api_key: CORE Memory API key (optional, loaded from config if not provided)
            base_url: MCP server base URL (optional, loaded from config if not provided)
            source: Source identifier for the requests (optional, loaded from config if not provided)
        """
        load_dotenv()

        # Load configuration
        config = Config()

        # Use provided values or load from config
        if api_key is None:
            api_key = config.get_core_memory_api_key()
        if base_url is None:
            base_url = config.get_core_memory_api_base_url()
        if source is None:
            source = config.get_core_memory_source()

        if not api_key:
            raise CoreMemoryError("API key is required.")

        self.api_key = api_key
        self.url = f"{base_url}?source={source}"
        self.session_id: Optional[str] = None
        self.tools: Dict[str, JSON] = {}
        self.timeout = config.get_core_memory_timeout()

        # Initialize the MCP session
        self._initialize_session()

    def _headers(self) -> Dict[str, str]:
        """Create headers for MCP requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _post(self, payload: JSON, stream: bool = False) -> requests.Response:
        """Send POST request to MCP server."""
        response = requests.post(
            self.url,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
            stream=stream,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise CoreMemoryError(
                f"HTTP error: {response.status_code} - {response.text}"
            )
        return response

    def _parse_rpc(self, resp: requests.Response) -> JSON:
        """Parse RPC response from MCP server."""
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip()

        if content_type == "application/json":
            msg = resp.json()
        elif content_type == "text/event-stream":
            msg = None
            for line in resp.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    break
            if msg is None:
                raise CoreMemoryError("No JSON in SSE stream")
        else:
            raise CoreMemoryError(f"Unexpected Content-Type: {content_type}")

        if "error" in msg:
            raise CoreMemoryError(f"MCP error: {msg['error']}")

        return msg["result"]

    def _initialize_session(self) -> None:
        """Initialize MCP session and list available tools."""
        # Initialize session
        init_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": True},
                "clientInfo": {"name": "Python-Script", "version": "1.0.0"},
            },
        }

        try:
            response = self._post(init_payload)
            self.session_id = response.headers.get("Mcp-Session-Id") or self.session_id
            self._parse_rpc(response)
        except Exception as e:
            raise CoreMemoryError(f"Failed to initialize MCP session: {e}")

        # List available tools
        tools_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }

        try:
            response = self._post(tools_payload)
            result = self._parse_rpc(response)
            self.tools = {t["name"]: t for t in result.get("tools", [])}
        except Exception as e:
            raise CoreMemoryError(f"Failed to list MCP tools: {e}")

        # Note: Direct API calls are used for core functionality, MCP tools are optional

    def _unwrap_tool_result(self, result: JSON) -> Any:
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

    def get_spaces(self) -> list:
        """
        Get available spaces in CORE Memory using MCP protocol (direct API endpoints are pending).

        Returns:
            List of available spaces
        """
        if "memory_get_spaces" not in self.tools:
            raise CoreMemoryError("Server missing required tool: memory_get_spaces")

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": "memory_get_spaces", "arguments": {}},
        }

        try:
            response = self._post(payload, stream=False)
            result = self._parse_rpc(response)
            unwrapped_result = self._unwrap_tool_result(result)

            # Extract spaces from the result
            if isinstance(unwrapped_result, dict) and "spaces" in unwrapped_result:
                return unwrapped_result["spaces"]
            elif isinstance(unwrapped_result, list):
                return unwrapped_result
            else:
                return []
        except Exception as e:
            raise CoreMemoryError(f"Failed to get spaces: {e}")

    def create_space(self, name: str, description: str = "") -> str:
        """
        Create a new space in CORE Memory.

        Args:
            name: Name of the space to create
            description: Optional description for the space

        Returns:
            Space ID of the created space

        Raises:
            CoreMemoryError: If space creation fails
        """
        # Use direct API call to create space (not through MCP tools)
        # Remove the MCP path and source parameter to get the base URL
        base_url = self.url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces"
        payload = {"name": name, "description": description}

        try:
            response = requests.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            space_id = data.get("space", {}).get("id") or data.get("id")
            if not space_id:
                raise CoreMemoryError(f"Unexpected create_space response: {data}")
            return space_id
        except Exception as e:
            raise CoreMemoryError(f"Failed to create space '{name}': {e}")

    def get_or_create_space(self, name: str, description: str = "") -> str:
        """
        Get existing space or create a new one.

        Args:
            name: Name of the space
            description: Description for new space if created

        Returns:
            Space ID
        """
        spaces = self.get_spaces()

        # Look for existing space
        for space in spaces:
            if space.get("name") == name and space.get("writable", True):
                return space.get("id") or space.get("spaceId")

        # Create new space if not found
        return self.create_space(name, description)

    def get_clinical_trials_space_id(self) -> str:
        """
        Get or create the clinical trials space ID from config.

        Returns:
            Space ID for the clinical trials space
        """
        config = Config()
        default_spaces = config.get_core_memory_default_spaces()
        space_name = default_spaces.get("clinical_trials", "Clinical Trials")
        return self.get_or_create_space(
            space_name,
            "Space for storing mCODE-aligned clinical trial protocols and eligibility criteria",
        )

    def get_patients_space_id(self) -> str:
        """
        Get or create the patients space ID from config.

        Returns:
            Space ID for the patients space
        """
        config = Config()
        default_spaces = config.get_core_memory_default_spaces()
        space_name = default_spaces.get("patients", "Patients")
        return self.get_or_create_space(
            space_name, "Space for storing mCODE-compliant patient oncology data"
        )

    def ingest(
        self,
        message: str,
        space_id: Optional[str] = None,
        source: str = "Python-Script",
        priority: str = "normal",
        tags: Optional[list] = None,
    ) -> dict:
        """
        Add data to CORE Memory using MCP protocol (direct API endpoints are pending).

        Args:
            message: The message to ingest
            space_id: Optional space ID to store the message in
            source: Source identifier for the message
            priority: Priority level ('low', 'normal', 'high') - ignored in MCP mode
            tags: Optional list of tags for the ingestion - ignored in MCP mode

        Returns:
            Dictionary with the ingestion result

        Raises:
            CoreMemoryError: If ingestion fails
        """
        if not message:
            raise CoreMemoryError("Message is required for ingestion.")

        # Prepare ingestion arguments for MCP
        ingest_args = {"message": message, "source": source}

        # Use provided space_id or get the clinical trials space ID
        if space_id:
            ingest_args["spaceId"] = space_id
        else:
            clinical_trials_space_id = self.get_clinical_trials_space_id()
            if clinical_trials_space_id:
                ingest_args["spaceId"] = clinical_trials_space_id

        # Call memory_ingest tool via MCP
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": "memory_ingest", "arguments": ingest_args},
        }

        try:
            response = self._post(payload, stream=True)
            result = self._parse_rpc(response)
            unwrapped_result = self._unwrap_tool_result(result)
            return unwrapped_result
        except Exception as e:
            raise CoreMemoryError(f"Failed to ingest message: {e}")

    def get_ingestion_status(self, space_id: Optional[str] = None) -> dict:
        """
        Get the current ingestion queue status for a space.

        Args:
            space_id: Optional space ID to check status for

        Returns:
            Dictionary with ingestion status information
        """
        # This is a placeholder - CORE Memory may not expose queue status via MCP
        # In a real implementation, you might need to check the CORE web interface
        # or use a different API endpoint
        return {
            "status": "queued",
            "message": "Ingestion is queued and will be processed asynchronously. Check the CORE web interface for completion status.",
        }

    def search(
        self, query: str, space_id: Optional[str] = None, limit: int = 10
    ) -> dict:
        """
        Search for memories in CORE Memory using MCP protocol (direct API endpoints are pending).

        Args:
            query: Search query
            space_id: Optional space ID to search in
            limit: Maximum number of results to return

        Returns:
            Dictionary with search results containing 'episodes' and 'facts' arrays
        """
        if "memory_search" not in self.tools:
            raise CoreMemoryError("Server missing required tool: memory_search")

        # Prepare search arguments
        search_args = {"query": query, "limit": limit}

        if space_id:
            search_args["spaceId"] = space_id

        # Call memory_search tool
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": "memory_search", "arguments": search_args},
        }

        try:
            response = self._post(payload, stream=False)
            result = self._parse_rpc(response)
            unwrapped_result = self._unwrap_tool_result(result)

            # Handle the actual search result format from CORE Memory
            # The result should be a dict with "episodes" and "facts" arrays
            if isinstance(unwrapped_result, dict):
                return unwrapped_result
            elif isinstance(unwrapped_result, list):
                # Fallback: if we get a list, wrap it in a dict structure
                return {"episodes": unwrapped_result, "facts": []}
            else:
                return {"episodes": [], "facts": []}
        except Exception as e:
            raise CoreMemoryError(f"Failed to search memory: {e}")

    def get_ingestion_logs(
        self,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list:
        """
        Get ingestion logs from CORE Memory (PENDING: Direct API endpoint not yet active).

        This method is currently a placeholder. Ingestion logs may not be exposed via API yet.

        Args:
            space_id: Optional space ID to filter logs
            limit: Maximum number of logs to return (default: 100)
            offset: Offset for pagination (default: 0)
            status: Optional status filter (e.g., 'success', 'failed', 'pending')
            start_date: Optional start date filter (ISO 8601 format)
            end_date: Optional end date filter (ISO 8601 format)

        Returns:
            List of ingestion log entries (currently returns placeholder data)

        Raises:
            CoreMemoryError: If the request fails
        """
        # PENDING: Direct API endpoint not yet active
        # For now, return placeholder data
        return [
            {
                "id": "placeholder_log_1",
                "message": "Ingestion logs API endpoint is pending implementation",
                "source": "system",
                "status": "pending",
                "created_at": "2025-01-01T00:00:00Z",
            }
        ]

    def get_episode_facts(
        self,
        episode_id: Optional[str] = None,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_metadata: bool = True,
    ) -> list:
        """
        Get episode facts from CORE Memory (PENDING: Direct API endpoint not yet active).

        This method is currently a placeholder. Episode facts API may not be exposed yet.

        Args:
            episode_id: Optional episode ID to filter facts
            space_id: Optional space ID to filter facts
            limit: Maximum number of facts to return (default: 100)
            offset: Offset for pagination (default: 0)
            include_metadata: Whether to include metadata in response (default: True)

        Returns:
            List of episode facts (currently returns placeholder data)

        Raises:
            CoreMemoryError: If the request fails
        """
        # PENDING: Direct API endpoint not yet active
        # For now, return placeholder data
        return [
            {
                "id": "placeholder_fact_1",
                "content": "Episode facts API endpoint is pending implementation",
                "episode_id": episode_id or "unknown",
                "confidence": 0.0,
                "metadata": {"status": "pending"},
            }
        ]

    def get_specific_log(self, log_id: str) -> dict:
        """
        Get a specific ingestion log by ID from CORE Memory (PENDING: Direct API endpoint not yet active).

        This method is currently a placeholder.

        Args:
            log_id: The ID of the ingestion log to retrieve

        Returns:
            Dictionary containing the specific log details (currently returns placeholder data)

        Raises:
            CoreMemoryError: If the request fails
        """
        if not log_id:
            raise CoreMemoryError("Log ID is required.")

        # PENDING: Direct API endpoint not yet active
        # For now, return placeholder data
        return {
            "id": log_id,
            "message": "Specific log API endpoint is pending implementation",
            "source": "system",
            "status": "pending",
            "created_at": "2025-01-01T00:00:00Z",
        }

    def delete_log_entry(self, log_id: str) -> dict:
        """
        Delete a specific ingestion log entry from CORE Memory (PENDING: Direct API endpoint not yet active).

        This method is currently a placeholder.

        Args:
            log_id: The ID of the ingestion log entry to delete

        Returns:
            Dictionary containing the deletion confirmation (currently returns placeholder data)

        Raises:
            CoreMemoryError: If the request fails
        """
        if not log_id:
            raise CoreMemoryError("Log ID is required.")

        # PENDING: Direct API endpoint not yet active
        # For now, return placeholder data
        return {
            "id": log_id,
            "status": "deleted",
            "message": "Log deletion API endpoint is pending implementation",
        }

    def get_user_profile(self) -> dict:
        """
        Get the current user's profile from CORE Memory (PENDING: Direct API endpoint not yet active).

        This method is currently a placeholder.

        Returns:
            Dictionary containing user profile information (currently returns placeholder data)

        Raises:
            CoreMemoryError: If the request fails
        """
        # PENDING: Direct API endpoint not yet active
        # For now, return placeholder data
        return {
            "id": "placeholder_user",
            "name": "API User",
            "email": "user@api.example.com",
            "status": "active",
            "message": "User profile API endpoint is pending implementation",
        }

    def get_space_details(self, space_id: str) -> dict:
        """
        Get details of a specific space from CORE Memory.

        Args:
            space_id: The ID of the space to retrieve details for

        Returns:
            Dictionary containing space details

        Raises:
            CoreMemoryError: If the request fails
        """
        if not space_id:
            raise CoreMemoryError("Space ID is required.")

        # Use direct API call (not through MCP tools)
        base_url = self.url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        try:
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise CoreMemoryError(f"Failed to get space details for '{space_id}': {e}")

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """
        Update a space in CORE Memory.

        Note: Space updates may not be supported by the current API version.
        This method attempts the update but may fail with a 400 error.

        Args:
            space_id: The ID of the space to update
            name: Optional new name for the space
            description: Optional new description for the space

        Returns:
            Dictionary containing the updated space information

        Raises:
            CoreMemoryError: If the request fails or space updates are not supported
        """
        if not space_id:
            raise CoreMemoryError("Space ID is required.")

        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description

        if not payload:
            raise CoreMemoryError(
                "At least one field (name or description) must be provided for update."
            )

        # Try PATCH first (more standard for updates)
        base_url = self.url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        try:
            response = requests.patch(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if e.response.status_code == 400:
                # Space updates may not be supported
                raise CoreMemoryError(
                    f"Space updates are not supported by the API (400 Bad Request): {e}"
                )
            elif e.response.status_code == 405:
                # Try PUT if PATCH is not allowed
                try:
                    response = requests.put(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e2:
                    raise CoreMemoryError(
                        f"Failed to update space '{space_id}' with PUT: {e2}"
                    )
            else:
                raise CoreMemoryError(f"Failed to update space '{space_id}': {e}")
        except Exception as e:
            raise CoreMemoryError(f"Failed to update space '{space_id}': {e}")
