import json
import os
import uuid
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

JSON = Dict[str, Any]


class CoreMemoryError(Exception):
    """Custom exception for CORE Memory errors."""

    pass


class CoreMemoryClient:
    """A lean client for interacting with the CORE Memory API using MCP protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://core.heysol.ai/api/v1/mcp",
        source: str = "Python-Script",
    ):
        """
        Initializes the CoreMemoryClient with MCP protocol support.

        Args:
            api_key: CORE Memory API key
            base_url: MCP server base URL
            source: Source identifier for the requests
        """
        if not api_key:
            raise CoreMemoryError("API key is required.")

        load_dotenv()
        self.api_key = api_key
        self.url = f"{base_url}?source={source}"
        self.session_id: Optional[str] = None
        self.tools: Dict[str, JSON] = {}

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
            self.url, json=payload, headers=self._headers(), timeout=60, stream=stream
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

        # Verify required tools are available
        if "memory_ingest" not in self.tools:
            raise CoreMemoryError("Server missing required tool: memory_ingest")

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
        Get available spaces in CORE Memory.

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
                timeout=60,
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
        Get or create the "mCODE Research Protocols" space ID.

        Returns:
            Space ID for the mCODE research protocols space
        """
        return self.get_or_create_space(
            "mCODE Research Protocols",
            "Space for storing mCODE-aligned clinical trial protocols and eligibility criteria",
        )

    def get_patients_space_id(self) -> str:
        """
        Get or create the "mCODE Patients" space ID.

        Returns:
            Space ID for the mCODE patients space
        """
        return self.get_or_create_space(
            "mCODE Patients", "Space for storing mCODE-compliant patient oncology data"
        )

    def ingest(
        self,
        message: str,
        space_id: Optional[str] = None,
        source: str = "Python-Script",
    ) -> dict:
        """
        Ingests a message into CORE Memory using the proper MCP protocol.

        Args:
            message: The message to ingest
            space_id: Optional space ID to store the message in
            source: Source identifier for the message

        Returns:
            Dictionary with the ingestion result

        Raises:
            CoreMemoryError: If ingestion fails
        """
        if not message:
            raise CoreMemoryError("Message is required for ingestion.")

        # Prepare ingestion arguments
        ingest_args = {"message": message, "source": source}

        # Use provided space_id or get the "Clinical Trials" space ID
        if space_id:
            ingest_args["spaceId"] = space_id
        else:
            clinical_trials_space_id = self.get_clinical_trials_space_id()
            if clinical_trials_space_id:
                ingest_args["spaceId"] = clinical_trials_space_id

        # Call memory_ingest tool
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
        Search for memories in CORE Memory.

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
