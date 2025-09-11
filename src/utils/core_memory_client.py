import os
import uuid
import json
import requests
from typing import Any, Dict, Optional
from dotenv import load_dotenv

JSON = Dict[str, Any]

class CoreMemoryError(Exception):
    """Custom exception for CORE Memory errors."""
    pass

class CoreMemoryClient:
    """A lean client for interacting with the CORE Memory API using MCP protocol."""
    
    def __init__(self, api_key: str, base_url: str = "https://core.heysol.ai/api/v1/mcp", source: str = "Python-Script"):
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
            self.url, 
            json=payload, 
            headers=self._headers(), 
            timeout=60, 
            stream=stream
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise CoreMemoryError(f"HTTP error: {response.status_code} - {response.text}")
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
            }
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
            "params": {}
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
    
    def ingest(self, message: str, space_id: Optional[str] = None, source: str = "Python-Script") -> dict:
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
        ingest_args = {
            "message": message,
            "source": source
        }
        
        if space_id:
            ingest_args["spaceId"] = space_id
        
        # Call memory_ingest tool
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "memory_ingest",
                "arguments": ingest_args
            }
        }
        
        try:
            response = self._post(payload, stream=True)
            result = self._parse_rpc(response)
            unwrapped_result = self._unwrap_tool_result(result)
            return unwrapped_result
        except Exception as e:
            raise CoreMemoryError(f"Failed to ingest message: {e}")
    
    def search(self, query: str, space_id: Optional[str] = None, limit: int = 10) -> dict:
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
        search_args = {
            "query": query,
            "limit": limit
        }
        
        if space_id:
            search_args["spaceId"] = space_id
        
        # Call memory_search tool
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "memory_search",
                "arguments": search_args
            }
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
    