"""
Main HeySol API client implementation.

This module contains the main HeySolClient class that provides access to all
HeySol API functionality including MCP protocol support, authentication,
memory management, and robust error handling.
"""

import json
import uuid
import time
import logging
from typing import Any, Dict, Optional, Union, List
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HeySolConfig
from .exceptions import (
    HeySolError,
    AuthenticationError,
    RateLimitError,
    APIError,
    ValidationError,
    ConnectionError,
    ServerError,
    NotFoundError,
)


class HeySolClient:
    """
    Main client for interacting with the HeySol API.

    This client provides a comprehensive interface to the HeySol API with support
    for MCP protocol, authentication, memory management, rate limiting, and
    robust error handling with retry mechanisms.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[HeySolConfig] = None,
        base_url: Optional[str] = None,
        source: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        oauth2_client_id: Optional[str] = None,
        oauth2_client_secret: Optional[str] = None,
        oauth2_redirect_uri: Optional[str] = None,
        use_oauth2: bool = False,
    ):
        """
        Initialize the HeySol API client.

        Args:
            api_key: HeySol API key (optional, can be loaded from config)
            config: HeySolConfig instance (optional)
            base_url: Base URL for the API (optional, can be loaded from config)
            source: Source identifier for requests (optional, can be loaded from config)
            timeout: Request timeout in seconds (optional, can be loaded from config)
            max_retries: Maximum number of retries (optional, can be loaded from config)
            oauth2_client_id: OAuth2 client ID (optional)
            oauth2_client_secret: OAuth2 client secret (optional)
            oauth2_redirect_uri: OAuth2 redirect URI (optional)
            use_oauth2: Whether to use OAuth2 authentication instead of API key

        Raises:
            AuthenticationError: If no authentication method is provided
        """
        # Load configuration
        if config is None:
            config = HeySolConfig.from_env()

        # Override config values with provided parameters
        if api_key is not None:
            config.api_key = api_key
        if base_url is not None:
            config.base_url = base_url
        if source is not None:
            config.source = source
        if timeout is not None:
            config.timeout = timeout
        if max_retries is not None:
            config.max_retries = max_retries
        if oauth2_client_id is not None:
            config.oauth2_client_id = oauth2_client_id
        if oauth2_client_secret is not None:
            config.oauth2_client_secret = oauth2_client_secret
        if oauth2_redirect_uri is not None:
            config.oauth2_redirect_uri = oauth2_redirect_uri

        self.config = config
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Any] = {}
        self._rate_limit_remaining = config.rate_limit_per_minute
        self._rate_limit_reset_time = time.time() + 60  # Reset every minute
        self._request_count = 0

        # OAuth2 setup
        self.use_oauth2 = use_oauth2 or bool(config.oauth2_client_id and config.oauth2_client_secret)
        self.oauth2_auth: Optional[OAuth2Authenticator] = None

        # Validate required configuration
        if not self.use_oauth2 and not config.api_key:
            raise AuthenticationError(
                "API key is required when not using OAuth2. Provide it as a parameter or set COREAI_API_KEY environment variable."
            )

        if self.use_oauth2 and not config.oauth2_client_id:
            raise AuthenticationError(
                "OAuth2 client ID is required for OAuth2 authentication. Set COREAI_OAUTH2_CLIENT_ID environment variable."
            )

        # Setup logging
        self._setup_logging()

        # Setup HTTP session with retry strategy
        self._setup_http_session()

        # Initialize authentication
        self._initialize_authentication()

        # Initialize MCP session
        self._initialize_session()

        self.logger.info("HeySol API client initialized successfully")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)

        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.config.log_to_file and self.config.log_file_path:
            file_handler = logging.FileHandler(self.config.log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _setup_http_session(self) -> None:
        """Setup HTTP session with retry strategy and timeouts."""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
            backoff_factor=self.config.backoff_factor,
        )

        # Create adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default timeout
        self.session.timeout = self.config.timeout

    def _initialize_authentication(self) -> None:
        """Initialize authentication method (API key or OAuth2)."""
        if self.use_oauth2:
            try:
                # Import OAuth2 classes locally to avoid circular imports
                from .oauth2 import OAuth2Authenticator, OAuth2ClientCredentialsAuthenticator, OAuth2Error

                if self.config.oauth2_client_secret:
                    # Use client credentials flow
                    self.oauth2_auth = OAuth2ClientCredentialsAuthenticator(self.config)
                    self.oauth2_auth.authenticate()
                    self.logger.info("OAuth2 client credentials authentication initialized")
                else:
                    # Use authorization code flow
                    self.oauth2_auth = OAuth2Authenticator(self.config)
                    self.logger.info("OAuth2 authorization code flow initialized (tokens will be obtained on first request)")
            except Exception as e:
                raise AuthenticationError(f"OAuth2 initialization failed: {e}")
        else:
            self.logger.info("API key authentication initialized")

    def _headers(self) -> Dict[str, str]:
        """Create headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "User-Agent": f"heysol-python-client/{self.config.source}",
        }

        # Set authorization header based on authentication method
        if self.use_oauth2 and self.oauth2_auth:
            try:
                headers["Authorization"] = self.oauth2_auth.get_authorization_header()
            except Exception as e:
                raise AuthenticationError(f"OAuth2 authentication failed: {e}")
        else:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        return headers

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        if not self.config.rate_limit_enabled:
            return

        current_time = time.time()

        # Reset rate limit counter if a minute has passed
        if current_time >= self._rate_limit_reset_time:
            self._rate_limit_remaining = self.config.rate_limit_per_minute
            self._rate_limit_reset_time = current_time + 60

        if self._rate_limit_remaining <= 0:
            reset_in = int(self._rate_limit_reset_time - current_time)
            raise RateLimitError(
                f"Rate limit exceeded. Resets in {reset_in} seconds.",
                retry_after=reset_in
            )

        self._rate_limit_remaining -= 1

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """
        Make an HTTP request with error handling and retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint (relative to base URL)
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response

        Returns:
            requests.Response: The HTTP response object

        Raises:
            HeySolError: For various API and network errors
        """
        # Fix URL construction to include source parameter like working client
        url = urljoin(self.config.base_url, endpoint.lstrip("/"))

        # Add source parameter to URL if not already present (like working client)
        if "?" not in url and self.config.source:
            url += f"?source={self.config.source}"
        elif self.config.source:
            # If URL already has query parameters, append source parameter
            url += f"&source={self.config.source}"

        # Check rate limit
        self._check_rate_limit()

        try:
            self.logger.debug(f"Making {method} request to {url}")

            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=self._headers(),
                timeout=self.config.timeout,
                stream=stream,
            )

            # Handle different HTTP status codes
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key or authentication failed")
            elif response.status_code == 404:
                raise NotFoundError("Requested resource not found")
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                retry_after = int(retry_after) if retry_after else None
                raise RateLimitError("Rate limit exceeded", retry_after=retry_after)
            elif response.status_code >= 500:
                raise ServerError(f"Server error: {response.status_code}")
            elif response.status_code >= 400:
                raise APIError(f"Client error: {response.status_code}")

            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            raise ConnectionError("Request timeout")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Connection error")
        except requests.exceptions.HTTPError as e:
            raise APIError(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Request failed: {e}")

    def _parse_json_response(self, response: requests.Response) -> Any:
        """
        Parse JSON response from the API.

        Args:
            response: HTTP response object

        Returns:
            Parsed JSON data

        Raises:
            APIError: If response cannot be parsed as JSON
        """
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise APIError(f"Failed to parse JSON response: {e}")

    def _parse_mcp_response(self, response: requests.Response) -> Any:
        """
        Parse MCP protocol response.

        Args:
            response: HTTP response object

        Returns:
            Parsed MCP response data

        Raises:
            HeySolError: If MCP response cannot be parsed or contains errors
        """
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()

        if content_type == "application/json":
            data = self._parse_json_response(response)
        elif content_type == "text/event-stream":
            # Handle Server-Sent Events (SSE) like working client
            data = None
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        break
                    except json.JSONDecodeError:
                        continue

            if data is None:
                raise APIError("No JSON data found in SSE stream")
        else:
            raise APIError(f"Unexpected Content-Type: {content_type}")

        # Check for MCP error
        if isinstance(data, dict) and "error" in data:
            raise APIError(f"MCP error: {data['error']}")

        # Extract result from MCP response
        if isinstance(data, dict) and "result" in data:
            return data["result"]

        return data

    def _initialize_session(self) -> None:
        """Initialize MCP session and list available tools."""
        self.logger.info("Initializing MCP session")

        # Initialize session
        init_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": True},
                "clientInfo": {
                    "name": "heysol-python-client",
                    "version": "1.0.0"
                },
            },
        }

        try:
            response = self._make_request("POST", "", data=init_payload)
            self.session_id = response.headers.get("Mcp-Session-Id")

            if self.session_id:
                self.logger.info(f"MCP session initialized with ID: {self.session_id}")
            else:
                self.logger.warning("No session ID returned from server")

        except Exception as e:
            raise APIError(f"Failed to initialize MCP session: {e}")

        # List available tools
        tools_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }

        try:
            response = self._make_request("POST", "", data=tools_payload)
            result = self._parse_mcp_response(response)

            if isinstance(result, dict) and "tools" in result:
                self.tools = {tool["name"]: tool for tool in result["tools"]}
                self.logger.info(f"Available tools: {list(self.tools.keys())}")
            else:
                self.logger.warning("No tools returned from server")

        except Exception as e:
            self.logger.warning(f"Failed to list MCP tools: {e}")

    def _call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            stream: Whether to stream the response

        Returns:
            Tool call result

        Raises:
            APIError: If tool is not available or call fails
        """
        if tool_name not in self.tools:
            raise APIError(f"Tool '{tool_name}' is not available")

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            },
        }

        response = self._make_request("POST", "", data=payload, stream=stream)
        return self._parse_mcp_response(response)

    def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get available memory spaces.

        Returns:
            List of available spaces

        Raises:
            HeySolError: If the request fails
        """
        try:
            result = self._call_tool("memory_get_spaces")

            # Handle MCP response format - extract spaces from nested structure
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        try:
                            # Parse JSON string from MCP response
                            spaces_data = json.loads(text_content)
                            if isinstance(spaces_data, list):
                                return spaces_data
                            elif isinstance(spaces_data, dict) and "spaces" in spaces_data:
                                return spaces_data["spaces"]
                        except json.JSONDecodeError:
                            continue

            # Fallback: return empty list if parsing fails
            self.logger.warning("Could not parse spaces response, returning empty list")
            return []

        except APIError as e:
            raise HeySolError(f"Failed to get spaces: {e}")

    def create_space(self, name: str, description: str = "") -> str:
        """
        Create a new memory space.

        Args:
            name: Name of the space to create
            description: Optional description for the space

        Returns:
            Space ID of the created space

        Raises:
            HeySolError: If space creation fails
        """
        if not name:
            raise ValidationError("Space name is required")

        # Use direct API call for space creation
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces"

        payload = {"name": name, "description": description}

        try:
            response = self._make_request("POST", url, data=payload)
            data = self._parse_json_response(response)

            space_id = data.get("space", {}).get("id") or data.get("id")
            if not space_id:
                raise APIError(f"Unexpected create_space response: {data}")

            self.logger.info(f"Created space '{name}' with ID: {space_id}")
            return space_id

        except Exception as e:
            raise HeySolError(f"Failed to create space '{name}': {e}")

    def get_or_create_space(self, name: str, description: str = "") -> str:
        """
        Get existing space or create a new one.

        Args:
            name: Name of the space
            description: Description for new space if created

        Returns:
            Space ID

        Raises:
            HeySolError: If the operation fails
        """
        try:
            spaces = self.get_spaces()

            # Look for existing space
            for space in spaces:
                if space.get("name") == name and space.get("writable", True):
                    space_id = space.get("id") or space.get("spaceId")
                    if space_id:
                        self.logger.info(f"Found existing space '{name}' with ID: {space_id}")
                        return space_id

            # Create new space if not found
            self.logger.info(f"Space '{name}' not found, creating new space")
            return self.create_space(name, description)

        except Exception as e:
            raise HeySolError(f"Failed to get or create space '{name}': {e}")

    def ingest(
        self,
        message: str,
        space_id: Optional[str] = None,
        source: Optional[str] = None,
        priority: str = "normal",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest data into CORE Memory.

        Args:
            message: The message to ingest
            space_id: Optional space ID to store the message in
            source: Source identifier for the message
            priority: Priority level ('low', 'normal', 'high')
            tags: Optional list of tags for the ingestion

        Returns:
            Dictionary with the ingestion result

        Raises:
            HeySolError: If ingestion fails
        """
        if not message:
            raise ValidationError("Message is required for ingestion")

        # Prepare ingestion arguments
        ingest_args = {
            "message": message,
            "source": source or self.config.source,
            "priority": priority,
        }

        if tags:
            ingest_args["tags"] = tags

        # Use provided space_id or get the default space ID
        if space_id:
            ingest_args["spaceId"] = space_id
        else:
            # Use clinical trials space as default
            try:
                clinical_trials_space_id = self.get_or_create_space(
                    self.config.default_spaces.get("clinical_trials", "Clinical Trials")
                )
                ingest_args["spaceId"] = clinical_trials_space_id
            except Exception:
                # Continue without space_id if default space creation fails
                self.logger.warning("Could not determine default space ID, proceeding without it")

        try:
            result = self._call_tool("memory_ingest", ingest_args, stream=True)
            self.logger.info("Successfully ingested message into CORE Memory")
            return result

        except Exception as e:
            raise HeySolError(f"Failed to ingest message: {e}")

    def search(
        self,
        query: str,
        space_id: Optional[str] = None,
        limit: int = 10,
        valid_at: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories in CORE Memory.

        Args:
            query: Search query
            space_id: Optional space ID to search in
            limit: Maximum number of results to return
            valid_at: Optional timestamp for time-based search
            start_time: Optional start time for time range search
            end_time: Optional end time for time range search

        Returns:
            Dictionary with search results containing 'episodes' and 'facts' arrays

        Raises:
            HeySolError: If search fails
        """
        if not query:
            raise ValidationError("Search query is required")

        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")

        # Prepare search arguments
        search_args = {
            "query": query,
            "limit": limit,
        }

        if space_id:
            search_args["spaceId"] = space_id
        if valid_at:
            search_args["validAt"] = valid_at
        if start_time:
            search_args["startTime"] = start_time
        if end_time:
            search_args["endTime"] = end_time

        try:
            result = self._call_tool("memory_search", search_args)

            # Ensure result has expected structure
            if not isinstance(result, dict):
                result = {"episodes": [], "facts": []}

            if "episodes" not in result:
                result["episodes"] = []
            if "facts" not in result:
                result["facts"] = []

            self.logger.info(f"Search completed. Found {len(result['episodes'])} episodes and {len(result['facts'])} facts")
            return result

        except Exception as e:
            raise HeySolError(f"Failed to search memory: {e}")

    def get_user_profile(self) -> Dict[str, Any]:
        """
        Get the current user's profile.

        Returns:
            Dictionary containing user profile information

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for user profile
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/profile"

        try:
            response = self._make_request("GET", url)
            profile = self._parse_json_response(response)
            self.logger.info("Successfully retrieved user profile")
            return profile

        except Exception as e:
            raise HeySolError(f"Failed to get user profile: {e}")

    def get_ingestion_logs(
        self,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get ingestion logs from CORE Memory.

        Args:
            space_id: Optional space ID to filter logs
            limit: Maximum number of logs to return (default: 100)
            offset: Offset for pagination (default: 0)
            status: Optional status filter (e.g., 'success', 'failed', 'pending')
            start_date: Optional start date filter (ISO 8601 format)
            end_date: Optional end date filter (ISO 8601 format)

        Returns:
            List of ingestion log entries

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for log retrieval
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/logs"

        params = {
            "limit": limit,
            "offset": offset,
        }

        if space_id:
            params["space_id"] = space_id
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            response = self._make_request("GET", url, params=params)
            logs = self._parse_json_response(response)
            return logs if isinstance(logs, list) else []
        except Exception as e:
            raise HeySolError(f"Failed to get ingestion logs: {e}")

    def get_specific_log(self, log_id: str) -> Dict[str, Any]:
        """
        Get a specific ingestion log by ID.

        Args:
            log_id: The ID of the ingestion log to retrieve

        Returns:
            Dictionary containing the specific log details

        Raises:
            HeySolError: If the request fails
        """
        if not log_id:
            raise ValidationError("Log ID is required")

        # Use direct API call for specific log retrieval
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/logs/{log_id}"

        try:
            response = self._make_request("GET", url)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to get log {log_id}: {e}")

    def delete_log_entry(self, log_id: str) -> Dict[str, Any]:
        """
        Delete a specific ingestion log entry.

        Args:
            log_id: The ID of the ingestion log entry to delete

        Returns:
            Dictionary containing the deletion confirmation

        Raises:
            HeySolError: If the request fails
        """
        if not log_id:
            raise ValidationError("Log ID is required")

        # Use direct API call for log deletion
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/logs/{log_id}"

        try:
            response = self._make_request("DELETE", url)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to delete log {log_id}: {e}")

    def bulk_space_operations(
        self,
        operations: List[Dict[str, Any]],
        continue_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform bulk operations on spaces.

        Args:
            operations: List of space operations to perform
            continue_on_error: Whether to continue if individual operations fail

        Returns:
            Dictionary with bulk operation results

        Raises:
            HeySolError: If the request fails
        """
        if not operations:
            raise ValidationError("Operations list cannot be empty")

        # Use direct API call for bulk space operations
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/bulk"

        payload = {
            "operations": operations,
            "continue_on_error": continue_on_error,
        }

        try:
            response = self._make_request("POST", url, data=payload)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to perform bulk space operations: {e}")

    def get_space_details(self, space_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific space.

        Args:
            space_id: The ID of the space to retrieve details for

        Returns:
            Dictionary containing space details

        Raises:
            HeySolError: If the request fails
        """
        if not space_id:
            raise ValidationError("Space ID is required")

        # Use direct API call for space details
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        try:
            response = self._make_request("GET", url)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to get space details for '{space_id}': {e}")

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update a space's properties.

        Args:
            space_id: The ID of the space to update
            name: Optional new name for the space
            description: Optional new description for the space
            metadata: Optional metadata updates

        Returns:
            Dictionary containing the updated space information

        Raises:
            HeySolError: If the request fails
        """
        if not space_id:
            raise ValidationError("Space ID is required")

        if not any([name, description, metadata]):
            raise ValidationError("At least one field must be provided for update")

        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if metadata is not None:
            payload["metadata"] = metadata

        # Use direct API call for space update
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        try:
            response = self._make_request("PATCH", url, data=payload)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to update space '{space_id}': {e}")

    def delete_space(self, space_id: str) -> Dict[str, Any]:
        """
        Delete a space and all its contents.

        Args:
            space_id: The ID of the space to delete

        Returns:
            Dictionary containing the deletion confirmation

        Raises:
            HeySolError: If the request fails
        """
        if not space_id:
            raise ValidationError("Space ID is required")

        # Use direct API call for space deletion
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        try:
            response = self._make_request("DELETE", url)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to delete space '{space_id}': {e}")

    def add_data_to_ingestion_queue(
        self,
        data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        space_id: Optional[str] = None,
        priority: str = "normal",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add data to the ingestion queue for processing.

        Args:
            data: Data to ingest (string, dict, or list of dicts)
            space_id: Optional space ID to store the data in
            priority: Priority level ('low', 'normal', 'high')
            tags: Optional list of tags for the data
            metadata: Optional metadata for the ingestion

        Returns:
            Dictionary with ingestion queue result

        Raises:
            HeySolError: If ingestion fails
        """
        # Use direct API call for queue ingestion
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/ingestion/queue"

        payload = {
            "data": data,
            "priority": priority,
        }

        if space_id:
            payload["space_id"] = space_id
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        try:
            response = self._make_request("POST", url, data=payload)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to add data to ingestion queue: {e}")

    def get_episode_facts(
        self,
        episode_id: Optional[str] = None,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get episode facts from CORE Memory.

        Args:
            episode_id: Optional episode ID to filter facts
            space_id: Optional space ID to filter facts
            limit: Maximum number of facts to return (default: 100)
            offset: Offset for pagination (default: 0)
            include_metadata: Whether to include metadata in response

        Returns:
            List of episode facts

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for episode facts
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/episodes/facts"

        params = {
            "limit": limit,
            "offset": offset,
            "include_metadata": include_metadata,
        }

        if episode_id:
            params["episode_id"] = episode_id
        if space_id:
            params["space_id"] = space_id

        try:
            response = self._make_request("GET", url, params=params)
            facts = self._parse_json_response(response)
            return facts if isinstance(facts, list) else []
        except Exception as e:
            raise HeySolError(f"Failed to get episode facts: {e}")

    def search_knowledge_graph(
        self,
        query: str,
        space_id: Optional[str] = None,
        limit: int = 10,
        depth: int = 2,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Search the knowledge graph for related concepts and entities.

        Args:
            query: Search query for knowledge graph
            space_id: Optional space ID to search in
            limit: Maximum number of results to return
            depth: Depth of graph traversal (1-5)
            include_metadata: Whether to include metadata in results

        Returns:
            Dictionary with knowledge graph search results

        Raises:
            HeySolError: If search fails
        """
        if not query:
            raise ValidationError("Search query is required")

        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")

        if depth < 1 or depth > 5:
            raise ValidationError("Depth must be between 1 and 5")

        # Use direct API call for knowledge graph search
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/knowledge-graph/search"

        params = {
            "q": query,
            "limit": limit,
            "depth": depth,
            "include_metadata": include_metadata,
        }

        if space_id:
            params["space_id"] = space_id

        try:
            response = self._make_request("GET", url, params=params)
            return self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to search knowledge graph: {e}")

    def authorize_oauth2_interactive(self, scope: str = "openid profile email api") -> bool:
        """
        Perform interactive OAuth2 authorization.

        Args:
            scope: OAuth2 scope for authorization

        Returns:
            True if authorization successful

        Raises:
            HeySolError: If authorization fails
        """
        if not self.use_oauth2:
            raise HeySolError("OAuth2 is not enabled for this client")

        if not self.oauth2_auth:
            raise HeySolError("OAuth2 authenticator not initialized")

        try:
            self.oauth2_auth.authorize_interactive(scope=scope)
            self.logger.info("OAuth2 authorization completed successfully")
            return True
        except Exception as e:
            raise HeySolError(f"OAuth2 authorization failed: {e}")

    def get_oauth2_user_info(self) -> Dict[str, Any]:
        """
        Get OAuth2 user information.

        Returns:
            User information dictionary

        Raises:
            HeySolError: If user info retrieval fails
        """
        if not self.use_oauth2 or not self.oauth2_auth:
            raise HeySolError("OAuth2 is not enabled for this client")

        try:
            return self.oauth2_auth.get_user_info()
        except Exception as e:
            raise HeySolError(f"Failed to get user info: {e}")

    def introspect_oauth2_token(self, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Introspect OAuth2 token.

        Args:
            token: Token to introspect (uses current token if None)

        Returns:
            Token introspection data

        Raises:
            HeySolError: If introspection fails
        """
        if not self.use_oauth2 or not self.oauth2_auth:
            raise HeySolError("OAuth2 is not enabled for this client")

        try:
            return self.oauth2_auth.introspect_token(token)
        except Exception as e:
            raise HeySolError(f"Token introspection failed: {e}")

    def refresh_oauth2_token(self) -> bool:
        """
        Refresh OAuth2 access token.

        Returns:
            True if refresh successful

        Raises:
            HeySolError: If refresh fails
        """
        if not self.use_oauth2 or not self.oauth2_auth:
            raise HeySolError("OAuth2 is not enabled for this client")

        try:
            self.oauth2_auth.refresh_access_token()
            self.logger.info("OAuth2 token refreshed successfully")
            return True
        except Exception as e:
            raise HeySolError(f"Token refresh failed: {e}")

    def authorize_oauth2_interactive(
        self,
        scope: str = "openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email"
    ) -> bool:
        """
        Perform interactive OAuth2 authorization with browser.

        Args:
            scope: OAuth2 scope for authorization

        Returns:
            True if authorization successful

        Raises:
            HeySolError: If authorization fails
        """
        if not self.use_oauth2:
            raise HeySolError("OAuth2 is not enabled for this client")

        try:
            # Import OAuth2 classes locally to avoid circular imports
            from .oauth2 import InteractiveOAuth2Authenticator

            # Create interactive OAuth2 authenticator
            interactive_auth = InteractiveOAuth2Authenticator(self.config)

            def progress_callback(message: str):
                self.logger.info(f"OAuth2 Progress: {message}")

            # Perform interactive authorization
            tokens = interactive_auth.authorize_interactive(
                scope=scope,
                on_progress=progress_callback
            )

            # Store tokens in the client
            if hasattr(self, 'oauth2_auth') and self.oauth2_auth:
                self.oauth2_auth.tokens = tokens

            self.logger.info("Interactive OAuth2 authorization completed successfully")
            return True

        except Exception as e:
            raise HeySolError(f"Interactive OAuth2 authorization failed: {e}")

    def close(self) -> None:
        """Close the HTTP session and clean up resources."""
        if hasattr(self, "session"):
            self.session.close()
            self.logger.info("HeySol API client session closed")