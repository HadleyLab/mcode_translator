# HeySol API Client - Comprehensive API Documentation

## Overview

The HeySol API client provides access to the HeySol platform through both MCP (Model Context Protocol) and direct REST API endpoints. This documentation covers all discovered and tested endpoints, their parameters, response formats, and usage examples.

## Base Configuration

- **MCP URL**: `https://core.heysol.ai/api/v1/mcp?source=Kilo-Code`
- **Protocol**: Server-Sent Events (SSE) for MCP, JSON for direct API
- **Authentication**: Bearer token authentication
- **Server**: `core-unified-mcp-server v1.0.0`

## Authentication

All API requests require authentication using a Bearer token:

```python
from heysol.client import HeySolClient

client = HeySolClient(api_key="your-api-key-here")
```

## MCP Tools

The following MCP tools are available:

### 1. memory_ingest
**Purpose**: Store conversation data, insights, and decisions in CORE Memory

**Parameters**:
```json
{
  "message": "The data to ingest in text format"
}
```

**Response**:
```json
{
  "success": true,
  "id": "run_cmftdi5b20ijw34nt21h6v89c"
}
```

**Usage**:
```python
result = client.ingest("Important conversation data to store")
```

### 2. memory_search
**Purpose**: Search memory for relevant project context, user preferences, and previous discussions

**Parameters**:
```json
{
  "query": "The search query in third person perspective",
  "validAt": "The valid at time in ISO format (optional)",
  "startTime": "The start time in ISO format (optional)",
  "endTime": "The end time in ISO format (optional)",
  "spaceIds": ["Array of strings representing UUIDs of spaces (optional)"]
}
```

**Response**:
```json
{
  "episodes": [],
  "facts": []
}
```

**Usage**:
```python
result = client.search("project requirements and user preferences", limit=10)
```

### 3. memory_get_spaces
**Purpose**: Retrieve list of memory organization spaces

**Parameters**:
```json
{
  "all": "Get all spaces (optional, default: true)"
}
```

**Response**:
```json
{
  "spaces": [
    {
      "id": "space-uuid",
      "name": "Space Name",
      "description": "Space Description"
    }
  ]
}
```

**Usage**:
```python
result = client.get_spaces()
```

### 4. get_user_profile
**Purpose**: Retrieve user profile and preferences for personalized interactions

**Parameters**:
```json
{
  "profile": "Get user profile (optional, default: true)"
}
```

**Response**:
```json
{
  "user_id": "user-uuid",
  "email": "user@example.com",
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}
```

**Usage**:
```python
result = client.get_user_profile()
```

## Direct API Endpoints

### Memory Endpoints

#### Search Knowledge Graph
- **Endpoint**: `/memory/search`
- **Method**: GET
- **Parameters**:
  - `query` (string): Search query
  - `limit` (integer, optional): Result limit (default: 10)
  - `offset` (integer, optional): Result offset (default: 0)
- **Response**: Search results with episodes and facts

#### Add Data to Ingestion Queue
- **Endpoint**: `/memory/ingest`
- **Method**: POST
- **Parameters**:
  - `message` (string): Data to ingest
- **Response**: Ingestion confirmation with ID

#### Get Episode Facts
- **Endpoint**: `/memory/episodes/{episode_id}/facts`
- **Method**: GET
- **Parameters**:
  - `limit` (integer, optional): Facts limit (default: 100)
  - `offset` (integer, optional): Facts offset (default: 0)
  - `include_metadata` (boolean, optional): Include metadata (default: true)
- **Response**: Episode facts and metadata

#### Get Ingestion Logs
- **Endpoint**: `/memory/logs`
- **Method**: GET
- **Parameters**:
  - `limit` (integer, optional): Logs limit (default: 50)
  - `offset` (integer, optional): Logs offset (default: 0)
- **Response**: List of ingestion logs

#### Get Specific Log
- **Endpoint**: `/memory/logs/{log_id}`
- **Method**: GET
- **Response**: Detailed log information

### Space Endpoints

#### Bulk Space Operations
- **Endpoint**: `/spaces/bulk`
- **Method**: PUT
- **Parameters**:
  - `operations` (array): Array of space operations
    - `action` (string): "create", "update", or "delete"
    - `name` (string): Space name (for create)
    - `id` (string): Space ID (for update/delete)
- **Response**: Operation results with created/updated/deleted counts

#### Get Space Details
- **Endpoint**: `/spaces/{space_id}/details`
- **Method**: GET
- **Parameters**:
  - `include_stats` (boolean, optional): Include statistics (default: true)
  - `include_metadata` (boolean, optional): Include metadata (default: true)
- **Response**: Space details with stats and metadata

#### Update Space
- **Endpoint**: `/spaces/{space_id}`
- **Method**: PUT
- **Parameters**:
  - `name` (string, optional): New space name
  - `description` (string, optional): New description
- **Response**: Updated space information

#### Delete Space
- **Endpoint**: `/spaces/{space_id}`
- **Method**: DELETE
- **Parameters**:
  - `confirm` (boolean): Must be true to confirm deletion
- **Response**: Deletion confirmation

### OAuth2 Endpoints

#### Get OAuth2 Authorization URL
- **Endpoint**: `/oauth2/authorize`
- **Method**: GET
- **Parameters**:
  - `client_id` (string): OAuth2 client ID
  - `redirect_uri` (string): Redirect URI
  - `state` (string, optional): State parameter
- **Response**: Authorization URL and state

#### OAuth2 Token Exchange
- **Endpoint**: `/oauth2/token`
- **Method**: POST
- **Parameters**:
  - `grant_type` (string): "authorization_code"
  - `code` (string): Authorization code
  - `client_id` (string): Client ID
  - `client_secret` (string): Client secret
- **Response**: Access token, refresh token, and metadata

#### OAuth2 Refresh Token
- **Endpoint**: `/oauth2/token`
- **Method**: POST
- **Parameters**:
  - `grant_type` (string): "refresh_token"
  - `refresh_token` (string): Refresh token
  - `client_id` (string): Client ID
  - `client_secret` (string): Client secret
- **Response**: New access token and metadata

#### OAuth2 Revoke Token
- **Endpoint**: `/oauth2/revoke`
- **Method**: POST
- **Parameters**:
  - `token` (string): Token to revoke
  - `client_id` (string): Client ID
  - `client_secret` (string): Client secret
- **Response**: Revocation confirmation

### Webhook Endpoints

#### Register Webhook
- **Endpoint**: `/webhooks`
- **Method**: POST
- **Parameters**:
  - `url` (string): Webhook URL
  - `events` (array): Array of event types
- **Response**: Webhook registration confirmation

#### List Webhooks
- **Endpoint**: `/webhooks`
- **Method**: GET
- **Parameters**:
  - `limit` (integer, optional): Results limit (default: 100)
  - `offset` (integer, optional): Results offset (default: 0)
- **Response**: List of registered webhooks

#### Delete Webhook
- **Endpoint**: `/webhooks/{webhook_id}`
- **Method**: DELETE
- **Parameters**:
  - `confirm` (boolean): Must be true to confirm deletion
- **Response**: Deletion confirmation

## Error Handling

### Common HTTP Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Invalid or missing authentication
- **404 Not Found**: Resource not found
- **408 Request Timeout**: Request timed out
- **500 Internal Server Error**: Server error

### Error Response Format

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

### Exception Types

- `HeySolError`: General API error
- `ValidationError`: Input validation error
- `AuthenticationError`: Authentication failure

## Usage Examples

### Basic Memory Operations

```python
from heysol.client import HeySolClient

client = HeySolClient(api_key="your-api-key")

# Search memory
results = client.search("project requirements", limit=10)

# Ingest data
result = client.ingest("Important project decision: use React for frontend")

# Get user profile
profile = client.get_user_profile()
```

### Space Management

```python
# Get space details
space = client.get_space_details("space-123")

# Update space
updated = client.update_space("space-123", {
    "name": "Updated Space Name",
    "description": "Updated description"
})

# Bulk operations
operations = [
    {"action": "create", "name": "New Space"},
    {"action": "update", "id": "space-456", "name": "Updated Space"}
]
results = client.bulk_space_operations(operations)
```

### Webhook Management

```python
# Register webhook
webhook = client.register_webhook(
    "https://example.com/webhook",
    ["memory.ingested", "memory.searched"]
)

# List webhooks
webhooks = client.list_webhooks()

# Delete webhook
client.delete_webhook("webhook-123", confirm=True)
```

## Performance Considerations

- **Response Time**: API responses typically under 2 seconds
- **Rate Limiting**: Monitor for rate limit headers
- **Timeout**: Default timeout is 30 seconds
- **Retries**: Implement exponential backoff for failed requests

## Security Best Practices

- Store API keys securely (environment variables recommended)
- Use HTTPS for all requests
- Validate SSL certificates
- Implement proper error handling
- Log errors without exposing sensitive data
- Rotate API keys regularly

## Testing

The API client includes comprehensive test suites:

```bash
# Run MCP URL tests
python tests/test_mcp_correct_url.py

# Run comprehensive tests
python tests/test_simple_comprehensive.py

# Run all tests
python -m pytest tests/
```

## Troubleshooting

### Common Issues

1. **404 Errors**: Check endpoint permissions and authentication
2. **Timeout Errors**: Increase timeout or check network connectivity
3. **Authentication Errors**: Verify API key validity
4. **MCP Connection Issues**: Check MCP URL and server availability

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For API issues or questions:
- Check the MCP server status
- Verify API key permissions
- Review error logs
- Test with minimal examples first

---

*This documentation is based on discovered endpoints and tested functionality as of the current implementation.*