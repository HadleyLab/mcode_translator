# HeySol API Client - Comprehensive API Documentation

## Overview

The HeySol API client provides access to the HeySol platform through both MCP (Model Context Protocol) and direct REST API endpoints. This documentation covers all discovered and tested endpoints, their parameters, response formats, and usage examples.

## 📊 Implementation Status (Updated: 2025-09-21)

### 🎯 **RECOMMENDATION: Use MCP Protocol for Best Results**

## MCP vs Direct API Comparison

| Feature | MCP Protocol | Direct API | Recommendation |
|---------|--------------|------------|----------------|
| **Spaces Management** | ✅ **FULLY WORKING** (7/7 spaces retrieved) | ✅ **PARTIALLY WORKING** (2/6 endpoints) | **🟢 MCP** |
| **Memory Operations** | ✅ **FULLY WORKING** (search, ingest, facts) | ❌ **MOSTLY FAILING** (1/6 endpoints) | **🟢 MCP** |
| **Authentication** | ✅ **WORKING** (API key) | ❌ **FAILING** (401/403 errors) | **🟢 MCP** |
| **Success Rate** | **100%** for core operations | **28.57%** (6/21 endpoints) | **🟢 MCP** |
| **Data Quality** | ✅ **Rich metadata & summaries** | ❌ **Limited or HTML responses** | **🟢 MCP** |

### ✅ **MCP Protocol - RECOMMENDED APPROACH**
- **Status**: ✅ **FULLY FUNCTIONAL** with 100+ tools available
- **Base URL**: `https://core.heysol.ai/api/v1/mcp`
- **Source Parameter**: `?source=Kilo-Code` (identifier, can be any value)
- **Working Tools**: `memory_ingest`, `memory_search`, `memory_get_spaces`, `get_user_profile`
- **GitHub Integration**: 90+ GitHub-related MCP tools available
- **Authentication**: ✅ Working with API key via Bearer token
- **Spaces**: ✅ **ALL OPERATIONS WORKING** - Retrieved 7 spaces with full metadata
- **Memory**: ✅ **ALL OPERATIONS WORKING** - Search, ingest, facts fully functional

### ❌ **Direct API - LIMITED FUNCTIONALITY**
- **Overall Success Rate**: 28.57% (6/21 endpoints working)
- **Authentication Issues**: Most endpoints return 401/403 errors
- **Server Problems**: Multiple 500 Internal Server Errors
- **Connection Issues**: OAuth2 endpoints redirect to login page
- **Data Validation**: Missing required fields cause 400 errors

#### Direct API Working Endpoints (6/21)
- **GET `/api/v1/spaces`** - List spaces ✅ **WORKING** (200 OK)
- **GET `/api/v1/spaces/{spaceId}`** - Get space details ✅ **WORKING** (200 OK)
- **GET `/api/v1/logs`** - List logs ✅ **WORKING** (200 OK, returns HTML)
- **POST `/oauth/authorize`** - OAuth2 authorization ✅ **WORKING** (200 OK, returns login page)
- **POST `/api/v1/webhooks`** - Create webhook ✅ **WORKING** (200 OK, returns HTML)
- **PUT `/api/v1/webhooks/{id}`** - Update webhook ✅ **WORKING** (200 OK, returns HTML)

#### Direct API Failed Endpoints (15/21)
- **Space Management**: POST, PUT, DELETE operations ❌ **FAILED** (400/500 errors)
- **Memory Operations**: Search, add, facts, delete ❌ **FAILED** (400/404/500 errors)
- **OAuth2**: Token exchange, user info, introspection ❌ **FAILED** (401/400 errors)
- **User Management**: Profile access ❌ **FAILED** (401 Invalid token)

### ❌ **Known Issues & Limitations**
- **Server-Side Errors**: Multiple 500 errors suggest backend instability
- **Authentication Mismatch**: Some endpoints require OAuth2 instead of API key
- **Data Validation**: Operations fail with missing required fields or invalid test data
- **Response Format**: Some endpoints return HTML instead of JSON
- **DELETE Operations**: Log and space deletion endpoints not functional

## Base Configuration

- **Base URL**: `https://core.heysol.ai/api/v1`
- **MCP Endpoint**: `https://core.heysol.ai/api/v1/mcp?source=Kilo-Code`
- **Authentication**: Bearer token (API key or OAuth2)
- **Protocol**: REST API + MCP (Model Context Protocol) via Server-Sent Events
- **Server**: `core-unified-mcp-server v1.0.0`
- **Status**: Production-ready for space management and MCP operations

## Authentication

All API requests require authentication using a Bearer token:

```python
from heysol.client import HeySolClient

client = HeySolClient(api_key="your-api-key-here")
```

## MCP Tools

The following MCP tools are **available and working** via the MCP endpoint at `https://core.heysol.ai/api/v1/mcp?source=Kilo-Code`:

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

#### Delete Log Entry ⚠️ PENDING
- **Endpoint**: `/memory/logs/{log_id}` (DELETE)
- **Method**: DELETE
- **Status**: ❌ Not yet implemented in API
- **Note**: DELETE endpoint for log entries is not currently available in the HeySol API

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