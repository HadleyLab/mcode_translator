# HeySol API Client Documentation

## Quick Start

```python
from heysol import HeySolClient

# Initialize with API key
client = HeySolClient(api_key="your-api-key")

# Get user profile
profile = client.get_user_profile()

# Create a memory space
space_id = client.create_space("Research", "Clinical trial data")

# Ingest data
client.ingest("New clinical findings", space_id=space_id, tags=["research"])

# Search memories
results = client.search("clinical trial", limit=5)
```

## API Access Methods

### ✅ **MCP Protocol (RECOMMENDED)**
- **Access**: `https://core.heysol.ai/api/v1/mcp`
- **Protocol**: Server-Sent Events with JSON-RPC
- **Tools**: 100+ available (memory, spaces, GitHub integration)
- **Status**: ✅ **Fully functional**

### ❌ **Direct API (Limited)**
- **Access**: `https://core.heysol.ai/api/v1/{endpoint}`
- **Protocol**: Standard REST API
- **Working**: Only 3/21 endpoints (14% success rate)
- **Status**: ❌ **Severely limited**

## Authentication

```python
# API Key (recommended for server applications)
client = HeySolClient(api_key="your-api-key")

# OAuth2 (recommended for user applications)
from heysol.oauth2 import InteractiveOAuth2Authenticator
auth = InteractiveOAuth2Authenticator(client_id="...", client_secret="...")
client = HeySolClient(oauth2_auth=auth)
```

## Authentication

All API requests require authentication using a Bearer token:

```python
from heysol.client import HeySolClient

client = HeySolClient(api_key="your-api-key-here")
```


## API Reference

### User Operations

| Method | Endpoint | MCP Access | Direct API | Description |
|--------|----------|------------|------------|-------------|
| `get_user_profile()` | `/user/profile` | ✅ **Working** | ❌ **Failed** | Get current user profile |

### Memory Operations

| Method | Endpoint | MCP Access | Direct API | Description |
|--------|----------|------------|------------|-------------|
| `search()` | `POST /search` | ✅ **Working** | ❌ **Failed** | Search memories |
| `ingest()` | `POST /add` | ✅ **Working** | ❌ **Failed** | Ingest data |
| `search_knowledge_graph()` | `POST /search` | ✅ **Working** | ❌ **Failed** | Search knowledge graph |
| `get_episode_facts()` | `GET /episodes/{id}/facts` | ✅ **Working** | ❌ **Failed** | Get episode facts |
| `get_ingestion_logs()` | `GET /logs` | ✅ **Working** | ❌ **Failed** | Get ingestion logs |
| `get_specific_log()` | `GET /logs/{id}` | ✅ **Working** | ❌ **Failed** | Get specific log |

### Space Operations

| Method | Endpoint | MCP Access | Direct API | Description |
|--------|----------|------------|------------|-------------|
| `get_spaces()` | `GET /spaces` | ✅ **Working** | ✅ **Working** | List spaces |
| `create_space()` | `POST /spaces` | ✅ **Working** | ❌ **Failed** | Create space |
| `get_space_details()` | `GET /spaces/{id}` | ✅ **Working** | ✅ **Working** | Get space details |
| `update_space()` | `PUT /spaces/{id}` | ✅ **Working** | ❌ **Failed** | Update space |
| `delete_space()` | `DELETE /spaces/{id}` | ✅ **Working** | ✅ **Working** | Delete space |
| `bulk_space_operations()` | `PUT /spaces` | ✅ **Working** | ❌ **Failed** | Bulk operations |

### Webhook Operations

| Method | Endpoint | MCP Access | Direct API | Description |
|--------|----------|------------|------------|-------------|
| `register_webhook()` | `POST /webhooks` | ❌ **N/A** | ❌ **Failed** | Register webhook |
| `list_webhooks()` | `GET /webhooks` | ❌ **N/A** | ❌ **Failed** | List webhooks |
| `get_webhook()` | `GET /webhooks/{id}` | ❌ **N/A** | ❌ **Failed** | Get webhook |
| `update_webhook()` | `PUT /webhooks/{id}` | ❌ **N/A** | ❌ **Failed** | Update webhook |
| `delete_webhook()` | `DELETE /webhooks/{id}` | ❌ **N/A** | ❌ **Failed** | Delete webhook |

### OAuth2 Operations

| Method | Endpoint | MCP Access | Direct API | Description |
|--------|----------|------------|------------|-------------|
| OAuth2 endpoints | Various | ❌ **N/A** | ❌ **Failed** | OAuth2 authentication |

## Error Handling

```python
from heysol import HeySolError, ValidationError, AuthenticationError

try:
    result = client.search("query")
except AuthenticationError:
    print("Invalid API key")
except ValidationError as e:
    print(f"Invalid input: {e}")
except HeySolError as e:
    print(f"API error: {e}")
```

## Practical Examples

### Memory Management

```python
# Create and manage memory spaces
space_id = client.create_space("Clinical Research", "Cancer trial data")
client.ingest("New treatment shows 85% efficacy", space_id=space_id, tags=["clinical", "treatment"])

# Search with filters
results = client.search("cancer treatment", space_id=space_id, limit=10)
for episode in results.get("episodes", []):
    print(f"- {episode.get('content', '')}")

# Get knowledge graph connections
kg_results = client.search_knowledge_graph("treatment efficacy", limit=5, depth=2)
```

### Space Operations

```python
# List all spaces
spaces = client.get_spaces()
print(f"Found {len(spaces)} spaces")

# Get space details
space_details = client.get_space_details(space_id)
print(f"Space: {space_details.get('name')}")

# Update space
client.update_space(space_id, name="Updated Research", description="Updated description")

# Delete space (requires confirmation)
client.delete_space(space_id, confirm=True)
```

### User Profile

```python
# Get current user profile
profile = client.get_user_profile()
print(f"User: {profile.get('name', 'Unknown')}")
print(f"Email: {profile.get('email', 'Not provided')}")
```


## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=heysol

# Run specific test categories
pytest -m "unit"    # Unit tests only
pytest -m "slow"    # Integration tests
```

## Support

- **Primary Method**: MCP Protocol (`https://core.heysol.ai/api/v1/mcp`)
- **Fallback**: Direct API (limited to 3/21 endpoints)
- **Issues**: Check API key and network connectivity first

---

*Documentation updated based on current testing results and MCP protocol functionality.*