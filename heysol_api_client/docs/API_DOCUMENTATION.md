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

### ✅ **Direct API (RECOMMENDED)**
- **Access**: `https://core.heysol.ai/api/v1/{endpoint}`
- **Protocol**: Standard REST API with Bearer token authentication
- **Status**: ✅ **Lean and reliable**

### ❌ **MCP Protocol (Available)**
- **Access**: `https://core.heysol.ai/api/v1/mcp`
- **Protocol**: Server-Sent Events with JSON-RPC
- **Tools**: 100+ available (memory, spaces, GitHub integration)
- **Status**: ✅ **Available but not used by default**

## Authentication

All API requests require authentication using a Bearer token:

```python
from heysol import HeySolClient

client = HeySolClient(api_key="your-api-key-here")
```

The client automatically handles token authentication for all API calls.


## API Reference

### User Operations

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `get_user_profile()` | `GET /api/profile` (OAuth) | ⚠️ **OAuth Pending** | Get current user profile (OAuth implementation pending) |

### Memory Operations

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `search()` | `POST /search` | ✅ **Working** | Search memories |
| `ingest()` | `POST /add` | ✅ **Working** | Ingest data |
| `search_knowledge_graph()` | `POST /search` | ✅ **Working** | Search knowledge graph |
| `get_episode_facts()` | `GET /episodes/{id}/facts` | ✅ **Working** | Get episode facts |
| `get_ingestion_logs()` | `GET /logs` | ✅ **Working** | Get ingestion logs |
| `get_specific_log()` | `GET /logs/{id}` | ✅ **Working** | Get specific log |
| `delete_log_entry()` | `DELETE /logs/{id}` | ✅ **Working** | Delete log entry |

### Space Operations

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `get_spaces()` | `GET /spaces` | ✅ **Working** | List spaces |
| `create_space()` | `POST /spaces` | ✅ **Working** | Create space |
| `get_space_details()` | `GET /spaces/{id}` | ✅ **Working** | Get space details |
| `update_space()` | `PUT /spaces/{id}` | ✅ **Working** | Update space |
| `delete_space()` | `DELETE /spaces/{id}` | ✅ **Working** | Delete space |
| `bulk_space_operations()` | `PUT /spaces` | ✅ **Working** | Bulk operations |

### Webhook Operations

| Method | Endpoint | Status | Description |
|--------|----------|--------|-------------|
| `register_webhook()` | `POST /webhooks` | ✅ **Working** | Register webhook |
| `list_webhooks()` | `GET /webhooks` | ✅ **Working** | List webhooks |
| `get_webhook()` | `GET /webhooks/{id}` | ✅ **Working** | Get webhook |
| `update_webhook()` | `PUT /webhooks/{id}` | ✅ **Working** | Update webhook |
| `delete_webhook()` | `DELETE /webhooks/{id}` | ✅ **Working** | Delete webhook |

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

- **Primary Method**: Direct API (`https://core.heysol.ai/api/v1/{endpoint}`)
- **Authentication**: Bearer token with API key
- **Issues**: Check API key and network connectivity first

---

*Documentation updated for lean, direct API implementation.*