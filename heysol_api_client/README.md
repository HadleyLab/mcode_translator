# HeySol API Client

[![PyPI version](https://badge.fury.io/py/heysol-api-client.svg)](https://badge.fury.io/py/heysol-api-client)
[![Python versions](https://img.shields.io/pypi/pyversions/heysol-api-client.svg)](https://pypi.org/project/heysol-api-client/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/heysol/heysol-python-client/workflows/CI/badge.svg)](https://github.com/heysol/heysol-python-client/actions)
[![Coverage](https://codecov.io/gh/heysol/heysol-python-client/branch/main/graph/badge.svg)](https://codecov.io/gh/heysol/heysol-python-client)

A comprehensive, production-ready Python client for the HeySol API with support for MCP (Model Context Protocol), authentication, memory management, and robust error handling.

## Features

- 🚀 **Full API Support**: Complete coverage of HeySol API endpoints
- 🔐 **Authentication**: API key authentication with Bearer token support
- 📝 **Memory Management**: Ingest, search, and manage memory spaces
- 🛡️ **Error Handling**: Comprehensive exception hierarchy with retry mechanisms
- 📊 **Rate Limiting**: Built-in rate limiting and throttling
- 📝 **Type Hints**: Full type annotation support
- 📚 **Documentation**: Comprehensive docstrings and examples
- 🧪 **Testing**: 80%+ test coverage with mocks and fixtures
- 📦 **PyPI Ready**: Complete packaging configuration
- 🖥️ **CLI Tool**: Command-line interface for all operations

## 📊 Implementation Status

### ✅ **Fully Working & Tested**
- **MCP Protocol**: 100+ tools available (memory, spaces, GitHub integration)
- **Memory Operations**: Complete coverage (ingest, search, knowledge graph, queue, episodes)
- **Space Management**: Full CRUD operations (create, read, update, delete, bulk ops)
- **User Profile**: Get user profile via MCP
- **Log Management**: Complete log operations with source filtering
- **Webhook Operations**: Full webhook lifecycle management
- **CLI Interface**: Complete command-line interface for all operations
- **Direct API**: All practical endpoints implemented and tested

### ✅ **Test Coverage**
- **53 CLI Tests**: Comprehensive test suite for all CLI commands
- **Unit Tests**: Extensive unit test coverage for all modules
- **Integration Tests**: End-to-end testing and validation
- **Error Handling**: Robust error scenarios and edge cases

## Installation

### From PyPI (Recommended)

```bash
pip install heysol-api-client
```

### From Source

```bash
git clone https://github.com/heysol/heysol-python-client.git
cd heysol-python-client
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from heysol import HeySolClient

# Initialize with API key
client = HeySolClient(api_key="your-api-key-here")

# Get user profile
profile = client.get_user_profile()
print(f"Hello, {profile['name']}!")

# Create or get a memory space
space_id = client.get_or_create_space("My Research", "Space for research data")

# Ingest data
result = client.ingest(
    message="Clinical trial shows promising results for new treatment",
    space_id=space_id
)

# Search for data
results = client.search("clinical trial", space_ids=[space_id], limit=5)
for episode in results["episodes"]:
    print(f"- {episode['content']}")

# Clean up
client.close()
```

### CLI Usage

```bash
# Setup: Get your API key from https://core.heysol.ai/settings/api
export HEYSOL_API_KEY="your-api-key-here"

# Get user profile
heysol-client profile get

# Space management
heysol-client spaces list
heysol-client spaces create "My Research" --description "Research data"
heysol-client spaces get <space-id>
heysol-client spaces update <space-id> --name "Updated Name"
heysol-client spaces delete <space-id> --confirm

# Memory operations
heysol-client memory ingest "Clinical trial data" --space-id <space-id>
heysol-client memory search "cancer research" --space-id <space-id> --limit 10
heysol-client memory search-graph "treatment outcomes" --depth 3
heysol-client memory queue "Batch data" --space-id <space-id> --tags clinical research
heysol-client memory episode <episode-id>

# Log management
heysol-client logs list --space-id <space-id> --source "heysol-python-client"
heysol-client logs get <log-id>
heysol-client logs delete "source-name" --confirm
heysol-client logs delete-entry <log-id> --confirm

# Webhook management
heysol-client webhooks create "https://myapp.com/webhook" --secret "webhook-secret"
heysol-client webhooks list
heysol-client webhooks update <webhook-id> "https://new-url.com" --events memory.created
heysol-client webhooks delete <webhook-id> --confirm

# MCP tools
heysol-client tools list
```

### Configuration

#### Environment Variables

```bash
export HEYSOL_API_KEY="your-api-key-here"
export HEYSOL_BASE_URL="https://core.heysol.ai/api/v1/mcp"
export HEYSOL_SOURCE="my-application"
export HEYSOL_LOG_LEVEL="INFO"
```

#### Configuration File

Create a `heysol_config.json`:

```json
{
  "api_key": "your-api-key-here",
  "base_url": "https://core.heysol.ai/api/v1/mcp",
  "source": "my-application",
  "timeout": 60,
  "max_retries": 3,
  "rate_limit_per_minute": 60,
  "log_level": "INFO"
}
```

Then load it:

```python
from heysol import HeySolConfig

config = HeySolConfig.from_file("heysol_config.json")
client = HeySolClient(config=config)
```

## API Reference

### HeySolClient

Main synchronous client for the HeySol API.

#### Methods

**User Operations:**
- `get_user_profile() -> Dict[str, Any]`: Get current user profile

**Memory Operations:**
- `ingest(message: str, space_id: str = None, session_id: str = None) -> Dict[str, Any]`: Ingest data
- `search(query: str, space_ids: List[str] = None, limit: int = 10, include_invalidated: bool = False) -> Dict[str, Any]`: Search memory
- `search_knowledge_graph(query: str, space_id: str = None, limit: int = 10, depth: int = 2, include_metadata: bool = True) -> Dict[str, Any]`: Search knowledge graph
- `add_data_to_ingestion_queue(data: Any, space_id: str = None, priority: str = "normal", tags: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]`: Add data to ingestion queue
- `get_episode_facts(episode_id: str, limit: int = 100, offset: int = 0, include_metadata: bool = True) -> List[Dict[str, Any]]`: Get episode facts

**Space Operations:**
- `get_spaces() -> List[Dict[str, Any]]`: Get available memory spaces
- `create_space(name: str, description: str = "") -> str`: Create a new space
- `get_space_details(space_id: str, include_stats: bool = True, include_metadata: bool = True) -> Dict[str, Any]`: Get space details
- `update_space(space_id: str, name: str = None, description: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]`: Update space
- `bulk_space_operations(intent: str, space_id: str = None, statement_ids: List[str] = None, space_ids: List[str] = None) -> Dict[str, Any]`: Bulk space operations
- `delete_space(space_id: str, confirm: bool = False) -> Dict[str, Any]`: Delete space

**Log Operations:**
- `get_ingestion_logs(space_id: str = None, limit: int = 100, offset: int = 0, status: str = None, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]`: Get ingestion logs
- `get_specific_log(log_id: str) -> Dict[str, Any]`: Get specific log by ID
- `delete_log_entry(log_id: str) -> Dict[str, Any]`: Delete log entry by ID
- `delete_logs_by_source(source: str, space_id: str = None, confirm: bool = False) -> Dict[str, Any]`: Delete logs by source

**Webhook Operations:**
- `register_webhook(url: str, events: List[str] = None, secret: str = "") -> Dict[str, Any]`: Register webhook
- `list_webhooks(space_id: str = None, active: bool = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]`: List webhooks
- `get_webhook(webhook_id: str) -> Dict[str, Any]`: Get webhook details
- `update_webhook(webhook_id: str, url: str, events: List[str], secret: str = "", active: bool = True) -> Dict[str, Any]`: Update webhook
- `delete_webhook(webhook_id: str, confirm: bool = False) -> Dict[str, Any]`: Delete webhook

**Utility Methods:**
- `close() -> None`: Close client and clean up resources
- `is_mcp_available() -> bool`: Check if MCP is available
- `get_available_tools() -> Dict[str, Any]`: Get available MCP tools

### Configuration

#### HeySolConfig

Configuration class supporting multiple sources:

- `HeySolConfig.from_env()` - Load from environment variables
- `HeySolConfig.from_file(path)` - Load from JSON file
- `HeySolConfig.from_dict(data)` - Load from dictionary

#### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_key` | str | None | HeySol API key |
| `base_url` | str | `https://core.heysol.ai/api/v1/mcp` | API base URL |
| `source` | str | `heysol-python-client` | Source identifier |
| `timeout` | int | 60 | Request timeout in seconds |
| `max_retries` | int | 3 | Maximum retry attempts |
| `rate_limit_per_minute` | int | 60 | Rate limit per minute |
| `log_level` | str | `INFO` | Logging level |

## Error Handling

The client provides a comprehensive exception hierarchy:

```python
from heysol import (
    HeySolError,
    AuthenticationError,
    RateLimitError,
    APIError,
    ValidationError,
    ConnectionError,
    ServerError,
    NotFoundError
)

try:
    client.search("query")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except HeySolError as e:
    print(f"API error: {e}")
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py` - Basic client operations and API usage
- `cli_usage.py` - Complete command-line interface examples
- `cli_source_filtering_demo.py` - CLI source filtering demonstrations
- `log_management.py` - Log management and deletion operations
- `source_filtering_demo.py` - Source-based filtering examples
- `oauth2_setup_guide.py` - OAuth2 setup and configuration (legacy)

## Documentation

See the `docs/` directory for comprehensive documentation:

- `API_DOCUMENTATION.md` - Complete API reference and usage guide
- `AUTHENTICATION_GUIDE.md` - Authentication methods and configuration
- `API_DISCOVERY.md` - API endpoint discovery and testing
- `TESTING_REPORT.md` - Testing results and coverage reports

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=heysol_api_client

# Run specific test categories
pytest -m "unit"  # Unit tests only
pytest -m "slow"   # Slow integration tests
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/heysol/heysol-python-client.git
cd heysol-python-client

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black heysol_api_client/
isort heysol_api_client/

# Type checking
mypy heysol_api_client/

# Linting
flake8 heysol_api_client/

# All checks
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email**: dev@heysol.ai
- 📖 **Documentation**: https://docs.heysol.ai/api-reference
- 🐛 **Issues**: https://github.com/heysol/heysol-python-client/issues
- 💬 **Discussions**: https://github.com/heysol/heysol-python-client/discussions

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**HeySol API Client** - A production-ready Python client for the HeySol API