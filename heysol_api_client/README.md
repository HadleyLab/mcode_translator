# HeySol API Client

[![PyPI version](https://badge.fury.io/py/heysol-api-client.svg)](https://badge.fury.io/py/heysol-api-client)
[![Python versions](https://img.shields.io/pypi/pyversions/heysol-api-client.svg)](https://pypi.org/project/heysol-api-client/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/heysol/heysol-python-client/workflows/CI/badge.svg)](https://github.com/heysol/heysol-python-client/actions)
[![Coverage](https://codecov.io/gh/heysol/heysol-python-client/branch/main/graph/badge.svg)](https://codecov.io/gh/heysol/heysol-python-client)

A comprehensive, production-ready Python client for the HeySol API with support for MCP (Model Context Protocol), authentication, memory management, and robust error handling.

## Features

- 🚀 **Full API Support**: Complete coverage of HeySol API endpoints
- 🔐 **Authentication**: API key and Bearer token authentication
- 📝 **Memory Management**: Ingest, search, and manage memory spaces
- ⚡ **Async Support**: Concurrent operations using aiohttp
- 🛡️ **Error Handling**: Comprehensive exception hierarchy with retry mechanisms
- 📊 **Rate Limiting**: Built-in rate limiting and throttling
- 📝 **Type Hints**: Full type annotation support
- 📚 **Documentation**: Comprehensive docstrings and examples
- 🧪 **Testing**: 80%+ test coverage with mocks and fixtures
- 📦 **PyPI Ready**: Complete packaging configuration

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
from heysol import HeySolClient, HeySolConfig

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
    space_id=space_id,
    tags=["clinical-trial", "research"]
)

# Search for data
results = client.search("clinical trial", space_id=space_id, limit=5)
for episode in results["episodes"]:
    print(f"- {episode['content']}")

# Clean up
client.close()
```

### Async Usage

```python
import asyncio
from heysol import AsyncHeySolClient

async def main():
    async with AsyncHeySolClient(api_key="your-api-key-here") as client:
        # Concurrent operations
        tasks = [
            client.ingest("Data point 1", tags=["batch"]),
            client.ingest("Data point 2", tags=["batch"]),
            client.ingest("Data point 3", tags=["batch"]),
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)
        print(f"Successfully ingested {len(results)} items")

asyncio.run(main())
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
  "log_level": "INFO",
  "async_enabled": false
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

- `get_user_profile() -> Dict[str, Any]`: Get current user profile
- `get_spaces() -> List[Dict[str, Any]]`: Get available memory spaces
- `create_space(name: str, description: str = "") -> str`: Create a new space
- `get_or_create_space(name: str, description: str = "") -> str`: Get existing or create space
- `ingest(message: str, space_id: str = None, source: str = None, priority: str = "normal", tags: List[str] = None) -> Dict[str, Any]`: Ingest data
- `search(query: str, space_id: str = None, limit: int = 10, **kwargs) -> Dict[str, Any]`: Search memory
- `search_knowledge_graph(query: str, space_id: str = None, limit: int = 10, depth: int = 2, include_metadata: bool = True) -> Dict[str, Any]`: Search knowledge graph
- `get_ingestion_logs(space_id: str = None, limit: int = 100, offset: int = 0, status: str = None, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]`: Get ingestion logs
- `get_specific_log(log_id: str) -> Dict[str, Any]`: Get specific log by ID
- `delete_log_entry(log_id: str) -> Dict[str, Any]`: Delete log entry
- `close() -> None`: Close client and clean up resources

### AsyncHeySolClient

Async client for concurrent operations.

#### Methods

All methods from `HeySolClient` available as async versions, plus:

- `async with client:` - Context manager support
- Concurrent execution of multiple operations

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
| `async_enabled` | bool | False | Enable async support |

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

- `basic_usage.py` - Basic client operations
- `async_usage.py` - Async operations and concurrency
- `log_management.py` - Log management and deletion operations
- `oauth2_google_demo.py` - Interactive Google OAuth2 demo
- `oauth2_simple_demo.py` - Simple working OAuth2 demo
- `oauth2_log_cli.py` - Command-line OAuth2 tool
- `oauth2_log_demo.py` - Standalone OAuth2 demo script
- `oauth2_setup_guide.py` - OAuth2 setup guide
- `oauth2_log_operations.ipynb` - Complete OAuth2 demo notebook

## Documentation

See the `docs/` directory for comprehensive documentation:

- `API_DOCUMENTATION.md` - Complete API reference and usage guide
- `OAUTH2_AUTHORIZATION_GUIDE.md` - OAuth2 setup and usage guide
- `AUTHENTICATION_GUIDE.md` - Authentication methods and configuration
- `API_DISCOVERY.md` - API endpoint discovery and testing
- `TESTING_REPORT.md` - Testing results and coverage reports

## Results and Logs

- `results/` - Test results, investigation reports, and performance data
- `logs/` - Application logs and debugging information

## OAuth2 Authentication

The HeySol API client supports OAuth2 authentication with Google accounts. See the unified OAuth2 implementation:

- **`examples/oauth2_log_operations.ipynb`** - Complete OAuth2 demo notebook
- **`examples/oauth2_log_demo.py`** - Standalone OAuth2 demo script
- **`examples/oauth2_log_cli.py`** - Command-line OAuth2 tool
- **`examples/oauth2_google_demo.py`** - Interactive Google OAuth2 demo
- **`examples/oauth2_simple_demo.py`** - Simple working OAuth2 demo
- **`docs/OAUTH2_AUTHORIZATION_GUIDE.md`** - OAuth2 setup and usage guide

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=heysol_api_client

# Run specific test categories
pytest -m "unit"  # Unit tests only
pytest -m "async"  # Async tests only
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