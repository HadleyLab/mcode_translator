# HeySol API Client Authentication Guide

This guide explains the two authentication methods supported by the HeySol API client and which endpoints work with each method.

## Authentication Methods

### 1. API Key Authentication (Recommended for Server Applications)
- **Type**: Bearer token authentication
- **Setup**: Simple - just provide an API key
- **Security**: Good for server-to-server communication
- **Usage**: Direct API calls without user interaction

```python
from heysol.client import HeySolClient

client = HeySolClient(api_key="your-api-key")
profile = client.get_user_profile()
```

### 2. OAuth2 Authentication (Recommended for User Applications)
- **Type**: Google OAuth2 flow with interactive browser authentication
- **Setup**: Requires OAuth2 client credentials and user consent
- **Security**: Higher security with user authorization
- **Usage**: Applications that need user-specific permissions

```python
from heysol.oauth2 import InteractiveOAuth2Authenticator

auth = InteractiveOAuth2Authenticator(
    client_id="your-client-id",
    client_secret="your-client-secret"
)

# Build authorization URL
auth_url = auth.build_authorization_url()

# After user authorizes, exchange code for tokens
tokens = auth.exchange_code_for_tokens(authorization_code)

# Use tokens with client
client = HeySolClient(api_key=tokens.access_token)
```

## Endpoint Compatibility Matrix

| Endpoint Category | Method | API Key Auth | OAuth2 Auth | Notes |
|------------------|--------|--------------|-------------|-------|
| **User** | `get_user_profile()` | ✅ | ✅ | Both return user profile data |
| **Memory** | `ingest()` | ✅ | ✅ | Ingest data into CORE Memory |
| **Memory** | `search()` | ✅ | ✅ | Search memories |
| **Memory** | `get_spaces()` | ✅ | ✅ | Get available memory spaces |
| **Memory** | `create_space()` | ✅ | ✅ | Create new memory space |
| **Memory** | `delete_log_entry()` | ❌ | ❌ | Delete log entries (API endpoint not available) |
| **Memory** | `search_knowledge_graph()` | ✅ | ✅ | Search knowledge graph |
| **Memory** | `add_data_to_ingestion_queue()` | ✅ | ✅ | Queue data for ingestion |
| **Memory** | `get_episode_facts()` | ✅ | ✅ | Get episode facts |
| **Memory** | `get_ingestion_logs()` | ✅ | ✅ | Get ingestion logs |
| **Memory** | `get_specific_log()` | ✅ | ✅ | Get specific log by ID |
| **Spaces** | `bulk_space_operations()` | ✅ | ✅ | Bulk space operations |
| **Spaces** | `get_space_details()` | ✅ | ✅ | Get space details |
| **Spaces** | `update_space()` | ✅ | ✅ | Update space properties |
| **Spaces** | `delete_space()` | ✅ | ✅ | Delete space (requires confirm=True) |
| **OAuth2** | `get_oauth2_authorization_url()` | ✅ | ✅ | Get OAuth2 authorization URL |
| **OAuth2** | `oauth2_authorization_decision()` | ✅ | ✅ | Make OAuth2 authorization decision |
| **OAuth2** | `oauth2_token_exchange()` | ✅ | ✅ | Exchange code for tokens |
| **OAuth2** | `get_oauth2_user_info()` | ✅ | ✅ | Get OAuth2 user info (uses access token) |
| **OAuth2** | `oauth2_refresh_token()` | ✅ | ✅ | Refresh access token |
| **OAuth2** | `oauth2_revoke_token()` | ✅ | ✅ | Revoke OAuth2 token |
| **OAuth2** | `oauth2_token_introspection()` | ✅ | ✅ | Introspect OAuth2 token |
| **Webhooks** | `register_webhook()` | ✅ | ✅ | Register new webhook |
| **Webhooks** | `list_webhooks()` | ✅ | ✅ | List webhooks |
| **Webhooks** | `get_webhook()` | ✅ | ✅ | Get webhook details |
| **Webhooks** | `update_webhook()` | ✅ | ✅ | Update webhook |
| **Webhooks** | `delete_webhook()` | ✅ | ✅ | Delete webhook (requires confirm=True) |

## Key Differences

### API Key Authentication
- **Pros**:
  - Simple setup and usage
  - No user interaction required
  - Good for automated systems
  - Direct server-to-server communication
- **Cons**:
  - Less granular permissions
  - API key must be securely stored
  - No user-specific context

### OAuth2 Authentication
- **Pros**:
  - User-specific permissions
  - More secure (tokens can be revoked)
  - User consent and authorization
  - Automatic token refresh
- **Cons**:
  - More complex setup
  - Requires user interaction
  - Additional OAuth2 credentials needed

## Environment Variables

### For API Key Authentication
```bash
export COREAI_API_KEY="your-api-key-here"
```

### For OAuth2 Authentication
```bash
export COREAI_OAUTH2_CLIENT_ID="your-client-id"
export COREAI_OAUTH2_CLIENT_SECRET="your-client-secret"
export COREAI_OAUTH2_REDIRECT_URI="http://localhost:8080/callback"
export COREAI_OAUTH2_SCOPE="openid profile email api"
```

## Testing

### Mock Tests (No API Key Required)
```bash
# Run comprehensive mock tests
python -m pytest tests/test_authentication_comprehensive.py -v

# Run all mock tests
python -m pytest tests/test_comprehensive_heysol_client.py -v
```

### Live API Tests (API Key Required)
```bash
# Set API key
export COREAI_API_KEY="your-valid-api-key"

# Run live integration tests
python -m pytest tests/test_integration_live_api.py -v
```

## Error Handling

Both authentication methods handle errors consistently:

- **API Key Errors**: Invalid or expired API keys return HTTP 401
- **OAuth2 Errors**: Expired tokens are automatically detected
- **Network Errors**: Connection timeouts and network issues are handled
- **Validation Errors**: Input validation errors are raised immediately

## Best Practices

1. **Use API Key Authentication** for:
   - Server-side applications
   - Automated processes
   - Background services
   - API-to-API integrations

2. **Use OAuth2 Authentication** for:
   - User-facing applications
   - Applications requiring user consent
   - Higher security requirements
   - User-specific data access

3. **Security Considerations**:
   - Never hardcode credentials in source code
   - Use environment variables for configuration
   - Rotate API keys regularly
   - Implement proper token refresh for OAuth2
   - Validate all inputs before API calls

4. **Testing Strategy**:
   - Use mock tests for development and CI/CD
   - Use live tests sparingly to avoid API rate limits
   - Test both authentication methods in your application
   - Monitor API usage and error rates

## Migration Between Authentication Methods

The HeySol API client is designed to work seamlessly with both authentication methods. You can switch between them by simply changing the client initialization:

```python
# API Key client
api_client = HeySolClient(api_key="your-api-key")

# OAuth2 client (after token exchange)
oauth_client = HeySolClient(api_key=tokens.access_token)

# Both clients have identical method signatures
profile1 = api_client.get_user_profile()
profile2 = oauth_client.get_user_profile()
```

This allows you to easily migrate between authentication methods or support both in the same application.