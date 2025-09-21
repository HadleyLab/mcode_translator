# HeySol API Client - Testing Guide (Updated)

## Overview

This document provides information about the current test suite for the HeySol API client after reorganization. The tests have been cleaned up to remove obsolete files and focus on what's actually working.

## Current Test Structure

### ✅ Core Test Files (All Working)

#### 1. Comprehensive Mock Tests
- **File**: `test_comprehensive_heysol_client.py`
- **Purpose**: Complete test suite with mocked HTTP responses
- **Coverage**: All 47 endpoints, authentication, error handling, performance
- **Status**: ✅ All tests passing
- **Benefits**: Fast execution, no API dependencies, deterministic results

#### 2. Live API Integration Tests
- **File**: `test_integration_live_api.py`
- **Purpose**: Test with real HeySol API endpoints
- **Coverage**: Working endpoints (spaces, webhooks, memory operations)
- **Status**: ✅ Working for available endpoints
- **Benefits**: Validates real-world behavior

#### 3. MCP Working vs Not Working Test
- **File**: `test_mcp_working_vs_not.py`
- **Purpose**: Demonstrate MCP functionality with correct vs incorrect URLs
- **Coverage**: MCP protocol testing, tool availability
- **Status**: ✅ Shows MCP working with correct URL
- **Benefits**: Clear demonstration of working vs broken MCP URLs

#### 4. API Working vs Not Working Test
- **File**: `test_api_working_vs_not.py`
- **Purpose**: Demonstrate API endpoint functionality
- **Coverage**: Working endpoints vs 404 endpoints
- **Status**: ✅ Shows spaces/webhooks working, others 404
- **Benefits**: Clear demonstration of available vs missing endpoints

#### 5. MCP Correct URL Test
- **File**: `test_mcp_correct_url.py`
- **Purpose**: Test MCP with the correct working URL
- **Coverage**: MCP initialization, tools listing, session management
- **Status**: ✅ All MCP functionality working
- **Benefits**: Validates MCP protocol implementation

## Current Status: What's Working vs Not Working

### ✅ **MCP Protocol: FULLY WORKING**
- **Correct URL**: `https://core.heysol.ai/api/v1/mcp?source=Kilo-Code`
- **Server**: `core-unified-mcp-server v1.0.0`
- **Available Tools**:
  - `memory_ingest` - Store conversation data
  - `memory_search` - Search memory
  - `memory_get_spaces` - Get available spaces
  - `get_user_profile` - Get user profile
- **Session Management**: Working with session IDs
- **JSON-RPC Protocol**: Valid responses

### ✅ **Direct API: PARTIALLY WORKING**
**Working Endpoints:**
- `/spaces` - List and manage spaces ✅
- `/webhooks` - Webhook operations ✅
- Memory operations via client fallback ✅

**Not Working Endpoints (404):**
- `/user/profile` - Endpoint doesn't exist ❌
- `/memory/*` - Direct memory endpoints missing ❌
- `/oauth2/*` - OAuth2 endpoints not implemented ❌

### Authentication Methods

#### 1. API Key Authentication (Working)
```python
client = HeySolClient(api_key="your-api-key")
```

#### 2. OAuth2 Authentication (Not Available)
- OAuth2 endpoints return 404
- No OAuth2 implementation available

## Running Tests

### Option 1: Comprehensive Mock Tests (Recommended)

```bash
cd heysol_api_client
python -m pytest tests/test_comprehensive_heysol_client.py -v
```

**Features:**
- 47 comprehensive tests covering all endpoints
- Mock-based testing (no API key required)
- Authentication, error handling, and performance tests
- Detailed reporting and coverage

### Option 2: Live API Integration Tests

```bash
cd heysol_api_client
export HEYSOL_API_KEY="your-api-key"
python -m pytest tests/test_integration_live_api.py -v
```

**Features:**
- Real API endpoint testing
- Tests working endpoints (spaces, webhooks, memory)
- Validates actual API behavior
- Requires valid API key

### Option 3: MCP Working vs Not Working Test

```bash
cd heysol_api_client
export HEYSOL_API_KEY="your-api-key"
python tests/test_mcp_working_vs_not.py
```

**Features:**
- Demonstrates MCP working with correct URL
- Shows MCP failing with incorrect URLs
- Tests both direct MCP calls and client integration
- Clear comparison of working vs broken

### Option 4: API Working vs Not Working Test

```bash
cd heysol_api_client
export HEYSOL_API_KEY="your-api-key"
python tests/test_api_working_vs_not.py
```

**Features:**
- Shows which API endpoints work vs return 404
- Tests both direct API calls and client integration
- Clear demonstration of available functionality
- Helps understand API limitations

### Option 5: MCP Correct URL Test

```bash
cd heysol_api_client
export HEYSOL_API_KEY="your-api-key"
python tests/test_mcp_correct_url.py
```

**Features:**
- Tests MCP with the correct working URL
- Validates MCP tools and session management
- Confirms MCP protocol implementation
- Shows full MCP functionality

## Current Test Coverage

### ✅ **MCP Protocol Tests**
- **File**: `test_mcp_correct_url.py`
- **Coverage**: MCP initialization, tools listing, session management
- **Status**: All tests passing ✅
- **Tools Tested**: memory_ingest, memory_search, memory_get_spaces, get_user_profile

### ✅ **Comprehensive Client Tests**
- **File**: `test_comprehensive_heysol_client.py`
- **Coverage**: All 47 endpoints with mock responses
- **Status**: All tests passing ✅
- **Categories**: Authentication, memory, spaces, OAuth2, webhooks, error handling, performance

### ✅ **Live API Integration Tests**
- **File**: `test_integration_live_api.py`
- **Coverage**: Real API endpoints
- **Working**: Spaces, webhooks, memory operations via client
- **Not Working**: User profile, direct memory endpoints, OAuth2 endpoints

### ✅ **Working vs Not Working Tests**
- **Files**: `test_mcp_working_vs_not.py`, `test_api_working_vs_not.py`
- **Purpose**: Clear demonstration of functionality
- **MCP**: Shows correct URL works, incorrect URLs fail
- **API**: Shows which endpoints work vs return 404

## Error Handling Tests

### HTTP Error Scenarios
- ✅ 404 Not Found
- ✅ 401 Unauthorized
- ✅ 500 Internal Server Error
- ✅ Connection timeout
- ✅ Invalid JSON responses

### Validation Error Tests
- ✅ Empty/invalid parameters
- ✅ Missing required fields
- ✅ Invalid data types
- ✅ Boundary condition testing

### Authentication Error Tests
- ✅ Missing API key
- ✅ Invalid API key format
- ✅ Expired tokens (OAuth2)

## Performance Tests

### Response Time Testing
- Average response time measurement
- Performance regression detection
- Load testing with multiple requests

### Memory Usage Testing
- Memory leak detection
- Resource cleanup verification
- Session management efficiency

## Configuration

### Environment Variables
```bash
export HEYSOL_API_KEY="your-api-key"
```

### Configuration File
The `.env` file in the project root contains:
```env
HEYSOL_API_KEY=your-api-key
HEYSOL_OAUTH2_CLIENT_ID=your-oauth2-client-id
HEYSOL_OAUTH2_CLIENT_SECRET=your-oauth2-client-secret
```

### Test-Specific Configuration
- **Mock Tests**: No API key required
- **Live API Tests**: Requires `HEYSOL_API_KEY`
- **MCP Tests**: Requires `HEYSOL_API_KEY`
- **Working vs Not Working Tests**: Requires `HEYSOL_API_KEY`

## Test Results Interpretation

### Mock Test Results
Mock tests should always pass if the client code is correct. Failures indicate:
- Code implementation issues
- Missing method implementations
- Incorrect parameter handling

### Live Test Results
Live test failures may indicate:
- API service issues
- Authentication problems
- Network connectivity issues
- API changes or deprecations

### Performance Test Results
Performance test failures may indicate:
- API service degradation
- Network latency issues
- Client-side performance regressions

## Troubleshooting

### Common Issues

#### 1. "HEYSOL_API_KEY environment variable not set"
**Solution**: Set your API key:
```bash
export HEYSOL_API_KEY="your-api-key"
```

#### 2. "Connection timeout"
**Solution**: Check network connectivity and API service status

#### 3. "401 Unauthorized"
**Solution**: Verify your API key is valid and has correct permissions

#### 4. "404 Not Found" for MCP
**Solution**: Use the correct MCP URL:
```python
# ✅ Correct
https://core.heysol.ai/api/v1/mcp?source=Kilo-Code

# ❌ Incorrect (will fail)
https://core.heysol.ai/api/v1/mcp
```

#### 5. Import Errors
**Solution**: Run tests from the project root:
```bash
cd heysol_api_client
python -m pytest tests/ -v
```

### Understanding Test Results

#### Mock Tests (test_comprehensive_heysol_client.py)
- Should always pass if client code is correct
- Failures indicate code implementation issues
- No API key required

#### Live API Tests (test_integration_live_api.py)
- Tests real API endpoints
- Some endpoints work (spaces, webhooks), others return 404
- Requires valid API key

#### MCP Tests (test_mcp_correct_url.py)
- Tests MCP with correct URL - should pass
- Validates MCP tools and session management
- Requires valid API key

#### Working vs Not Working Tests
- Clearly shows what's functional vs broken
- Helps understand API limitations
- Requires valid API key

## Continuous Integration

### GitHub Actions Example
```yaml
name: API Client Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          cd heysol_api_client
          pip install -r requirements.txt
      - name: Run mock tests
        run: |
          cd tests
          python test_simple_runner.py
      - name: Run live tests
        run: |
          cd tests
          python test_comprehensive_runner.py --api-key ${{ secrets.HEYSOL_API_KEY }} --live-only
```

## Contributing

### Adding New Tests

1. Add test methods to `TestHeySolClientComprehensive` class
2. Include both mock and live test scenarios
3. Add proper error handling tests
4. Update this documentation

### Test Standards

- All tests should be idempotent
- Mock tests should not require external dependencies
- Live tests should handle API failures gracefully
- Tests should clean up after themselves
- Error messages should be descriptive and actionable

## API Compatibility

### Version Compatibility
- Current API version: MCP Protocol v1
- Base URL: `https://core.heysol.ai/api/v1/mcp?source=heysol-python-client`
- Authentication: Bearer token with API key

### Breaking Changes
- OAuth2 endpoints may change behavior
- MCP protocol updates may affect functionality
- New required parameters may be added

## Support

For issues with the test suite:
1. Check the troubleshooting section above
2. Verify your API key and permissions
3. Check the HeySol API status
4. Review recent changes to the API client

For API-specific issues:
- Check HeySol API documentation
- Verify endpoint compatibility
- Test with different API keys if available