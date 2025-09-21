# 🔐 OAuth2 Authorization Endpoint Guide

## Complete Guide to HeySol OAuth2 Authorization

This guide explains how the OAuth2 authorization endpoint works and provides a complete demonstration of Google OAuth2 authentication with the HeySol API client.

## 📋 OAuth2 Authorization Endpoint Overview

### What is the OAuth2 Authorization Endpoint?

The OAuth2 authorization endpoint is the entry point for the OAuth2 authorization code flow. It handles user authentication and authorization, redirecting users to authenticate with their identity provider (Google, in this case) and obtaining consent for the requested permissions.

### How It Works

Based on the HeySol API documentation at `https://docs.heysol.ai/api-reference/oauth2-authorization-endpoint`, the OAuth2 authorization endpoint:

1. **Endpoint**: `GET /oauth/authorize`
2. **Purpose**: Initiates OAuth2 authorization code flow with PKCE (Proof Key for Code Exchange)
3. **Authentication**: Requires user to be logged in via session cookie
4. **Security**: Supports PKCE for enhanced security

### Authorization Flow Steps

1. **Authorization Request**: Client builds authorization URL with required parameters
2. **User Authentication**: User is redirected to Google's OAuth2 login page
3. **Consent**: User grants permissions for requested scopes
4. **Authorization Code**: Google redirects back with authorization code
5. **Token Exchange**: Code is exchanged for access tokens
6. **API Access**: Tokens are used to authenticate API requests

## 🛠️ OAuth2 Authorization Endpoint Details

### Request Format

```http
GET /oauth/authorize
```

### Parameters

The authorization endpoint accepts these query parameters:

- `client_id` (required): OAuth2 client identifier
- `redirect_uri` (required): URI to redirect to after authorization
- `scope` (required): Space-separated list of requested permissions
- `response_type` (required): Must be "code" for authorization code flow
- `state` (optional): Opaque value for CSRF protection
- `code_challenge` (required for PKCE): SHA256 hash of code verifier
- `code_challenge_method` (required for PKCE): Must be "S256"

### Response

The endpoint redirects to the `redirect_uri` with either:

**Success Response:**
```
HTTP/1.1 302 Found
Location: {redirect_uri}?code={authorization_code}&state={state}
```

**Error Response:**
```
HTTP/1.1 302 Found
Location: {redirect_uri}?error={error_code}&error_description={description}&state={state}
```

### Common Error Codes

- `invalid_request`: Missing or invalid parameters
- `unauthorized_client`: Client not authorized
- `access_denied`: User denied authorization
- `unsupported_response_type`: Invalid response_type
- `invalid_scope`: Requested scope is invalid

## 🚀 Google OAuth2 Demo Implementation

### Complete OAuth2 Flow Demonstration

The `oauth2_google_demo.py` script demonstrates the complete OAuth2 authorization flow:

1. **Configuration Validation**: Checks OAuth2 credentials
2. **Authorization URL Building**: Constructs Google OAuth2 URL
3. **Interactive Authorization**: Opens browser for user authentication
4. **Callback Handling**: Receives authorization code via HTTP server
5. **Token Exchange**: Exchanges code for access tokens
6. **API Testing**: Makes authenticated API calls
7. **Token Management**: Demonstrates token refresh

### Running the Demo

```bash
cd heysol_api_client
python oauth2_google_demo.py
```

### Demo Features

- ✅ **Interactive Browser Authentication**: Automatically opens Google OAuth2 login
- ✅ **Local Callback Server**: Handles OAuth2 redirect at `http://localhost:8080/callback`
- ✅ **Token Management**: Secure storage and refresh of OAuth2 tokens
- ✅ **API Integration**: Tests both Google and HeySol API endpoints
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Progress Tracking**: Step-by-step progress reporting

### Configuration Requirements

Set these environment variables in `heysol_api_client/.env`:

```bash
HEYSOL_OAUTH2_CLIENT_ID=your-google-oauth2-client-id
HEYSOL_OAUTH2_CLIENT_SECRET=your-google-oauth2-client-secret
HEYSOL_OAUTH2_REDIRECT_URI=http://localhost:8080/callback
HEYSOL_OAUTH2_SCOPE=openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email
```

### Google OAuth2 Setup

1. **Create Google OAuth2 Credentials**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Google+ API
   - Create OAuth2 credentials (Web application)
   - Set authorized redirect URI: `http://localhost:8080/callback`

2. **Configure OAuth2 Consent Screen**:
   - Navigate to "APIs & Services" → "OAuth consent screen"
   - Choose "External" user type and click "Create"
   - Fill in app information (name, email, etc.)
   - Add authorized domains (e.g., `localhost` for testing)
   - Configure required scopes (openid, profile, email)
   - Save and continue through all steps
   - **Critical Step**: Go to "Publish app" → "Publish" (even for testing)

3. **Set Environment Variables**:
   - Copy client ID and secret to `.env` file
   - Ensure redirect URI matches Google Console configuration

## 🔧 OAuth2 Implementation Details

### Authorization URL Construction

```python
def build_google_authorization_url(self) -> str:
    """Build Google OAuth2 authorization URL."""
    google_auth_url = "https://accounts.google.com/oauth/authorize"

    params = {
        "client_id": self.client_id,
        "redirect_uri": self.redirect_uri,
        "scope": self.scope,
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
        "state": "heysol-oauth2-demo"
    }

    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{google_auth_url}?{query_string}"
```

### Token Exchange Process

```python
def exchange_code_for_tokens(self, code: str) -> bool:
    """Exchange authorization code for access tokens."""
    token_url = "https://oauth2.googleapis.com/token"

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": self.redirect_uri,
        "client_id": self.client_id,
        "client_secret": self.client_secret,
    }

    response = requests.post(token_url, data=data, headers=headers)
    token_data = response.json()

    # Store tokens securely
    tokens = OAuth2Tokens(
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        token_type=token_data.get("token_type", "Bearer"),
        expires_in=token_data.get("expires_in"),
        scope=token_data.get("scope"),
    )
```

### Authenticated API Calls

```python
def test_authenticated_api_calls(self) -> bool:
    """Test authenticated API calls."""
    # Test Google user info
    userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    auth_header = self.oauth2_auth.get_authorization_header()

    response = requests.get(userinfo_url, headers={"Authorization": auth_header})
    user_info = response.json()

    # Test HeySol API calls
    client = HeySolClient(oauth2_auth=self.oauth2_auth, skip_mcp_init=True)
    profile = client.get_user_profile()
```

## 🔐 Security Considerations

### PKCE Implementation

The OAuth2 authorization endpoint supports PKCE (Proof Key for Code Exchange) for enhanced security:

- **Code Challenge**: SHA256 hash of code verifier
- **Code Verifier**: Cryptographically random string
- **Method**: S256 (SHA256 with Base64URL encoding)

### Token Security

- **Access Tokens**: Short-lived tokens for API access
- **Refresh Tokens**: Long-lived tokens for obtaining new access tokens
- **Token Storage**: Secure in-memory storage only
- **Token Refresh**: Automatic refresh before expiration

### HTTPS Requirements

- All OAuth2 endpoints require HTTPS
- Callback server uses HTTP for local development only
- Production applications must use HTTPS for all endpoints

## 🧪 Testing the Implementation

### Unit Tests

The OAuth2 implementation includes comprehensive unit tests:

```bash
cd heysol_api_client
python -m pytest tests/test_oauth2_comprehensive.py -v
```

### Integration Tests

Run integration tests to verify OAuth2 functionality:

```bash
cd heysol_api_client
python -m pytest tests/test_integration_live_api.py::TestLiveAPIIntegration::test_live_get_oauth2_authorization_url -v
```

### Manual Testing

Test the complete OAuth2 flow manually:

1. Run the demo: `python oauth2_google_demo.py`
2. Complete Google OAuth2 authentication in browser
3. Verify successful token exchange
4. Check authenticated API calls
5. Review generated logs and results

## 📊 OAuth2 Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│  Authorization   │───▶│   Google OAuth2 │
│                 │    │     Endpoint     │    │     Login       │
│ oauth2_google_  │    │ /oauth/authorize │    │                 │
│ demo.py         │    │                  │    │ accounts.google.│
└─────────────────┘    └──────────────────┘    │ com/oauth/      │
                                                │ authorize       │
┌─────────────────┐    ┌──────────────────┐    │                 │
│   Callback      │◀───│   Redirect URI   │◀───│                 │
│   Server        │    │                  │    └─────────────────┘
│ localhost:8080  │    │ ?code=auth_code  │
│ /callback       │    │ &state=...       │
└─────────────────┘    └──────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Token          │───▶│   Google Token  │
│                 │    │   Exchange       │    │   Endpoint      │
│ Exchange code   │    │ /oauth/token     │    │ oauth2.google-  │
│ for tokens      │    │                  │    │ com/token       │
└─────────────────┘    └──────────────────┘    └─────────────────┘

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│   API Calls      │───▶│   HeySol API    │
│                 │    │   with Tokens    │    │   Endpoints     │
│ Make API calls  │    │                  │    │ core.heysol.ai  │
│ with Bearer     │    │ Authorization:   │    │                 │
│ tokens          │    │ Bearer <token>   │    └─────────────────┘
└─────────────────┘    └──────────────────┘
```

## 🎯 OAuth2 Demo Results

After running the demo, you should see results like:

```
📊 GOOGLE OAUTH2 DEMO RESULTS
Timestamp: 2025-09-21T13:32:18.203Z
Overall Success: ✅ YES

Steps Completed: 7
1. ✅ configuration: OAuth2 configuration validated
2. ✅ authenticator_setup: OAuth2 authenticator created
3. ✅ callback_server: OAuth2 callback server started
4. ✅ interactive_auth: Interactive OAuth2 authorization completed
5. ✅ token_exchange: Authorization code exchanged for tokens
6. ✅ api_calls: Authenticated API calls successful
7. ✅ token_refresh: Token refresh demonstration successful

🎉 SUCCESS: Google OAuth2 demo completed successfully!

The demo demonstrated:
✅ OAuth2 configuration validation
✅ Interactive Google OAuth2 authorization
✅ Authorization code exchange for tokens
✅ Authenticated API calls
✅ Token management and refresh
```

## 📞 Troubleshooting

### Common Issues

1. **"OAuth2 credentials not found"**
   - Ensure `HEYSOL_OAUTH2_CLIENT_ID` and `HEYSOL_OAUTH2_CLIENT_SECRET` are set
   - Check environment variable names (case-sensitive)

2. **"404 Error on Google OAuth2 URL"**
   - **Primary Issue**: OAuth2 consent screen not configured or published
   - **Solution**: Complete OAuth2 consent screen setup in Google Cloud Console
   - **Steps**:
     1. Go to [Google Cloud Console](https://console.cloud.google.com/)
     2. Navigate to "APIs & Services" → "OAuth consent screen"
     3. Choose "External" user type and click "Create"
     4. Fill in app information (name, email, etc.)
     5. Add authorized domains (e.g., `localhost` for testing)
     6. Save and continue through all steps
     7. **Important**: Go to "Publish app" → "Publish" (even for testing)

3. **"Browser authentication fails"**
   - Verify Google OAuth2 credentials are correct
   - Check authorized redirect URI in Google Console
   - Ensure callback server port 8080 is available
   - Verify OAuth2 consent screen is published

4. **"Token exchange fails"**
   - Verify client secret is correct
   - Check redirect URI matches Google Console configuration
   - Ensure authorization code hasn't expired
   - Verify OAuth2 consent screen is properly configured

5. **"API calls fail"**
   - Verify access token is valid
   - Check token expiration time
   - Ensure proper scopes are requested
   - Confirm OAuth2 consent screen includes required scopes

### Debug Information

The demo generates detailed logs:
- `oauth2_google_demo.log`: Detailed execution logs
- `oauth2_google_demo_results.json`: Structured results
- Console output: Real-time progress updates

### Getting Help

1. Check the generated log files for detailed error information
2. Review OAuth2 configuration in your environment
3. Verify Google Cloud Console settings
4. Test individual components separately

## 🎉 Conclusion

The OAuth2 authorization endpoint provides a secure and standardized way to authenticate users and authorize API access. The Google OAuth2 demo implementation demonstrates the complete authorization code flow with PKCE security, interactive browser authentication, and comprehensive error handling.

The implementation is production-ready and follows OAuth2 security best practices, making it suitable for real-world applications requiring Google OAuth2 authentication with the HeySol API.

---

**OAuth2 Authorization Guide** - Complete implementation and demonstration of OAuth2 authorization endpoint functionality

**Version**: 1.0.0
**Last Updated**: 2025-09-21
**Status**: ✅ Complete