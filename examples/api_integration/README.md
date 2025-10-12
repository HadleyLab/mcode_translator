# API Integration Example

This example demonstrates how to integrate the mCODE Translator with external APIs, including ClinicalTrials.gov, CORE Memory, and webhook systems.

## What You'll Learn

- API connectivity testing and configuration
- CORE Memory integration for persistent storage
- Webhook setup for real-time notifications
- Rate limiting and error handling strategies
- Integration testing patterns

## Quick Start

```bash
cd examples/api_integration
python api_client_demo.py
```

## Expected Output

```
🚀 mCODE Translator - API Integration Demo
============================================================

📋 Configuration loaded
   • APIs configured: 3
   • CORE Memory: Enabled

1️⃣ Testing ClinicalTrials.gov API Integration
──────────────────────────────────────────────────
   🔍 Checking API connectivity...
   ✅ ClinicalTrials.gov API accessible
   📊 Rate limit status: OK
   🌐 Endpoint: https://clinicaltrials.gov/api/v2

2️⃣ Testing CORE Memory API Integration
──────────────────────────────────────────────────
   🔑 API Key: Configured
   🌐 Base URL: https://core.heysol.ai/api/v1
   📚 Available operations:
      • Memory ingestion
      • Semantic search
      • Webhook integration
      • Space management

   📊 Memory Statistics:
      • Total episodes: 1,247
      • Active spaces: 5
      • Storage used: 45.2 MB
      • Search latency: 120ms

3️⃣ Testing Webhook Integration
──────────────────────────────────────────────────
   🔗 Webhook Configuration:
      • URL: https://your-app.com/webhooks/mcode-results
      • Events: trial.processed, batch.completed, error.occurred
      • Authentication: HMAC-SHA256

   📤 Sample Webhook Payload:
      {
        "event": "trial.processed",
        "timestamp": "2024-01-15T10:30:00Z",
        "data": {
          "trial_id": "NCT02364999",
          "mcode_elements": 5,
          "processing_time": 0.15,
          "confidence_score": 0.94
        }
      }

4️⃣ Testing API Rate Limiting
──────────────────────────────────────────────────
   🚦 Rate Limiting Configuration:
      • Clinicaltrials Api:
         - requests per second: 10
         - requests per hour: 1000
         - burst limit: 50
      • Core Memory Api:
         - requests per minute: 60
         - requests per hour: 1000
         - concurrent requests: 5

5️⃣ Testing Error Handling and Retry Logic
──────────────────────────────────────────────────
   🛠️  Error Handling Strategies:
      • Network timeout (retry with backoff)
      • API rate limit exceeded (exponential backoff)
      • Invalid API key (fail fast)
      • Malformed response (parse error handling)
      • Service unavailable (circuit breaker)

6️⃣ Running Integration Tests
──────────────────────────────────────────────────
   🧪 Integration Test Results:
      • ClinicalTrials.gov API: ✅ PASS
      • CORE Memory API: ⚠️  SKIP (no API key)
      • Webhook delivery: ✅ PASS (mock)
      • Rate limiting: ✅ PASS
      • Error recovery: ✅ PASS

🎉 API Integration Demo completed!
```

## API Integration Points

### ClinicalTrials.gov API

The system integrates with the official ClinicalTrials.gov API v2:

```python
from src.utils.api_manager import APIManager

api_manager = APIManager(config)
trial_data = api_manager.fetch_trial("NCT02364999")
```

**Features:**
- Automatic rate limiting (10 req/sec, 1000 req/hour)
- Response caching to reduce API calls
- Error handling with exponential backoff
- Structured data parsing

### CORE Memory API

Persistent storage and semantic search capabilities:

```python
from src.services.heysol_client import HeySolClient

client = HeySolClient(api_key="your-key")
client.ingest_trial_data(processed_results)
search_results = client.search("breast cancer trials")
```

**Features:**
- Long-term data persistence
- Semantic search across all stored content
- Webhook notifications for data changes
- Multi-space organization

### Webhook System

Real-time notifications for processing events:

```python
# Webhook payload structure
{
    "event": "trial.processed",
    "timestamp": "2024-01-15T10:30:00Z",
    "data": {
        "trial_id": "NCT02364999",
        "mcode_elements": 5,
        "processing_time": 0.15,
        "confidence_score": 0.94
    }
}
```

**Supported Events:**
- `trial.processed` - Individual trial completion
- `batch.completed` - Batch processing finished
- `error.occurred` - Processing errors
- `memory.ingested` - Data stored in CORE Memory

## Configuration

### Environment Variables

```bash
# Required for CORE Memory features
export HEYSOL_API_KEY="your-api-key-here"

# Optional: API rate limiting
export API_RATE_LIMIT="10"
export API_TIMEOUT="30"
```

### Configuration Files

API settings are configured in `src/config/apis_config.json`:

```json
{
  "clinicaltrials": {
    "base_url": "https://clinicaltrials.gov/api/v2",
    "rate_limit": 10,
    "timeout": 30,
    "retry_attempts": 3
  },
  "core_memory": {
    "base_url": "https://core.heysol.ai/api/v1",
    "rate_limit": 60,
    "timeout": 15
  }
}
```

## Rate Limiting

The system implements comprehensive rate limiting:

| API | Limit | Window | Burst |
|-----|-------|--------|-------|
| ClinicalTrials.gov | 10 req/sec | 1 hour | 50 |
| CORE Memory | 60 req/min | 1 hour | 10 |
| Webhooks | 100 req/min | 1 hour | 20 |

## Error Handling

### Retry Strategies

- **Network timeouts**: Exponential backoff (1s, 2s, 4s, 8s)
- **Rate limits**: Wait for reset period + buffer
- **Server errors**: Retry up to 3 times
- **Auth errors**: Fail fast, no retry

### Circuit Breaker

- Automatically opens after 5 consecutive failures
- Half-open retry after 60 seconds
- Full recovery after successful request

## Files in This Example

- `api_client_demo.py` - Main demonstration script
- `README.md` - This documentation

## Testing Integration

### Unit Tests

```bash
# Test API clients individually
python -m pytest tests/unit/test_api_manager.py -v
python -m pytest tests/unit/test_heysol_client.py -v
```

### Integration Tests

```bash
# Test full API workflows
python -m pytest tests/integration/test_api_integration.py -v
```

### Load Testing

```bash
# Test rate limiting and performance
python -m pytest tests/performance/test_api_performance.py -v
```

## Production Deployment

### Monitoring

```python
# Enable API monitoring
from src.utils.metrics import APIMetrics

metrics = APIMetrics()
metrics.track_api_call("clinicaltrials", "fetch_trial", 0.15, success=True)
```

### Logging

```python
# Structured API logging
import logging

logger = logging.getLogger("api_integration")
logger.info("API call completed", extra={
    "api": "clinicaltrials",
    "endpoint": "studies",
    "duration": 0.15,
    "status": 200
})
```

## Troubleshooting

### Common Issues

- **API Key Errors**: Verify `HEYSOL_API_KEY` is set correctly
- **Rate Limiting**: Implement backoff strategies
- **Network Issues**: Check proxy/firewall settings
- **Webhook Failures**: Verify endpoint URLs and authentication

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python api_client_demo.py

# Test specific API
python -c "from src.utils.api_manager import APIManager; m = APIManager(); print('API OK')"
```

## Next Steps

After running this example:

1. **Configure API Keys**: Set up real API credentials
2. **Implement Webhooks**: Create endpoints for notifications
3. **Set up Monitoring**: Add dashboards and alerts
4. **Load Testing**: Test with production-scale data
5. **Security Review**: Audit API key management and data handling