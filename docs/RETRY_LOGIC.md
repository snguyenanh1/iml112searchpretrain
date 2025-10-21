# Retry Logic for API Errors

## Overview

The system implements intelligent retry logic to handle API failures, with special handling for 503 Service Unavailable errors from Google Gemini and other LLM providers.

## Implementation

### Location
`src/iML/llm/base_chat.py` - `BaseAssistantChat.assistant_chat()` method

### Features

#### 1. **503 Error Detection**
The `is_503_error()` function detects Service Unavailable errors through multiple methods:
- HTTP 503 status code in exception message
- `status_code` attribute check
- `response.status_code` attribute check (for requests library)
- gRPC UNAVAILABLE errors (code 14)

#### 2. **Intelligent Retry Strategy**

##### For 503 Service Unavailable Errors:
- **Wait Time**: 5 minutes (300 seconds)
- **Reason**: API is overloaded and needs time to recover
- **Max Retries**: 10 attempts (configurable)
- **Total Max Wait**: Up to 50 minutes

Example log output:
```
🔄 Attempt 1/10 - Service Unavailable (503). API is overloaded.
⏳ Waiting 300 seconds (5 minutes) before retry...
🔁 Retrying request (attempt 2/10)...
```

##### For Other Errors:
- **Wait Time**: Exponential backoff (32s, 64s, 128s, 128s, ...)
- **Formula**: `min(128, 32 * 2^(attempt-1))`
- **Max Retries**: 10 attempts (configurable)

Example log output:
```
⚠️  Attempt 1/10 failed: ValueError: Invalid response format
⏳ Waiting 32 seconds before retry...
🔁 Retrying request (attempt 2/10)...
```

## Usage

The retry logic is automatically applied to all LLM API calls through the `assistant_chat()` method:

```python
# Standard usage - retry logic is automatic
response = llm.assistant_chat("Your prompt here")

# Custom max retries (default is 10)
response = llm.assistant_chat("Your prompt here", max_retries=15)
```

## Error Types Handled

### 503 Service Unavailable
- Google Gemini API overload
- Vertex AI service unavailable
- Other cloud provider temporary outages

### Other Errors (with exponential backoff)
- Rate limiting (429)
- Temporary network issues
- Transient API failures
- Timeout errors

## Configuration

You can customize the retry behavior:

```python
# In your agent code
response = llm.assistant_chat(
    message="Your prompt",
    max_retries=20,  # Increase max retries for critical operations
    max_lines=2000   # Also supports line truncation
)
```

## Benefits

1. **Resilience**: Automatically recovers from temporary API failures
2. **Cost Efficiency**: Avoids wasting expensive compute by handling overload gracefully
3. **User Experience**: No manual intervention required for transient errors
4. **Logging**: Clear visibility into retry attempts and wait times
5. **Flexibility**: Different strategies for different error types

## Example Scenarios

### Scenario 1: Gemini API Overload
```
Request 1 → 503 Error → Wait 5 min → Retry → Success
Total time: ~5 minutes
```

### Scenario 2: Transient Network Error
```
Request 1 → Network Error → Wait 32s → Retry → Success
Total time: ~32 seconds
```

### Scenario 3: Persistent 503 Error
```
Request 1 → 503 → Wait 5 min → Retry
Request 2 → 503 → Wait 5 min → Retry
Request 3 → 503 → Wait 5 min → Retry
...
Request 10 → 503 → Fail (max retries reached)
Total time: ~45 minutes before giving up
```

## Monitoring

All retry attempts are logged with emoji indicators for easy monitoring:
- 🔄 = 503 Service Unavailable detected
- ⏳ = Waiting before retry
- 🔁 = Retrying request
- ⚠️  = Non-503 error occurred
- ❌ = Max retries reached, giving up

## Best Practices

1. **Set appropriate max_retries** for your use case:
   - Production: 10-15 retries (default: 10)
   - Development: 5-8 retries
   - Critical operations: 20+ retries

2. **Monitor logs** for patterns:
   - Frequent 503 errors → Consider rate limiting or API quota increase
   - Frequent other errors → Check network/code issues

3. **Plan for wait times**:
   - 503 errors can take up to 50 minutes to resolve (10 retries × 5 min)
   - Factor this into your pipeline timeouts

4. **Use appropriate timeout settings**:
   - Ensure `per_execution_timeout` in config is large enough to accommodate retries
   - Recommended: At least 60 minutes for critical operations

## Future Enhancements

Potential improvements to consider:
- Configurable wait time for 503 errors
- Adaptive retry based on error patterns
- Circuit breaker pattern for persistent failures
- Metrics collection for retry statistics

