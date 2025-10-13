# LLM Plugin Development Research Notes

## Simon Willison's LLM Tool Research

### Tool Overview
- **Name**: LLM (Command Line Interface for Large Language Models)
- **Author**: Simon Willison
- **Repository**: https://github.com/simonw/llm
- **Documentation**: https://llm.datasette.io/

### Plugin Architecture
- Uses `@llm.hookimpl` decorator for plugin registration
- Main hook: `register_models(register)` function
- Supports sync and async model variants
- Pydantic-based options validation
- Environment variable based authentication
- Automatic plugin discovery after installation

### Plugin Installation Pattern
```bash
llm install llm-provider-name
```

## Reference Plugin Analysis

### llm-anthropic (Primary Reference)
**Repository**: https://github.com/simonw/llm-anthropic

**Key Implementation Details**:
- File: `llm_anthropic.py` (single file implementation)
- Model Classes: `ClaudeMessages` (sync), `AsyncClaudeMessages` (async)
- Options: `ClaudeOptions` class with Pydantic validation
- Authentication: `_Shared` class with API key configuration
- Registration: Multiple models with aliases and capabilities

**Code Patterns**:
```python
@llm.hookimpl
def register_models(register):
    register(
        ClaudeMessages("claude-3-opus-20240229"),
        AsyncClaudeMessages("claude-3-opus-20240229"),
    )

class _Shared:
    needs_key = "anthropic"
    key_env_var = "ANTHROPIC_API_KEY"

class ClaudeOptions(llm.Options):
    max_tokens: Optional[int] = Field(...)
    temperature: Optional[float] = Field(...)
```

### Other Reference Plugins
- **llm-openai**: Advanced features, tools, vision support
- **llm-gemini**: Google integration, search capabilities
- **llm-ollama**: Local model provider pattern

## Z.ai API Research

### API Configuration
- **Base URL**: `https://api.z.ai/api/paas/v4/`
- **Chat Endpoint**: `POST /chat/completions`
- **Authentication**: Bearer token
- **Documentation**: https://docs.z.ai/

### Available Models
1. **GLM-4.6** - Latest text generation model
2. **GLM-4.5V** - Vision model (supports images)
3. **GLM-4-32b-0414-128K** - Large context (128K tokens)
4. **CogVideoX-3** - Video generation model

### SDKs and Integration
- **Python SDK**: `pip install zai-sdk`
- **Java SDK**: Available via Maven
- **OpenAI Compatibility**: Supports OpenAI SDKs for migration

### Authentication
```python
import os
api_key = os.environ.get("ZAI_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
```

### API Request Format (Expected)
```python
{
    "model": "glm-4.6",
    "messages": [
        {"role": "user", "content": "Your prompt here"}
    ],
    "temperature": 1.0,
    "max_tokens": 4096,
    "stream": false
}
```

## Plugin Development Strategy

### File Structure
```
llm-zai/
├── llm_zai.py          # Main plugin implementation
├── __init__.py         # Package metadata
├── test_llm_zai.py     # Unit tests
├── README.md           # Documentation
├── pyproject.toml      # Modern Python packaging
└── LICENSE             # Open source license
```

### Core Components
1. **Authentication Class**: `_Shared` with API key configuration
2. **Options Class**: `ZaiOptions` with Pydantic validation
3. **Model Classes**: `ZaiChat` (sync) and `AsyncZaiChat` (async)
4. **Registration Hook**: `register_models` function
5. **Error Handling**: Comprehensive HTTP error management

### Key Implementation Tasks
1. Convert LLM message format to Z.ai API format
2. Handle authentication and API key management
3. Implement request/response conversion
4. Add error handling for various HTTP status codes
5. Support streaming if available
6. Register multiple models with appropriate capabilities

## Technical Considerations

### Model Capabilities to Configure
- `can_stream`: True/False based on Z.ai streaming support
- `supports_images`: True for GLM-4.5V
- `supports_pdf`: Determine from Z.ai capabilities
- `default_max_tokens`: Model-specific defaults
- `supports_tools`: If Z.ai supports function calling

### Error Scenarios to Handle
- 401/403: Authentication failures
- 429: Rate limiting
- 500/502/503: Server errors
- Network timeouts
- Invalid parameters

### Testing Strategy
1. Mock HTTP requests for unit tests
2. Test message format conversion
3. Test response parsing
4. Integration tests with real API (if possible)
5. Manual testing with LLM CLI

## Development Environment

### Setup Commands
```bash
# Create project
mkdir llm-zai
cd llm-zai

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Development dependencies
pip install llm httpx pydantic pytest pytest-mock

# Development installation
pip install -e .
```

### Testing Commands
```bash
# Unit tests
pytest

# Plugin registration test
llm --help | grep zai

# Manual testing
llm -m zai-glm-4.6 "Test message"
```

## Next Steps

1. **Verify Z.ai API Details**: Confirm exact request/response formats
2. **Test API Authentication**: Obtain test API key
3. **Check Streaming Support**: Determine if Z.ai supports streaming
4. **Identify Coder-Specific Models**: Confirm which models are for coding
5. **Rate Limit Research**: Understand API limits for different plans

## Resources

### Documentation
- LLM Plugin Tutorial: https://llm.datasette.io/en/stable/plugins.html
- Z.ai API Docs: https://docs.z.ai/
- Pydantic Documentation: https://pydantic-docs.helpmanual.io/

### Code References
- llm-anthropic source: https://github.com/simonw/llm-anthropic/blob/main/llm_anthropic.py
- LLM base classes: Available in the llm package source

---

*Research conducted on October 12, 2025*