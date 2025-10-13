# LLM Plugin for Z.ai Coder Plans - Development Task Checklist

Based on research conducted on October 12, 2025, this checklist provides a comprehensive roadmap for creating a plugin to add Z.ai's coder models to Simon Willison's LLM command line tool.

## Research Summary

### LLM Tool Architecture
- Created by Simon Willison
- Plugin-based architecture using `@llm.hookimpl` decorators
- Model registration through `register_models()` function
- Supports sync/async model variants
- Uses Pydantic for options validation
- Authentication via API keys and environment variables

### Z.ai API Details
- **Base URL**: `https://api.z.ai/api/paas/v4/`
- **Authentication**: Bearer token (ZAI_API_KEY)
- **Available Models**:
  - GLM-4.6 (latest text model)
  - GLM-4.5V (vision model, supports images)
  - GLM-4-32b-0414-128K (large context, 128K tokens)
  - Additional models for specific use cases

### Reference Implementations
- **llm-anthropic**: https://github.com/simonw/llm-anthropic
- **llm-openai**: https://github.com/simonw/llm-openai
- **llm-gemini**: https://github.com/simonw/llm-gemini
- **llm-ollama**: https://github.com/simonw/llm-ollama

---

## Phase 1: Project Setup and Structure

### 1.1 Initialize Plugin Project
- [ ] Create new directory: `llm-zai`
- [ ] Initialize Python package with `pyproject.toml` or `setup.py`
- [ ] Set up package metadata (name, version, author, description)
- [ ] Add dependencies: `llm` (as plugin dependency), `httpx` (for API calls), `pydantic` (for options validation)

### 1.2 Create Basic Plugin Structure
- [ ] Create main plugin file: `llm_zai.py`
- [ ] Create `__init__.py` with package metadata
- [ ] Create `README.md` with installation and usage instructions
- [ ] Create `LICENSE` file (recommend MIT or Apache-2.0 like other plugins)

## Phase 2: Core Plugin Implementation

### 2.1 Import Dependencies and Base Classes
- [ ] Import required LLM base classes and utilities
- [ ] Import HTTP client (httpx) for API calls
- [ ] Import Pydantic for model options validation
- [ ] Set up logging for debugging

### 2.2 Define Authentication and Configuration
- [ ] Create `_Shared` class with authentication settings:
  - [ ] Set `needs_key = "zai"`
  - [ ] Set `key_env_var = "ZAI_API_KEY"`
  - [ ] Configure API base URL: `https://api.z.ai/api/paas/v4/`
  - [ ] Set default capabilities (can_stream, supports_tools, etc.)

### 2.3 Create Model Options Class
- [ ] Define `ZaiOptions` class inheriting from `llm.Options`
- [ ] Add Pydantic fields for:
  - [ ] `temperature` (float, default 1.0, range 0.0-2.0)
  - [ ] `max_tokens` (int, default None/4096)
  - [ ] `top_p` (float, default None, range 0.0-1.0)
  - [ ] `stream` (bool, default False)
  - [ ] Any Z.ai-specific parameters

### 2.4 Implement Model Classes
- [ ] Create base `ZaiChat` class inheriting from `llm.Model`
- [ ] Implement synchronous `__call__` method:
  - [ ] Convert LLM prompt format to Z.ai API format
  - [ ] Make HTTP POST request to chat completions endpoint
  - [ ] Handle authentication headers
  - [ ] Process response and convert to LLM response format
  - [ ] Handle errors and HTTP status codes

- [ ] Create async variant `AsyncZaiChat` class:
  - [ ] Implement async `__call__` method using httpx async client
  - [ ] Mirror sync implementation but with async/await patterns

- [ ] Add streaming support if supported by Z.ai:
  - [ ] Implement `stream_chat` method for streaming responses
  - [ ] Handle server-sent events or chunked responses

### 2.5 Model Registration
- [ ] Implement `register_models` hook function:
  - [ ] Register GLM-4.6 model with appropriate aliases
  - [ ] Register GLM-4.5V model with vision support flag
  - [ ] Register GLM-4-32b-0414-128K model with large context
  - [ ] Set appropriate defaults for each model (max_tokens, etc.)
  - [ ] Configure model capabilities (supports_images, supports_pdf, etc.)

## Phase 3: API Integration Details

### 3.1 Request/Response Handling
- [ ] Implement message format conversion:
  - [ ] Convert LLM message format to Z.ai chat format
  - [ ] Handle system messages, user messages, assistant messages
  - [ ] Process attachments/images for vision models

- [ ] Implement response parsing:
  - [ ] Extract generated text from Z.ai response
  - [ ] Handle usage statistics (tokens used, etc.)
  - [ ] Parse any model-specific metadata

### 3.2 Error Handling
- [ ] Implement comprehensive error handling:
  - [ ] HTTP 401/403 for authentication issues
  - [ ] HTTP 429 for rate limiting
  - [ ] HTTP 500/502/503 for server errors
  - [ ] Network timeouts and connection issues
  - [ ] Invalid request parameters

### 3.3 Configuration Management
- [ ] Support environment variable configuration
- [ ] Add model-specific parameter validation
- [ ] Implement reasonable defaults for all parameters

## Phase 4: Testing and Validation

### 4.1 Unit Tests
- [ ] Create test file: `test_llm_zai.py`
- [ ] Write tests for message format conversion
- [ ] Write tests for response parsing
- [ ] Mock HTTP requests for isolated testing
- [ ] Test error handling scenarios

### 4.2 Integration Tests
- [ ] Test plugin registration with LLM tool
- [ ] Test basic chat completion functionality
- [ ] Test streaming functionality (if implemented)
- [ ] Test vision capabilities with GLM-4.5V
- [ ] Test authentication with valid API key

### 4.3 Manual Testing
- [ ] Install plugin in development mode
- [ ] Test with `llm --help` to verify model registration
- [ ] Test basic prompts: `llm -m zai-glm-4.6 "Hello, world!"`
- [ ] Test with different options: `llm -m zai-glm-4.6 --temp 0.7 "Write code"`
- [ ] Test streaming: `llm -m zai-glm-4.6 --stream "Explain APIs"`

## Phase 5: Documentation and Release

### 5.1 Documentation
- [ ] Complete `README.md` with:
  - [ ] Installation instructions
  - [ ] Supported models and their capabilities
  - [ ] Configuration options
  - [ ] Usage examples
  - [ ] API key setup instructions

- [ ] Add inline code documentation
- [ ] Document any limitations or known issues

### 5.2 Package Publication
- [ ] Register package on PyPI as `llm-zai`
- [ ] Verify installation via `llm install llm-zai`
- [ ] Test plugin discovery and model registration
- [ ] Create GitHub repository for source code

## Phase 6: Advanced Features (Optional)

### 6.1 Enhanced Capabilities
- [ ] Add support for function calling if Z.ai supports it
- [ ] Implement tool use capabilities
- [ ] Add support for structured outputs
- [ ] Implement thinking mode if available

### 6.2 Performance Optimizations
- [ ] Add request/response caching where appropriate
- [ ] Optimize for large context handling
- [ ] Implement connection pooling for HTTP requests

### 6.3 Developer Experience
- [ ] Add debug logging options
- [ ] Create setup scripts for easy configuration
- [ ] Add configuration file support for advanced users

---

## Key Technical Details

### API Endpoint Configuration
```
Base URL: https://api.z.ai/api/paas/v4/
Chat Completions: POST /chat/completions
Authentication: Bearer token (ZAI_API_KEY)
```

### Model Aliases to Implement
- `zai-glm-4.6` → GLM-4.6 (primary text model)
- `zai-glm-4.5v` → GLM-4.5V (vision model)
- `zai-glm-4-32b` → GLM-4-32b-0414-128K (large context)
- `zai-coder` → Primary coding model (identify specific coder model)

### Reference Implementation Pattern
Use `llm-anthropic` as the primary reference:
- File structure and naming conventions
- Authentication patterns
- Model registration approach
- Error handling strategies
- Options validation with Pydantic

### Development Environment Setup
```bash
# Create development environment
python -m venv llm-zai-env
source llm-zai-env/bin/activate  # On Windows: llm-zai-env\Scripts\activate

# Install development dependencies
pip install llm httpx pydantic pytest pytest-mock

# Install in development mode
pip install -e .
```

### Testing Commands
```bash
# Run unit tests
pytest test_llm_zai.py

# Test plugin registration
llm --help | grep zai

# Test basic functionality
llm -m zai-glm-4.6 "Test prompt"
```

---

## Resources and References

### Official Documentation
- **LLM Tool**: https://llm.datasette.io/
- **LLM Plugins**: https://llm.datasette.io/en/stable/plugins.html
- **Z.ai API**: https://docs.z.ai/
- **Z.ai Models**: https://z.ai/model-api

### Reference Plugins
- **llm-anthropic**: https://github.com/simonw/llm-anthropic
- **llm-openai**: https://github.com/simonw/llm-openai
- **llm-gemini**: https://github.com/simonw/llm-gemini
- **llm-ollama**: https://github.com/simonw/llm-ollama

### Key Files to Study
- `llm_anthropic.py` - Complete reference implementation
- `llm_openai.py` - Advanced features like tools
- `setup.py` files from reference plugins
- `pyproject.toml` for modern Python packaging

This checklist provides a comprehensive roadmap for developing a production-ready LLM plugin for Z.ai's coder models that follows established patterns from existing plugins in the ecosystem.