# llm-zai

A plugin for [Simon Willison's LLM](https://llm.datasette.io/) command-line tool that adds support for [Z.ai](https://z.ai/)'s GLM language models.

## Installation

First, install the LLM tool:

```bash
pip install llm
```

Then install this plugin:

```bash
pip install llm-zai
```

## Configuration

You'll need a Z.ai API key. You can get an API key by signing up at [Z.ai](https://z.ai/).

### Method 1: Use LLM's Native Secrets Management (Recommended)

Store your API key using LLM's built-in secrets management:

```bash
llm keys set zai
# Prompt will ask for your API key
```

This method is more secure as the key is stored encrypted and only used by the LLM tool.

### Method 2: Environment Variable (Legacy)

You can also set it as an environment variable:

```bash
export ZAI_API_KEY="your-api-key-here"
```

### Method 3: Pass Key Directly

Or pass the key directly with each request:

```bash
llm --key "your-api-key-here" -m zai-glm-4.6 "Your prompt here"
```

**Priority Order**: The plugin will look for API keys in this order:
1. Key passed with `--key` option
2. Key stored with `llm keys set zai`
3. `ZAI_API_KEY` environment variable

## Supported Models

This plugin supports the following Z.ai models:

### GLM-4.6 (zai-glm-4.6)
- **Description**: Latest text generation model
- **Aliases**: `glm-4.6`
- **Streaming**: Yes
- **Max Tokens**: 4096 (default)

```bash
llm -m zai-glm-4.6 "Explain quantum computing"
```

### GLM-4.5V (zai-glm-4.5v)
- **Description**: Vision model with image support
- **Aliases**: `glm-4.5v`
- **Streaming**: Yes
- **Max Tokens**: 4096 (default)
- **Images**: Yes

```bash
llm -m zai-glm-4.5v --image photo.jpg "Describe this image"
```

### GLM-4-32b (zai-glm-4-32b)
- **Description**: Large context model (128K tokens)
- **Aliases**: `glm-4-32b`, `glm-4-32b-0414-128k`
- **Streaming**: Yes
- **Max Tokens**: 8192 (default)

```bash
llm -m zai-glm-4-32b "Analyze this long document"
```

### Z.ai Coder (zai-coder)
- **Description**: Specialized coding model
- **Aliases**: `coder`, `zai-coder-llm`
- **Streaming**: Yes
- **Max Tokens**: 4096 (default)

```bash
llm -m zai-coder "Write a Python function to sort a list"
```

## Usage Examples

### Basic Usage

```bash
# Simple text generation
llm -m zai-glm-4.6 "What is machine learning?"

# With temperature control
llm -m zai-glm-4.6 --temp 0.7 "Write a creative story"

# With max tokens limit
llm -m zai-glm-4.6 --max-tokens 1000 "Explain the solar system"
```

### Streaming

```bash
# Stream the response as it's generated
llm -m zai-glm-4.6 --stream "Describe the history of computing"
```

### Vision Model

```bash
# Analyze images with GLM-4.5V
llm -m zai-glm-4.5v --image cat.jpg "What's in this image?"

# Multiple images
llm -m zai-glm-4.5v --image img1.jpg --image img2.jpg "Compare these images"
```

### Coding Assistant

```bash
# Use the specialized coder model
llm -m zai-coder "Write a REST API in Python using FastAPI"

# Code review
llm -m zai-coder --temp 0.2 < code.py "Review this code for bugs and improvements"
```

### Large Context Processing

```bash
# Process long documents with GLM-4-32b
cat long_document.txt | llm -m zai-glm-4-32b "Summarize this document"
```

## Model Options

All models support the following options:

### Temperature
Controls randomness in the output. Range: 0.0 to 2.0
- `0.0`: Deterministic, focused output
- `1.0`: Default balance
- `2.0`: Maximum creativity

```bash
llm -m zai-glm-4.6 --temp 0.1 "Write formal documentation"
llm -m zai-glm-4.6 --temp 1.5 "Brainstorm creative ideas"
```

### Max Tokens
Maximum number of tokens to generate.

```bash
llm -m zai-glm-4.6 --max-tokens 500 "Brief summary of AI"
```

### Top P
Nucleus sampling parameter. Range: 0.0 to 1.0

```bash
llm -m zai-glm-4.6 --top-p 0.9 "Generate text with focused vocabulary"
```

## Available Models

List all available models:

```bash
llm models | grep zai
```

Output:
```
zai-glm-4.6        Z.ai: zai-glm-4.6
zai-glm-4.5v       Z.ai: zai-glm-4.5v
zai-glm-4-32b      Z.ai: zai-glm-4-32b
zai-coder          Z.ai: zai-coder
```

## API Key Setup

### Temporary Setup (current session)

```bash
export ZAI_API_KEY="your-api-key-here"
```

### Persistent Setup

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
echo 'export ZAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Windows Setup

```cmd
set ZAI_API_KEY=your-api-key-here
```

Or for persistent setup in PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable('ZAI_API_KEY', 'your-api-key-here', 'User')
```

## Troubleshooting

### Invalid API Key
```
ValueError: Invalid Z.ai API key
```
- Verify your API key is correct
- Check that you've set the key using `llm keys set zai` or the `ZAI_API_KEY` environment variable
- If using stored keys, verify the key is set: `llm keys list`

### Rate Limiting
```
ValueError: Rate limit exceeded. Please try again later.
```
- Wait a few minutes before making more requests
- Consider upgrading your Z.ai plan for higher limits

### Network Issues
```
ValueError: Network error connecting to Z.ai
```
- Check your internet connection
- Verify firewall settings allow outbound HTTPS connections
- Try again later if the issue persists

### Model Not Found
```
ValueError: Z.ai API error: Model not found
```
- Check if the model name is spelled correctly
- Verify the model is available in your Z.ai plan

## Development

### Setup Development Environment

```bash
git clone https://github.com/simonw/llm-zai
cd llm-zai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[test]

# Run tests
pytest
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_zai

# Run specific test
pytest test_llm_zai.py::TestZaiChat::test_model_init -v
```

### Code Quality

```bash
# Format code
black llm_zai.py test_llm_zai.py

# Lint code
flake8 llm_zai.py test_llm_zai.py

# Type checking
mypy llm_zai.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References

- [LLM Tool Documentation](https://llm.datasette.io/)
- [Z.ai API Documentation](https://docs.z.ai/)
- [LLM Plugin Development Guide](https://llm.datasette.io/en/stable/plugins.html)
- [Z.ai Models](https://z.ai/model-api)

## Support

- For issues with this plugin, please [open an issue on GitHub](https://github.com/simonw/llm-zai/issues)
- For Z.ai API issues, contact [Z.ai support](https://z.ai/support)
- For LLM tool issues, see the [LLM documentation](https://llm.datasette.io/)