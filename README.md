# llm-zai

This plugin connects [Simon Willison's LLM](https://llm.datasette.io/) command-line tool to Z.ai's GLM language models.

## Installation

Install LLM first:

```bash
pip install llm
```

Then install the plugin:

```bash
pip install llm-zai
```

## Configuration

You need a Z.ai API key. Sign up at [Z.ai](https://z.ai/) to get one.

The plugin uses this priority order for API keys:
1. Key passed with `--key` option
2. Key stored with `llm keys set zai`
3. `ZAI_API_KEY` environment variable

### Set API Key (Recommended)

```bash
llm keys set zai
# Enter your API key when prompted
```

### Use Environment Variable

```bash
export ZAI_API_KEY="your-api-key-here"
```

### Pass Key Directly

```bash
llm --key "your-api-key-here" -m zai-glm-4.6 "Your prompt here"
```

## Models

The plugin supports these Z.ai models:

- **zai-glm-4.6** - Latest text generation model
- **zai-glm-4.5v** - Vision model with image support
- **zai-glm-4.5** - Standard text generation model
- **zai-glm-4.5-air** - Lightweight text generation model
- **zai-glm-4-32b** - Large context model (128K tokens)

All models disable streaming. Default max tokens: 4096 (8192 for GLM-4-32b).

```bash
# Text generation
llm -m zai-glm-4.6 "Explain quantum computing"

# Vision model
llm -m zai-glm-4.5v --image photo.jpg "Describe this image"
```


## Usage

```bash
# Text generation
llm -m zai-glm-4.6 "What is machine learning?"

# Control randomness (0.0-2.0)
llm -m zai-glm-4.6 --temp 0.7 "Write a creative story"

# Limit response length
llm -m zai-glm-4.6 --max-tokens 1000 "Explain the solar system"

# Analyze images
llm -m zai-glm-4.5v --image cat.jpg "What's in this image?"

# Compare images
llm -m zai-glm-4.5v --image img1.jpg --image img2.jpg "Compare these images"

# Process long documents
cat long_document.txt | llm -m zai-glm-4-32b "Summarize this document"
```

## Options

- **--temp** - Controls randomness (0.0 to 2.0, default 1.0)
- **--max-tokens** - Maximum tokens to generate
- **--top-p** - Nucleus sampling (0.0 to 1.0)

## List Models

```bash
llm models | grep zai
```


## Troubleshooting

**Invalid API Key** - Verify your key is correct and stored: `llm keys list`

**Rate Limiting** - Wait a few minutes or upgrade your Z.ai plan

**Network Issues** - Check your internet connection and firewall settings

**Model Not Found** - Check model spelling and availability in your Z.ai plan

## Development

Clone and set up the development environment:

```bash
git clone https://github.com/divsmith/llm-zai.git
cd llm-zai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[test]
```

### Testing

```bash
pytest
pytest -v  # Verbose output
pytest --cov=llm_zai  # With coverage
```

### Code Quality

```bash
black .  # Format code
flake8 .  # Lint
mypy llm_zai/  # Type check
```

### Build

```bash
python -m build
twine check dist/*  # Verify packages
```

## Contributing

Submit Pull Requests for contributions. Open an issue first for major changes.

## License

Apache License 2.0

## Support

- Plugin issues: [GitHub Issues](https://github.com/divsmith/llm-zai/issues)
- Z.ai API: [Z.ai Support](https://z.ai/support)
- LLM tool: [Documentation](https://llm.datasette.io/)