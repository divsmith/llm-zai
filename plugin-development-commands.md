# LLM Z.ai Plugin Development Commands

## Project Setup Commands

```bash
# Create project directory
mkdir llm-zai
cd llm-zai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install llm httpx pydantic pytest pytest-mock

# Create basic project structure
touch llm_zai.py __init__.py test_llm_zai.py README.md LICENSE
```

## Development Commands

```bash
# Install plugin in development mode
pip install -e .

# Test plugin registration
llm --help | grep -i zai

# List available models
llm models

# Test basic functionality
llm -m zai-glm-4.6 "Hello, world!"

# Test with options
llm -m zai-glm-4.6 --temp 0.7 "Write a Python function"

# Test streaming (if supported)
llm -m zai-glm-4.6 --stream "Explain machine learning"

# Test vision model (if image support available)
llm -m zai-glm-4.5v --image path/to/image.jpg "Describe this image"
```

## Testing Commands

```bash
# Run unit tests
pytest test_llm_zai.py -v

# Run tests with coverage
pytest test_llm_zai.py --cov=llm_zai

# Run specific test
pytest test_llm_zai.py::test_message_conversion -v

# Mock API tests
pytest test_llm_zai.py -m "mock_api"
```

## Environment Setup

```bash
# Set Z.ai API key
export ZAI_API_KEY="your-api-key-here"

# For Windows
set ZAI_API_KEY=your-api-key-here

# Add to shell profile for persistence
echo 'export ZAI_API_KEY="your-api-key-here"' >> ~/.zshrc
```

## Package Publishing Commands

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to test PyPI
twine upload --repository testpypi dist/*

# Upload to production PyPI
twine upload dist/*

# Install from test PyPI
pip install --index-url https://test.pypi.org/simple/ llm-zai

# Install from production PyPI
pip install llm-zai
```

## Debugging Commands

```bash
# Enable verbose logging
export LLM_LOG_LEVEL=DEBUG

# Test plugin loading
python -c "import llm_zai; print('Plugin loaded successfully')"

# Check model registration
python -c "
import llm
print([m.model_id for m in llm.get_models() if 'zai' in m.model_id])
"

# Test API connection
curl -X POST https://api.z.ai/api/paas/v4/chat/completions \
  -H "Authorization: Bearer $ZAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }'
```

## Git Commands

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: LLM Z.ai plugin structure"

# Create meaningful commits
git add llm_zai.py
git commit -m "Implement basic ZaiChat model class"

git add test_llm_zai.py
git commit -m "Add unit tests for message conversion"

# Create release tag
git tag -a v1.0.0 -m "First release of llm-zai plugin"
git push origin v1.0.0
```

## Useful One-Liners

```bash
# Check if plugin is registered
llm models | grep zai

# Test all models quickly
for model in zai-glm-4.6 zai-glm-4.5v zai-glm-4-32b; do
  echo "Testing $model..."
  llm -m $model "test" --no-stream
done

# Validate API key
curl -H "Authorization: Bearer $ZAI_API_KEY" https://api.z.ai/api/paas/v4/models

# Check Python path for plugin
python -c "import llm_zai; print(llm_zai.__file__)"

# Monitor API usage (if endpoint available)
curl -H "Authorization: Bearer $ZAI_API_KEY" https://api.z.ai/api/paas/v4/usage
```

## Code Quality Commands

```bash
# Format code with black
black llm_zai.py test_llm_zai.py

# Lint with flake8
flake8 llm_zai.py test_llm_zai.py

# Type checking with mypy
mypy llm_zai.py

# Security check with bandit
bandit -r llm_zai.py

# Dependency check
pip-audit
```

## Clean Up Commands

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info/

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Clean test coverage
rm -rf .coverage htmlcov/

# Uninstall development version
pip uninstall llm-zai -y
```

## IDE/Editor Integration

### VS Code Setup
```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.flake8
code --install-extension ms-python.mypy-type-checker
```

### PyCharm Setup
- Configure Python interpreter to use venv
- Enable pytest test runner
- Set up code formatting with Black
- Configure type checking with mypy

---

**Note**: These commands assume a Unix-like environment (macOS/Linux). Adjust paths and syntax for Windows as needed.