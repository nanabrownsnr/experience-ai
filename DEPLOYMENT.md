# ExperienceAI - Deployment Guide

## Installation Options

### Option 1: Install from Local Source (For Development)

```bash
# Direct installation
pip install /Users/nanabrown/experience-ai

# Editable installation (changes reflect immediately)
pip install -e /Users/nanabrown/experience-ai

# Install with development dependencies
pip install -e "/Users/nanabrown/experience-ai[dev]"
```

### Option 2: Install from Built Package

```bash
# Install the wheel file directly
pip install /Users/nanabrown/experience-ai/dist/experienced_ai-0.1.0-py3-none-any.whl
```

### Option 3: Deploy to PyPI (Recommended)

#### Prerequisites
1. Create accounts on [PyPI](https://pypi.org/account/register/) and [TestPyPI](https://test.pypi.org/account/register/)
2. Set up API tokens for secure authentication

#### Step 1: Test on TestPyPI First

```bash
# Upload to TestPyPI (testing repository)
python3 -m twine upload --repository testpypi dist/*

# You'll be prompted for credentials:
# Username: __token__
# Password: [your TestPyPI API token]
```

#### Step 2: Test Installation from TestPyPI

```bash
# Create a new virtual environment for testing
python3 -m venv test_env
source test_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ experience-ai

# Test the package
python -c "from experience_ai import EvolvingPrompt; print('✅ Package works!')"
```

#### Step 3: Deploy to Production PyPI

```bash
# Upload to production PyPI
python3 -m twine upload dist/*

# You'll be prompted for credentials:
# Username: __token__
# Password: [your PyPI API token]
```

#### Step 4: Install from PyPI

After successful deployment, anyone can install with:

```bash
pip install experience-ai
```

## Setting Up PyPI API Tokens

### For PyPI (Production)
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Give it a name like "experience-ai-upload"
5. Select "Entire account" or scope to specific project
6. Copy the token (starts with `pypi-`)

### For TestPyPI
1. Go to https://test.pypi.org/manage/account/
2. Follow the same steps as above

### Configure credentials (Optional)
Create a `~/.pypirc` file to avoid entering credentials each time:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

## Usage After Installation

### Basic Usage

```python
import os
from openai import OpenAI
from experience_ai import EvolvingPrompt, LocalStorageAdapter

# Set up LLM client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create storage and prompt manager
storage = LocalStorageAdapter("./my_ai_interactions.json")
base_prompt = "You are a helpful AI assistant."

prompt_manager = EvolvingPrompt(base_prompt, storage, client)

# Use in your application
current_prompt = prompt_manager.get_prompt()
# ... use prompt with your AI model ...

# Record successful interactions
prompt_manager.record_interaction(
    conversation="User asked for help with Python",
    outcome="task_completed",
    metadata={"language": "python"}
)
```

### Integration Examples

#### With Streamlit
```python
import streamlit as st
from experience_ai import EvolvingPrompt, LocalStorageAdapter

@st.cache_resource
def get_prompt_manager():
    storage = LocalStorageAdapter("streamlit_interactions.json")
    return EvolvingPrompt(base_prompt, storage, openai_client)

prompt_manager = get_prompt_manager()
```

#### With FastAPI
```python
from fastapi import FastAPI
from experience_ai import EvolvingPrompt, LocalStorageAdapter

app = FastAPI()

# Initialize once at startup
prompt_manager = EvolvingPrompt(base_prompt, storage, client)

@app.post("/chat")
async def chat(message: str):
    current_prompt = prompt_manager.get_prompt()
    # ... handle chat ...
    prompt_manager.record_interaction(message, "helpful_response")
    return response
```

## Package Information

- **PyPI Name**: `experience-ai`
- **Import Name**: `experience_ai`
- **Version**: 0.1.0
- **Python Requirements**: >=3.8
- **Dependencies**: `openai>=1.0.0`

## Updating the Package

1. Update version in `pyproject.toml` and `setup.py`
2. Rebuild the package: `python3 -m build`
3. Upload new version: `python3 -m twine upload dist/*`

## Troubleshooting

### Common Issues

1. **Package name conflicts**: If `experience-ai` is taken, modify the name in `setup.py` and `pyproject.toml`
2. **Import errors**: Make sure to import as `from experience_ai import ...` (the module name uses underscores)
3. **Permission errors**: Use API tokens instead of username/password
4. **Missing dependencies**: Ensure `openai` is installed in your environment

### Support

- GitHub Issues: https://github.com/nanabrown/experience-ai/issues
- Documentation: See README.md in the repository
