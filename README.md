# GNU Radio RAG Assistant

This project is an AI-powered assistant designed to help developers write Python code for GNU Radio. By utilizing Retrieval-Augmented Generation (RAG), it combines Large Language Models (LLMs) with specific GNU Radio documentation to provide accurate code suggestions and technical support.

## Features

- **RAG-Powered Chat**: Specialized assistance for GNU Radio Python development.
- **Document Integration**: Uses processed documentation for context-aware responses.
- **Hybrid UI**: Includes both a Desktop GUI and a Web interface.

## Installation

### 1. Prerequisites

Ensure you have Python installed. This project uses `uv` for efficient dependency management.

### 2. Install Packages

You can use `uv` for fast installation:

```bash
uv sync
```

Or use `pip` to install the requirements manually:

```bash
pip install flask pymupdf numpy sentence-transformers requests openai torch transformers scikit-learn
```

## Configuration

### 1. Knowledge Base Setup

The project includes processed documentation in the `summary/` directory. These files need to be indexed into the vector database.

Please extract all the compressed files in file `summary/` to the current folder.(Please ensure that all PDF files are in the current folder after decompression)

### 2. AI Model Configuration (config.py)

You can customize the LLM providers and models in `config.py`.

#### API Base URL & Provider Settings

Locate the `LLM_PROVIDERS` dictionary to modify API endpoints:

```python
LLM_PROVIDERS = {
    "openai": {
        "display_name": "OpenAI",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "ChatGPT-4o-Latest"],
        "api_base": "https://"  # Modify API address here
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-chat-light"],
        "api_base": "https://api.deepseek.com/v1"  # Modify API address here
    }
}
```

#### Model Selection & Parameters

Update the `LLM_CONFIG` to select your preferred model:

```python
LLM_CONFIG = {
    "provider": "openai",            # Choose "openai" or "deepseek"
    "model_name": "gemini-3-flash", # Choose specific model name
    "temperature": 0.7,
    "max_tokens": 1000,
}
```

#### API Key Setup

Insert your API keys in the `LLM_CONFIG` section:

```python
LLM_CONFIG = {
    # ... other config
    "openai_api_key": "your-openai-api-key-here",
    "deepseek_api_key": "your-deepseek-api-key-here",
}
```

## Usage

To launch the application:

```bash
uv run app.py
```

or

```bash
python app.py
```

From the main window, you can manage the knowledge base and click **"Open Web UI"** to use the browser-based chat interface.

`Survey Questionnaire/` includes survey questionnaires for some participants
