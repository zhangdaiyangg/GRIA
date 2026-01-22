"""
Configuration file: Stores all adjustable parameters for the system.
"""

# Vector Database Configuration
VECTOR_DB_CONFIG = {
    "vector_store_path": "./vector_store",  # Location to store the vector database
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",  # Selection of embedding model
    "embedding_dimension": 768,  # Dimension of embedding vectors
    "distance_metric": "cosine",  # Similarity calculation method: cosine, euclidean, dot_product
    "top_k_results": 8,  # Maximum number of results returned by retrieval
    "use_gpu": True,  # Whether to use GPU (if available)
    "batch_size": 32,  # Batch size, used to control GPU memory usage
}

# Document Processing Configuration
DOCUMENT_PROCESSOR_CONFIG = {
    "chunk_size": 1000,  # Text chunk size (number of characters)
    "chunk_overlap": 200,  # Chunk overlap size (number of characters)
    "supported_extensions": [".pdf"],  # Supported file types
    "max_file_size_mb": 50,  # Maximum size for a single file (MB)
}

# Available LLM providers and their models
LLM_PROVIDERS = {
    "openai": {
        "display_name": "OpenAI",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo","ChatGPT-4o-Latest"],
        "api_base": "https://api.poe.com/v1"
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-chat-light"],
        "api_base": "https://api.deepseek.com/v1"
    }
}


#Enter your key here
# Large Language Model Configuration
LLM_CONFIG = {
    "provider": "openai",  # Model provider: openai, deepseek
    "model_name": "gemini-3-flash",  # Model name
    "temperature": 1.0,  # Temperature parameter (0-1), higher means more creative, lower means more deterministic
    "max_tokens": 50000,  # Maximum number of tokens to generate
    "top_p": 0.95,  # Nucleus sampling parameter (0-1)
    "frequency_penalty": 0,  # Frequency penalty (-2.0-2.0)
    "presence_penalty": 0,  # Presence penalty (-2.0-2.0)
    "openai_api_key": " ",  # OpenAI API Key Enter your key here
    "deepseek_api_key": "",  # DeepSeek API Key
    "system_prompt": """You are a helpful assistant who answers questions based on the information provided in the context.
Please ensure your answers are concise, clear, and relevant to the question."""
}

# RAG Configuration
RAG_CONFIG = {
    "similarity_threshold": 0.45,  # Similarity threshold, retrieval results below this value will be filtered
    "max_context_length": 4000,  # Maximum context length (number of characters)
    "context_header": "Based on the following information:\n\n",  # Context header
    "context_footer": "\n\nAnswer the question: ",  # Context footer
}

# Application Configuration
APP_CONFIG = {
    "app_title": "PDF RAG System",
    "app_width": 1200,
    "app_height": 800,
    "theme": "light",  # light or dark
    "debug_mode": False,  # Whether to show debug information
}