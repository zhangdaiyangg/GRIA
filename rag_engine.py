"""
RAG Engine Module: Responsible for integrating retrieval and generation
"""

import os
import logging
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
import openai
from vector_store import VectorStore
from config import RAG_CONFIG, LLM_CONFIG, LLM_PROVIDERS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, vector_store: VectorStore, rag_config=None, llm_config=None):
        """Initialize RAG Engine"""
        self.vector_store = vector_store
        self.rag_config = rag_config or RAG_CONFIG
        self.llm_config = llm_config or LLM_CONFIG

        # Initialize API settings
        self._setup_api()

    # def _setup_api(self):
    #     """Setup API configuration"""
    #     # OpenAI config
    #     openai.api_key = self.llm_config.get("openai_api_key", "")
    #
    #     logger.info(f"RAG engine initialized, using provider: {self.llm_config['provider']}")
    def _setup_api(self):
        """Setup API configuration"""
        provider = self.llm_config.get("provider")

        if provider == "openai":
            # 1. Set API Key (read from LLM_CONFIG)
            api_key = self.llm_config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment variables.")
            openai.api_key = api_key

            # 2. Set API Base URL (read from LLM_PROVIDERS, this is key!)
            # This makes the code more flexible, address is determined by config
            api_base = LLM_PROVIDERS.get(provider, {}).get("api_base")
            if api_base:
                openai.api_base = api_base
                logger.info(f"Using custom API base for OpenAI: {api_base}")
            else:
                # If not configured, use official default address
                openai.api_base = 'https://api.openai.com/v1'
                logger.info("Using official OpenAI API endpoint.")

        # deepseek logic remains unchanged
        elif provider == "deepseek":
            logger.info("DeepSeek provider selected. API calls will be handled by requests library.")

        logger.info(f"RAG engine initialized, current provider: {provider}")

    def _rewrite_query(self, original_query: str) -> str:
        """
        Use LLM in the background to polish the original query for optimized vector retrieval.
        This is an internal optimization, transparent to the user.
        """
        # System prompt designed specifically for query rewriting tasks
        rewrite_prompt = f"""
        You are a top-tier GNU Radio expert system analyst. Your mission is to deconstruct a user's request into a detailed, technical search query optimized for a vector database containing Python code examples.

        When the user asks for a Python code solution, follow these steps:
        1.  **Identify the Core Task:** Clearly state the primary goal (e.g., "Channel Energy Detection", "FM Demodulation").
        2.  **List Key GNU Radio Blocks:** Name the specific, essential GNU Radio blocks required. Use their full names (e.g., `blocks.complex_to_mag_squared`, `filter.moving_average_ff`).
        3.  **Outline the Signal Chain:** Describe the step-by-step connection plan (signal flow) from the source to the sink. This is the implementation logic.
        4.  **Mention Critical Parameters:** If applicable, include key parameters that are essential for the implementation, such as `samp_rate`.

        The output MUST be a concise, dense paragraph of technical keywords, block names, and processing steps. Do NOT write Python code or provide conversational explanations in the rewritten query.

        ---
        **EXAMPLE:**

        **Original Query:** "Please implement a channel energy detection module and show the output."

        **Rewritten Query:** "Implementation of a channel energy detector in GNU Radio. The signal processing chain is: `analog.sig_source_c` for signal generation, passed to `blocks.throttle` for rate control. Signal power is calculated with `blocks.complex_to_mag_squared`. The result is smoothed using a `filter.moving_average_ff` block. The final real-time energy value is displayed with a `qtgui.number_sink`."
        ---

        **YOUR TASK:**

        **Original Query:** "{original_query}"
        **Rewritten Query:**
        """

        try:
            # Call using the currently selected LLM provider
            provider = self.llm_config["provider"]

            # Use specific parameters to ensure quality and speed of rewriting
            rewrite_params = {
                "temperature": 0.4,  # Deterministic output
                "max_tokens": 2000,  # Limit output length
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }

            rewritten_query = original_query  # Default to original query
            rewritten_query = rewritten_query + original_query
            if provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.llm_config["model_name"],
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    **rewrite_params
                )
                rewritten_query = response.choices[0].message.content.strip()

            elif provider == "deepseek":
                messages = [{"role": "user", "content": rewrite_prompt}]
                data = {
                    "model": self.llm_config["model_name"],
                    "messages": messages,
                    **rewrite_params
                }
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.llm_config['deepseek_api_key']}"
                }
                api_base = LLM_PROVIDERS["deepseek"]["api_base"]

                response = requests.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    data=json.dumps(data)
                )
                response.raise_for_status()
                rewritten_query = response.json()["choices"][0]["message"]["content"].strip()

            # Clean up possible irrelevant prefixes, e.g., "Rewritten Query:"
            if "Rewritten Query:" in rewritten_query:
                rewritten_query = rewritten_query.split("Rewritten Query:")[-1].strip()

            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten_query}'")
            return rewritten_query

        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}. Using original query for retrieval.")
            return original_query  # Safely return original query in case of any error

    def query(self, query: str) -> Dict[str, Any]:
        """Process query and return augmented answer"""
        logger.info(f"Processing query: {query}")
        rewritten_query = self._rewrite_query(query)

        print(f"{rewritten_query}")

        # Retrieve relevant documents
        search_results = self.vector_store.search(
            rewritten_query,
            top_k=None,
            threshold=self.rag_config["similarity_threshold"]
        )

        if not search_results:
            logger.warning("No relevant documents found")
            return {
                "answer": "Sorry, I could not find information relevant to your question in the knowledge base.",
                "contexts": [],
                "query": query
            }

        # Build context
        context = self._build_context(search_results)

        # Generate answer
        answer = self._generate_answer(query, context)

        # Return result
        result = {
            "answer": answer,
            "contexts": [
                {
                    "text": item["text"],
                    "source": item["file_name"],
                    "score": item["score"]
                }
                for item in search_results
            ],
            "query": query
        }

        return result

    def _build_context(self, search_results: List[Dict]) -> str:
        """Build context from search results"""
        # Add context header
        context = self.rag_config["context_header"]

        # Add content of each document chunk
        total_length = len(context)
        max_length = self.rag_config["max_context_length"]

        for i, result in enumerate(search_results):
            # Extract text and source information
            text = result["text"]
            source = result["file_name"]

            # Format context entry
            entry = f"Document {i + 1} (from {source}):\n{text}\n\n"

            # Check if max context length is exceeded
            if total_length + len(entry) > max_length:
                # If exceeded, truncate this entry to fit context
                available_space = max_length - total_length
                if available_space > 100:  # Ensure enough space for meaningful content
                    entry = f"Document {i + 1} (from {source}):\n{text[:available_space - 50]}...\n\n"
                    context += entry
                break

            # Add entry to context
            context += entry
            total_length += len(entry)

        # Add context footer
        context += self.rag_config["context_footer"]

        return context

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using selected model"""
        provider = self.llm_config["provider"]

        try:
            # Choose API call method based on provider
            if provider == "openai":
                return self._generate_with_openai(query, context)
            elif provider == "deepseek":
                return self._generate_with_deepseek(query, context)
            else:
                raise ValueError(f"Unsupported model provider: {provider}")

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _generate_with_openai(self, query: str, context: str) -> str:
        """Generate answer using OpenAI API"""
        # Build full prompt
        messages = [
            {"role": "system", "content": self.llm_config["system_prompt"]},
            {"role": "user", "content": f"{context}{query}"}
        ]

        # Call OpenAI API
        # response = openai.ChatCompletion.create(
        #     model=self.llm_config["model_name"],
        #     messages=messages,
        #     temperature=self.llm_config["temperature"],
        #     max_tokens=self.llm_config["max_tokens"],
        #     top_p=self.llm_config["top_p"],
        #     frequency_penalty=self.llm_config["frequency_penalty"],
        #     presence_penalty=self.llm_config["presence_penalty"]
        # )

        response = openai.ChatCompletion.create(
            model=self.llm_config["model_name"],
            messages=messages,
            temperature=self.llm_config["temperature"],
            max_tokens=self.llm_config["max_tokens"],
            top_p=self.llm_config["top_p"]
            # Removed frequency_penalty and presence_penalty
        )

        # Extract and return answer
        return response.choices[0].message.content

    def _generate_with_deepseek(self, query: str, context: str) -> str:
        """Generate answer using DeepSeek API"""
        # Build full prompt
        messages = [
            {"role": "system", "content": self.llm_config["system_prompt"]},
            {"role": "user", "content": f"{context}{query}"}
        ]

        # Prepare request data
        data = {
            "model": self.llm_config["model_name"],
            "messages": messages,
            "temperature": self.llm_config["temperature"],
            "max_tokens": self.llm_config["max_tokens"],
            "top_p": self.llm_config["top_p"],
            "frequency_penalty": self.llm_config["frequency_penalty"],
            "presence_penalty": self.llm_config["presence_penalty"]
        }

        # Get API base URL and key
        api_base = LLM_PROVIDERS["deepseek"]["api_base"]
        api_key = self.llm_config["deepseek_api_key"]

        if not api_key:
            raise ValueError("DeepSeek API key not set")

        # Set request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Send request to DeepSeek API
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )

        # Check response
        response.raise_for_status()
        response_data = response.json()

        # Extract and return answer
        return response_data["choices"][0]["message"]["content"]

    def update_rag_config(self, new_config: Dict) -> None:
        """Update RAG config"""
        self.rag_config.update(new_config)
        logger.info(f"RAG config updated: {self.rag_config}")

    def update_llm_config(self, new_config: Dict) -> None:
        """Update LLM config"""
        # Check if provider has changed
        provider_changed = ("provider" in new_config and
                            new_config["provider"] != self.llm_config.get("provider"))

        # Update config
        self.llm_config.update(new_config)

        # If provider changed, reset API
        if provider_changed:
            self._setup_api()

        logger.info(f"LLM config updated, current provider: {self.llm_config['provider']}")
