"""
LLM adapters for different AI providers.

This module provides a unified interface for different LLM providers,
allowing the EvolvingPrompt system to work with any AI model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    
    All LLM providers should implement this interface to work with ExperienceAI.
    """
    
    @abstractmethod
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str, 
                     max_tokens: Optional[int] = None,
                     temperature: Optional[float] = None) -> str:
        """
        Generate text using the LLM.
        
        Args:
            system_prompt: The system message/prompt
            user_prompt: The user message/prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If the LLM call fails
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI GPT models (GPT-3.5, GPT-4, etc.).
    """
    
    def __init__(self, client: Any, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI adapter.
        
        Args:
            client: OpenAI client instance (openai.OpenAI())
            model: Model name to use (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        self.client = client
        self.model = model
    
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str, 
                     max_tokens: Optional[int] = 300,
                     temperature: Optional[float] = 0.3) -> str:
        """Generate text using OpenAI's API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class GeminiAdapter(LLMAdapter):
    """
    Adapter for Google Gemini models.
    """
    
    def __init__(self, client: Any, model: str = "gemini-pro"):
        """
        Initialize the Gemini adapter.
        
        Args:
            client: Google GenerativeAI client instance
            model: Model name to use (e.g., "gemini-pro", "gemini-1.5-pro")
        """
        self.client = client
        self.model = model
    
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str, 
                     max_tokens: Optional[int] = 300,
                     temperature: Optional[float] = 0.3) -> str:
        """Generate text using Google Gemini API."""
        try:
            # Configure the model
            generation_config = {
                "temperature": temperature or 0.3,
                "max_output_tokens": max_tokens or 300,
                "top_p": 0.8,
                "top_k": 40
            }
            
            model = self.client.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
                system_instruction=system_prompt
            )
            
            response = model.generate_content(user_prompt)
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")


class ClaudeAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude models.
    """
    
    def __init__(self, client: Any, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize the Claude adapter.
        
        Args:
            client: Anthropic client instance
            model: Model name to use (e.g., "claude-3-sonnet-20240229", "claude-3-haiku-20240307")
        """
        self.client = client
        self.model = model
    
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str, 
                     max_tokens: Optional[int] = 300,
                     temperature: Optional[float] = 0.3) -> str:
        """Generate text using Anthropic's Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 300,
                temperature=temperature or 0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise Exception(f"Claude API error: {e}")


class HuggingFaceAdapter(LLMAdapter):
    """
    Adapter for Hugging Face models (local or API).
    """
    
    def __init__(self, 
                 model_name: str,
                 use_api: bool = False,
                 api_token: Optional[str] = None):
        """
        Initialize the Hugging Face adapter.
        
        Args:
            model_name: Hugging Face model name
            use_api: Whether to use Hugging Face API or load locally
            api_token: API token for Hugging Face (if using API)
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_token = api_token
        
        if use_api:
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(
                    model=model_name,
                    token=api_token
                )
            except ImportError:
                raise ImportError("huggingface_hub is required for API usage. Install with: pip install huggingface_hub")
        else:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device_map="auto"
                )
            except ImportError:
                raise ImportError("transformers is required for local usage. Install with: pip install transformers torch")
    
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str, 
                     max_tokens: Optional[int] = 300,
                     temperature: Optional[float] = 0.3) -> str:
        """Generate text using Hugging Face model."""
        try:
            # Combine system and user prompts
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            if self.use_api:
                response = self.client.text_generation(
                    full_prompt,
                    max_new_tokens=max_tokens or 300,
                    temperature=temperature or 0.3,
                    return_full_text=False
                )
                return response.strip()
            else:
                # Local pipeline
                response = self.pipeline(
                    full_prompt,
                    max_new_tokens=max_tokens or 300,
                    temperature=temperature or 0.3,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                generated_text = response[0]['generated_text']
                # Extract just the assistant's response
                return generated_text.split("Assistant:")[-1].strip()
        except Exception as e:
            raise Exception(f"Hugging Face error: {e}")


class MockAdapter(LLMAdapter):
    """
    Mock adapter for testing and demonstration purposes.
    
    This adapter returns predefined responses instead of calling a real LLM.
    """
    
    def __init__(self, responses: Optional[List[str]] = None):
        """
        Initialize the mock adapter.
        
        Args:
            responses: List of predefined responses to cycle through
        """
        self.responses = responses or [
            "1. Focus on providing clear, actionable examples with step-by-step explanations.",
            "2. Include error handling and best practices to increase user satisfaction.", 
            "3. Break down complex topics into digestible parts for better understanding.",
            "4. Provide context about why certain approaches work to help users learn."
        ]
        self.current_index = 0
    
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str, 
                     max_tokens: Optional[int] = 300,
                     temperature: Optional[float] = 0.3) -> str:
        """Return a predefined mock response."""
        response = self.responses[self.current_index % len(self.responses)]
        self.current_index += 1
        return response


# Factory function for easy adapter creation
def create_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    """
    Factory function to create LLM adapters.
    
    Args:
        provider: LLM provider name ("openai", "gemini", "claude", "huggingface", "mock")
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMAdapter instance
        
    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        "openai": OpenAIAdapter,
        "gemini": GeminiAdapter,
        "claude": ClaudeAdapter,
        "huggingface": HuggingFaceAdapter,
        "mock": MockAdapter
    }
    
    if provider.lower() not in providers:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(providers.keys())}")
    
    return providers[provider.lower()](**kwargs)
