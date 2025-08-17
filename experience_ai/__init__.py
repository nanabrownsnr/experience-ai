"""
ExperienceAI - A package for creating system prompts that evolve over time.

This package allows you to create system prompts that learn from user interactions
and evolve to better handle future requests.
"""

from .prompt import EvolvingPrompt
from .storage import LocalStorageAdapter
from .llm_adapters import (
    LLMAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    ClaudeAdapter,
    HuggingFaceAdapter,
    MockAdapter,
    create_llm_adapter
)
from .interaction_classifier import (
    AutoInteractionClassifier,
    InteractionClassification,
    classify_interaction
)

__version__ = "0.3.2"
__all__ = [
    "EvolvingPrompt", 
    "LocalStorageAdapter",
    "LLMAdapter",
    "OpenAIAdapter",
    "GeminiAdapter", 
    "ClaudeAdapter",
    "HuggingFaceAdapter",
    "MockAdapter",
    "create_llm_adapter",
    "AutoInteractionClassifier",
    "InteractionClassification",
    "classify_interaction"
]
