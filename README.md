# ExperienceAI 🧠

A Python package that makes AI systems learn and improve from user interactions automatically. Create chatbots that remember your preferences, adapt to your communication style, and evolve their responses over time.

## 🌟 What makes ExperienceAI special?

- **🤖 Intelligent Learning**: Uses LLM-powered classification to understand what users actually want
- **🎯 Smart Preference Detection**: Automatically detects when users state preferences ("call me Sarah", "I prefer brief answers")
- **🔄 Self-Improving Prompts**: System prompts that get smarter with every conversation
- **⚡ Zero Configuration**: Just plug in your API key and start learning

## ✨ Key Features

- **🎯 Smart Preference Detection**: Automatically detects user preferences, communication styles, and instructions
- **🤖 LLM-Powered Classification**: Uses the same LLM as your chatbot for intelligent learning decisions
- **🌐 Multi-LLM Support**: Works with OpenAI, Gemini, Claude, and custom LLMs
- **📊 Automatic Learning**: No manual labeling - it learns what matters automatically
- **💾 Persistent Memory**: Saves learnings between sessions
- **📈 Learning Analytics**: Monitor how your AI is improving over time

## Installation

```bash
# Install from local development
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## 🎯 Smart Preference Detection

**New in v0.2.0!** ExperienceAI now automatically detects and prioritizes user preferences without any manual classification needed.

### What Gets Detected Automatically:

```python
from experience_ai import AutoInteractionClassifier

classifier = AutoInteractionClassifier()

# Name preferences - HIGH PRIORITY
"call me Nana" → Detected as name_preference
"refer to me as Dr. Smith" → Detected as name_preference
"my name is John" → Detected as name_preference

# Communication style preferences - HIGH PRIORITY
"I like short answers" → Detected as style_preference
"I prefer one-line responses for historical facts" → Detected as style_preference
"keep your responses concise" → Detected as style_preference

# Context and instructions - HIGH PRIORITY
"just so you know, I work in tech" → Detected as general_preference
"remember, I'm a beginner" → Detected as general_preference

# Generic conversations - FILTERED OUT
"hello" → Classified as simple_greeting (not learned)
"thanks" → Classified as positive feedback (lower priority)
```

### Automatic Learning in Action:

```python
# User states preferences - immediately learned!
User: "call me Nana"
AI: "Hello Nana! How can I assist you?"
🧠 Learned: User wants to be called 'Nana'

User: "just so you know, I like short answers for historical questions"
AI: "Understood! I'll keep historical responses concise."
🧠 Learned: User prefers short answers for historical questions

# Check what was learned
User: "prompt"
--- Learned Experience ---
1. Address the user as 'Nana' in all interactions.
2. User preference: short answers for historical questions
```

## Quick Start

### Automatic Chatbot Example
```python
import asyncio
from experience_ai import (
    EvolvingPrompt, LocalStorageAdapter, OpenAIAdapter, 
    AutoInteractionClassifier
)

class SmartChatbot:
    def __init__(self):
        # Set up components
        self.classifier = AutoInteractionClassifier()
        self.storage = LocalStorageAdapter("./memory.json")
        self.llm_adapter = OpenAIAdapter(openai_client, model="gpt-3.5-turbo")
        
        # Create evolving prompt
        self.prompt_manager = EvolvingPrompt(
            base_prompt="You are a helpful AI assistant...",
            storage_adapter=self.storage,
            llm_adapter=self.llm_adapter
        )
    
    async def chat(self, message: str) -> str:
        # Get evolved prompt
        system_prompt = self.prompt_manager.get_prompt()
        
        # Generate response
        response = self.llm_adapter.generate_text(
            system_prompt=system_prompt,
            user_prompt=message
        )
        
        # Auto-classify and learn from preferences
        classification = self.classifier.classify_interaction(
            user_message=message,
            agent_response=response
        )
        
        # Learn if it's a user preference
        if classification.outcome == 'user_preference_stated':
            self.prompt_manager.record_interaction(
                conversation=f"User: {message}\nAI: {response}",
                outcome='user_preference_stated',
                metadata=classification.metadata
            )
            print(f"🧠 Learned: {classification.reasoning}")
        
        return response

# Usage
chatbot = SmartChatbot()
response = await chatbot.chat("call me Alex")
# Output: 🧠 Learned: User wants to be called 'alex'
```

### Manual OpenAI Example
```python
import os
from openai import OpenAI
from experience_ai import EvolvingPrompt, LocalStorageAdapter, OpenAIAdapter

# Set up your LLM client and adapter
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm_adapter = OpenAIAdapter(client, model="gpt-4")

# Create storage adapter
storage = LocalStorageAdapter("./interactions.json")

# Define your base system prompt
base_prompt = """You are a helpful AI assistant that provides clear, 
accurate responses and helps users accomplish their tasks efficiently."""

# Create the evolving prompt system
prompt_manager = EvolvingPrompt(
    base_prompt=base_prompt,
    storage_adapter=storage,
    llm_adapter=llm_adapter
)

# Get the current prompt (initially just the base prompt)
current_prompt = prompt_manager.get_prompt()
print("Current prompt:", current_prompt)

# Record interactions as they happen
prompt_manager.record_interaction(
    conversation="User asked for Python code to parse JSON files",
    outcome="code_block_copied",  # Indicates success
    metadata={"language": "python", "topic": "file_parsing"}
)

# After recording successful interactions, the prompt evolves
evolved_prompt = prompt_manager.get_prompt()
print("Evolved prompt:", evolved_prompt)

# Check learning statistics
stats = prompt_manager.get_stats()
print("Learning stats:", stats)
```

## Core Concepts

### Base Prompt
The static foundation of your system prompt that never changes.

### Experience Prompt
Dynamically generated insights from successful interactions that get appended to your base prompt.

### Interaction Recording
Every interaction should be recorded with:
- **Conversation**: Summary or full conversation content
- **Outcome**: Success/failure indicator (e.g., 'code_block_copied', 'task_completed', 'user_rephrased')
- **Metadata**: Optional additional context

### Success Outcomes
The system learns from interactions marked with positive outcomes:
- `code_block_copied`
- `resolved_and_ended`
- `task_completed`
- `user_satisfied`
- `solution_accepted`
- `helpful_response`

## Multi-LLM Support

ExperienceAI works with **any LLM provider** through a unified adapter interface. No need to be locked into OpenAI!

### Supported Providers

#### OpenAI (GPT-3.5, GPT-4, etc.)
```python
from openai import OpenAI
from experience_ai import OpenAIAdapter

client = OpenAI(api_key="your-key")
llm_adapter = OpenAIAdapter(client, model="gpt-4")
```

#### Google Gemini
```python
import google.generativeai as genai
from experience_ai import GeminiAdapter

genai.configure(api_key="your-key")
llm_adapter = GeminiAdapter(genai, model="gemini-pro")
```

#### Anthropic Claude
```python
from anthropic import Anthropic
from experience_ai import ClaudeAdapter

client = Anthropic(api_key="your-key")
llm_adapter = ClaudeAdapter(client, model="claude-3-sonnet-20240229")
```

#### Hugging Face Models
```python
from experience_ai import HuggingFaceAdapter

# Using Hugging Face API
llm_adapter = HuggingFaceAdapter(
    model_name="microsoft/DialoGPT-medium",
    use_api=True,
    api_token="your-token"
)

# Or load locally (requires more resources)
llm_adapter = HuggingFaceAdapter(
    model_name="microsoft/DialoGPT-medium",
    use_api=False
)
```

#### Easy Setup with Factory Function
```python
from experience_ai import create_llm_adapter

# Automatically creates the right adapter
llm_adapter = create_llm_adapter("openai", client=openai_client, model="gpt-4")
llm_adapter = create_llm_adapter("gemini", client=genai, model="gemini-pro")
llm_adapter = create_llm_adapter("claude", client=claude_client)
```

### Custom LLM Integration

Want to use a different LLM? Just implement the `LLMAdapter` interface:

```python
from experience_ai import LLMAdapter

class MyCustomAdapter(LLMAdapter):
    def __init__(self, client):
        self.client = client
    
    def generate_text(self, system_prompt: str, user_prompt: str, 
                     max_tokens: int = 300, temperature: float = 0.3) -> str:
        # Your custom LLM API call here
        response = self.client.generate(
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.text

# Use it just like any other adapter
llm_adapter = MyCustomAdapter(my_client)
prompt_manager = EvolvingPrompt(base_prompt, storage, llm_adapter)
```

## Advanced Usage

### Custom Success Outcomes

```python
custom_outcomes = ['user_thanked', 'problem_solved', 'question_answered']

prompt_manager = EvolvingPrompt(
    base_prompt=base_prompt,
    storage_adapter=storage,
    llm_adapter=llm_adapter,
    success_outcomes=custom_outcomes
)
```

### Recording Complex Conversations

```python
# Record structured conversation data
conversation_data = [
    {"role": "user", "content": "How do I handle errors in Python?"},
    {"role": "assistant", "content": "You can use try/except blocks..."},
    {"role": "user", "content": "Perfect, that worked!"}
]

prompt_manager.record_interaction(
    conversation=conversation_data,
    outcome="solution_accepted",
    metadata={"domain": "python", "difficulty": "beginner"}
)
```

### Analytics and Monitoring

```python
# Get detailed statistics
stats = prompt_manager.get_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Total interactions: {stats['total_interactions']}")

# Export all interactions for analysis
all_interactions = prompt_manager.export_interactions()

# Clear history if needed
prompt_manager.clear_history()
```

## Architecture

The package is built with a modular architecture:

- **`EvolvingPrompt`**: Core orchestration class
- **`LocalStorageAdapter`**: JSON file-based storage
- **Storage Adapter Pattern**: Easily extend with database backends
- **LLM Client Agnostic**: Works with any LLM client that follows the OpenAI API pattern

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Type checking
mypy experience_ai/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
