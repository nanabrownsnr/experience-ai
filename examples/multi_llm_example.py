"""
Multi-LLM Example for ExperienceAI

This example shows how to use ExperienceAI with different LLM providers:
- OpenAI GPT models
- Google Gemini
- Anthropic Claude
- Hugging Face models
- Mock adapter (for testing)
"""

import os
import sys
from pathlib import Path

# Add the package to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from experience_ai import (
    EvolvingPrompt, 
    LocalStorageAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    ClaudeAdapter,
    HuggingFaceAdapter,
    MockAdapter,
    create_llm_adapter
)


def setup_openai_example():
    """Example using OpenAI GPT models."""
    print("🤖 OpenAI Example")
    print("-" * 30)
    
    try:
        from openai import OpenAI
        
        # Setup OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create adapter
        llm_adapter = OpenAIAdapter(client, model="gpt-3.5-turbo")
        
        # Setup ExperienceAI
        storage = LocalStorageAdapter("openai_interactions.json")
        prompt_manager = EvolvingPrompt(
            base_prompt="You are a helpful coding assistant.",
            storage_adapter=storage,
            llm_adapter=llm_adapter
        )
        
        print("✅ OpenAI setup complete")
        return prompt_manager
        
    except ImportError:
        print("❌ OpenAI not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
        return None


def setup_gemini_example():
    """Example using Google Gemini."""
    print("🧠 Gemini Example")
    print("-" * 30)
    
    try:
        import google.generativeai as genai
        
        # Setup Gemini client
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Create adapter
        llm_adapter = GeminiAdapter(genai, model="gemini-pro")
        
        # Setup ExperienceAI
        storage = LocalStorageAdapter("gemini_interactions.json")
        prompt_manager = EvolvingPrompt(
            base_prompt="You are a helpful coding assistant.",
            storage_adapter=storage,
            llm_adapter=llm_adapter
        )
        
        print("✅ Gemini setup complete")
        return prompt_manager
        
    except ImportError:
        print("❌ Google GenerativeAI not installed. Install with: pip install google-generativeai")
        return None
    except Exception as e:
        print(f"❌ Gemini setup failed: {e}")
        return None


def setup_claude_example():
    """Example using Anthropic Claude."""
    print("🎭 Claude Example")
    print("-" * 30)
    
    try:
        from anthropic import Anthropic
        
        # Setup Claude client
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Create adapter
        llm_adapter = ClaudeAdapter(client, model="claude-3-haiku-20240307")
        
        # Setup ExperienceAI
        storage = LocalStorageAdapter("claude_interactions.json")
        prompt_manager = EvolvingPrompt(
            base_prompt="You are a helpful coding assistant.",
            storage_adapter=storage,
            llm_adapter=llm_adapter
        )
        
        print("✅ Claude setup complete")
        return prompt_manager
        
    except ImportError:
        print("❌ Anthropic not installed. Install with: pip install anthropic")
        return None
    except Exception as e:
        print(f"❌ Claude setup failed: {e}")
        return None


def setup_huggingface_example():
    """Example using Hugging Face models."""
    print("🤗 Hugging Face Example")
    print("-" * 30)
    
    try:
        # Using Hugging Face API (lighter weight)
        llm_adapter = HuggingFaceAdapter(
            model_name="microsoft/DialoGPT-medium",
            use_api=True,
            api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
        
        # Setup ExperienceAI
        storage = LocalStorageAdapter("huggingface_interactions.json")
        prompt_manager = EvolvingPrompt(
            base_prompt="You are a helpful coding assistant.",
            storage_adapter=storage,
            llm_adapter=llm_adapter
        )
        
        print("✅ Hugging Face setup complete")
        return prompt_manager
        
    except Exception as e:
        print(f"❌ Hugging Face setup failed: {e}")
        return None


def setup_mock_example():
    """Example using mock adapter for testing."""
    print("🎲 Mock Example")
    print("-" * 30)
    
    # Create mock adapter with custom responses
    custom_responses = [
        "Based on successful interactions, here are key insights:\n1. Users appreciate clear, step-by-step explanations.",
        "Key learnings from interactions:\n1. Code examples should include error handling.\n2. Explanations work best when broken into small chunks.",
        "Experience summary:\n1. Users prefer practical examples over theory.\n2. Context about 'why' improves understanding."
    ]
    
    llm_adapter = MockAdapter(responses=custom_responses)
    
    # Setup ExperienceAI
    storage = LocalStorageAdapter("mock_interactions.json")
    prompt_manager = EvolvingPrompt(
        base_prompt="You are a helpful coding assistant.",
        storage_adapter=storage,
        llm_adapter=llm_adapter
    )
    
    print("✅ Mock setup complete")
    return prompt_manager


def demonstrate_evolution(prompt_manager, name):
    """Demonstrate prompt evolution with sample interactions."""
    print(f"\n📈 Demonstrating Evolution - {name}")
    print("-" * 40)
    
    # Initial prompt
    print("Initial prompt:")
    print(prompt_manager.get_prompt()[:100] + "...")
    
    # Add some sample interactions
    interactions = [
        ("User asked for Python help with loops", "task_completed"),
        ("Explained list comprehensions with examples", "code_block_copied"), 
        ("Helped debug a function", "resolved_and_ended"),
        ("Provided best practices guide", "helpful_response")
    ]
    
    for conversation, outcome in interactions:
        prompt_manager.record_interaction(conversation, outcome)
    
    # Show evolved prompt
    print("\nEvolved prompt:")
    evolved = prompt_manager.get_prompt()
    print(evolved[:200] + "..." if len(evolved) > 200 else evolved)
    
    # Show stats
    stats = prompt_manager.get_stats()
    print(f"\nStats: {stats['total_interactions']} total, {stats['successful_interactions']} successful ({stats['success_rate']:.1%})")


def factory_example():
    """Example using the factory function."""
    print("\n🏭 Factory Function Example")
    print("-" * 40)
    
    # Using factory function for easy setup
    try:
        llm_adapter = create_llm_adapter("mock")
        
        storage = LocalStorageAdapter("factory_interactions.json")
        prompt_manager = EvolvingPrompt(
            base_prompt="You are a helpful assistant.",
            storage_adapter=storage,
            llm_adapter=llm_adapter
        )
        
        print("✅ Factory setup complete")
        demonstrate_evolution(prompt_manager, "Factory")
        
    except Exception as e:
        print(f"❌ Factory example failed: {e}")


def main():
    """Run all examples."""
    print("🌟 ExperienceAI Multi-LLM Examples")
    print("=" * 50)
    
    # Try each provider
    examples = [
        ("OpenAI", setup_openai_example),
        ("Gemini", setup_gemini_example), 
        ("Claude", setup_claude_example),
        ("Hugging Face", setup_huggingface_example),
        ("Mock", setup_mock_example)
    ]
    
    successful_setups = []
    
    for name, setup_func in examples:
        try:
            prompt_manager = setup_func()
            if prompt_manager:
                successful_setups.append((name, prompt_manager))
            print()
        except Exception as e:
            print(f"❌ {name} example failed: {e}\n")
    
    # Demonstrate evolution with successful setups
    if successful_setups:
        print("\n" + "=" * 50)
        print("🧪 Running Evolution Demonstrations")
        print("=" * 50)
        
        for name, prompt_manager in successful_setups:
            try:
                demonstrate_evolution(prompt_manager, name)
                # Clean up demo files
                prompt_manager.clear_history()
            except Exception as e:
                print(f"❌ Evolution demo failed for {name}: {e}")
    
    # Factory example
    factory_example()
    
    print("\n" + "=" * 50)
    print("🎯 Usage Summary")
    print("=" * 50)
    print("""
To use ExperienceAI with different LLMs:

1. OpenAI:
   from experience_ai import OpenAIAdapter
   adapter = OpenAIAdapter(openai_client, "gpt-4")

2. Gemini:
   from experience_ai import GeminiAdapter  
   adapter = GeminiAdapter(genai, "gemini-pro")

3. Claude:
   from experience_ai import ClaudeAdapter
   adapter = ClaudeAdapter(anthropic_client, "claude-3-sonnet-20240229")

4. Any LLM via factory:
   from experience_ai import create_llm_adapter
   adapter = create_llm_adapter("openai", client=client, model="gpt-4")
""")


if __name__ == "__main__":
    main()
