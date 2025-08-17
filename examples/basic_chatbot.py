#!/usr/bin/env python3
"""
Smart AI Chatbot with ExperienceAI

A production-ready AI chatbot that learns from your conversations and gets smarter over time.

Setup:
1. Install dependencies: pip install openai google-generativeai anthropic python-dotenv
2. Set your API key: export OPENAI_API_KEY="your-key" (or GEMINI_API_KEY, ANTHROPIC_API_KEY)
3. Run: python smart_chatbot.py

The chatbot will:
- Chat with you using real AI models
- Automatically learn from your interactions  
- Evolve its responses to be more helpful over time
- Save its learnings between sessions
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Add package to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from experience_ai import (
    EvolvingPrompt, 
    LocalStorageAdapter, 
    OpenAIAdapter,
    GeminiAdapter, 
    ClaudeAdapter,
    AutoInteractionClassifier
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("💡 Tip: Install python-dotenv for .env file support: pip install python-dotenv")

# LLM client imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class SmartChatbot:
    """
    Production-ready AI chatbot that learns from conversations.
    """
    
    def __init__(self, debug_mode=False):
        self.conversation_history = []
        self.classifier = AutoInteractionClassifier()
        self.debug_mode = debug_mode
        
        # Set up LLM
        self.llm_adapter = self._setup_llm()
        if not self.llm_adapter:
            print("❌ No AI provider available. Please install a client and set API key.")
            sys.exit(1)
        
        # Set up storage 
        self.storage = LocalStorageAdapter("./chatbot_memory.json")
        
        # Professional system prompt
        self.base_prompt = """You are a highly capable AI assistant designed to be helpful, accurate, and personable.

Your core abilities:
- Answer questions clearly and comprehensively
- Help with writing, analysis, coding, math, and creative tasks
- Provide step-by-step guidance for complex problems
- Adapt your communication style to what works best for each user
- Learn from interactions to improve future responses

Communication principles:
- Be direct and concise while being thorough
- Use examples and analogies when they clarify concepts
- Ask follow-up questions when requests need clarification
- Acknowledge uncertainty rather than guessing
- Maintain a helpful and professional tone

You continuously learn from successful interactions to become more effective at helping users accomplish their goals."""

        # Set up the evolving prompt system
        self.prompt_manager = EvolvingPrompt(
            base_prompt=self.base_prompt,
            storage_adapter=self.storage, 
            llm_adapter=self.llm_adapter,
            success_outcomes=[
                'task_completed',
                'user_satisfied',
                'helpful_response', 
                'question_answered',
                'problem_solved'
            ]
        )
        
        # Load previous learnings
        stats = self.prompt_manager.get_stats()
        if stats['total_interactions'] > 0:
            print(f"🧠 Loaded {stats['total_interactions']} previous interactions (success rate: {stats['success_rate']:.1%})")
        else:
            print("🆕 Starting fresh - ready to learn from our conversations!")
    
    def _setup_llm(self):
        """Set up the best available LLM provider."""
        
        # Try OpenAI first
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    # Test the connection
                    client.models.list()
                    print("✅ Using OpenAI GPT")
                    return OpenAIAdapter(client, model="gpt-3.5-turbo")
                except Exception as e:
                    print(f"⚠️ OpenAI setup failed: {e}")
        
        # Try Gemini
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    # Test the connection by listing models
                    list(genai.list_models())
                    print("✅ Using Google Gemini")
                    return GeminiAdapter(genai, model="gemini-pro")
                except Exception as e:
                    print(f"⚠️ Gemini setup failed: {e}")
        
        # Try Claude
        if CLAUDE_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    client = Anthropic(api_key=api_key)
                    print("✅ Using Anthropic Claude")
                    return ClaudeAdapter(client)
                except Exception as e:
                    print(f"⚠️ Claude setup failed: {e}")
        
        return None
    
    def _get_current_prompt(self) -> str:
        """Get the current evolved system prompt."""
        return self.prompt_manager.get_prompt()
    
    async def chat(self, message: str) -> str:
        """
        Chat with the AI and automatically learn from the interaction.
        
        Args:
            message: User's message
            
        Returns:
            AI's response
        """
        try:
            # Get current evolved prompt
            system_prompt = self._get_current_prompt()
            
            # Generate response
            response = self.llm_adapter.generate_text(
                system_prompt=system_prompt,
                user_prompt=message,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Store conversation turn
            self.conversation_history.append({
                'user': message,
                'assistant': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            
            # Still record the failed interaction
            self.conversation_history.append({
                'user': message,
                'assistant': error_response,
                'timestamp': datetime.now().isoformat(),
                'error': True
            })
            
            return error_response
    
    def _learn_from_interaction(self, user_msg: str, assistant_response: str, follow_up: str):
        """Automatically learn from a completed interaction."""
        
        # Classify the interaction
        classification = self.classifier.classify_interaction(
            user_message=user_msg,
            agent_response=assistant_response,
            follow_up_message=follow_up,
            conversation_history=[f"User: {t['user']}" for t in self.conversation_history[-5:]]
        )
        
        # Map to our success outcomes
        outcome_mapping = {
            'task_completed': 'task_completed',
            'user_satisfied': 'user_satisfied',
            'helpful_response': 'helpful_response',
            'simple_greeting': 'helpful_response',
            'question_answered': 'question_answered'
        }
        
        outcome = outcome_mapping.get(classification.outcome, 'neutral')
        
        # Record learning if confident and successful
        if classification.confidence > 0.5 and outcome in self.prompt_manager.success_outcomes:
            conversation = f"User: {user_msg}\nAssistant: {assistant_response}"
            
            self.prompt_manager.record_interaction(
                conversation=conversation,
                outcome=outcome,
                metadata={
                    'confidence': classification.confidence,
                    'reasoning': classification.reasoning,
                    'auto_classified': True,
                    'follow_up': follow_up
                }
            )
            
            print(f"\n🧠 Learned from interaction: {classification.reasoning}")
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        base_stats = self.prompt_manager.get_stats()
        base_stats['conversations'] = len(self.conversation_history)
        return base_stats
    
    def show_learning_progress(self):
        """Show current learning progress."""
        stats = self.get_stats()
        
        print(f"\n📊 Learning Progress:")
        print(f"   💬 Conversations: {stats['conversations']}")
        print(f"   📈 Learning interactions: {stats['total_interactions']}")
        print(f"   ✅ Success rate: {stats['success_rate']:.1%}")
        
        if stats['outcome_breakdown']:
            print(f"   📋 Success types: {stats['outcome_breakdown']}")
        
        # Check if prompt has evolved
        current_prompt = self._get_current_prompt()
        if "--- Learned Experience ---" in current_prompt:
            print(f"   ✨ Status: AI has learned and evolved!")
        else:
            print(f"   ⏳ Status: Gathering experience to evolve")
    
    def show_current_prompt(self):
        """Display the current system prompt being used."""
        current_prompt = self._get_current_prompt()
        
        print("\n" + "="*60)
        print("🔍 CURRENT SYSTEM PROMPT")
        print("="*60)
        print(current_prompt)
        print("="*60)
    
    async def run_chat_session(self):
        """Run the main chat session."""
        
        print("\n" + "="*60)
        print("🤖 Smart AI Chatbot - Powered by ExperienceAI")
        print("="*60)
        print("Start chatting! The AI will learn from our conversation and get better over time.")
        print("")
        print("💡 Available Commands:")
        print("   'stats' - Show learning progress")
        print("   'prompt' - View current system prompt")
        print("   'debug' - Toggle debug mode")
        print("   'quit' - Exit chatbot")
        print("="*60)
        
        pending_interaction = None  # Track interaction waiting for follow-up
        
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for chatting! Your AI has learned from our conversation.")
                    self.show_learning_progress()
                    break
                
                if user_input.lower() == 'stats':
                    self.show_learning_progress()
                    continue
                    
                if user_input.lower() == 'prompt':
                    self.show_current_prompt()
                    continue
                    
                if user_input.lower() == 'debug':
                    self.debug_mode = not self.debug_mode
                    print(f"\n🔧 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                    continue
                
                # Check if we should learn from the previous interaction
                if pending_interaction and len(self.conversation_history) > 0:
                    prev_turn = self.conversation_history[-1]
                    learned = self._learn_from_interaction(
                        prev_turn['user'], 
                        prev_turn['assistant'],
                        user_input
                    )
                
                # Get AI response
                print("\n🤖 AI: ", end="", flush=True)
                response = await self.chat(user_input)
                print(response)
                
                # Mark this interaction as pending (will learn from next user message)
                pending_interaction = True
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let's continue chatting...")


async def main():
    """Main function."""
    
    print("🚀 Initializing Smart AI Chatbot...")
    
    # Check if any LLM libraries are available
    if not any([OPENAI_AVAILABLE, GEMINI_AVAILABLE, CLAUDE_AVAILABLE]):
        print("""
❌ No AI provider libraries found!

Please install at least one:
- OpenAI: pip install openai  
- Gemini: pip install google-generativeai
- Claude: pip install anthropic

Then set your API key as an environment variable:
- export OPENAI_API_KEY="your-openai-key"
- export GEMINI_API_KEY="your-gemini-key" 
- export ANTHROPIC_API_KEY="your-anthropic-key"
""")
        return
    
    # Initialize chatbot
    try:
        chatbot = SmartChatbot()
        await chatbot.run_chat_session()
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("""
Please make sure you have:
1. Installed the required packages
2. Set your API key as an environment variable
3. Have a valid API key with available credits
""")


if __name__ == "__main__":
    asyncio.run(main())
