"""
Fully Automatic Learning AI Agent

This example demonstrates a complete AI agent that:
1. Automatically classifies user interactions as successful/unsuccessful
2. Continuously learns and evolves its system prompt
3. Adapts its behavior based on what works best
4. Provides insights into its learning process

Perfect example for users to understand how to leverage the ExperienceAI package.
"""

import os
import sys
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add package to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

# ExperienceAI imports
from experience_ai import (
    EvolvingPrompt, 
    LocalStorageAdapter, 
    GeminiAdapter,
    OpenAIAdapter,
    MockAdapter,
    AutoInteractionClassifier,
    classify_interaction
)

# For demo purposes - in real use, you'd use your actual LLM client
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    user_message: str
    agent_response: str
    timestamp: datetime
    classification: Optional[str] = None
    confidence: Optional[float] = None


class AutoLearningAgent:
    """
    A complete AI agent that automatically learns from interactions.
    
    Key Features:
    - Automatic interaction classification (no manual feedback needed)
    - Continuous prompt evolution based on successful patterns
    - Conversation context awareness
    - Learning analytics and insights
    """
    
    def __init__(self, llm_provider: str = "mock", model: str = None):
        """
        Initialize the auto-learning agent.
        
        Args:
            llm_provider: "openai", "gemini", or "mock" for demonstration
            model: Specific model name (optional)
        """
        self.conversation_history: List[ConversationTurn] = []
        self.conversation_buffer: List[str] = []  # For classification context
        
        # Set up LLM adapter
        self.llm_adapter = self._setup_llm_adapter(llm_provider, model)
        
        # Set up automatic interaction classifier
        self.classifier = AutoInteractionClassifier()
        
        # Set up storage
        self.storage = LocalStorageAdapter("./auto_learning_interactions.json")
        
        # Enhanced base prompt for learning
        self.base_prompt = """You are an intelligent AI assistant that continuously learns and adapts.

Core Principles:
- Always strive to be helpful, accurate, and clear in your responses
- Pay attention to user preferences and adapt accordingly  
- If you don't know something, admit it rather than guessing
- Provide practical, actionable information when possible
- Be conversational but professional
- Learn from each interaction to improve future responses

Communication Style:
- Be concise but thorough
- Use examples when they help clarify concepts
- Ask clarifying questions when requests are ambiguous
- Acknowledge when you've learned something new"""
        
        # Set up evolving prompt system
        self.success_outcomes = [
            'task_completed',
            'user_satisfied', 
            'helpful_response',
            'question_answered',
            'problem_solved',
            'engagement_positive'
        ]
        
        self.prompt_manager = EvolvingPrompt(
            base_prompt=self.base_prompt,
            storage_adapter=self.storage,
            llm_adapter=self.llm_adapter,
            success_outcomes=self.success_outcomes
        )
        
        print("🤖 Auto-Learning Agent initialized successfully!")
        self._print_initialization_status()
    
    def _setup_llm_adapter(self, provider: str, model: str = None):
        """Set up the appropriate LLM adapter."""
        
        if provider.lower() == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = OpenAI(api_key=api_key)
                return OpenAIAdapter(client, model or "gpt-3.5-turbo")
            else:
                print("⚠️ OpenAI API key not found, falling back to mock adapter")
        
        elif provider.lower() == "gemini" and GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                return GeminiAdapter(genai, model or "gemini-pro")
            else:
                print("⚠️ Gemini API key not found, falling back to mock adapter")
        
        # Fall back to mock adapter for demonstration
        return MockAdapter(responses=[
            "I understand your request. Let me help you with that by providing a clear, step-by-step approach.",
            "That's a great question! Here's what I think would work best for your situation...",
            "I can definitely assist you with that. Based on what you've described, I'd recommend...",
            "Thank you for that information. Let me provide you with a comprehensive answer...",
            "I see what you're looking for. Here's a practical solution that should work well..."
        ])
    
    def _print_initialization_status(self):
        """Print the current status of the agent."""
        print("-" * 60)
        print(f"🧠 LLM Provider: {type(self.llm_adapter).__name__}")
        print(f"📊 Previous interactions: {len(self.storage.read_interactions())}")
        
        stats = self.prompt_manager.get_stats()
        if stats['total_interactions'] > 0:
            print(f"📈 Success rate: {stats['success_rate']:.1%}")
            print("✅ Prompt has learned from previous sessions")
        else:
            print("🆕 Fresh start - ready to learn!")
        print("-" * 60)
    
    async def chat(self, message: str) -> str:
        """
        Main chat method with automatic learning.
        
        Args:
            message: User's message
            
        Returns:
            Agent's response
        """
        # Generate response using current evolved prompt
        response = await self._generate_response(message)
        
        # Create conversation turn
        turn = ConversationTurn(
            user_message=message,
            agent_response=response,
            timestamp=datetime.now()
        )
        
        # Store in conversation history
        self.conversation_history.append(turn)
        self.conversation_buffer.extend([f"User: {message}", f"Agent: {response}"])
        
        # Keep buffer manageable (last 10 exchanges)
        if len(self.conversation_buffer) > 20:
            self.conversation_buffer = self.conversation_buffer[-20:]
        
        print(f"👤 User: {message}")
        print(f"🤖 Agent: {response}")
        
        return response
    
    async def _generate_response(self, message: str) -> str:
        """Generate response using the LLM adapter."""
        try:
            current_prompt = self.prompt_manager.get_prompt()
            
            # Use the LLM adapter to generate response
            response = self.llm_adapter.generate_text(
                system_prompt=current_prompt,
                user_prompt=message,
                max_tokens=500,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Could you please try rephrasing your question?"
    
    async def process_follow_up(self, follow_up_message: str) -> str:
        """
        Process a follow-up message and automatically learn from the interaction.
        
        Args:
            follow_up_message: User's follow-up message
            
        Returns:
            Agent's response to the follow-up
        """
        if not self.conversation_history:
            return await self.chat(follow_up_message)
        
        # Get the previous interaction for classification
        previous_turn = self.conversation_history[-1]
        
        # Classify the previous interaction based on the follow-up
        classification = self.classifier.classify_interaction(
            user_message=previous_turn.user_message,
            agent_response=previous_turn.agent_response,
            follow_up_message=follow_up_message,
            conversation_history=self.conversation_buffer[-10:]  # Recent context
        )
        
        # Update the previous turn with classification
        previous_turn.classification = classification.outcome
        previous_turn.confidence = classification.confidence
        
        # Record the learning automatically
        self._record_automatic_learning(previous_turn, classification)
        
        # Now handle the follow-up as a new conversation
        response = await self.chat(follow_up_message)
        
        return response
    
    def _record_automatic_learning(self, turn: ConversationTurn, classification):
        """Record learning based on automatic classification."""
        
        # Map classifier outcomes to our success outcomes
        outcome_mapping = {
            'task_completed': 'task_completed',
            'user_satisfied': 'user_satisfied', 
            'helpful_response': 'helpful_response',
            'simple_greeting': 'helpful_response',
            'user_unsatisfied': 'user_unsatisfied',
            'insufficient_response': 'needs_improvement',
            'neutral_interaction': 'neutral'
        }
        
        mapped_outcome = outcome_mapping.get(classification.outcome, 'neutral')
        
        # Only record if it's a meaningful interaction
        if classification.confidence > 0.4 and mapped_outcome in self.success_outcomes:
            conversation_text = f"User: {turn.user_message}\nAgent: {turn.agent_response}"
            
            self.prompt_manager.record_interaction(
                conversation=conversation_text,
                outcome=mapped_outcome,
                metadata={
                    'auto_classified': True,
                    'confidence': classification.confidence,
                    'reasoning': classification.reasoning,
                    'timestamp': turn.timestamp.isoformat(),
                    'classification_metadata': classification.metadata
                }
            )
            
            print(f"🧠 Learned: {classification.reasoning} (confidence: {classification.confidence:.2f})")
            print(f"📈 Recorded as: {mapped_outcome}")
        
        else:
            print(f"🤔 Interaction noted but not learned from (confidence too low: {classification.confidence:.2f})")
    
    def get_learning_insights(self) -> Dict:
        """Get insights about what the agent has learned."""
        stats = self.prompt_manager.get_stats()
        
        insights = {
            'total_conversations': len(self.conversation_history),
            'learning_stats': stats,
            'recent_classifications': [],
            'learning_rate': 0.0,
            'top_success_patterns': []
        }
        
        # Analyze recent classifications
        recent_turns = self.conversation_history[-10:]
        classified_turns = [t for t in recent_turns if t.classification]
        
        if classified_turns:
            insights['recent_classifications'] = [
                {
                    'message': t.user_message[:50] + "..." if len(t.user_message) > 50 else t.user_message,
                    'classification': t.classification,
                    'confidence': t.confidence
                }
                for t in classified_turns
            ]
            
            # Calculate learning rate (how often we learn something)
            learning_interactions = len([t for t in classified_turns if t.confidence and t.confidence > 0.4])
            insights['learning_rate'] = learning_interactions / len(recent_turns) if recent_turns else 0
        
        return insights
    
    def print_learning_status(self):
        """Print detailed learning status."""
        print("\n" + "=" * 70)
        print("🧠 AUTO-LEARNING AGENT STATUS")
        print("=" * 70)
        
        insights = self.get_learning_insights()
        
        # Basic stats
        print(f"💬 Total conversations: {insights['total_conversations']}")
        print(f"📊 Total learning interactions: {insights['learning_stats']['total_interactions']}")
        print(f"✅ Successful interactions: {insights['learning_stats']['successful_interactions']}")
        print(f"📈 Success rate: {insights['learning_stats']['success_rate']:.1%}")
        print(f"🎯 Learning rate: {insights['learning_rate']:.1%}")
        
        # Recent classifications
        if insights['recent_classifications']:
            print(f"\n🔍 Recent Automatic Classifications:")
            for item in insights['recent_classifications'][-5:]:
                print(f"   • {item['message']}")
                print(f"     → {item['classification']} (confidence: {item['confidence']:.2f})")
        
        # Current prompt evolution
        current_prompt = self.prompt_manager.get_prompt()
        if "--- Learned Experience ---" in current_prompt:
            print(f"\n✨ Prompt Evolution: ACTIVE")
            print(f"   The agent has learned from experience and evolved its approach")
        else:
            print(f"\n⏳ Prompt Evolution: LEARNING")
            print(f"   The agent is gathering experience to evolve its approach")
        
        # Outcome breakdown
        if insights['learning_stats']['outcome_breakdown']:
            print(f"\n📋 Learning Breakdown:")
            for outcome, count in insights['learning_stats']['outcome_breakdown'].items():
                print(f"   • {outcome}: {count}")
        
        print("=" * 70)
    
    async def demo_conversation(self):
        """Run a demo conversation to show automatic learning."""
        print("\n🎭 DEMO: Automatic Learning in Action")
        print("-" * 50)
        
        # Simulate a realistic conversation flow
        demo_flow = [
            # Initial request
            ("Can you help me understand how machine learning works?", None),
            # Positive follow-up
            ("That's really helpful! Can you give me a specific example?", "task_completed"),
            
            # New topic
            ("I'm also trying to learn Python programming", None), 
            # Engagement
            ("Great! What's the best way to start?", "helpful_response"),
            
            # Problem scenario
            ("I'm getting an error when I run my code", None),
            # Resolution
            ("Perfect! That fixed it. Thank you!", "problem_solved"),
            
            # Gratitude
            ("You've been really helpful today", "user_satisfied"),
        ]
        
        for i, (message, expected_outcome) in enumerate(demo_flow):
            print(f"\n--- Demo Step {i+1}/{len(demo_flow)} ---")
            
            if i == 0:
                # First message
                await self.chat(message)
            else:
                # Follow-up messages (triggers automatic classification)
                await self.process_follow_up(message)
            
            # Brief pause for readability
            await asyncio.sleep(1)
        
        print(f"\n✅ Demo completed! The agent has learned automatically from the conversation flow.")


async def interactive_session(agent: AutoLearningAgent):
    """Run an interactive session with the agent."""
    print(f"\n🎯 Interactive Session Started!")
    print(f"Commands: 'status' (show learning), 'insights' (detailed analysis), 'quit' (exit)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Session ended. Thanks for chatting!")
                break
            elif user_input.lower() == 'status':
                agent.print_learning_status()
                continue
            elif user_input.lower() == 'insights':
                insights = agent.get_learning_insights()
                print(f"\n📊 Detailed Insights:")
                print(f"JSON: {insights}")
                continue
            elif not user_input:
                continue
            
            # Check if this is a follow-up (we have conversation history)
            if agent.conversation_history:
                await agent.process_follow_up(user_input)
            else:
                await agent.chat(user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main function demonstrating the auto-learning agent."""
    print("🌟 ExperienceAI - Fully Automatic Learning Agent")
    print("=" * 70)
    print("This agent automatically learns from interactions without manual feedback!")
    print("=" * 70)
    
    # Determine which LLM to use
    llm_provider = "mock"  # Default to mock for demo
    
    if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
        llm_provider = "openai"
        print("🔑 OpenAI API key detected - using GPT for responses")
    elif (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) and GEMINI_AVAILABLE:
        llm_provider = "gemini" 
        print("🔑 Gemini API key detected - using Gemini for responses")
    else:
        print("🎭 Using mock responses for demonstration")
    
    # Create the agent
    agent = AutoLearningAgent(llm_provider=llm_provider)
    
    # Show current learning state
    agent.print_learning_status()
    
    # Ask user what they want to do
    print(f"\n🚀 What would you like to do?")
    print(f"1. Run demo conversation (shows automatic learning)")
    print(f"2. Interactive chat session")
    print(f"3. Both (demo first, then chat)")
    
    while True:
        choice = input(f"\nEnter choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3")
    
    # Execute based on choice
    if choice in ['1', '3']:
        await agent.demo_conversation()
        agent.print_learning_status()
    
    if choice in ['2', '3']:
        await interactive_session(agent)
    
    # Final learning summary
    print(f"\n🎯 Final Learning Summary:")
    agent.print_learning_status()
    
    # Show the evolved prompt
    final_prompt = agent.prompt_manager.get_prompt()
    print(f"\n📝 Final Evolved Prompt:")
    print("-" * 40)
    print(final_prompt[:500] + "..." if len(final_prompt) > 500 else final_prompt)
    
    print(f"\n✨ The agent has automatically learned from {len(agent.conversation_history)} conversations!")


if __name__ == "__main__":
    print("🤖 Loading Automatic Learning Agent...")
    asyncio.run(main())
