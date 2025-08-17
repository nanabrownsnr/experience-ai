"""
Example: Integrating ExperienceAI with Google ADK (Agent Development Kit)

This example shows how to use ExperienceAI to create evolving prompts
that learn from agent interactions over time.
"""

from typing import Any, Dict, List
import json
import asyncio
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams, 
    StdioConnectionParams, 
    SseConnectionParams
)
from mcp import StdioServerParameters
from experience_ai import EvolvingPrompt, LocalStorageAdapter, GeminiAdapter
import google.generativeai as genai
import os

load_dotenv()

class EvolvingADKAgent:
    """
    An ADK agent that learns and evolves its system prompt over time.
    """
    
    def __init__(self):
        self.setup_environment()
        self.setup_evolving_prompt()
        self.setup_mcp_toolsets()
        self.create_agent()
        
    def setup_environment(self):
        """Setup API keys and environment variables."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
    def setup_evolving_prompt(self):
        """Setup the evolving prompt system."""
        # Create LLM adapter
        self.llm_adapter = GeminiAdapter(genai, model="gemini-2.0-flash")
        
        # Create storage
        self.storage = LocalStorageAdapter("./adk_agent_interactions.json")
        
        # Define base prompt with tool awareness
        self.base_prompt = """You are a helpful AI assistant with access to various tools through MCP (Model Context Protocol).

Key guidelines:
- Always check what tools are available to you before responding
- Use tools when they can help answer the user's question or complete their task
- Be explicit about which tools you're using and why
- If you don't have the right tools for a task, explain what tools would be needed
- Provide clear, actionable responses based on tool outputs
- Ask clarifying questions when the user's request is ambiguous

Remember: You are more helpful when you actively use the tools available to you."""
        
        # Create prompt manager with custom success outcomes for ADK context
        self.success_outcomes = [
            'task_completed',
            'tool_used_successfully', 
            'user_satisfied',
            'problem_solved',
            'helpful_response',
            'correct_tool_selection',
            'clear_explanation_provided'
        ]
        
        self.prompt_manager = EvolvingPrompt(
            base_prompt=self.base_prompt,
            storage_adapter=self.storage,
            llm_adapter=self.llm_adapter,
            success_outcomes=self.success_outcomes
        )
        
    def setup_mcp_toolsets(self):
        """Setup MCP toolsets from environment configuration."""
        # Load MCP server configurations
        mcp_server_list = os.getenv("MCP_SERVER_LIST")
        if not mcp_server_list:
            print("Warning: MCP_SERVER_LIST not found in environment.")
            self.mcp_servers = []
        else:
            try:
                self.mcp_servers = json.loads(mcp_server_list)
                print(f"Loaded {len(self.mcp_servers)} MCP server configurations")
            except json.JSONDecodeError as e:
                print(f"Error parsing MCP_SERVER_LIST JSON: {e}")
                self.mcp_servers = []
        
        # Create toolsets
        self.mcp_toolsets = []
        for server_config in self.mcp_servers:
            toolset = self._create_toolset(server_config)
            if toolset:
                self.mcp_toolsets.append(toolset)
                
        print(f"Created {len(self.mcp_toolsets)} MCP toolsets")
        
    def _create_toolset(self, server_config: Dict) -> MCPToolset:
        """Create a single MCP toolset from configuration."""
        try:
            if server_config["type"] == "streamable_http":
                return MCPToolset(
                    connection_params=StreamableHTTPConnectionParams(
                        url=server_config["url"],
                        timeout=server_config.get("timeout", 3000),
                    ),
                    tool_filter=server_config.get("tool_filter")
                )
            elif server_config["type"] == "stdio":
                return MCPToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command=server_config["command"],
                            args=server_config["args"],
                        ),
                        timeout=server_config.get("timeout", 30.0),
                    ),
                    tool_filter=server_config.get("tool_filter")
                )
            elif server_config["type"] == "sse":
                return MCPToolset(
                    connection_params=SseConnectionParams(
                        url=server_config["url"],
                        timeout=server_config.get("timeout", 30.0),
                    )
                )
            else:
                print(f"Unknown server type '{server_config['type']}' for {server_config.get('name', 'unknown')}")
                return None
                
        except Exception as e:
            print(f"Failed to create toolset for {server_config.get('name', 'unknown')}: {e}")
            return None
    
    def create_agent(self):
        """Create the ADK agent with evolving prompt."""
        self.agent = LlmAgent(
            model="gemini-2.0-flash",
            name="evolving_assistant",
            instruction=self.prompt_manager.get_prompt(),
            tools=self.mcp_toolsets
        )
        
    def update_agent_prompt(self):
        """Update the agent's instruction with the latest evolved prompt."""
        new_prompt = self.prompt_manager.get_prompt()
        self.agent.instruction = new_prompt
        print("🔄 Agent prompt updated with latest learnings")
        
    async def chat(self, message: str, session_id: str = "default") -> str:
        """
        Chat with the agent and record the interaction for learning.
        
        Args:
            message: User message
            session_id: Session identifier for tracking conversations
            
        Returns:
            Agent response
        """
        try:
            # Get response from agent
            response = await self.agent.run(message)
            
            # Extract response text (adjust based on ADK response format)
            response_text = str(response) if response else "No response generated"
            
            # For now, we'll assume successful interaction
            # In a real implementation, you'd want to analyze the response quality
            self.record_interaction(
                conversation=f"User: {message}\nAgent: {response_text}",
                outcome="helpful_response",  # Default to successful
                metadata={
                    "session_id": session_id,
                    "tools_available": len(self.mcp_toolsets),
                    "response_length": len(response_text)
                }
            )
            
            return response_text
            
        except Exception as e:
            # Record failed interaction
            self.record_interaction(
                conversation=f"User: {message}\nError: {str(e)}",
                outcome="error_occurred",
                metadata={
                    "session_id": session_id,
                    "error_type": type(e).__name__
                }
            )
            return f"Sorry, I encountered an error: {str(e)}"
    
    def record_interaction(self, conversation: str, outcome: str, metadata: Dict = None):
        """Record an interaction for learning."""
        self.prompt_manager.record_interaction(
            conversation=conversation,
            outcome=outcome,
            metadata=metadata or {}
        )
        
        # Update agent prompt if we have new learnings
        self.update_agent_prompt()
    
    def record_successful_tool_use(self, tool_name: str, task_description: str, outcome: str):
        """Record successful tool usage for learning."""
        self.record_interaction(
            conversation=f"Used tool '{tool_name}' for: {task_description}",
            outcome="tool_used_successfully",
            metadata={
                "tool_name": tool_name,
                "task_type": "tool_usage",
                "outcome": outcome
            }
        )
    
    def get_learning_stats(self) -> Dict:
        """Get current learning statistics."""
        return self.prompt_manager.get_stats()
    
    def get_current_prompt(self) -> str:
        """Get the current evolved prompt."""
        return self.prompt_manager.get_prompt()
    
    def clear_learning_history(self):
        """Clear all learning history (use with caution)."""
        self.prompt_manager.clear_history()
        self.update_agent_prompt()


# Example usage and testing
async def main():
    """Example of how to use the EvolvingADKAgent."""
    
    # Create the evolving agent
    agent = EvolvingADKAgent()
    
    print("🤖 Evolving ADK Agent initialized!")
    print("-" * 50)
    
    # Show initial prompt
    print("📝 Initial Prompt:")
    print(agent.get_current_prompt())
    print("\n" + "-" * 50)
    
    # Simulate some interactions
    print("💬 Simulating interactions...")
    
    # Example interactions (you'd replace with real user interactions)
    test_interactions = [
        ("What tools do you have available?", "helpful_response"),
        ("Can you help me with file management?", "task_completed"),
        ("Search for information about Python", "tool_used_successfully"),
        ("I need help with data analysis", "user_satisfied")
    ]
    
    for message, expected_outcome in test_interactions:
        print(f"\nUser: {message}")
        
        # In real usage, you'd call await agent.chat(message)
        # For demo, we'll just record the interaction
        agent.record_interaction(
            conversation=f"User: {message}\nAgent: I'll help you with that using my available tools.",
            outcome=expected_outcome,
            metadata={"demo": True}
        )
        print(f"✅ Recorded interaction with outcome: {expected_outcome}")
    
    print("\n" + "-" * 50)
    
    # Show evolved prompt
    print("🚀 Evolved Prompt:")
    print(agent.get_current_prompt())
    
    print("\n" + "-" * 50)
    
    # Show learning stats
    stats = agent.get_learning_stats()
    print("📊 Learning Statistics:")
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Successful interactions: {stats['successful_interactions']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Outcome breakdown: {stats['outcome_breakdown']}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
