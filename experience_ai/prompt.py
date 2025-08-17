"""
Core prompt management for evolving system prompts.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from .storage import LocalStorageAdapter
from .llm_adapters import LLMAdapter


class EvolvingPrompt:
    """
    A system prompt that learns from user interactions and evolves over time.
    
    This class manages a base system prompt and an experience-based prompt that
    is synthesized from successful past interactions using an LLM.
    """
    
    def __init__(self, 
                 base_prompt: str, 
                 storage_adapter: LocalStorageAdapter, 
                 llm_adapter: LLMAdapter,
                 success_outcomes: Optional[List[str]] = None):
        """
        Initialize the EvolvingPrompt system.
        
        Args:
            base_prompt (str): The static base system prompt
            storage_adapter (LocalStorageAdapter): Storage adapter for persisting interactions
            llm_adapter (LLMAdapter): LLM adapter for synthesizing experience (supports any LLM provider)
            success_outcomes (List[str], optional): List of outcome tags considered successful.
                                                   Defaults to common success indicators.
        """
        self.base_prompt = base_prompt
        self.storage_adapter = storage_adapter
        self.llm_adapter = llm_adapter
        
        # Default success outcomes that indicate positive interactions
        self.success_outcomes = success_outcomes or [
            'user_preference_stated',    # HIGHEST PRIORITY - user stated a preference
            'user_instruction_given',    # User gave specific instructions
            'user_suggestion_made',      # User made suggestions for improvement
            'user_feedback_provided',    # User provided feedback on responses
            'code_block_copied',
            'resolved_and_ended',
            'task_completed',
            'user_satisfied',
            'solution_accepted',
            'helpful_response'
        ]
    
    def record_interaction(self, 
                         conversation: Union[str, List[Dict[str, str]]], 
                         outcome: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a user interaction for future learning.
        
        Args:
            conversation: The conversation content (string summary or list of messages)
            outcome (str): The outcome tag (e.g., 'code_block_copied', 'user_rephrased', etc.)
            metadata (Dict[str, Any], optional): Additional metadata about the interaction
        """
        interaction_data = {
            'conversation': conversation,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.storage_adapter.write_interaction(interaction_data)
    
    def _synthesize_experience(self, interactions: List[Dict[str, Any]]) -> str:
        """
        Synthesize experience-based rules from successful interactions.
        
        Args:
            interactions (List[Dict[str, Any]]): List of successful interaction objects
            
        Returns:
            str: Synthesized experience prompt or empty string if no interactions
        """
        if not interactions:
            return ""
        
        # Separate different types of learning interactions
        preference_interactions = []
        instruction_interactions = []
        feedback_interactions = []
        other_interactions = []
        
        for interaction in interactions:
            outcome = interaction.get('outcome', '')
            if outcome == 'user_preference_stated':
                preference_interactions.append(interaction)
            elif outcome == 'user_instruction_given':
                instruction_interactions.append(interaction)
            elif outcome == 'user_feedback_provided':
                feedback_interactions.append(interaction)
            else:
                other_interactions.append(interaction)
        
        # Start with guidelines from learnable interactions
        guidelines = []
        
        # Process user preferences first (highest priority)
        if preference_interactions:
            for interaction in preference_interactions[-5:]:  # Last 5 preferences
                metadata = interaction.get('metadata', {})
                message_type = metadata.get('message_type')
                message_content = metadata.get('message_content')
                original_message = metadata.get('original_message', '')
                
                if message_type == 'preference':
                    # Use the LLM-extracted content if available, otherwise original message
                    content = message_content if message_content and message_content != original_message else original_message
                    
                    # Check for name preferences specifically
                    if any(phrase in original_message.lower() for phrase in ['call me', 'refer to me', 'my name is', 'prefer to be called']):
                        # Extract name from content
                        import re
                        name_match = re.search(r'\b(?:call me|refer to me as|my name is|prefer to be called)\s+(\w+)', original_message, re.IGNORECASE)
                        if name_match:
                            name = name_match.group(1)
                            guidelines.append(f"Address the user as '{name}' in all interactions.")
                        else:
                            guidelines.append(f"User naming preference: {content}")
                    else:
                        guidelines.append(f"User preference: {content}")
        
        # Process instructions
        if instruction_interactions:
            for interaction in instruction_interactions[-3:]:  # Last 3 instructions
                metadata = interaction.get('metadata', {})
                message_content = metadata.get('message_content', '')
                original_message = metadata.get('original_message', '')
                content = message_content if message_content and message_content != original_message else original_message
                guidelines.append(f"User instruction: {content}")
        
        # Process feedback for behavioral adjustments
        if feedback_interactions:
            for interaction in feedback_interactions[-3:]:  # Last 3 feedback items
                metadata = interaction.get('metadata', {})
                message_content = metadata.get('message_content', '')
                original_message = metadata.get('original_message', '')
                content = message_content if message_content and message_content != original_message else original_message
                guidelines.append(f"User feedback: {content}")
        
        # If we have direct preferences, return them immediately
        if guidelines:
            return "\n".join([f"{i+1}. {guideline}" for i, guideline in enumerate(guidelines)])
        
        # Fallback to LLM synthesis for other interactions
        if other_interactions:
            meta_prompt = """Based on the following successful interactions, identify specific user communication preferences and behavioral patterns. Focus ONLY on:

1. How the user likes to be addressed or referenced
2. Preferred response formats (short, detailed, etc.)
3. Communication style preferences
4. Specific instructions the user has given

Successful Interactions:
"""
            
            # Add other interactions
            for i, interaction in enumerate(other_interactions[-5:], 1):
                conversation_summary = interaction.get('conversation', '')
                if isinstance(conversation_summary, list):
                    conversation_summary = f"Multi-turn conversation with {len(conversation_summary)} exchanges"
                
                meta_prompt += f"""
{i}. Outcome: {interaction.get('outcome', 'unknown')}
   Conversation: {conversation_summary[:300]}...
"""
            
            meta_prompt += """\n\nGenerate 2-4 specific guidelines about this user's preferences. Each should be actionable and specific. Format as numbered points."""
            
            try:
                system_prompt = "You are an AI assistant that extracts specific user preferences and communication styles from successful interactions."
                
                response = self.llm_adapter.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=meta_prompt,
                    max_tokens=200,
                    temperature=0.2
                )
                
                return response
            
            except Exception as e:
                print(f"Warning: Could not synthesize experience: {e}")
                return ""
        
        return ""
    
    def get_prompt(self) -> str:
        """
        Get the complete evolved system prompt.
        
        Returns:
            str: The complete system prompt including base prompt and synthesized experience
        """
        # Read all interactions
        all_interactions = self.storage_adapter.read_interactions()
        
        # Filter for successful interactions
        successful_interactions = [
            interaction for interaction in all_interactions
            if interaction.get('outcome', '').lower() in [outcome.lower() for outcome in self.success_outcomes]
        ]
        
        # Synthesize experience from successful interactions
        experience_prompt = self._synthesize_experience(successful_interactions)
        
        # Combine base prompt with experience
        if experience_prompt:
            return f"""{self.base_prompt}

--- Learned Experience ---
{experience_prompt}

Apply these learnings to provide more helpful and targeted responses."""
        else:
            return self.base_prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored interactions.
        
        Returns:
            Dict[str, Any]: Dictionary containing interaction statistics
        """
        all_interactions = self.storage_adapter.read_interactions()
        
        # Count outcomes
        outcome_counts = {}
        for interaction in all_interactions:
            outcome = interaction.get('outcome', 'unknown')
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        # Count successful interactions
        successful_count = sum(
            1 for interaction in all_interactions
            if interaction.get('outcome', '').lower() in [outcome.lower() for outcome in self.success_outcomes]
        )
        
        return {
            'total_interactions': len(all_interactions),
            'successful_interactions': successful_count,
            'success_rate': successful_count / len(all_interactions) if all_interactions else 0,
            'outcome_breakdown': outcome_counts
        }
    
    def clear_history(self) -> None:
        """
        Clear all stored interaction history.
        """
        self.storage_adapter.clear_interactions()
    
    def export_interactions(self) -> List[Dict[str, Any]]:
        """
        Export all stored interactions for analysis or backup.
        
        Returns:
            List[Dict[str, Any]]: List of all stored interactions
        """
        return self.storage_adapter.read_interactions()
