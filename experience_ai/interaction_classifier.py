"""
Automatic interaction classifier for determining success/failure of AI interactions.

This module provides tools to automatically analyze user interactions and determine
whether they were successful or not, enabling fully automatic learning.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InteractionClassification:
    """Result of interaction classification."""
    outcome: str
    confidence: float
    reasoning: str
    metadata: Dict


class AutoInteractionClassifier:
    """
    Automatically classifies interactions as successful or unsuccessful
    based on conversation patterns, user responses, and sentiment.
    """
    
    def __init__(self, llm_adapter=None):
        self.llm_adapter = llm_adapter
        # User preference patterns - HIGH PRIORITY for learning
        self.preference_patterns = [
            # Name and identity preferences - capture the name
            r'\b(?:call me|refer to me as|my name is|i\'?m)\s+(\w+)',
            r'\b(?:address me as|use the name)\s+(\w+)',
            r'\bi prefer to be called\s+(\w+)',
            
            # Communication style preferences with context
            r'\bi (?:like|prefer|want|need)\s+(.{5,100})\s+(?:responses?|answers?|when)',
            r'\bplease\s+(.{5,50})\s+(?:when|in|for)',
            r'\bcan you\s+(.{5,50})\s+(?:instead|rather than)',
            r'\bjust so you know,?\s+i\s+(.{10,150})',
            r'\bremember,?\s+i\s+(.{10,150})',
            
            # Format preferences - capture full context including conditions
            r'\bi (?:like|prefer|want)\s+((?:short|brief|concise|one[- ]?line).{0,100})',
            r'\bkeep (?:it|your responses?)\s+(.{5,50})',
            r'\bmake (?:it|your responses?)\s+(.{5,50})',
            
            # Topic-specific preferences - capture both topic and preference
            r'\bfor\s+(\w+(?:\s+\w+)*)\s+(?:questions?|topics?),?\s+i (?:like|prefer|want)\s+(.{5,80})',
            r'\bwhen i ask about\s+(\w+(?:\s+\w+)*),?\s+(.{10,80})',
            r'\bi (?:like|prefer|want)\s+(.{0,50}?)\s+(?:answers?|responses?)\s+(?:only\s+)?when\s+i\s+ask\s+about\s+(\w+(?:\s+\w+)*)',
        ]
        
        self.positive_patterns = [
            # Direct positive responses
            r'\b(thank(?:s|you)|great|perfect|awesome|excellent|amazing|wonderful)\b',
            r'\b(works?|working|solved|fixed|helpful|useful|exactly)\b',
            r'\b(love|like|appreciate|brilliant|fantastic|superb)\b',
            
            # Confirmation patterns
            r'\b(yes|yeah|yep|correct|right|exactly|precisely)\b',
            r'\b(got it|understand|makes sense|clear)\b',
            
            # Success indicators
            r'\b(success|successful|done|completed|finished)\b',
            r'\b(that.{0,20}(?:worked|works|helped|perfect))\b',
        ]
        
        self.negative_patterns = [
            # Direct negative responses
            r'\b(no|nope|wrong|incorrect|bad|terrible|awful)\b',
            r'\b(doesn.?t work|not working|failed|error|problem)\b',
            r'\b(confused|confusing|unclear|don.?t understand)\b',
            
            # Frustration indicators
            r'\b(frustrated|annoying|annoyed|stuck|help)\b',
            r'\b(still not|never mind|forget it|give up)\b',
            
            # Correction patterns
            r'\b(actually|no,|wait,|that.?s not|not what)\b',
            r'\b(let me try again|different|another way)\b',
            
            # Repetition (user repeating similar requests)
            r'\b(again|repeat|once more|still)\b',
        ]
        
        self.task_completion_patterns = [
            r'\b(copied|copy|saved|downloaded|installed|created)\b',
            r'\b(file|code|script|program).{0,20}(?:works|working|runs)\b',
            r'\b(?:it|this|that).{0,10}(?:works|worked|fixed)\b',
        ]
        
        self.engagement_patterns = [
            r'\b(?:what|how|why|when|where).{0,50}\?',
            r'\b(?:can you|could you|would you|please)\b',
            r'\b(?:also|additionally|furthermore|moreover|next)\b',
            r'\b(?:another|more|different|other)\b',
        ]
    
    def classify_message_type(self, user_message: str, llm_adapter=None) -> Dict[str, any]:
        """
        Use LLM to intelligently classify the type of user message.
        
        Args:
            user_message: The user's message to analyze
            llm_adapter: LLM adapter for classification (optional)
            
        Returns:
            Dict containing classification info
        """
        if not llm_adapter:
            # Fallback to simple heuristics if no LLM available
            return self._fallback_classification(user_message)
        
        classification_prompt = f"""Classify the following user message into ONE of these categories:

1. PREFERENCE - User is stating how they want to be addressed, communication style preferences, or format preferences
   Examples: "call me John", "I prefer short answers", "I like detailed explanations when asking about science"

2. INSTRUCTION - User is giving specific directions about how to behave or respond
   Examples: "always ask for clarification", "don't use technical jargon", "remember this for future conversations"

3. SUGGESTION - User is proposing an idea or recommendation
   Examples: "you might want to consider", "it would be better if", "perhaps you could"

4. FEEDBACK - User is commenting on previous responses or interactions
   Examples: "that was helpful", "not quite what I meant", "perfect explanation"

5. RESPONSE - User is responding to a question or continuing normal conversation
   Examples: Questions, requests for information, normal conversational responses

User message: "{user_message}"

Classification: [PREFERENCE|INSTRUCTION|SUGGESTION|FEEDBACK|RESPONSE]
Confidence: [0.1-1.0]
Reasoning: [brief explanation]
Key Content: [extract the most important part for learning]"""
        
        try:
            system_prompt = "You are a precise message classifier. Respond in the exact format requested."
            response = llm_adapter.generate_text(
                system_prompt=system_prompt,
                user_prompt=classification_prompt,
                max_tokens=150,
                temperature=0.1
            )
            
            # Parse the LLM response
            return self._parse_classification_response(response, user_message)
            
        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
            return self._fallback_classification(user_message)
    
    def _parse_classification_response(self, response: str, original_message: str) -> Dict[str, any]:
        """Parse the structured LLM classification response."""
        classification = {
            'type': 'response',
            'confidence': 0.5,
            'reasoning': 'Default classification',
            'content': original_message,
            'should_learn': False,
            'classification_method': 'llm'  # Mark this as LLM-based classification
        }
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Classification:'):
                class_type = line.split(':', 1)[1].strip().lower()
                if class_type in ['preference', 'instruction', 'suggestion', 'feedback']:
                    classification['type'] = class_type
                    classification['should_learn'] = True
                elif 'preference' in class_type:
                    classification['type'] = 'preference'
                    classification['should_learn'] = True
                elif 'instruction' in class_type:
                    classification['type'] = 'instruction'
                    classification['should_learn'] = True
                    
            elif line.startswith('Confidence:'):
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    classification['confidence'] = float(confidence_str)
                except:
                    pass
                    
            elif line.startswith('Reasoning:'):
                classification['reasoning'] = line.split(':', 1)[1].strip()
                
            elif line.startswith('Key Content:'):
                classification['content'] = line.split(':', 1)[1].strip()
        
        return classification
    
    def _fallback_classification(self, user_message: str) -> Dict[str, any]:
        """Fallback classification using simple heuristics when LLM is unavailable."""
        user_text = user_message.lower()
        
        # Simple heuristic checks
        if any(phrase in user_text for phrase in ['call me', 'refer to me', 'my name is', 'i prefer', 'i like', 'i want']):
            return {
                'type': 'preference',
                'confidence': 0.7,
                'reasoning': 'Contains preference indicators',
                'content': user_message,
                'should_learn': True,
                'classification_method': 'heuristic'
            }
        
        elif any(phrase in user_text for phrase in ['remember', 'always', 'never', 'don\'t', 'please']):
            return {
                'type': 'instruction',
                'confidence': 0.6,
                'reasoning': 'Contains instruction indicators',
                'content': user_message,
                'should_learn': True,
                'classification_method': 'heuristic'
            }
        
        elif any(phrase in user_text for phrase in ['you should', 'you could', 'might want', 'suggest']):
            return {
                'type': 'suggestion',
                'confidence': 0.6,
                'reasoning': 'Contains suggestion indicators',
                'content': user_message,
                'should_learn': True,
                'classification_method': 'heuristic'
            }
        
        elif any(phrase in user_text for phrase in ['good', 'bad', 'wrong', 'right', 'helpful', 'not what', 'too technical', 'too complex', 'too simple', 'better', 'clearer']):
            return {
                'type': 'feedback',
                'confidence': 0.6,
                'reasoning': 'Contains feedback indicators',
                'content': user_message,
                'should_learn': True,
                'classification_method': 'heuristic'
            }
        
        else:
            return {
                'type': 'response',
                'confidence': 0.8,
                'reasoning': 'Normal conversational response',
                'content': user_message,
                'should_learn': False,
                'classification_method': 'heuristic'
            }
    
    def classify_interaction(self, 
                           user_message: str, 
                           agent_response: str, 
                           follow_up_message: Optional[str] = None,
                           conversation_history: Optional[List[str]] = None,
                           llm_adapter=None) -> InteractionClassification:
        """
        Classify an interaction as successful or unsuccessful.
        
        Args:
            user_message: The user's original message
            agent_response: The agent's response
            follow_up_message: User's follow-up (if any)
            conversation_history: Previous messages in conversation
            llm_adapter: LLM adapter for intelligent classification
            
        Returns:
            InteractionClassification with outcome and reasoning
        """
        
        # First check message type using LLM-based classification - HIGHEST PRIORITY
        # Use provided llm_adapter or fall back to class-level one
        effective_llm_adapter = llm_adapter or self.llm_adapter
        message_classification = self.classify_message_type(user_message, effective_llm_adapter)
        
        # Set classification method explicitly on the metadata
        classification_method = 'llm' if effective_llm_adapter else 'heuristic'
        
        if message_classification['should_learn']:
            # Map message types to outcomes that match EvolvingPrompt success_outcomes
            outcome_mapping = {
                'preference': 'user_preference_stated',
                'instruction': 'instruction_received', 
                'suggestion': 'suggestion_received',
                'feedback': 'feedback_received'
            }
            
            outcome = outcome_mapping.get(message_classification['type'], 'user_preference_stated')
            
            return InteractionClassification(
                outcome=outcome,
                confidence=message_classification['confidence'],
                reasoning=message_classification['reasoning'],
                metadata={
                    'message_type': message_classification['type'],
                    'message_content': message_classification['content'],
                    'should_learn': True,
                    'original_message': user_message,
                    'auto_classified': True,
                    'classification_method': classification_method
                }
            )
        
        # Combine all user text for analysis
        user_text = user_message.lower()
        if follow_up_message:
            user_text += " " + follow_up_message.lower()
        
        agent_text = agent_response.lower()
        
        # Calculate scores for different indicators
        positive_score = self._calculate_pattern_score(user_text, self.positive_patterns)
        negative_score = self._calculate_pattern_score(user_text, self.negative_patterns)
        task_completion_score = self._calculate_pattern_score(user_text, self.task_completion_patterns)
        engagement_score = self._calculate_pattern_score(user_text, self.engagement_patterns)
        
        # Analyze conversation flow
        flow_analysis = self._analyze_conversation_flow(
            user_message, agent_response, follow_up_message
        )
        
        # Check for specific success indicators
        success_indicators = self._check_success_indicators(
            user_text, agent_text, conversation_history or []
        )
        
        # Determine outcome based on weighted scoring
        total_positive = positive_score + task_completion_score + flow_analysis['positive'] + success_indicators['positive']
        total_negative = negative_score + flow_analysis['negative'] + success_indicators['negative']
        
        # Engagement is positive but less weighted
        total_positive += engagement_score * 0.5
        
        # Classification logic
        if total_positive > total_negative and total_positive > 0.3:
            if task_completion_score > 0.2:
                outcome = "task_completed"
                confidence = min(0.9, total_positive)
                reasoning = "User indicated task completion or success"
            elif positive_score > 0.4:
                outcome = "user_satisfied"
                confidence = min(0.85, total_positive)
                reasoning = "Strong positive sentiment detected"
            elif engagement_score > 0.3:
                outcome = "helpful_response"
                confidence = min(0.75, total_positive)
                reasoning = "User engaged with follow-up questions"
            else:
                outcome = "helpful_response"
                confidence = min(0.7, total_positive)
                reasoning = "Generally positive interaction"
        
        elif total_negative > total_positive and total_negative > 0.3:
            outcome = "user_unsatisfied"
            confidence = min(0.8, total_negative)
            reasoning = "Negative sentiment or confusion detected"
        
        else:
            # Neutral or unclear
            if self._is_simple_greeting(user_message):
                outcome = "simple_greeting"
                confidence = 0.6
                reasoning = "Simple greeting or acknowledgment"
            elif len(agent_response) < 50 and not follow_up_message:
                outcome = "insufficient_response"
                confidence = 0.5
                reasoning = "Response may be too brief"
            else:
                outcome = "neutral_interaction"
                confidence = 0.4
                reasoning = "Interaction sentiment unclear"
        
        return InteractionClassification(
            outcome=outcome,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'positive_score': positive_score,
                'negative_score': negative_score,
                'task_completion_score': task_completion_score,
                'engagement_score': engagement_score,
                'flow_analysis': flow_analysis,
                'success_indicators': success_indicators,
                'user_message_length': len(user_message),
                'agent_response_length': len(agent_response),
                'has_follow_up': follow_up_message is not None
            }
        )
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate how well text matches a set of patterns."""
        if not text:
            return 0.0
            
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.0
    
    def _analyze_conversation_flow(self, user_msg: str, agent_resp: str, follow_up: Optional[str]) -> Dict:
        """Analyze the flow of conversation for success indicators."""
        analysis = {'positive': 0.0, 'negative': 0.0}
        
        if not follow_up:
            # No follow-up could mean satisfaction or disengagement
            if len(agent_resp) > 100:  # Substantial response
                analysis['positive'] += 0.1
            return analysis
        
        follow_up_lower = follow_up.lower()
        
        # Positive flow indicators
        if any(word in follow_up_lower for word in ['thank', 'great', 'perfect', 'works']):
            analysis['positive'] += 0.3
        
        if '?' in follow_up:  # Follow-up question indicates engagement
            analysis['positive'] += 0.2
        
        # Negative flow indicators
        if any(word in follow_up_lower for word in ['not', 'wrong', 'error', 'problem']):
            analysis['negative'] += 0.3
        
        # Repetition of similar request
        user_words = set(user_msg.lower().split())
        follow_up_words = set(follow_up.lower().split())
        overlap = len(user_words & follow_up_words)
        if overlap > len(user_words) * 0.5:  # High overlap might indicate repetition
            analysis['negative'] += 0.2
        
        return analysis
    
    def _check_success_indicators(self, user_text: str, agent_text: str, history: List[str]) -> Dict:
        """Check for specific success indicators."""
        indicators = {'positive': 0.0, 'negative': 0.0}
        
        # Check if agent provided code/commands that user acknowledged
        if any(marker in agent_text for marker in ['```', 'command:', 'run:', '$ ']):
            if any(word in user_text for word in ['worked', 'works', 'success', 'done']):
                indicators['positive'] += 0.3
        
        # Check for problem resolution
        problem_words = ['error', 'issue', 'problem', 'bug', 'broken']
        solution_words = ['fixed', 'resolved', 'solved', 'working']
        
        if any(word in ' '.join(history[-3:]).lower() for word in problem_words):
            if any(word in user_text for word in solution_words):
                indicators['positive'] += 0.4
        
        return indicators
    
    def _is_simple_greeting(self, message: str) -> bool:
        """Check if message is a simple greeting."""
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        message_lower = message.lower().strip()
        
        return (message_lower in greetings or 
                len(message.split()) <= 3 and 
                any(greeting in message_lower for greeting in greetings))


# Utility function for easy access
def classify_interaction(user_message: str, 
                        agent_response: str, 
                        follow_up_message: Optional[str] = None,
                        conversation_history: Optional[List[str]] = None) -> InteractionClassification:
    """
    Convenience function to classify an interaction.
    
    Args:
        user_message: The user's original message
        agent_response: The agent's response  
        follow_up_message: User's follow-up message (optional)
        conversation_history: Previous conversation context (optional)
        
    Returns:
        InteractionClassification with outcome and reasoning
    """
    classifier = AutoInteractionClassifier()
    return classifier.classify_interaction(
        user_message, agent_response, follow_up_message, conversation_history
    )
