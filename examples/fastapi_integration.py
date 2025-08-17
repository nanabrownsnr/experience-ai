"""
FastAPI Integration Example for ExperienceAI

This example provides a lightweight HTTP API for integrating ExperienceAI's
EvolvingPrompt and AutoInteractionClassifier into existing applications.

Features:
- POST /chat: Chat with the assistant using the current evolved prompt
- POST /record: Record arbitrary interactions/outcomes
- GET /prompt: Fetch the current evolved prompt
- GET /stats: Inspect learning stats
- POST /clear: Clear learning history (use with caution)

LLM selection:
- If OPENAI_API_KEY is set and openai is installed, uses OpenAIAdapter
- Else if GEMINI_API_KEY is set and google-generativeai is installed, uses GeminiAdapter
- Else falls back to MockAdapter (no external API required)

Run:
  export OPENAI_API_KEY={{OPENAI_API_KEY}}  # or GEMINI_API_KEY={{GEMINI_API_KEY}}
  uvicorn examples.fastapi_integration:app --host 0.0.0.0 --port 8000 --reload

Note: Replace {{OPENAI_API_KEY}}/{{GEMINI_API_KEY}} with your actual secret values at runtime.
"""

import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from experience_ai import (
    EvolvingPrompt,
    LocalStorageAdapter,
    AutoInteractionClassifier,
    OpenAIAdapter,
    GeminiAdapter,
    MockAdapter,
)

# Load environment variables from .env if present
load_dotenv()

# -------- LLM Adapter Selection --------
_openai_available = False
_openai_client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        import openai
        from openai import OpenAI
        _openai_client = OpenAI()
        _openai_available = True
    except Exception:
        _openai_available = False

_gemini_available = False
_gemini_client = None
if os.getenv("GEMINI_API_KEY"):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _gemini_client = genai
        _gemini_available = True
    except Exception:
        _gemini_available = False

if _openai_available:
    LLM_PROVIDER = "openai"
    llm_adapter = OpenAIAdapter(client=_openai_client, model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
elif _gemini_available:
    LLM_PROVIDER = "gemini"
    llm_adapter = GeminiAdapter(client=_gemini_client, model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
else:
    LLM_PROVIDER = "mock"
    llm_adapter = MockAdapter()

# -------- Evolving Prompt & Classifier --------
storage_path = os.getenv("EA_API_STORAGE", "./api_interactions.json")
storage = LocalStorageAdapter(storage_path)

success_outcomes = [
    "task_completed",
    "tool_used_successfully",
    "user_satisfied",
    "problem_solved",
    "helpful_response",
    "correct_tool_selection",
    "clear_explanation_provided",
    # AutoInteractionClassifier can emit these when learning preferences/instructions
    "user_preference_stated",
    "instruction_received",
    "suggestion_received",
    "feedback_received",
]

BASE_PROMPT = (
    "You are a helpful AI assistant. Be concise, accurate, and adapt to user preferences "
    "learned from past interactions. Ask clarifying questions when needed."
)

prompt_manager = EvolvingPrompt(
    base_prompt=BASE_PROMPT,
    storage_adapter=storage,
    llm_adapter=llm_adapter,
    success_outcomes=success_outcomes,
)

classifier = AutoInteractionClassifier(llm_adapter=llm_adapter)

# -------- FastAPI App --------
app = FastAPI(title="ExperienceAI FastAPI Integration", version="0.1.0")

# CORS (configure as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.3


class ChatResponse(BaseModel):
    response: str
    classification: Dict[str, Any]
    provider: str
    prompt_version: int


class RecordRequest(BaseModel):
    conversation: str
    outcome: str
    metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "provider": LLM_PROVIDER}


@app.get("/prompt")
def get_prompt() -> Dict[str, Any]:
    return {"prompt": prompt_manager.get_prompt(), "stats": prompt_manager.get_stats()}


@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    return prompt_manager.get_stats()


@app.post("/clear")
def clear_history() -> Dict[str, Any]:
    prompt_manager.clear_history()
    return {"status": "cleared"}


@app.post("/record")
def record_interaction(body: RecordRequest = Body(...)) -> Dict[str, Any]:
    prompt_manager.record_interaction(
        conversation=body.conversation,
        outcome=body.outcome,
        metadata=body.metadata or {},
    )
    # Update prompt after recording
    prompt = prompt_manager.get_prompt()
    return {"status": "recorded", "prompt": prompt}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest = Body(...)) -> ChatResponse:
    # Use the current evolved prompt as the system instruction
    system_prompt = prompt_manager.get_prompt()

    # Generate a response via the selected LLM adapter
    response_text = llm_adapter.generate_text(
        system_prompt=system_prompt,
        user_prompt=body.message,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )

    # Classify interaction outcome and record it
    classification_obj = classifier.classify_interaction(
        user_message=body.message,
        agent_response=response_text,
        llm_adapter=llm_adapter,
    )

    prompt_manager.record_interaction(
        conversation=f"User: {body.message}\nAgent: {response_text}",
        outcome=classification_obj.outcome,
        metadata={
            "session_id": body.session_id,
            "response_length": len(response_text),
            "classification": {
                "outcome": classification_obj.outcome,
                "confidence": classification_obj.confidence,
                "reasoning": classification_obj.reasoning,
                **classification_obj.metadata,
            },
        },
    )

    # Update prompt (implicitly when next get_prompt() is called)
    # Return the response and classification metadata
    stats = prompt_manager.get_stats()
    prompt_version = stats.get("total_interactions", 0)

    return ChatResponse(
        response=response_text,
        classification={
            "outcome": classification_obj.outcome,
            "confidence": classification_obj.confidence,
            "reasoning": classification_obj.reasoning,
            **classification_obj.metadata,
        },
        provider=LLM_PROVIDER,
        prompt_version=prompt_version,
    )

