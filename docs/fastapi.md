# FastAPI Integration

This guide shows how to integrate ExperienceAI into an existing FastAPI application. It exposes HTTP endpoints for chatting with an evolving assistant, recording interactions, and inspecting learning stats.

Example file: examples/fastapi_integration.py

Prerequisites
- Python 3.8+
- Install dependencies:
  - pip install fastapi uvicorn python-dotenv
  - plus your LLM client:
    - OpenAI: pip install openai
    - Gemini: pip install google-generativeai

Environment variables
- Choose one LLM provider:
  - export OPENAI_API_KEY={{OPENAI_API_KEY}}
  - or export GEMINI_API_KEY={{GEMINI_API_KEY}}
- Optional storage path for interactions:
  - export EA_API_STORAGE=/var/data/experience-ai/api_interactions.json
- Optional model overrides:
  - export OPENAI_MODEL=gpt-4o-mini
  - export GEMINI_MODEL=gemini-2.0-flash
- Optional CORS configuration:
  - export CORS_ALLOW_ORIGINS="http://localhost:3000,https://yourapp.com"

Run the server
- uvicorn examples.fastapi_integration:app --host 0.0.0.0 --port 8000 --reload

Endpoints
- GET /health
  - Returns { status, provider }
- GET /prompt
  - Returns the current evolved prompt and stats
- GET /stats
  - Returns learning statistics
- POST /record
  - Body: { conversation: string, outcome: string, metadata?: object }
  - Records an interaction and refreshes the prompt
- POST /clear
  - Clears all learning history (use with caution)
- POST /chat
  - Body: { message: string, session_id?: string, max_tokens?: number, temperature?: number }
  - Returns { response, classification, provider, prompt_version }
  - Automatically classifies the interaction and records it with metadata

Example requests
- curl http://localhost:8000/health
- curl -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"Call me Nana and keep replies short."}'
- curl http://localhost:8000/prompt
- curl http://localhost:8000/stats

Notes
- Storage is file-backed via LocalStorageAdapter; for production, you may implement a DB-backed storage adapter.
- Keep secrets out of logs and source control. Configure environment variables securely.
- The FastAPI example uses the same LLM adapter for chat generation and interaction classification to minimize configuration and cost.

