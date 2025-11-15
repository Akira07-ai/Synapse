# main.py — Synapse Medical Assistant Backend (gemini-2.0-flash compatible)

import os
import time
import logging
from typing import Optional, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------------------------
# Setup
# -------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=API_KEY)

# Select model (your choice: gemini-2.0-flash)
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="Synapse Medical Assistant", version="2.0")


# -------------------------------------------
# Static UI Mount
# -------------------------------------------

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "UI not found. Place index.html in ./static folder."}


# -------------------------------------------
# CORS
# -------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


# -------------------------------------------
# Rate Limit (lightweight in-memory)
# -------------------------------------------

RATE_LIMIT_WINDOW = 20
RATE_LIMIT_MAX = 18
rate_store: Dict[str, list] = {}


def apply_rate_limit(ip: str):
    now = time.time()
    entries = rate_store.get(ip, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    entries.append(now)
    rate_store[ip] = entries


# -------------------------------------------
# Emergency & Remote Detection
# -------------------------------------------

EMERGENCY_KEYWORDS = [
    "chest pain", "unconscious", "not breathing", "severe bleeding",
    "difficulty breathing", "shortness of breath", "heart attack",
    "stroke", "seizure", "fainting", "loss of consciousness",
]

REMOTE_KEYWORDS = [
    "remote", "alone", "forest", "mountain", "hiking", "camp", "no hospital",
    "far away", "30 km", "hours away", "village", "off-grid"
]


def detect_keywords(text: str, keywords: list):
    text = text.lower()
    return any(k in text for k in keywords)


def emergency_instructions():
    return (
        "⚠️ **Emergency detected — act immediately**\n\n"
        "**Steps until help arrives:**\n"
        "1. **Call your local emergency number now.**\n"
        "2. If severe bleeding: apply **firm direct pressure**.\n"
        "3. If unconscious but breathing: place in **recovery position**.\n"
        "4. If not breathing: begin **hands-only CPR** (100–120 per min).\n"
        "5. Keep them warm, monitor breathing, do NOT give food/drink.\n\n"
        "**These steps save lives. Call emergency services immediately.**"
    )


# -------------------------------------------
# Prompt Template
# -------------------------------------------

SYSTEM_PROMPT = """
You are Synapse — a calm, practical medical assistant.
Your role:
- Provide first-aid guidance and remote/field assistance.
- Triage severity clearly.
- Provide improvised steps for remote, off-grid situations.
- When emergency detected, give immediate lifesaving steps.
- Do NOT provide drug dosing or invasive procedures.
- Use short, clear, step-by-step instructions.
"""


def build_prompt(user_msg: str, context: Optional[str], remote: bool) -> str:
    ctx = f"Context: {context}\n" if context else ""
    rm = "The user is in a remote or field situation.\n" if remote else ""
    return f"{SYSTEM_PROMPT}\n{ctx}{rm}User: {user_msg}\nAssistant:"


# -------------------------------------------
# Request/Response Models
# -------------------------------------------

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    remote_mode: Optional[bool] = False


class ChatResponse(BaseModel):
    reply: str
    warning: bool
    ts: float


# -------------------------------------------
# Chat Endpoint
# -------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: Request, body: ChatRequest):

    client_ip = request.client.host if request.client else "local"
    apply_rate_limit(client_ip)

    msg = (body.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message.")

    # Emergency short circuit
    if detect_keywords(msg, EMERGENCY_KEYWORDS):
        reply = emergency_instructions()
        footer = "\n\n— Synapse: Get emergency help immediately."
        return ChatResponse(reply=reply + footer, warning=True, ts=time.time())

    # Remote mode auto-detection
    remote = body.remote_mode or detect_keywords(msg, REMOTE_KEYWORDS)

    # Build final prompt
    prompt = build_prompt(msg, body.context, remote)

    # Call Gemini API
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 600,
                "temperature": 0.15,
                "top_p": 0.9
            }
        )

        reply_text = response.text.strip() if hasattr(response, "text") else str(response)

    except Exception as e:
        logging.exception("Gemini Error")
        raise HTTPException(status_code=502, detail=f"AI backend error: {e}")

    # Standard footer
    footer = (
        "\n\n— Synapse reminder: I'm not a medical professional. "
        "Use this to stabilize until real help is available."
    )

    return ChatResponse(
        reply=reply_text + footer,
        warning=False,
        ts=time.time()
    )
