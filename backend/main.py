# -*- coding: utf-8 -*-
import os, logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve, build_context_snippets  # NEW

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ai-patient-inquiry")

# --- Env & OpenAI ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to backend/.env")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI ---
app = FastAPI(title="AI Patient Inquiry Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Inquiry(BaseModel):
    question: str

SYSTEM_PROMPT = (
    "You are an assistant for the Signifier sleep therapy device and related service ONLY.\n"
    "Answer strictly using the provided CONTEXT. If the answer is not in CONTEXT, reply:\n"
    "\"I can help with the Signifier device and support only. For other topics, please contact your clinician.\"\n"
    "NEVER provide medical advice, diagnosis, or treatment instructions. Keep answers to 1–3 sentences.\n"
    "At the end, include short bracket citations like [1], [2] that correspond to the provided context items."
)

ALLOWED_KEYWORDS: List[str] = [
    "signifier", "sleep", "apnea", "device", "therapy", "tongue", "usage", "use",
    "setup", "install", "pair", "charge", "battery", "clean", "maintain", "maintenance",
    "support", "appointment", "book", "reschedule", "cancel", "hours", "contact", "warranty",
    "replacement", "spare", "parts", "manual", "guide", "troubleshoot", "error", "issue",
]

def in_scope(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ALLOWED_KEYWORDS)

def normalize_text(s: str) -> str:
    for a,b in {"\u2019":"'", "\u2018":"'", "\u201C":'"', "\u201D":'"', "\u00A0":" "}.items():
        s = s.replace(a,b)
    return s.strip()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inquiry")
def inquiry(payload: Inquiry):
    q = (payload.question or "").strip()
    if not q:
        ans = "Please enter a question about the Signifier device or support."
        log.info("Q:<empty> | A:%s", ans); return {"answer": ans}

    log.info("Q: %s", q)

    if not in_scope(q):
        ans = ("I can help with the Signifier sleep therapy device and support only "
               "(setup, usage, troubleshooting, appointments). For other topics, please contact your clinician.")
        log.info("A (scope-refusal): %s", ans); return {"answer": ans}

    # RAG: retrieve relevant FAQ chunks
    snippets = []
    cites_list = []
    try:
        results = retrieve(q, k=4)
        context_block, cites_list = build_context_snippets(results)
    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        context_block = ""
        cites_list = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"CONTEXT:\n{context_block or 'NO RELEVANT CONTEXT FOUND'}"},
        {"role": "user", "content": q},
    ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=220,
        )
        answer = normalize_text(chat.choices[0].message.content or "")
        log.info("A: %s", answer)
        return {"answer": answer, "citations": cites_list}
    except Exception as e:
        demo_reply = ("I can help with the Signifier device and support only. "
                      "For medical or out-of-scope questions, please contact your clinician.")
        log.error("OpenAI error: %s", str(e))
        log.info("A (fallback): %s", demo_reply)
        return {"answer": demo_reply, "note": "demo_fallback", "error": str(e), "citations": cites_list}

# Serve frontend at /app
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    log.warning("Frontend directory not found at %s", FRONTEND_DIR)
# -*- coding: utf-8 -*-
import os, logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve, build_context_snippets  # NEW

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ai-patient-inquiry")

# --- Env & OpenAI ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to backend/.env")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI ---
app = FastAPI(title="AI Patient Inquiry Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Inquiry(BaseModel):
    question: str

SYSTEM_PROMPT = (
    "You are an assistant for the Signifier sleep therapy device and related service ONLY.\n"
    "Answer strictly using the provided CONTEXT. If the answer is not in CONTEXT, reply:\n"
    "\"I can help with the Signifier device and support only. For other topics, please contact your clinician.\"\n"
    "NEVER provide medical advice, diagnosis, or treatment instructions. Keep answers to 1–3 sentences.\n"
    "At the end, include short bracket citations like [1], [2] that correspond to the provided context items."
)

ALLOWED_KEYWORDS: List[str] = [
    "signifier", "sleep", "apnea", "device", "therapy", "tongue", "usage", "use",
    "setup", "install", "pair", "charge", "battery", "clean", "maintain", "maintenance",
    "support", "appointment", "book", "reschedule", "cancel", "hours", "contact", "warranty",
    "replacement", "spare", "parts", "manual", "guide", "troubleshoot", "error", "issue",
]

def in_scope(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ALLOWED_KEYWORDS)

def normalize_text(s: str) -> str:
    for a,b in {"\u2019":"'", "\u2018":"'", "\u201C":'"', "\u201D":'"', "\u00A0":" "}.items():
        s = s.replace(a,b)
    return s.strip()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inquiry")
def inquiry(payload: Inquiry):
    q = (payload.question or "").strip()
    if not q:
        ans = "Please enter a question about the Signifier device or support."
        log.info("Q:<empty> | A:%s", ans); return {"answer": ans}

    log.info("Q: %s", q)

    if not in_scope(q):
        ans = ("I can help with the Signifier sleep therapy device and support only "
               "(setup, usage, troubleshooting, appointments). For other topics, please contact your clinician.")
        log.info("A (scope-refusal): %s", ans); return {"answer": ans}

    # RAG: retrieve relevant FAQ chunks
    snippets = []
    cites_list = []
    try:
        results = retrieve(q, k=4)
        context_block, cites_list = build_context_snippets(results)
    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        context_block = ""
        cites_list = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"CONTEXT:\n{context_block or 'NO RELEVANT CONTEXT FOUND'}"},
        {"role": "user", "content": q},
    ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=220,
        )
        answer = normalize_text(chat.choices[0].message.content or "")
        log.info("A: %s", answer)
        return {"answer": answer, "citations": cites_list}
    except Exception as e:
        demo_reply = ("I can help with the Signifier device and support only. "
                      "For medical or out-of-scope questions, please contact your clinician.")
        log.error("OpenAI error: %s", str(e))
        log.info("A (fallback): %s", demo_reply)
        return {"answer": demo_reply, "note": "demo_fallback", "error": str(e), "citations": cites_list}

# Serve frontend at /app
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    log.warning("Frontend directory not found at %s", FRONTEND_DIR)
# -*- coding: utf-8 -*-
import os, logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve, build_context_snippets  # NEW

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ai-patient-inquiry")

# --- Env & OpenAI ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to backend/.env")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI ---
app = FastAPI(title="AI Patient Inquiry Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Inquiry(BaseModel):
    question: str

SYSTEM_PROMPT = (
    "You are an assistant for the Signifier sleep therapy device and related service ONLY.\n"
    "Answer strictly using the provided CONTEXT. If the answer is not in CONTEXT, reply:\n"
    "\"I can help with the Signifier device and support only. For other topics, please contact your clinician.\"\n"
    "NEVER provide medical advice, diagnosis, or treatment instructions. Keep answers to 1–3 sentences.\n"
    "At the end, include short bracket citations like [1], [2] that correspond to the provided context items."
)

ALLOWED_KEYWORDS: List[str] = [
    "signifier", "sleep", "apnea", "device", "therapy", "tongue", "usage", "use",
    "setup", "install", "pair", "charge", "battery", "clean", "maintain", "maintenance",
    "support", "appointment", "book", "reschedule", "cancel", "hours", "contact", "warranty",
    "replacement", "spare", "parts", "manual", "guide", "troubleshoot", "error", "issue",
]

def in_scope(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ALLOWED_KEYWORDS)

def normalize_text(s: str) -> str:
    for a,b in {"\u2019":"'", "\u2018":"'", "\u201C":'"', "\u201D":'"', "\u00A0":" "}.items():
        s = s.replace(a,b)
    return s.strip()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inquiry")
def inquiry(payload: Inquiry):
    q = (payload.question or "").strip()
    if not q:
        ans = "Please enter a question about the Signifier device or support."
        log.info("Q:<empty> | A:%s", ans); return {"answer": ans}

    log.info("Q: %s", q)

    if not in_scope(q):
        ans = ("I can help with the Signifier sleep therapy device and support only "
               "(setup, usage, troubleshooting, appointments). For other topics, please contact your clinician.")
        log.info("A (scope-refusal): %s", ans); return {"answer": ans}

    # RAG: retrieve relevant FAQ chunks
    snippets = []
    cites_list = []
    try:
        results = retrieve(q, k=4)
        context_block, cites_list = build_context_snippets(results)
    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        context_block = ""
        cites_list = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"CONTEXT:\n{context_block or 'NO RELEVANT CONTEXT FOUND'}"},
        {"role": "user", "content": q},
    ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=220,
        )
        answer = normalize_text(chat.choices[0].message.content or "")
        log.info("A: %s", answer)
        return {"answer": answer, "citations": cites_list}
    except Exception as e:
        demo_reply = ("I can help with the Signifier device and support only. "
                      "For medical or out-of-scope questions, please contact your clinician.")
        log.error("OpenAI error: %s", str(e))
        log.info("A (fallback): %s", demo_reply)
        return {"answer": demo_reply, "note": "demo_fallback", "error": str(e), "citations": cites_list}

# Serve frontend at /app
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    log.warning("Frontend directory not found at %s", FRONTEND_DIR)

