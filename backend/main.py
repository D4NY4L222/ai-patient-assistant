# -*- coding: utf-8 -*-
import os, logging, re, difflib
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve, build_context_snippets  # RAG helpers

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
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class Inquiry(BaseModel):
    question: str

# --- Prompt (softer than strict-context; still safe) ---
SYSTEM_PROMPT = (
    "You are an assistant for the Signifier sleep therapy device and related service ONLY.\n"
    "PRIORITIES:\n"
    "1) Safety: Do NOT give medical advice, diagnosis, or treatment instructions.\n"
    "2) Helpfulness: Answer concisely (1–3 sentences), using the provided CONTEXT if relevant.\n"
    "3) Scope: If the user asks about non-Signifier topics, politely redirect to clinician/official resources.\n"
    "If there is no relevant context for the question, provide a short general, non-medical answer "
    "and suggest contacting support/clinician as appropriate.\n"
    "At the end, include bracket citations [1], [2] for any facts taken from CONTEXT; if none used, omit citations."
)

# --- Keyword / typo tolerant scope checks ---
ALLOWED_KEYWORDS: List[str] = [
    # device / product
    "signifier","sleep","apnea","apnoea","device","therapy","tongue","mouthpiece","control","unit",
    # usage / setup
    "usage","use","using","setup","set","install","pair","connect","charge","battery","intensity",
    # care / maintenance
    "clean","cleaning","care","maintain","maintenance",
    # support / logistics
    "support","appointment","book","reschedule","cancel","hours","contact","warranty",
    "replacement","spare","parts","manual","guide","troubleshoot","troubleshooting","error","issue","problem",
]
GREETING_WORDS = {"hi","hello","hey","howdy","good","morning","afternoon","evening"}

def normalize_text(s: str) -> str:
    for a,b in {"\u2019":"'", "\u2018":"'", "\u201C":'"', "\u201D":'"', "\u00A0":" "}.items():
        s = s.replace(a,b)
    return s.strip()

def tokens(s: str) -> List[str]:
    s = s.lower()
    # keep words only
    return re.findall(r"[a-z]+", s)

def tokens_have_allowed_with_fuzzy(tok_list: List[str], allowed: List[str], ratio: float = 0.8) -> bool:
    # direct hit
    if any(t in allowed for t in tok_list): 
        return True
    # fuzzy hit (typo tolerance)
    for t in tok_list:
        matches = difflib.get_close_matches(t, allowed, n=1, cutoff=ratio)
        if matches:
            return True
    return False

def is_greeting(tok_list: List[str]) -> bool:
    return any(t in GREETING_WORDS for t in tok_list)

# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/inquiry")
def inquiry(payload: Inquiry):
    q_raw = (payload.question or "").strip()
    if not q_raw:
        ans = "Please enter a question about the Signifier device or support."
        log.info("Q:<empty> | A:%s", ans); 
        return {"answer": ans}

    q = normalize_text(q_raw)
    tks = tokens(q)
    log.info("Q: %s", q)

    # Friendly greeting if that's all we got
    if is_greeting(tks) and not tokens_have_allowed_with_fuzzy(tks, ALLOWED_KEYWORDS):
        ans = ("Hi! I’m the assistant for the Signifier sleep therapy device. "
               "Ask me about setup, usage, troubleshooting, appointments or support.")
        log.info("A (greeting): %s", ans)
        return {"answer": ans}

    # Scope check with typo tolerance
    in_scope = tokens_have_allowed_with_fuzzy(tks, ALLOWED_KEYWORDS, ratio=0.78)
    if not in_scope:
        ans = ("I can help with the Signifier sleep therapy device and support only "
               "(setup, usage, troubleshooting, appointments). For other topics, please contact your clinician.")
        log.info("A (scope-refusal): %s", ans)
        return {"answer": ans}

    # RAG: retrieve context (but don't fail if empty)
    try:
        results = retrieve(q, k=4)
        context_block, cites_list = build_context_snippets(results)
    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        context_block, cites_list = ("", [])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"CONTEXT:\n{context_block or 'NO RELEVANT CONTEXT FOUND'}"},
        {"role": "user", "content": q},
    ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.15,
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
        return {"answer": demo_reply, "note": "demo_fallback", "error": str(e)}

# Serve frontend at /app
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    log.warning("Frontend directory not found at %s", FRONTEND_DIR)
