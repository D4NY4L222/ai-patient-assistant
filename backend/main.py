# -*- coding: utf-8 -*-
import os, logging, re, difflib
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve, build_context_snippets  # RAG helpers

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ai-patient-inquiry")

# ---------- Env & OpenAI ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to backend/.env or Render env vars (mark 'available during build').")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI ----------
app = FastAPI(title="AI Patient Inquiry Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- Models ----------
class Inquiry(BaseModel):
    question: str

# ---------- Prompt (concise, on-topic, cites context only) ----------
SYSTEM_PROMPT = (
    "You are the Signifier eXciteOSA assistant.\n"
    "Think step-by-step internally using the provided CONTEXT to locate the most relevant facts, "
    "but only output a concise answer (1–4 sentences).\n"
    "RULES:\n"
    "- Use CONTEXT as source of truth (therapy timing, app as remote, mouthpiece replacement, LED indicators).\n"
    "- No medical advice, diagnosis, or individualized treatment.\n"
    "- If off-topic, politely redirect.\n"
    "- If CONTEXT lacks the answer, say so briefly and recommend official support; do not invent details.\n"
    "- Include bracket citations [1], [2] only when you used CONTEXT; otherwise omit citations."
)

# ---------- Scope / fuzzy matching ----------
ALLOWED_KEYWORDS: List[str] = [
    # brand/product
    "signifier","exciteosa","exiteosa","device","therapy","tongue","mouthpiece","control","unit",
    # usage / therapy timing
    "usage","use","using","session","daytime","awake","20","twenty","minutes","duration","timer","schedule",
    # app / connectivity
    "app","application","remote","bluetooth","pair","connect","connection","intensity","start","pause",
    # care / maintenance
    "clean","cleaning","care","maintain","maintenance","replace","replacement","90","ninety","days","wear",
    # indicators / errors / power
    "led","light","indicator","status","blink","blinking","flash","flashing","solid","error","code",
    "battery","charge","charging","power","port","usb",
    # support/logistics
    "support","appointment","book","reschedule","cancel","hours","contact","warranty",
    "manual","guide","troubleshoot","troubleshooting","issue","problem",
]
GREETING_WORDS = {"hi","hello","hey","howdy","good","morning","afternoon","evening"}

def normalize_text(s: str) -> str:
    for a,b in {"\u2019":"'", "\u2018":"'", "\u201C":'"', "\u201D":'"', "\u00A0":" "}.items():
        s = s.replace(a,b)
    return s.strip()

def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z]+", s.lower())

def tokens_have_allowed_with_fuzzy(tok_list: List[str], allowed: List[str], ratio: float = 0.76) -> bool:
    if any(t in allowed for t in tok_list):
        return True
    for t in tok_list:
        if difflib.get_close_matches(t, allowed, n=1, cutoff=ratio):
            return True
    return False

def is_greeting(tok_list: List[str]) -> bool:
    return any(t in GREETING_WORDS for t in tok_list)

# ---------- Smart LED lookup (covers color-only queries) ----------
COLOR_WORDS = {"blue","green","red","yellow","amber"}
STATE_WORDS = {
    "solid": {"solid","steady","constant"},
    "flashing": {"flash","flashing","blink","blinking","pulsing","pulse"},
}
LED_MEANINGS = {
    ("green","solid"):   "Battery full / ready.",
    ("green","flashing"): "Charging in progress.",
    ("red","solid"):     "Low battery or error.",
    ("red","flashing"):  "Low battery or error.",
    ("blue","solid"):    "Connected to the app (Bluetooth).",
    ("blue","flashing"): "Bluetooth pairing/connection mode.",
    # If you later confirm yellow/amber meanings, add them here.
}

def led_smart_lookup(q: str) -> Optional[str]:
    ql = q.lower()
    if ("led" not in ql) and ("light" not in ql) and ("indicator" not in ql) and ("status" not in ql):
        return None
    color = None
    for c in COLOR_WORDS:
        if c in ql:
            color = c
            break
    if not color:
        return None
    state = None
    for k, words in STATE_WORDS.items():
        if any(w in ql for w in words):
            state = k
            break
    if state:
        meaning = LED_MEANINGS.get((color, state))
        if meaning:
            return meaning
    # Color with no state → present both common meanings when known
    opts = []
    for st in ("solid","flashing"):
        m = LED_MEANINGS.get((color, st))
        if m:
            opts.append(f"{st.capitalize()} {color}: {m}")
    if opts:
        return " / ".join(opts)
    return None

# ---------- Topic hints (fallback if RAG has nothing) ----------
def topic_hint(q: str) -> Optional[str]:
    ql = q.lower()
    # therapy / usage
    if re.search(r"\b(20\s*min|twenty\s*min|use|using|usage|session|how to use)\b", ql):
        return ("Standard use is daytime therapy: 20 minutes per day while awake. "
                "Start/pause with the control unit or the eXciteOSA app. For personalized guidance, follow your clinician’s instructions.")
    # app remote
    if ("app" in ql) or ("remote" in ql) or ("bluetooth" in ql):
        return ("The eXciteOSA app works as a remote: pair via Bluetooth, then start/pause sessions, adjust intensity, and view reminders/progress.")
    # mouthpiece
    if ("mouthpiece" in ql) and (("replace" in ql) or "replacement" in ql or "how long" in ql or "when" in ql)):
        return ("Replace the mouthpiece every 90 days (or sooner if worn/damaged). Inspect regularly.")
    return None

# ---------- API ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)

@app.post("/inquiry")
def inquiry(payload: Inquiry):
    q_raw = (payload.question or "").strip()
    if not q_raw:
        ans = "Please enter a question about the eXciteOSA device or support."
        log.info("Q:<empty> | A:%s", ans)
        return {"answer": ans}

    q = normalize_text(q_raw)
    tks = tokens(q)
    log.info("Q: %s", q)

    # Greeting-only → friendly nudge
    if is_greeting(tks) and not tokens_have_allowed_with_fuzzy(tks, ALLOWED_KEYWORDS):
        ans = ("Hi! I’m the assistant for the eXciteOSA device. "
               "Ask me about daytime therapy (20 minutes while awake), the app as a remote, mouthpiece replacement, "
               "LED indicators, setup, troubleshooting, appointments or support.")
        return {"answer": ans}

    # Scope check (typo tolerant)
    if not tokens_have_allowed_with_fuzzy(tks, ALLOWED_KEYWORDS, ratio=0.76):
        ans = ("I can help with the eXciteOSA device and support only "
               "(therapy timing, app, mouthpiece replacement, LED meanings, setup, troubleshooting, appointments).")
        return {"answer": ans}

    # Fast LED path (works even if RAG misses)
    led = led_smart_lookup(q)
    if led:
        return {"answer": led}

    # RAG (safe if empty)
    try:
        results = retrieve(q, k=4)
        context_block, cites_list = build_context_snippets(results)
    except Exception as e:
        log.error("RAG retrieval failed: %s", e)
        context_block, cites_list = ("", [])

    used_context = bool(context_block.strip())
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
        # If the model basically says “no context,” fall back to hints instead of surfacing that text.
        if ("no relevant context" in answer.lower()) or ("context does not provide" in answer.lower()):
            hint = topic_hint(q)
            if hint:
                return {"answer": hint}
            return {"answer": "I don’t have that detail yet. Please contact official support for specifics."}
        return {"answer": answer, "citations": (cites_list if used_context else [])}
    except Exception as e:
        log.error("OpenAI error: %s", str(e))
        hint = topic_hint(q)
        if hint:
            return {"answer": hint}
        return {"answer": "I’m having trouble right now. Please try again or contact support."}

# ---------- Serve frontend at root ----------
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    log.info("Serving frontend from: %s", FRONTEND_DIR)
else:
    log.warning("Frontend directory not found at %s", FRONTEND_DIR)


