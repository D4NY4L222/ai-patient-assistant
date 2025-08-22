# -*- coding: utf-8 -*-
# backend/rag.py
import os, json, re, uuid
from typing import List, Tuple, Dict
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "store.json")
FAQ_PATH   = os.path.join(os.path.dirname(__file__), "data", "faqs.md")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _clean(text: str) -> str:
    text = text.replace("\u00A0"," ").replace("\r","")
    return re.sub(r"\s+", " ", text).strip()

def _chunk_markdown(md: str, max_chars: int = 900) -> List[str]:
    parts, buf = [], ""
    for line in md.split("\n"):
        if line.strip().startswith("#") and buf:
            parts.append(buf.strip()); buf = ""
        if len(buf) + len(line) + 1 > max_chars:
            parts.append(buf.strip()); buf = ""
        buf += line + "\n"
    if buf.strip(): parts.append(buf.strip())
    return [p for p in parts if p.strip()]

def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ingest_faqs(faq_path: str = FAQ_PATH, store_path: str = STORE_PATH) -> Dict:
    if not os.path.isfile(faq_path):
        raise FileNotFoundError(f"FAQ file not found at {faq_path}")
    with open(faq_path, "r", encoding="utf-8") as f:
        md = f.read()
    chunks = [_clean(c) for c in _chunk_markdown(md)]
    vecs = embed(chunks)
    records = []
    for text, vec in zip(chunks, vecs):
        records.append({"id": str(uuid.uuid4()), "text": text, "embedding": vec})
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL, "records": records}, f)
    return {"count": len(records), "store": store_path}

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def retrieve(query: str, k: int = 4) -> List[Tuple[str, float]]:
    if not os.path.isfile(STORE_PATH):
        return []
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    q_emb = embed([query])[0]
    q = np.array(q_emb, dtype="float32")
    scored = []
    for rec in store["records"]:
        v = np.array(rec["embedding"], dtype="float32")
        scored.append((rec["text"], _cosine(q, v)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

def build_context_snippets(snips: List[Tuple[str,float]]) -> Tuple[str, List[str]]:
    cites = []
    lines = []
    for i, (text, score) in enumerate(snips, start=1):
        tag = f"[{i}]"
        cites.append(tag + " " + text.split("\n")[0][:80].strip())
        lines.append(f"{tag} {text}")
    return "\n\n".join(lines), cites
