# -*- coding: utf-8 -*-
# backend/ingest.py
from rag import ingest_faqs

if __name__ == "__main__":
    out = ingest_faqs()
    print(f"Ingested {out['count']} chunks into {out['store']}")
