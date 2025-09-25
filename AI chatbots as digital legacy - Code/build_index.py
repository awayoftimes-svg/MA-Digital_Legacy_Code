# build_index.py
import csv
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"  
def ensure_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS style_data (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        language TEXT,
        source TEXT,
        line INTEGER,
        timestamp TEXT,
        embedding BLOB
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pairs_chat (
        id INTEGER PRIMARY KEY,
        context_text TEXT NOT NULL,
        reply_text   TEXT NOT NULL,
        language TEXT,
        source TEXT,
        gap_lines INTEGER,
        gap_seconds TEXT,
        embedding_context BLOB
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pairs_others (
        id INTEGER PRIMARY KEY,
        context_text TEXT NOT NULL,
        reply_text   TEXT NOT NULL,
        language TEXT,
        source TEXT,
        gap_lines INTEGER,
        gap_seconds TEXT,
        embedding_context BLOB
    );
    """)
    conn.commit()

def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    # OpenAI v1 SDK: batch embeddings
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def to_blob(vec: list[float]) -> bytes:
    # store as comma-joined floats (simple & portable)
    return (",".join(f"{x:.7f}" for x in vec)).encode("utf-8")

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    for col in df.columns:
        if isinstance(df[col].iloc[0] if len(df) else "", str):
            df[col] = df[col].fillna("").astype(str)
    return df

def chunk(lst, n=128):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=api_key)

    out_db = Path("data/out_data/luni_rag.db")
    out_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_db)
    ensure_db(conn)

    # --- STYLE (L) ---
    style_csv = "data/out_data/style_data.csv"
    if Path(style_csv).exists():
        df = load_csv(style_csv)
        # expected cols: text, language, source, line, timestamp
        rows = df.to_dict(orient="records")
        texts = [r.get("text", "") for r in rows if r.get("text","").strip()]
        ids = []
        with conn:
            for batch in chunk([r for r in rows if r.get("text","").strip()], 64):
                embs = embed_texts(client, [r["text"] for r in batch])
                for r, e in zip(batch, embs):
                    conn.execute(
                        "INSERT INTO style_data(text,language,source,line,timestamp,embedding) VALUES (?,?,?,?,?,?)",
                        (r.get("text",""), r.get("language",""), r.get("source",""),
                         int(r.get("line",0) or 0), r.get("timestamp",""), to_blob(e))
                    )

    # --- C→L ---
    chat_csv = "data/out_data/pairs_chat.csv"
    if Path(chat_csv).exists():
        df = load_csv(chat_csv)
        # expected: context_text, reply_text, language, source, gap_lines, gap_seconds
        rows = [r for r in df.to_dict(orient="records") if r.get("context_text","").strip() and r.get("reply_text","").strip()]
        for batch in chunk(rows, 64):
            embs = embed_texts(client, [r["context_text"] for r in batch])
            with conn:
                for r, e in zip(batch, embs):
                    conn.execute(
                        "INSERT INTO pairs_chat(context_text,reply_text,language,source,gap_lines,gap_seconds,embedding_context) VALUES (?,?,?,?,?,?,?)",
                        (r.get("context_text",""), r.get("reply_text",""), r.get("language",""),
                         r.get("source",""), int(r.get("gap_lines",0) or 0), r.get("gap_seconds",""),
                         to_blob(e))
                    )
        curated_csv = "data/out_data/pairs_curated.csv"
    if Path(curated_csv).exists():
        df = load_csv(curated_csv)
        rows = [r for r in df.to_dict(orient="records")
                if r.get("context_text","").strip() and r.get("reply_text","").strip()]
        for batch in chunk(rows, 64):
            embs = embed_texts(client, [r["context_text"] for r in batch])
            with conn:
                for r, e in zip(batch, embs):
                    conn.execute(
                        "INSERT INTO pairs_chat(context_text,reply_text,language,source,gap_lines,gap_seconds,embedding_context) VALUES (?,?,?,?,?,?,?)",
                        (r.get("context_text",""),
                         r.get("reply_text",""),
                         r.get("language",""),
                         r.get("source","curated"),
                         int(r.get("gap_lines",0) or 0),
                         r.get("gap_seconds",""),
                         to_blob(e))
                    )

    # --- O→L (filtered later at query time) ---
    others_csv = "data/out_data/pairs_others.csv"
    if Path(others_csv).exists():
        df = load_csv(others_csv)
        rows = [r for r in df.to_dict(orient="records") if r.get("context_text","").strip() and r.get("reply_text","").strip()]
        for batch in chunk(rows, 64):
            embs = embed_texts(client, [r["context_text"] for r in batch])
            with conn:
                for r, e in zip(batch, embs):
                    conn.execute(
                        "INSERT INTO pairs_others(context_text,reply_text,language,source,gap_lines,gap_seconds,embedding_context) VALUES (?,?,?,?,?,?,?)",
                        (r.get("context_text",""), r.get("reply_text",""), r.get("language",""),
                         r.get("source",""), int(r.get("gap_lines",0) or 0), r.get("gap_seconds",""),
                         to_blob(e))
                    )

    print(f"✅ Indexed into {out_db}")

if __name__ == "__main__":
    main()
