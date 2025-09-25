# rag_chat_sqlite.py
import os, sqlite3, json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0  # deterministic

DB_PATH = Path("data/out_data/luni_rag.db")
PROMPT_PATH = Path("prompt/system_prompt.md")

MAX_SYSTEM_CHARS = 9000   # hartes Sicherheitslimit, falls tiktoken nicht installiert ist
#USE_TIKTOKEN = True       # falls installiert, genaueres Token-Budget nutzen
MAX_SYSTEM_TOKENS = 5500  # Zielbudget für Systemprompt (Rest bleibt für User+Snips)


# ---- Retrieval settings ----
MIN_SIM = 0.25
MIN_HITS = 2
TOP_K = 5
INCLUDE_OTHERS = False       # include O→L contexts?
OTHERS_PENALTY = 0.12       # downweight for O-origin snippets
DEDUP_SIM_THRESHOLD = 0.985 # (text-y) near-duplicate removal

# ---- Language & persona settings ----w
EMBED_MODEL = "text-embedding-3-small"  # 1536
GEN_MODEL = "gpt-4o-mini"
MAX_SNIPPETS_CHARS = 900

ANSWER_MODE = "follow-user"  # "follow-user" | "bilingual" | "force-de" | "force-en"
LANG_BOOST = 0.12            # +12% for matching language
LANG_SOFT_PENALTY = 0.06     # -6% for non-matching, still retrievable
MIXED_TOLERANCE = 0.03       # |p(de)-p(en)| < 0.03 => "mixed"
MIN_DE_CONF = 0.55           # minimum to call German
MIN_EN_CONF = 0.55           # minimum to call English

SYSTEM_PERSONA = """Du bist 'Lunibot', ein Chatbot der aus der Social-Media-/Twitch-Persona luniferi entstanden ist. Du antwortest auf Fragen und Nachrichten von Nutzern basierend auf relevanten Snippets aus luniferis Chat- und anderen Kontexten. Deine Antworten sollen informativ, freundlich und im Stil von luniferi sein.
Priorität:
1) Antworte nur mit Aussagen, die aus den REPLY-Snippets (meine eigenen Antworten) ableitbar sind.
2) Nutze CTX nur als Hinweis, formuliere aber NICHT in der Stimme anderer (keine wörtlichen Zitate von Chat/Others).
3) Wenn Snippets unpassend sind, sag ehrlich, dass du es on-stream nicht weißt.
4) Keine privaten Details; alles muss on-stream plausibel sein; kein Romance/Flirt.
Antwortstil: locker, kurz–mittel (5–10 Sätze), freundlich."""

USER_TMPL = """Frage/Nachricht:
{query}

Relevante Snippets (paraphrasieren erlaubt, nichts erfinden):
{snippets}
"""

def parse_blob_to_vec(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    arr = np.fromstring(blob.decode("utf-8"), sep=",", dtype=np.float32)
    return arr

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n if n > 0 else 1.0)

def cosine_sim_matrix(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    # assuming L2-normalized embeddings: cosine == dot
    return X @ q

def _lang_probs(text: str):
    text = (text or "").strip()
    if not text:
        return {}
    try:
        probs = {r.lang: r.prob for r in detect_langs(text)}
        s = sum(probs.values()) or 1.0
        return {k: v/s for k,v in probs.items()}
    except Exception:
        return {}

def detect_language_label(text: str) -> str:
    probs = _lang_probs(text)
    if not probs:
        return "unknown"
    p_de = probs.get("de", 0.0)
    p_en = probs.get("en", 0.0)
    if p_de >= MIN_DE_CONF and p_de > p_en + MIXED_TOLERANCE:
        return "de"
    if p_en >= MIN_EN_CONF and p_en > p_de + MIXED_TOLERANCE:
        return "en"
    if abs(p_de - p_en) < MIXED_TOLERANCE and (p_de > 0.35 or p_en > 0.35):
        return "mixed"
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    if best in ("de","en"):
        return best
    return "unknown"

def decide_output_language(user_text: str, mode: str = "follow-user") -> str:
    if mode == "force-de":
        return "de"
    if mode == "force-en":
        return "en"
    if mode == "bilingual":
        return "mixed"
    lbl = detect_language_label(user_text)
    return lbl if lbl in ("de","en","mixed") else "de"

def infer_snippet_lang(meta_lang: str, text: str) -> str:
    meta_lang = (meta_lang or "").strip().lower()
    if meta_lang in ("de","en","mixed","unknown"):
        return meta_lang
    if meta_lang.startswith("de"):
        return "de"
    if meta_lang.startswith("en"):
        return "en"
    return detect_language_label(text)

def lang_boost_factor(snippet_lang: str, target_lang: str) -> float:
    if snippet_lang not in ("de","en"):
        return 1.0
    if target_lang == "mixed":
        return 1.0 + (LANG_BOOST * 0.5)
    if snippet_lang == target_lang:
        return 1.0 + LANG_BOOST
    else:
        return 1.0 - LANG_SOFT_PENALTY

def load_pairs(conn: sqlite3.Connection) -> Tuple[np.ndarray, list]:
    cur = conn.cursor()
    rows = []
    # chat
    cur.execute("SELECT context_text, reply_text, language, source, gap_lines, gap_seconds, embedding_context FROM pairs_chat")
    rows += [dict(
        context_text=r[0] or "",
        reply_text=r[1] or "",
        language=r[2] or "",
        source=r[3] or "",
        gap_lines=int(r[4] or 0),
        gap_seconds=str(r[5] or ""),
        emb_blob=r[6],
        origin="C",
        kind="context"
    ) for r in cur.fetchall()]

    # others
    if INCLUDE_OTHERS:
        cur.execute("SELECT context_text, reply_text, language, source, gap_lines, gap_seconds, embedding_context FROM pairs_others")
        rows += [dict(
            context_text=r[0] or "",
            reply_text=r[1] or "",
            language=r[2] or "",
            source=r[3] or "",
            gap_lines=int(r[4] or 0),
            gap_seconds=str(r[5] or ""),
            emb_blob=r[6],
            origin="O",
            kind="context"
        ) for r in cur.fetchall()]

    embs, recs = [], []
    for r in rows:
        v = parse_blob_to_vec(r["emb_blob"])
        if v is None or v.size == 0:
            continue
        # embeddings already normalized in build_index.py; normalize again to be safe:
        v = l2_normalize(v)
        embs.append(v)
        recs.append({
            "text": r["context_text"],
            "reply": r["reply_text"],
            "meta": {
                "language": r["language"],
                "source": r["source"],
                "gap_lines": r["gap_lines"],
                "gap_seconds": r["gap_seconds"],
                "kind": r["kind"],
                "origin": r["origin"],
            }
        })

    if not embs:
        raise RuntimeError("Keine Embeddings in der DB gefunden. Bitte erst build_index.py laufen lassen.")
    X = np.vstack(embs).astype(np.float32)
    return X, recs

def embed_query(client: OpenAI, q: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    v = np.array(e, dtype=np.float32)
    return l2_normalize(v)

def rerank(scores: np.ndarray, recs: list, target_lang: str) -> np.ndarray:
    # 1) O-Downweight
    pen_origin = np.array([1.0 - OTHERS_PENALTY if r["meta"]["origin"] == "O" else 1.0 for r in recs], dtype=np.float32)
    scores = scores * pen_origin
    # 2) Language-aware factor
    factors = []
    for r in recs:
        sn_lang = infer_snippet_lang(r["meta"].get("language",""), r["text"])
        factors.append(lang_boost_factor(sn_lang, target_lang))
    scores = scores * np.array(factors, dtype=np.float32)
    return scores

def topk_unique(scores: np.ndarray, recs: list, k: int) -> List[Dict[str,Any]]:
    idx = np.argsort(-scores)
    out = []
    for i in idx:
        if len(out) >= k: break
        rec = recs[i]
        t = rec["text"].strip()
        if not t: continue
        # simple near-dup filter by text prefixes/inclusion
        is_dup = any(
            (len(t2) > 0 and (t[:120] == t2[:120] or t in t2 or t2 in t))
            for t2 in (r["text"] for r in out)
        )
        if is_dup: continue
        out.append({"score": float(scores[i]), **rec})
    return out

def format_snippets(snips: List[Dict[str,Any]], max_chars=MAX_SNIPPETS_CHARS) -> str:
    parts=[]; size=0
    for i,s in enumerate(snips,1):
        piece = f"[{i}] ({s['meta']['kind']}, {s['meta']['origin']}, src={s['meta']['source']}) {s['text']}"
        if size + len(piece) > max_chars: break
        parts.append(piece); size += len(piece)
    return "\n---\n".join(parts)

def call_llm(client: OpenAI, system: str, user: str) -> str:
    r = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=0.7
    )
    return r.choices[0].message.content

def augment_persona_for_language(system_prompt: str, target_lang: str) -> str:
    if target_lang == "de":
        tip = "Antworte bitte auf Deutsch."
    elif target_lang == "en":
        tip = "Please answer in English."
    else:
        tip = ("Antworte überwiegend in der Nutzersprache, "
               "aber du darfst authentisch gelegentlich zwischen Deutsch und Englisch wechseln (kein Satz-Mischmasch).")
    return system_prompt + "\n\n" + tip

def rag_answer(client: OpenAI, X: np.ndarray, recs: list, query: str) -> Tuple[str, List[Dict[str,Any]]]:
    target_lang = decide_output_language(query, mode=ANSWER_MODE)
    qv = embed_query(client, query)
    scores = cosine_sim_matrix(qv, X)
    scores = rerank(scores, recs, target_lang=target_lang)
    snips_all = topk_unique(scores, recs, TOP_K)
    snips = [s for s in snips_all if s["score"] >= MIN_SIM]
    use_snips = (len(snips) >= MIN_HITS)

    if not use_snips:
        system = augment_persona_for_language(SYSTEM_PERSONA, target_lang)
        user = USER_TMPL.format(
            query=query,
            snippets="(Keine verlässlichen Snippets – antworte nur mit deiner öffentlichen Persona, keine privaten Details.)"
        )
        ans = call_llm(client, system, user) 
        return ans, []
    system = augment_persona_for_language(SYSTEM_PERSONA, target_lang)
    user = USER_TMPL.format(query=query, snippets=format_snippets(snips))
    ans = call_llm(client, system, user)
    return ans, snips

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY fehlt")
    client = OpenAI(api_key=api_key)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB nicht gefunden: {DB_PATH}. Bitte build_index.py ausführen.")

    with sqlite3.connect(DB_PATH) as conn:
        X, recs = load_pairs(conn)

    print("RAG bereit. Tippe eine Frage (leer = Ende).")
    while True:
        try:
            q = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ans, sn = rag_answer(client, X, recs, q)
        print("\n--- Antwort ---\n", ans)
        print("\n--- Snippets ---")
        for i,s in enumerate(sn,1):
            print(f"[{i}] score={s['score']:.3f} {s['meta']['origin']} :: {s['text'][:120]}...")

if __name__ == "__main__":
    main()
