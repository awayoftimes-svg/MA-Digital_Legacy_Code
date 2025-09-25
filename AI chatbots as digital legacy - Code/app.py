# app.py
import os, re, sqlite3, json, random
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# --- Logging for the study ---
from datetime import datetime
import csv

LOG_PATH = Path("data/study_logs/log.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_turn(session_id: str | None,
             user_text: str,
             reply_text: str,
             sources: list[str] | None,
             used_snips: bool,
             extra: dict | None = None) -> None:
    """Append one chat turn to a CSV (semicolon; Windows-safe)."""
    try:
        is_new = not LOG_PATH.exists()
        with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            if is_new:
                w.writerow(["ts","session_id","user","reply","sources","used_snippets","extra_json"])
            w.writerow([
                datetime.utcnow().isoformat(),
                session_id or "",
                (user_text or "").replace("\n", " ")[:4000],
                (reply_text or "").replace("\n", " ")[:4000],
                ",".join(sources or []),
                int(bool(used_snips)),
                (extra and str(extra)) or ""
            ])
    except Exception as e:
        print("[LOG] failed:", e)

# ── ENV ──────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY fehlt. Trage ihn in .env ein.")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Pfade / Modelle ─────────────────────────────────────────────────────────
DB_PATH = Path("data/out_data/luni_rag.db")
PROMPT_PATH = Path("prompt/system_prompt.md")
OVERRIDES_PATH = Path("prompt/admin_overrides.md")

EMBED_MODEL = "text-embedding-3-small"   # 1536-dim, multilingual
DEFAULT_MODEL = "gpt-4.1"                # Model can be freely chosen

# ── RAG Settings ────────────────────────────────────────────────────────────
TOP_K = 8
INCLUDE_OTHERS = True
OTHERS_PENALTY = 0.40
LANG_BOOST = 0.0               
LANG_SOFT_PENALTY = 0.0        
MAX_SNIPPETS_CHARS = 1000
MIN_SIM = 0.18
MIN_HITS = 1

# --- Style Seeds / Profanity ---
STYLE_SEEDS_FROM_DB = True
STYLE_SEEDS_LIMIT = 8
STYLE_SEEDS_MAX_CHARS = 140
STYLE_SEEDS_LANG_BIAS = False   
STYLE_SEEDS_PROFANITY_ONLY = True
SWEAR_ENABLED = True
STYLE_SEEDS_PRIORITY_KEYWORDS = ["fucking"]
STYLE_SEEDS_MANUAL_FALLBACK = [
    "ja das ist fucking schwer.",
    "das ist fucking cool.",
    "ehh.. fucking wild.",
    "ok, das ist halt fucking weird.",
]

# Logging of sources in the final reply
ANNOTATE_SOURCES = False
LOG_RAG_TOPK = False

# --- Typo Jabs ---
TYPO_JABS_ENABLED = True
TYPO_JABS_PROB = 0.85 #this is too high and should be toned down to 0.3 or 0.4 in production
TYPO_JABS_MAX_CHARS = 120
SERIOUS_KEYWORDS = [
    "suizid","suicide","selbstmord","hilfe","depression","krise",
    "therapie","trauma","verzweifelt","akut","notfall"
]

# --- Output length / bubbles ---
MAX_TOKENS_PER_REPLY = 350   
MAX_TOTAL_CHARS = 900        # Max lengtghh of full reply
MAX_BUBBLES = 3              # max. Messages
MAX_BUBBLE_CHARS = 320       # max. length per bubble

# ── Tokenbudget for System-Prompt ───────────────────────────────────────────
USE_TIKTOKEN = True
MAX_SYSTEM_TOKENS = 5500
MAX_SYSTEM_CHARS = 9000

# ── Converation memory─────────────────────────────────────────
CONV_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_TURNS = 20

def trim_history(messages: List[Dict[str, str]], max_turns: int = MAX_TURNS) -> List[Dict[str, str]]:
    sys_msgs = [m for m in messages if m["role"] == "system"]
    dialog = [m for m in messages if m["role"] != "system"]
    keep = dialog[-2 * max_turns :]
    return sys_msgs + keep

# ── Request/Response Schemas ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    temperature: float | None = 0.75
    lang: str | None = None          
    session_id: Optional[str] = None
    bubbles_by: Optional[str] = "\n\n"

class TeachNote(BaseModel):
    note: str

class ResetBody(BaseModel):
    session_id: str

# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="LuniBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ── UI Mount ────────────────────────────────────────────────────────────────
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    index_file = Path("public") / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>UI fehlt</h1><p>Lege <code>public/index.html</code> an.</p>", status_code=200)
    return index_file.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root_html():
    index_file = Path("public") / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return HTMLResponse("<h1>UI fehlt</h1><p>Lege <code>public/index.html</code> an.</p>", status_code=200)

# ── Utils ────────────────────────────────────────
def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / (n if n > 0 else 1.0)

def cosine_sim_matrix(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ q  # Cosine = Dot (L2-normalisiert)

def lang_boost_factor(snippet_lang: str, target_lang: str) -> float:
    # not used for Auto-language but i kept it in case of later use
    ml = (snippet_lang or "").strip().lower()
    if ml.startswith("de"):
        sn = "de"
    elif ml.startswith("en"):
        sn = "en"
    else:
        return 1.0
    if target_lang == "de":
        return 1.0 + LANG_BOOST if sn == "de" else 1.0 - LANG_SOFT_PENALTY
    if target_lang == "en":
        return 1.0 + LANG_BOOST if sn == "en" else 1.0 - LANG_SOFT_PENALTY
    return 1.0

def get_admin_overrides(max_chars: int = 1200) -> str:
    """Liest prompt/admin_overrides.md, kürzt hart und säubert Leerzeilen."""
    try:
        if OVERRIDES_PATH.exists():
            txt = OVERRIDES_PATH.read_text(encoding="utf-8").strip()
            txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
            return txt[:max_chars]
    except Exception as e:
        print("WARN admin_overrides:", e)
    return ""

def load_style_lines(*,
                     target_lang: str | None,
                     limit: int = STYLE_SEEDS_LIMIT,
                     profanity_only: bool = STYLE_SEEDS_PROFANITY_ONLY) -> list[str]:
    """
    Holt kurze Beispielzeilen (L-Stil) aus style_data.
    - priorisiert STYLE_SEEDS_PRIORITY_KEYWORDS (z. B. 'fucking')
    - optional: nur Zeilen mit milden Flüchen
    - keine Sprach-Bias (Auto-Sprache)
    """
    if limit <= 0:
        return []

    patt_profanity = re.compile(
        r"\b("
        r"fuck|fucking|wtf|shit|damn|dammit|holy\s+shit|bitch|asshole|"
        r"schei[ßss]e?|verdammt|kacke|arsch|fick(en)?"
        r")\b", flags=re.I
    )

    seeds: list[str] = []
    seen = set()

    def push_text(t: str):
        nonlocal seeds, seen
        t = (t or "").strip()
        if not t or "http://" in t or "https://" in t:
            return
        if profanity_only and not patt_profanity.search(t):
            return
        t = re.sub(r"\s+", " ", t)[:STYLE_SEEDS_MAX_CHARS].strip()
        key = t.lower()
        if not t or key in seen:
            return
        seen.add(key)
        seeds.append(t)

    rows_priority: list[tuple[str,str]] = []
    rows_random: list[tuple[str,str]] = []

    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            cur = con.cursor()
        
            for kw in STYLE_SEEDS_PRIORITY_KEYWORDS:
                cur.execute("""
                    SELECT text, COALESCE(language,'') FROM style_data
                    WHERE text!='' AND lower(text) LIKE ?
                    ORDER BY RANDOM() LIMIT ?
                """, (f"%{kw.lower()}%", max(limit*3, 30)))
                rows_priority.extend(cur.fetchall())
        
            cur.execute("""
                SELECT text, COALESCE(language,'') FROM style_data
                WHERE text!='' ORDER BY RANDOM() LIMIT ?
            """, (max(limit*6, 60),))
            rows_random = cur.fetchall()

    for t, _lang in rows_priority:
        push_text(t)
        if len(seeds) >= limit: break

    if len(seeds) < limit:
        for t, _lang in rows_random:
            push_text(t)
            if len(seeds) >= limit: break

    if len(seeds) < limit and STYLE_SEEDS_MANUAL_FALLBACK:
        for t in STYLE_SEEDS_MANUAL_FALLBACK:
            push_text(t)
            if len(seeds) >= limit: break

    return seeds[:limit]

def has_serious_topic(t: str) -> bool:
    s = (t or "").lower()
    return any(k in s for k in SERIOUS_KEYWORDS)

def generate_typo_jab_via_model(user_text: str) -> str:
    """
    Liefert EINE kurze, spielerische Jab-Zeile bei offensichtlichen Typos.
    Gibt "" zurück, wenn keine Typos/ernstes Thema/Fehler.
    """
    if not user_text or has_serious_topic(user_text):
        return ""

    try:
        prompt_sys = (
            "You are a style helper for a streamer persona. "
            "If the user's message contains OBVIOUS typos/spelling mistakes, "
            "produce ONE playful PG-13 jab in THE USER'S LANGUAGE (German or English). "
            f"Constraints: friendly-snarky, no slurs, no protected-class insults, <= {TYPO_JABS_MAX_CHARS} chars, no emojis. "
            "If there are no obvious typos or the topic is serious, output an empty string.\n"
            'Return strictly JSON: {"jab": "<line or empty>"}'
        )

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": user_text}
            ]
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        jab = (data.get("jab") or "").strip()
        if jab and len(jab) > TYPO_JABS_MAX_CHARS:
            jab = jab[:TYPO_JABS_MAX_CHARS-1] + "…"
        return jab
    except Exception:
        return ""

# ── Prompt: Load Markdown ─────────────────────────────────────
def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")
    except FileNotFoundError:
        return ""

def strip_md_comments_and_noise(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"^---.*?---\s*", "", md, flags=re.S)
    md = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

def split_md_sections(md: str):
    parts = re.split(r"(?m)^(#{1,2})\s+(.*)$", md)
    sections = []
    preface = parts[0].strip()
    if preface:
        sections.append(("preface", preface))
    for i in range(1, len(parts), 3):
        title = (parts[i+1] or "").strip()
        content = (parts[i+2] or "").strip()
        sections.append((title, content))
    return sections

def reorder_and_limit_sections(sections, max_chars=MAX_SYSTEM_CHARS):
    out, size = [], 0
    for title, content in sections:
        content = re.sub(
            r"```.+?```",
            lambda m: (m.group(0)[:800] + "\n```…cut…```\n") if len(m.group(0)) > 1000 else m.group(0),
            content, flags=re.S
        )
        block = f"## {title}\n{content}".strip()
        if size + len(block) > max_chars:
            remain = max_chars - size
            if remain > 200:
                out.append(block[:remain] + "\n…")
            break
        out.append(block)
        size += len(block)
    return "\n\n".join(out)

def maybe_token_limit(text: str, max_tokens=MAX_SYSTEM_TOKENS):
    if not USE_TIKTOKEN:
        return text[:MAX_SYSTEM_CHARS] if len(text) > MAX_SYSTEM_CHARS else text
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(DEFAULT_MODEL)
    except Exception:
        return text[:MAX_SYSTEM_CHARS] if len(text) > MAX_SYSTEM_CHARS else text
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])

def build_system_prompt(target_lang: str, rag_mode: str = "hybrid") -> str:
    base = read_text_safe(PROMPT_PATH)
    if not base.strip():
        base = "Du bist 'LuniBot', ein Clon der social media persona Luniferi. Antworte knapp, direkt, öffentlich, ohne Privates. Du bist NICHT allwissend, und antwortest auf komplexere Fragen mit 'puh, keine Ahnung'."
    base = strip_md_comments_and_noise(base)
    sections = split_md_sections(base)
    base_compact = reorder_and_limit_sections(sections, max_chars=MAX_SYSTEM_CHARS)

    # Auto-Sprache → neutral formulieren
    out = base_compact.replace("<<LANG>>", "the user's language").replace("<<RAG_MODE>>", rag_mode)

    lang_tip = "Mirror the user's language (German or English)."
    reply_first_tip = (
        "WICHTIG: Formuliere Inhalte ausschließlich aus den REPLY-Snippets (eigene Aussagen). "
        "CTX-Snippets sind nur Hintergrund; übernimm nicht deren Wortlaut/Stimme. "
        "(alle 78 Karten, upright/reversed, kurze Bedeutungen in Nutzersprache, keine [#]-Zitate)."
    )

    if SWEAR_ENABLED:
        out += (
            "\n\n## Profanity & Tone (runtime)\n"
            "- Profanity is allowed if it matches my on-stream voice. Keep it playful, not mean.\n"
            "- No hateful slurs or protected-class attacks. PG-13 vibe; 1–2 profane words max unless a snippet shows more.\n"
        )

    if STYLE_SEEDS_FROM_DB:
        seeds = load_style_lines(target_lang=None, limit=STYLE_SEEDS_LIMIT)
        if seeds:
            seed_block = "\n".join(f"- {s}" for s in seeds)
            out += (
                "\n\n## style seeds (runtime, own on-stream lines)\n"
                "Use these as tone cues; do not quote them verbatim. Keep my voice consistent.\n"
                + seed_block
            )

    final = out.strip() + "\n\n" + lang_tip + "\n" + reply_first_tip
    final = maybe_token_limit(final, max_tokens=MAX_SYSTEM_TOKENS)
    return final

# ── DB / Load Embeddings  ───────────────────────────────────────────────────
def parse_blob_to_vec(blob: bytes) -> np.ndarray:
    if not blob:
        return None
    return np.fromstring(blob.decode("utf-8"), sep=",", dtype=np.float32)

def load_pairs(conn: sqlite3.Connection) -> tuple[np.ndarray, list[dict]]:
    cur = conn.cursor()
    rows = []
    cur.execute("SELECT context_text, reply_text, language, source, gap_lines, gap_seconds, embedding_context FROM pairs_chat")
    rows += [dict(
        context_text=r[0] or "", reply_text=r[1] or "", language=r[2] or "", source=r[3] or "",
        gap_lines=int(r[4] or 0), gap_seconds=str(r[5] or ""), emb_blob=r[6], origin="C"
    ) for r in cur.fetchall()]
    if INCLUDE_OTHERS:
        cur.execute("SELECT context_text, reply_text, language, source, gap_lines, gap_seconds, embedding_context FROM pairs_others")
        rows += [dict(
            context_text=r[0] or "", reply_text=r[1] or "", language=r[2] or "", source=r[3] or "",
            gap_lines=int(r[4] or 0), gap_seconds=str(r[5] or ""), emb_blob=r[6], origin="O"
        ) for r in cur.fetchall()]
    embs, recs = [], []
    for r in rows:
        v = parse_blob_to_vec(r["emb_blob"])
        if v is None or v.size == 0: continue
        v = l2_normalize(v)
        embs.append(v)
        recs.append({
            "text": r["context_text"],
            "reply": r["reply_text"],
            "meta": {
                "language": r["language"], "source": r["source"],
                "gap_lines": r["gap_lines"], "gap_seconds": r["gap_seconds"],
                "kind": "context", "origin": r["origin"],
            }
        })
    if not embs:
        raise RuntimeError("Keine Embeddings in der DB gefunden. Bitte erst build_index.py laufen lassen.")
    X = np.vstack(embs).astype(np.float32)
    return X, recs

# global laden beim Start
if not DB_PATH.exists():
    print(f"WARN: DB nicht gefunden unter {DB_PATH}. /chat wird ohne RAG arbeiten.")
    X_EMB, RECS = None, []
else:
    with sqlite3.connect(DB_PATH) as _conn:
        X_EMB, RECS = load_pairs(_conn)
    print(f"RAG geladen: {len(RECS)} Snippets")

# ── RAG Main function ──────────────────────────────────────────────────────
def embed_query(q: str) -> np.ndarray:
    e = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    return l2_normalize(np.array(e, dtype=np.float32))

def rerank(scores: np.ndarray, recs: list[dict], target_lang: str) -> np.ndarray:
    # OTHERS-Penalty remains
    pen_origin = np.array([1.0 - OTHERS_PENALTY if r["meta"]["origin"] == "O" else 1.0 for r in recs], dtype=np.float32)
    scores = scores * pen_origin
  
    if target_lang == "auto":
        return scores
    factors = [lang_boost_factor(r["meta"].get("language",""), target_lang) for r in recs]
    return scores * np.array(factors, dtype=np.float32)

def topk_unique(scores: np.ndarray, recs: list[dict], k: int) -> List[Dict[str, Any]]:
    idx = np.argsort(-scores)
    out = []
    for i in idx:
        if len(out) >= k: break
        rec = recs[i]
        t = (rec["text"] or "").strip()
        if not t: continue
        is_dup = any((len(t2) > 0 and (t[:120]==t2[:120] or t in t2 or t2 in t)) for t2 in (r["text"] for r in out))
        if is_dup: continue
        out.append({"score": float(scores[i]), **rec})
    return out

def format_snippets_reply_first(snips: List[Dict[str, Any]], max_chars=MAX_SNIPPETS_CHARS) -> str:
    parts, size = [], 0
    for i, s in enumerate(snips, 1):
        reply = (s.get("reply","") or "").strip()
        ctx = (s.get("text","") or "").strip()
        if len(ctx) > 200: ctx = ctx[:200] + " ..."
        piece = (f"[{i}] (origin={s['meta']['origin']}, src={s['meta']['source']})\n"
                 f"REPLY: {reply}\nCTX: {ctx}")
        if size + len(piece) > max_chars: break
        parts.append(piece); size += len(piece)
    return "\n---\n".join(parts)

# ── Building messages using the RAG ────────────────────────────────────────────────
def build_messages_with_rag(user_text: str, target_lang: str, session_id: Optional[str], temperature: float) -> tuple[list[dict], dict]:
    snips, snippets_text = [], "(Keine verlässlichen Snippets – antworte nur mit deiner öffentlichen Persona.)"
    rag_mode = "prompt_first"
    if X_EMB is not None and len(RECS) > 0:
        qv = embed_query(user_text)
        scores = cosine_sim_matrix(qv, X_EMB)
        scores = rerank(scores, RECS, target_lang=target_lang)
        sn_all = topk_unique(scores, RECS, TOP_K)
        if LOG_RAG_TOPK:
            dbg = ", ".join([f"{s['score']:.2f}:{s['meta']['origin']}:{s['meta']['source']}" for s in sn_all[:5]])
            print(f"[RAG] query='{user_text[:60]}...'  top={dbg}")
        snips = [s for s in sn_all if s["score"] >= MIN_SIM]
        use_snips = (len(snips) >= MIN_HITS)
        if use_snips:
            snippets_text = format_snippets_reply_first(snips, max_chars=MAX_SNIPPETS_CHARS)
            rag_mode = "hybrid"

    system_prompt = build_system_prompt(target_lang=target_lang, rag_mode=rag_mode)

    # Admin-Overrides get highest priority
    overrides = get_admin_overrides()
    messages: List[Dict[str, str]] = []
    if overrides:
        messages.append({
            "role": "system",
            "content": (
                "ADMIN LIVE NOTES — HIGHEST PRIORITY.\n"
                "These bullets override all other rules in this chat:\n"
                f"{overrides}"
            )
        })

    # Normal system prompt message
    messages.append({"role": "system", "content": system_prompt})

    # existing chat history
    if session_id and session_id in CONV_HISTORY:
        messages.extend(CONV_HISTORY[session_id])

    # User-Payload 
    user_payload = f"Frage/Nachricht:\n{user_text}\n\nSnippets:\n{snippets_text}"
    messages.append({"role": "user", "content": user_payload})

    return trim_history(messages), {
        "snippets_used": bool(snips),
        "sources": sorted({s["meta"].get("source","?") for s in snips}) if snips else [],
        "snippets": snips
    }

# ── Bubble-Split & limit ───────────────────────────────────────────────
def split_bubbles(text: str, sep: Optional[str]) -> List[str]:
    if not sep:
        return [text]
    parts = [p.strip() for p in text.split(sep)]
    return [p for p in parts if p]

def limit_bubbles_and_text(reply: str, sep: Optional[str]) -> List[str]:
    """Begrenzt Gesamttext, splittet in Bubbles, kappt Anzahl und Länge je Bubble."""
    if not reply:
        return [""]
    r = reply.strip()
    if len(r) > MAX_TOTAL_CHARS:
        r = r[:MAX_TOTAL_CHARS - 1] + "…"
    r = re.sub(r"\n{3,}", "\n\n", r)
    bubbles = split_bubbles(r, sep)
    bubbles = bubbles[:MAX_BUBBLES]
    trimmed = []
    for b in bubbles:
        b = b.strip()
        if len(b) > MAX_BUBBLE_CHARS:
            b = b[:MAX_BUBBLE_CHARS - 1] + "…"
        trimmed.append(b)
    return trimmed

# ── /chat (with RAG) ─────────────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    user_msg = (req.message or "").strip()
    if not user_msg:
        return {
            "reply": "(leere Nachricht)",
            "bubbles": ["(leere Nachricht)"],
            "session_id": req.session_id,
            "used_snippets": False,
            "sources": []
        }

    try:

        target_lang = "auto"

        # build RAG + Prompt
        messages, rag_meta = build_messages_with_rag(
            user_text=user_msg,
            target_lang=target_lang,
            session_id=req.session_id,
            temperature=req.temperature or 0.7
        )

        # Main reply with limited tokens
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=req.temperature or 0.7,
            messages=messages,
            max_tokens=MAX_TOKENS_PER_REPLY,
        )
        reply = resp.choices[0].message.content or ""

        # Typo-Jab as first line (optional)
        if TYPO_JABS_ENABLED and (random.random() < TYPO_JABS_PROB):
            jab = generate_typo_jab_via_model(user_msg)
            if jab:
                reply = f"{jab}\n\n{reply}"

        # (optional) Add source annotation
        if ANNOTATE_SOURCES and rag_meta.get("snippets_used"):
            srcs = ", ".join(rag_meta.get("sources") or []) or "n/a"
            reply = f"{reply}\n\n_(Kontext genutzt aus: {srcs})_"

        # >>> Limit bubbles & text
        bubbles = limit_bubbles_and_text(reply, req.bubbles_by)
        reply = "\n\n".join(bubbles) 

        # Logging (best effort)
        try:
            log_turn(
                req.session_id,
                user_msg,
                reply,
                rag_meta.get("sources"),
                rag_meta.get("snippets_used"),
                extra={}
            )
        except Exception:
            pass

        # Memory aktualisieren
        if req.session_id:
            hist = CONV_HISTORY.get(req.session_id, [])
            hist.append({"role": "user", "content": user_msg})
            hist.append({"role": "assistant", "content": reply})
            CONV_HISTORY[req.session_id] = trim_history(hist)

        return {
            "reply": reply,
            "bubbles": bubbles,
            "session_id": req.session_id,
            "used_snippets": rag_meta.get("snippets_used", False),
            "sources": rag_meta.get("sources", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR /chat:", repr(e))
        # Niemals null an die UI senden
        return {
            "reply": "[Serverfehler]",
            "bubbles": ["[Serverfehler]"],
            "session_id": req.session_id,
            "used_snippets": False,
            "sources": []
        }

# ── /chat-stream (deactivated) ──────────────────────────────────────────────
@app.post("/chat-stream")
def chat_stream(req: ChatRequest):
    raise HTTPException(status_code=501, detail="chat-stream derzeit deaktiviert (RAG). Nutze /chat.")

# ── Admin: Teach ────────────────────────────────────────────────────────────
@app.post("/admin/teach")
def admin_teach(body: TeachNote):
    note = (body.note or "").strip()
    if not note:
        raise HTTPException(status_code=400, detail="note ist leer")
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OVERRIDES_PATH.open("a", encoding="utf-8") as f:
        f.write("\n- " + note + "\n")
    return {"ok": True, "added": note}

# ── Admin: Prompt/Overrides Preview ─────────────────────────────────────────
@app.get("/admin/system")
def admin_system():
    base = build_system_prompt(target_lang="auto", rag_mode="hybrid")
    overrides = get_admin_overrides()
    return {
        "overrides_first_system_msg": bool(overrides),
        "overrides_preview": overrides[:4000] if overrides else "",
        "system_prompt_preview": base[:4000]
    }

# ── Admin: Session reset ────────────────────────────────────────────────────
@app.post("/admin/reset-session")
def admin_reset_session(body: ResetBody):
    sid = body.session_id
    if sid in CONV_HISTORY:
        del CONV_HISTORY[sid]
    return {"ok": True, "session_cleared": sid}

# ── How to start───────────────────────────────────────────────────────────
# Starten with:
# uvicorn app:app --host 127.0.0.1 --port 8000
