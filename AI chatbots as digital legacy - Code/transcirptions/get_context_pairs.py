# build_pairs_from_labeled_csv.py
import pandas as pd
from pathlib import Path
from datetime import timedelta

# ==== config ====
INPUT_DIR = Path("transcirptions\labeled")  
OUT_DIR = Path("data/out_data")
CONTEXT_JOIN = "\n"                      # joiner for multi-line context
MAX_CHARS_PER_CONTEXT = 2000             # cap context length

# conservative O->L filters; seconds will be auto-disabled if timestamps unusable
O_MAX_GAP_LINES = 1
O_MAX_GAP_SECONDS = 45  # will be ignored if timestamps are all "00:00:00" / empty

# =================

def parse_hms(s: str) -> timedelta | None:
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    # treat "00:00:00" as unusable (i.e., None)
    if s == "00:00:00":
        return None
    try:
        hh, mm, ss = s.split(":")
        return timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss))
    except Exception:
        return None

def td_to_seconds(td: timedelta | None) -> int | str:
    if td is None:
        return ""
    return int(td.total_seconds())

def timestamps_usable(df: pd.DataFrame) -> bool:
    """Return True if 'timestamp' exists and at least one value is a valid, non-00:00:00 time."""
    if "timestamp" not in df.columns:
        return False
    series = df["timestamp"].astype(str).fillna("")
    for val in series:
        if parse_hms(val) is not None:
            return True
    return False

def load_all_csvs(input_dir: Path) -> pd.DataFrame:
    """Liest alle *.csv aus input_dir, normalisiert Spalten & hängt 'source' = Dateiname an."""
    paths = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"Keine CSVs in {input_dir} gefunden.")
    frames = []
    for p in paths:
        # robustes Einlesen; wenn exotisches Encoding, ersetze unlesbare Zeichen
        df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
        # minimale Spalten sicherstellen
        for col in ["timestamp","speaker","text","source","language"]:
            if col not in df.columns:
                df[col] = ""
        df["source"] = df["source"].replace("", p.name)
        # Typen/NA aufräumen
        for col in ["timestamp","speaker","text","source","language"]:
            df[col] = df[col].fillna("").astype(str)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def collapse_blocks(df: pd.DataFrame, target_context: str, use_ts: bool) -> list[dict]:
    """
    Build (context_block -> next L reply) for target_context in {"C","O"}.
    Collapses consecutive target_context lines until next L and pairs them.
    If use_ts==False, gap_seconds will be "" and not meaningful.
    """
    rows = df.to_dict(orient="records")
    out = []
    buf = []             # accumulating C or O block
    buf_langs = []       # languages in block
    buf_source = None
    buf_last_ts = None   # last ts in the block

    def flush_when_reply(l_row: dict):
        nonlocal buf, buf_langs, buf_source, buf_last_ts
        if not buf:
            return
        ctx = CONTEXT_JOIN.join([r["text"] for r in buf])[:MAX_CHARS_PER_CONTEXT]
        lang = (pd.Series(buf_langs).mode().iloc[0] if buf_langs else l_row.get("language",""))
        source = buf_source or l_row.get("source","")

        # gaps
        gap_lines = l_row["__idx__"] - buf[-1]["__idx__"]
        gap_seconds = ""
        if use_ts:
            ts_last_ctx = parse_hms(buf_last_ts)
            ts_reply = parse_hms(l_row.get("timestamp",""))
            if ts_last_ctx is not None and ts_reply is not None:
                gap_seconds = td_to_seconds(ts_reply - ts_last_ctx)

        out.append({
            "context_text": ctx,
            "reply_text": l_row.get("text",""),
            "language": lang,
            "source": source,
            "gap_lines": gap_lines,
            "gap_seconds": gap_seconds
        })
        # reset buffer
        buf, buf_langs, buf_source, buf_last_ts = [], [], None, None

    for i, r in enumerate(rows):
        r["__idx__"] = i
        spk = (r.get("speaker","") or "").strip().upper()
        if spk == target_context:
            buf.append(r)
            buf_langs.append(r.get("language",""))
            buf_source = r.get("source", buf_source)
            buf_last_ts = r.get("timestamp", buf_last_ts)
        elif spk == "L":
            flush_when_reply(r)
        else:
            continue

    return out  # trailing context without reply is dropped

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # alle CSVs laden
    df = load_all_csvs(INPUT_DIR)

    use_ts = timestamps_usable(df)
    if not use_ts:
        print("ℹTimestamps appear unusable (all 00:00:00 or empty). gap_seconds will be skipped.")

    # --- STYLE: all L lines
    style = df[df["speaker"].str.upper() == "L"][["text","language","source","timestamp"]].copy()
    style["line"] = style.index
    style.to_csv(OUT_DIR / "style_data.csv", index=False)

    # --- C→L pairs
    pairs_c = collapse_blocks(df, target_context="C", use_ts=use_ts)
    pairs_chat = pd.DataFrame(pairs_c)

    # --- O→L pairs
    pairs_o = collapse_blocks(df, target_context="O", use_ts=use_ts)
    pairs_others = pd.DataFrame(pairs_o)

    # --- filters for O→L (seconds filter auto-disabled if timestamps unusable)
    if not pairs_others.empty:
        m = pd.Series([True] * len(pairs_others))
        if O_MAX_GAP_LINES is not None:
            m &= (pairs_others["gap_lines"].astype("int", errors="ignore") <= O_MAX_GAP_LINES)

        if use_ts and O_MAX_GAP_SECONDS is not None:
            secs = pairs_others["gap_seconds"].astype(str)
            known_mask = secs != ""
            keep_known = pd.to_numeric(secs.where(known_mask), errors="coerce") <= O_MAX_GAP_SECONDS
            m &= (~known_mask) | (keep_known.fillna(True))

        pairs_others = pairs_others[m]

    # --- save individual files
    (OUT_DIR / "style_data.csv").write_text(style.to_csv(index=False), encoding="utf-8")
    if not pairs_chat.empty:
        (OUT_DIR / "pairs_chat.csv").write_text(pairs_chat.to_csv(index=False), encoding="utf-8")
    if not pairs_others.empty:
        (OUT_DIR / "pairs_others.csv").write_text(pairs_others.to_csv(index=False), encoding="utf-8")

    print("✅ Wrote:")
    print(" -", OUT_DIR / "style_data.csv")
    if not pairs_chat.empty: print(" -", OUT_DIR / "pairs_chat.csv")
    if not pairs_others.empty: print(" -", OUT_DIR / "pairs_others.csv")

    # --- unified pairs.csv for RAG ---
    def _load_df(p: Path) -> pd.DataFrame:
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame()

    chat_path  = OUT_DIR / "pairs_chat.csv"
    oth_path   = OUT_DIR / "pairs_others.csv"
    df_chat = _load_df(chat_path)
    df_oth  = _load_df(oth_path)

    if not df_chat.empty:
        df_chat["origin"] = "C"
    if not df_oth.empty:
        df_oth["origin"] = "O"

    pairs = pd.concat([df_chat, df_oth], ignore_index=True) if (not df_chat.empty or not df_oth.empty) else pd.DataFrame()

    if not pairs.empty:
        pairs = pairs.rename(columns={"context_text": "context", "reply_text": "reply"})
        for col in ["language","source","gap_lines","gap_seconds","origin"]:
            if col not in pairs.columns:
                pairs[col] = ""
        pairs["context"] = pairs["context"].astype(str).str.strip()
        pairs["reply"]   = pairs["reply"].astype(str).str.strip()
        pairs = pairs[(pairs["context"]!="") & (pairs["reply"]!="")]
        pairs = pairs.drop_duplicates(subset=["context","reply"])

        out_pairs = OUT_DIR / "pairs.csv"
        pairs.to_csv(out_pairs, index=False, encoding="utf-8")
        print(" -", out_pairs)
    else:
        print("ℹ️  Keine Pairs zum Vereinigen gefunden.")

if __name__ == "__main__":
    main()
