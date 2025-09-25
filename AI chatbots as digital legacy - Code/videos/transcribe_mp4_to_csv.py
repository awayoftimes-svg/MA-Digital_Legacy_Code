import os, json, csv, subprocess, sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ===== Settings =====
CHUNK_SECONDS = 600               # 10 Minuten
SAMPLE_RATE = 16000               # 16 kHz
MODEL = "whisper-1"               # liefert Segment-Timestamps (verbose_json)
AUDIO_EXT = "wav"                 # pro Chunk: PCM WAV mono/16k
# ====================

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def run_ffmpeg(cmd: list):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg Fehler:\n{proc.stderr}")

def split_to_chunks(input_mp4: Path, out_dir: Path) -> List[Path]:
    """
    Extrahiert Audio, konvertiert zu mono/16k WAV und splittet in Chunks à CHUNK_SECONDS.
    Gibt die Liste der Chunk-Dateien in zeitlicher Reihenfolge zurück.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # -vn (kein Video), -ac 1 (mono), -ar 16000 (16kHz), -f segment + -segment_time
    pattern = out_dir / f"{input_mp4.stem}_%03d.{AUDIO_EXT}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_mp4),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
        "-reset_timestamps", "1",
        str(pattern)
    ]
    run_ffmpeg(cmd)
    chunks = sorted(out_dir.glob(f"{input_mp4.stem}_*.{AUDIO_EXT}"))
    if not chunks:
        raise RuntimeError("Keine Chunk-Dateien erzeugt.")
    return chunks

def human_ts(seconds: float) -> str:
    if seconds is None:
        return ""
    s = max(0.0, float(seconds))
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def transcribe_chunk(client: OpenAI, wav_path: Path) -> Dict[str, Any]:
    with wav_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model=MODEL,
            file=f,
            response_format="verbose_json",  # wichtig für Segmente+Timestamps
        )
    return json.loads(resp.model_dump_json())

def process_video(input_mp4: Path, out_csv: Path, work_dir: Optional[Path] = None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY fehlt (in .env setzen).")
    client = OpenAI(api_key=api_key)

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg nicht gefunden. Bitte ffmpeg installieren und in PATH aufnehmen.")

    # 1) Splitten
    tmp_dir = work_dir or (input_mp4.parent / "_chunks")
    print(f"→ Splitte in Chunks ({CHUNK_SECONDS}s): {input_mp4.name}")
    chunks = split_to_chunks(input_mp4, tmp_dir)

    # 2) Chunks transkribieren und CSV-Zeilen bauen
    all_rows: List[Dict[str, Any]] = []
    line_no = 1
    for idx, chunk in enumerate(chunks):
        chunk_offset = idx * CHUNK_SECONDS
        print(f"  • Transkribiere Chunk {idx+1}/{len(chunks)}: {chunk.name}")
        try:
            data = transcribe_chunk(client, chunk)
        except Exception as e:
            print(f"    [WARN] Chunk {chunk.name} übersprungen: {e}")
            continue

        segments = data.get("segments") or []
        if not segments:
            # Fallback: Ganztext -> in einen Satz packen (ohne feine Timestamps)
            full_text = data.get("text", "").strip()
            if full_text:
                all_rows.append({
                    "timestamp": human_ts(chunk_offset),
                    "speaker": "",
                    "text": full_text,
                    "source": input_mp4.name,
                    "line": line_no,
                })
                line_no += 1
            continue

        for seg in segments:
            start = seg.get("start")  # Sekunden innerhalb des Chunks
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            global_start = chunk_offset + (start or 0.0)
            # Optional: in Sätze weiter splitten – hier lassen wir den Whisper-Segmenttext als Zeile
            all_rows.append({
                "timestamp": human_ts(global_start),
                "speaker": "",              # später L/C/O taggen
                "text": txt,
                "source": input_mp4.name,
                "line": line_no,
            })
            line_no += 1

    if not all_rows:
        raise RuntimeError("Keine Transkript-Zeilen generiert.")

    # 3) Schreiben
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","speaker","text","source","line"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"✅ Fertig: {len(all_rows)} Zeilen → {out_csv}")

if __name__ == "__main__":
    # === HIER DEINE DATEI UND AUSGABE EINTRAGEN ===
    PROJECT = Path(__file__).parent
    INPUT_FILE = PROJECT / "videos" / "GOWstream.mp4"   # <— anpassen
    OUTPUT_CSV = PROJECT / "videos" / "GOWStream"   # <— anpassen
    WORK_DIR   = PROJECT / "videos" / "_chunks"         # temp-Ordner für Chunks

    process_video(INPUT_FILE, OUTPUT_CSV, WORK_DIR)
