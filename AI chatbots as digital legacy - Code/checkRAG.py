import sqlite3, json
con = sqlite3.connect("data/out_data/luni_rag.db")
cur = con.cursor()
for t in ("pairs_chat","pairs_others", "style_data"):
    try:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        print(t, "→", cols)
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print("rows:", cur.fetchone()[0])
    except Exception as e:
        print(t, "ERROR", e)
con.close()