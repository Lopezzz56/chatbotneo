import sqlite3
import json
import numpy as np
from config.config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      doc_id TEXT,
      chunk_text TEXT,
      page INTEGER,
      char_start INTEGER,
      char_end INTEGER,
      embedding_json TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_chunks(doc_id: str, chunks: list):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for c in chunks:
        cur.execute("""
          INSERT INTO chunks (doc_id, chunk_text, page, char_start, char_end, embedding_json)
          VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, c["text"], c["page"], c["char_start"], c["char_end"], json.dumps(c["emb"])))
    conn.commit()
    conn.close()

def semantic_search(doc_id: str, query_emb: list, top_k: int = 3):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT chunk_text, embedding_json, page FROM chunks WHERE doc_id=?", (doc_id,))
    rows = cur.fetchall()
    conn.close()
    
    scored = []
    q = np.array(query_emb)
    for text, emb_json, page in rows:
        emb = np.array(json.loads(emb_json))
        score = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb))
        scored.append({"text": text, "page": page, "score": score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]