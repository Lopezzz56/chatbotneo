from google import genai
from config.config import GEMINI_API_KEY
import time
import streamlit as st

client = genai.Client(api_key=GEMINI_API_KEY)

def get_embeddings(texts: list[str]):
    if not texts:
        return []

    all_embeddings = []
    BATCH_SIZE = 50
    
    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [t for t in texts[i : i + BATCH_SIZE] if t.strip()]
            if not batch: continue

            resp = client.models.embed_content(
                model="gemini-embedding-001", 
                contents=batch,
            )
            
            if resp and resp.embeddings:
                batch_embs = [list(emb.values) for emb in resp.embeddings]
                all_embeddings.extend(batch_embs)
            
            time.sleep(0.5) 
            
        return all_embeddings
    except Exception as e:
        print(f"--- EMBEDDING API ERROR: {e} ---")
        st.error(f"API Error: {e}")
        return []