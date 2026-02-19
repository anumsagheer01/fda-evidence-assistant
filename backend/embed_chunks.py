import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg2://fda:fda_password@localhost:5432/fda_db"
)

engine = create_engine(DB_URL, future=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims

def fetch_chunks(limit: int = 200):
    sql = """
    SELECT id, content
    FROM label_chunks
    WHERE embedding IS NULL
    ORDER BY id ASC
    LIMIT :limit;
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]

def update_embedding(chunk_id: int, emb):
    # pgvector accepts string like: '[0.1,0.2,...]'
    emb_str = "[" + ",".join([str(float(x)) for x in emb]) + "]"
    sql = "UPDATE label_chunks SET embedding = :emb WHERE id = :id;"
    with engine.connect() as conn:
        conn.execute(text(sql), {"emb": emb_str, "id": chunk_id})
        conn.commit()

def main():
    model = SentenceTransformer(MODEL_NAME)

    rows = fetch_chunks(limit=300)
    if not rows:
        print("No chunks without embeddings.")
        return

    texts = [r["content"] for r in rows]
    ids = [r["id"] for r in rows]

    embeddings = model.encode(texts, normalize_embeddings=True)

    for chunk_id, emb in zip(ids, embeddings):
        update_embedding(chunk_id, emb)

    print(f"Embedded and stored {len(ids)} chunks")

if __name__ == "__main__":
    main()
