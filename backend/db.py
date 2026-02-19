import json
import hashlib
from sqlalchemy import create_engine, text


DB_URL = "postgresql+psycopg2://fda:fda_password@localhost:5432/fda_db"

engine = create_engine(DB_URL, future=True)


def init_db():
    
    create_chunks_sql = """
    CREATE TABLE IF NOT EXISTS label_chunks (
        id SERIAL PRIMARY KEY,
        label_id INT NOT NULL REFERENCES drug_labels(id) ON DELETE CASCADE,
        section TEXT NOT NULL,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        char_start INT,
        char_end INT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(label_id, section, chunk_index)
    );
    """

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS drug_labels (
        id SERIAL PRIMARY KEY,
        drug_query TEXT NOT NULL,
        query_used TEXT,
        brand_name TEXT,
        generic_name TEXT,
        manufacturer TEXT,
        effective_time TEXT,
        set_id TEXT,
        spl_id TEXT,
        sections JSONB NOT NULL,
        raw_result JSONB NOT NULL,
        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.execute(text(create_chunks_sql))
        conn.commit()


def save_label_summary(drug_query: str, query_used: str, citation: dict, sections: dict, raw_result: dict):
    insert_sql = """
    INSERT INTO drug_labels
    (drug_query, query_used, brand_name, generic_name, manufacturer, effective_time, set_id, spl_id, sections, raw_result)
    VALUES
    (:drug_query, :query_used, :brand_name, :generic_name, :manufacturer, :effective_time, :set_id, :spl_id,
    CAST(:sections AS jsonb), CAST(:raw_result AS jsonb));
    """

    payload = {
        "drug_query": drug_query,
        "query_used": query_used,
        "brand_name": citation.get("brand_name", ""),
        "generic_name": citation.get("generic_name", ""),
        "manufacturer": citation.get("manufacturer", ""),
        "effective_time": citation.get("effective_time", ""),
        "set_id": citation.get("set_id", ""),
        "spl_id": citation.get("spl_id", ""),
        "sections": json.dumps(sections),
        "raw_result": json.dumps(raw_result),
    }

    with engine.connect() as conn:
        conn.execute(text(insert_sql), payload)
        conn.commit()
from sqlalchemy import text

def get_recent_labels(limit: int = 10):
    sql = """
    SELECT id, drug_query, brand_name, generic_name, manufacturer, effective_time, fetched_at
    FROM drug_labels
    ORDER BY id DESC
    LIMIT :limit;
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]


def get_label_by_id(label_id: int):
    sql = """
    SELECT id, drug_query, brand_name, generic_name, manufacturer, effective_time, fetched_at, sections
    FROM drug_labels
    WHERE id = :label_id;
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"label_id": label_id}).mappings().first()
        return dict(row) if row else None

def simple_text_chunks(text: str, max_chars: int = 900, overlap: int = 120):
   
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def insert_chunks_for_label(label_id: int, sections: dict):
    if isinstance(sections, str):
        sections = json.loads(sections)

    """
    Takes stored label sections {section: text} and writes chunks into label_chunks.
    """
    insert_sql = """
    INSERT INTO label_chunks
    (label_id, section, chunk_index, content, content_hash, char_start, char_end)
    VALUES
    (:label_id, :section, :chunk_index, :content, :content_hash, :char_start, :char_end)
    ON CONFLICT (label_id, section, chunk_index) DO NOTHING;
    """

    rows_to_insert = []

    for section_name, section_text in (sections or {}).items():
        chunk_tuples = simple_text_chunks(section_text)
        for idx, (chunk_text, cs, ce) in enumerate(chunk_tuples):
            h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            rows_to_insert.append({
                "label_id": label_id,
                "section": section_name,
                "chunk_index": idx,
                "content": chunk_text,
                "content_hash": h,
                "char_start": cs,
                "char_end": ce
            })

    if not rows_to_insert:
        return 0

    with engine.connect() as conn:
        conn.execute(text(insert_sql), rows_to_insert)
        conn.commit()

    return len(rows_to_insert)


def get_latest_label_with_sections():
    sql = """
    SELECT id, sections
    FROM drug_labels
    ORDER BY id DESC
    LIMIT 1;
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
        return dict(row) if row else None


def count_chunks():
    sql = "SELECT COUNT(*) AS n FROM label_chunks;"
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
        return int(row["n"]) if row else 0
    
    
def get_chunk_by_id(chunk_id: int):
    sql = """
    SELECT id, label_id, section, chunk_index, content, char_start, char_end, created_at
    FROM label_chunks
    WHERE id = :chunk_id;
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"chunk_id": chunk_id}).mappings().first()
        return dict(row) if row else None
