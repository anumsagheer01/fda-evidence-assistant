from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings
import httpx
from db import init_db
from db import save_label_summary
from db import get_recent_labels, get_label_by_id
from db import get_latest_label_with_sections, insert_chunks_for_label, count_chunks
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from db import get_chunk_by_id










def pick_first(item, default=""):
  
    if item is None:
        return default
    if isinstance(item, list) and len(item) > 0:
        return item[0]
    if isinstance(item, str):
        return item
    return default


def build_citation(result: dict) -> dict:
    """
    Build a simple citation object from openFDA fields.
    """
    openfda = result.get("openfda", {}) or {}

    brand = pick_first(openfda.get("brand_name"), "")
    generic = pick_first(openfda.get("generic_name"), "")
    manufacturer = pick_first(openfda.get("manufacturer_name"), "")
    effective_time = result.get("effective_time", "")

    # openFDA records sometimes include a "spl_id" / "set_id"
    set_id = result.get("set_id", "")
    spl_id = result.get("spl_id", "")

  
    return {
        "brand_name": brand,
        "generic_name": generic,
        "manufacturer": manufacturer,
        "effective_time": effective_time,
        "set_id": set_id,
        "spl_id": spl_id,
        "source": "openFDA drug/label",
    }


def extract_sections(result: dict) -> dict:
    """
    Extract important sections from an openFDA label record.
    Each field is often a list[str]. 
    """
    return {
        "boxed_warning": pick_first(result.get("boxed_warning"), ""),
        "contraindications": pick_first(result.get("contraindications"), ""),
        "warnings_and_precautions": pick_first(result.get("warnings_and_precautions"), ""),
        "drug_interactions": pick_first(result.get("drug_interactions"), ""),
        "adverse_reactions": pick_first(result.get("adverse_reactions"), ""),
        "indications_and_usage": pick_first(result.get("indications_and_usage"), ""),
        "dosage_and_administration": pick_first(result.get("dosage_and_administration"), ""),
        "use_in_specific_populations": pick_first(result.get("use_in_specific_populations"), ""),
    }


app = FastAPI(title="FDA Evidence Assistant API", version="0.1")
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()

def rewrite_query_for_retrieval(user_q: str) -> str:
    """
    Rewrite the user question into a short, retrieval-friendly query.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's question into a short search query for FDA drug label evidence. "
         "Keep key medical terms. Remove extra words. Output only the rewritten query."),
        ("human", "{q}")
    ])
    chain = prompt | LLM
    resp = chain.invoke({"q": user_q})
    return (resp.content or "").strip()

def evidence_is_weak(matches: list, min_matches: int = 3, max_distance: float = 0.55) -> bool:
    """
    Baby-simple heuristic:
    - weak if too few chunks
    - or if best chunk is still far away (high distance)
    """
    if not matches or len(matches) < min_matches:
        return True

    try:
        best_dist = float(matches[0].get("distance", 999))
    except Exception:
        best_dist = 999

    return best_dist > max_distance



LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)



class Settings(BaseSettings):
    openfda_base_url: str = "https://api.fda.gov"

settings = Settings()
@app.on_event("startup")
def on_startup():
    init_db()



@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tools/openfda/label")
async def fetch_label(drug_name: str, limit: int = 1):
    if not drug_name or not drug_name.strip():
        raise HTTPException(status_code=400, detail="drug_name is required")

    drug_name = drug_name.strip()

    candidates = [
        f'openfda.generic_name:"{drug_name}"',
        f'openfda.brand_name:"{drug_name}"',
        f'openfda.substance_name:"{drug_name}"',
    ]

    async with httpx.AsyncClient(timeout=20) as client:
        last_error = None
        for q in candidates:
            url = f"{settings.openfda_base_url}/drug/label.json"
            params = {"search": q, "limit": limit}
            r = await client.get(url, params=params)

            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if results:
                    return {
                        "query_used": q,
                        "meta": data.get("meta", {}),
                        "result": results[0],
                    }

            last_error = {"status_code": r.status_code, "body": r.text}

    raise HTTPException(
        status_code=404,
        detail={
            "message": f"No label found for '{drug_name}'. Try generic or brand name.",
            "last_error": last_error,
        },
    )
@app.get("/assist/label_summary")
async def label_summary(drug_name: str):
    tool_resp = await fetch_label(drug_name=drug_name, limit=1)
    result = tool_resp["result"]

    citation = build_citation(result)
    sections = extract_sections(result)

    # remove empty sections BEFORE saving/returning
    sections = {k: v for k, v in sections.items() if v and v.strip()}

    db_saved = True
    db_error = ""

    try:
        save_label_summary(
            drug_query=drug_name,
            query_used=tool_resp.get("query_used", ""),
            citation=citation,
            sections=sections,
            raw_result=result
        )
    except Exception as e:
        db_saved = False
        db_error = str(e)

    return {
        "drug_query": drug_name,
        "query_used": tool_resp.get("query_used", ""),
        "citation": citation,
        "sections": sections,
        "db_saved": db_saved,
        "db_error": db_error,
    }
@app.get("/db/recent_labels")
def recent_labels(limit: int = 10):
    return {"items": get_recent_labels(limit=limit)}


@app.get("/db/label/{label_id}")
def label_details(label_id: int):
    row = get_label_by_id(label_id)
    if not row:
        raise HTTPException(status_code=404, detail="Label not found")
    return row

@app.post("/rag/chunk_latest")
def chunk_latest():
    row = get_latest_label_with_sections()
    if not row:
        raise HTTPException(status_code=404, detail="No saved labels found. Search a drug first.")

    label_id = row["id"]
    sections = row["sections"]

    inserted = insert_chunks_for_label(label_id=label_id, sections=sections)
    total = count_chunks()

    return {"label_id": label_id, "chunks_inserted": inserted, "total_chunks": total}

@app.get("/rag/search")
def rag_search(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")

    q = q.strip()

    # 1) embed query (384 dims)
    emb = EMBED_MODEL.encode([q], normalize_embeddings=True)[0]
    emb_str = "[" + ",".join([str(float(x)) for x in emb]) + "]"

    # 2) vector similarity search
    sql = """
    SELECT
        c.id,
        c.label_id,
        c.section,
        c.chunk_index,
        c.content,
        (c.embedding <=> :q_emb) AS distance
    FROM label_chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> :q_emb
    LIMIT :k;
    """

    from db import engine

    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"q_emb": emb_str, "k": k}).mappings().all()
        items = [dict(r) for r in rows]

    return {"query": q, "k": k, "matches": items}

@app.get("/assist/answer")
def assist_answer(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")

    # 1) retrieve evidence chunks
    rag = rag_search(q=q, k=k)
    matches = rag.get("matches", [])

    if not matches:
        return {
            "question": q,
            "answer": "I couldn't find saved FDA evidence chunks yet. Try saving/chunking/embedding a label first.",
            "citations": [],
            "used_k": k
        }

    # 2) build evidence context
    evidence_lines = []
    citations = []
    for i, m in enumerate(matches, start=1):
        section = m.get("section", "unknown")
        content = m.get("content", "")
        label_id = m.get("label_id", None)

        evidence_lines.append(f"[{i}] Section: {section}\n{content}\n")
        citations.append({
            "id": i,
            "label_id": label_id,
            "section": section,
            "chunk_index": m.get("chunk_index", None),
            "distance": m.get("distance", None)
        })

    evidence_text = "\n".join(evidence_lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an FDA evidence assistant. Educational only, not medical advice. "
         "Answer ONLY using the provided evidence. If evidence is insufficient, say so. "
         "Always cite sources like [1], [2] in the answer."),
        ("human",
         "Question: {question}\n\nEvidence:\n{evidence}\n\n"
         "Write a clear answer in plain English with citations.")
    ])

    chain = prompt | LLM
    resp = chain.invoke({"question": q, "evidence": evidence_text})

    return {
        "question": q,
        "answer": resp.content,
        "citations": citations,
        "used_k": k
    }

@app.get("/assist/answer_agentic")
def assist_answer_agentic(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")

    user_q = q.strip()
    trace = []

    # Attempt 1: normal retrieval
    r1 = rag_search(q=user_q, k=k)
    m1 = r1.get("matches", [])
    trace.append({
        "attempt": 1,
        "query": user_q,
        "k": k,
        "num_matches": len(m1),
        "best_distance": (m1[0].get("distance") if m1 else None),
    })

    # If evidence is weak, attempt 2: rewrite query and increase k
    final_q = user_q
    final_k = k

    if evidence_is_weak(m1):
        rewritten = rewrite_query_for_retrieval(user_q)
        final_q = rewritten if rewritten else user_q
        final_k = max(8, k)

        r2 = rag_search(q=final_q, k=final_k)
        m2 = r2.get("matches", [])
        trace.append({
            "attempt": 2,
            "query": final_q,
            "k": final_k,
            "num_matches": len(m2),
            "best_distance": (m2[0].get("distance") if m2 else None),
        })

        # Choose better attempt (more matches wins; if tied, better distance wins)
        # Pick the best attempt by distance (lower = better), but only if attempt 2 returned something
        pick_2 = False
        if m2:
            try:
                d1 = float(m1[0]["distance"]) if m1 else 999
                d2 = float(m2[0]["distance"])
                pick_2 = d2 < d1
            except Exception:
                pick_2 = True

        if pick_2:
            final_q, final_k = final_q, final_k
        else:
            final_q, final_k = user_q, k



    # Final: generate answer 
    answer_payload = assist_answer(q=final_q, k=final_k)
    answer_payload["trace"] = trace
    answer_payload["final_query_used"] = final_q
    answer_payload["final_k_used"] = final_k

    return answer_payload

@app.get("/db/chunk/{chunk_id}")
def read_chunk(chunk_id: int):
    row = get_chunk_by_id(chunk_id)
    if not row:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return row


