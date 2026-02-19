**Agentic RAG Pipeline: FDA Drug Label Assistant**

I built this because drug information felt like a mix of random (and suspicious) websites with half-explanations. I wanted something that answers normal questions like warnings, dosage, side effects, precautions, and interactions, but still shows exactly where the answer came from. The goal is simple: give a clear response in plain language, and back it up with the official label text so the user can verify it.

As a user, you type a medicine name and a concern or question you might have about that drug. The app pulls the official FDA label text using the openFDA Drug Label API, saves it, breaks it into smaller readable sections (or chunks), and then searches those chunks when you ask a question. After that, it uses Gemini to write an explanation in simple terms based on the most relevant chunks. Along with the answer, it returns the exact label sections or chunks used, so the user is assured that this does not rely on random internet sources.

The main idea is RAG, which stands for retrieval augmented generation. In this project, retrieval means finding the best matching pieces of FDA label text for the user's input question. Generation means asking an LLM to explain the answer using only those retrieved pieces. This keeps the explanation grounded in the label instead of the model making assumptions.

Quick FDA context: the FDA is the U.S. Food and Drug Administration. They publish official drug labeling that includes sections like indications and usage, dosage and administration, warnings and precautions, adverse reactions, and other safety notes. openFDA is an FDA-supported way to access parts of that labeling data through an API.

**Components:**

Backend
FastAPI with Uvicorn runs the API endpoints. It fetches labels from openFDA, extracts relevant sections, saves them into PostgreSQL, chunks the sections, runs retrieval, and returns answers with evidence.

Database
PostgreSQL stores saved labels and metadata, plus the chunked label text. It also stores embeddings for each chunk so retrieval is fast and repeatable.

Frontend
Streamlit provides the UI. Users can search a drug, ask a question, control top k evidence chunks, and view recent saved labels from the database.

**Models used**

Embeddings model for retrieval
Sentence Transformers all MiniLM L6 v2. It produces 384-dimensional vectors.

LLM for final explanation
Google Gemini, 2.5 Flash. It takes the retrieved FDA label chunks and explains them in simple words while keeping the response tied to evidence.

**Tech Stack**

- Python FastAPI and Uvicorn (backend API)
- Streamlit (frontend UI)
- PostgreSQL (storage)
- SQLAlchemy (DB access)
- Sentence-transformers and PyTorch (embeddings)
- Google Gemini API (generation)
-  openFDA Drug Label API (data source)
-  Docker 



**Data source link for openFDA**
https://open.fda.gov/

openFDA Drug Label API documentation
https://open.fda.gov/apis/drug/label/

Drug label endpoint reference
https://api.fda.gov/drug/label.json

**Why this is useful**

It answers drug label questions using official labeling text and shows the exact evidence it relied on. That makes it easier to trust, easier to verify, and much safer than reading summaries with no source attached. It is designed for fast lookups and clarity, not for replacing medical advice.

**Note**

This tool is for informational use based on FDA labeling text. It is not medical advice!
