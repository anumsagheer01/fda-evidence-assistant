import requests
import streamlit as st
st.set_page_config(
    page_title="FDA Drug Label Assistant",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 35%, #FFF7FB 100%);
    }


    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 860px;
    }

   
    html, body, [class*="css"]  {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }

 
    .hero {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(170, 190, 255, 0.35);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 10px 25px rgba(60, 90, 180, 0.08);
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: -0.2px;
        margin-bottom: 4px;
    }

    .hero-sub {
        font-size: 1.02rem;
        color: rgba(20, 20, 40, 0.75);
        margin-bottom: 10px;
    }

    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        background: rgba(120, 150, 255, 0.12);
        border: 1px solid rgba(120, 150, 255, 0.25);
        color: rgba(35, 55, 130, 0.95);
        margin-right: 6px;
        margin-top: 4px;
    }


    div.stButton > button {
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 650 !important;
    }

    /* Text inputs */
    .stTextInput input, .stNumberInput input {
        border-radius: 12px !important;
    }

    /* Sliders */
    .stSlider {
        padding-top: 0.35rem;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(120, 150, 255, 0.22);
        box-shadow: 0 8px 18px rgba(60, 90, 180, 0.06);
    }

    /* Error box softer */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header / intro ----------
st.markdown(
    """
    <div class="hero">
  <div class="hero-title"><span style="color:#3B5BDB;">FDA Drug Label Assistant</span></div>

  <div class="hero-sub">
    As a user, please type a medicine name and a concern or question you might have about this drug. This agentic RAG assistant pulls info like warnings, dosage, side effects, and precautions straight from the official FDA drug label text (via the openFDA Drug Label API). Then, using Gemini, it explains the answer in simple words and shows the exact label sections/chunks it used so you’re not guessing or relying on random internet sources.

  </div>

  <span class="pill">Verified label text</span>
  <span class="pill">Cited evidence</span>
  <span class="pill">Saved history</span>

  <hr style="border:none;border-top:1px solid rgba(120,150,255,0.18);margin:12px 0 10px 0;" />

  <div style="color:rgba(20, 20, 40, 0.78); font-size:0.95rem; line-height:1.35;">
    <b>What’s FDA?</b> The <b>Food and Drug Administration</b> is the U.S. government agency that publishes official drug labels.
    Those labels include warnings, precautions, dosage info, and safety notes.
  </div>
</div>

    """,
    unsafe_allow_html=True,
)




BACKEND_URL = st.text_input("Backend URL", "http://127.0.0.1:8000")
drug_name = st.text_input("Enter a drug name", "ibuprofen")
question = st.text_input("Ask a question", "What are the warnings and precautions?")
k = st.slider("Top-k chunks", 1, 10, 5)

if st.button("Ask Agentic FDA Assistant"):
    # 1) (optional but recommended) fetch + save label first
    r1 = requests.get(f"{BACKEND_URL}/assist/label_summary", params={"drug_name": drug_name})
    if r1.status_code != 200:
        st.error(f"Label fetch failed: {r1.status_code}")
        st.stop()

    # 2) chunk and embed latest label (so retrieval works)
    r2 = requests.post(f"{BACKEND_URL}/rag/chunk_latest")
    if r2.status_code != 200:
        st.error(f"Chunking failed: {r2.status_code} {r2.text}")
        st.stop()

    r3 = requests.post(f"{BACKEND_URL}/rag/embed_pending")
    if r3.status_code != 200:
        st.error(f"Embedding failed: {r3.status_code} {r3.text}")
        st.stop()

    # 3) ask agentic answer
    r4 = requests.get(f"{BACKEND_URL}/assist/answer_agentic", params={"q": question, "k": k})

    if r4.status_code != 200:
        st.error(f"Answer failed: {r4.status_code} {r4.text}")
        st.stop()

    data = r4.json()

    st.subheader("Answer")
    st.write(data.get("answer", ""))

    st.subheader("Citations")
    for c in data.get("citations", []):
        st.caption(f"Section: {c.get('section')} | chunk: {c.get('chunk_index')} | distance: {round(c.get('distance', 0), 4)}")


col1, col2 = st.columns([1, 1])
with col1:
    show_n = st.number_input("Show last N records", min_value=1, max_value=50, value=10)
with col2:
    refresh = st.button("Refresh saved list")

try:
    r = requests.get(f"{BACKEND_URL}/db/recent_labels", params={"limit": int(show_n)}, timeout=10)
    if r.status_code == 200:
        items = r.json().get("items", [])

        if not items:
            st.info("No saved labels yet. Search a drug above to save one.")
        else:
            st.write("Latest saved labels:")
            st.dataframe(items, use_container_width=True)

            # Let user pick one ID to view sections
            label_ids = [str(x["id"]) for x in items if "id" in x]
            chosen = st.selectbox("Pick a saved record to view sections", label_ids)

            if chosen:
                details = requests.get(f"{BACKEND_URL}/db/label/{chosen}", timeout=10)
                if details.status_code == 200:
                    d = details.json()
                    st.markdown(f"### Record {d['id']} - {d.get('drug_query','')}")
                    st.write(f"**Brand:** {d.get('brand_name','')}")
                    st.write(f"**Generic:** {d.get('generic_name','')}")
                    st.write(f"**Effective time:** {d.get('effective_time','')}")
                    with st.expander("Saved sections"):
                        st.json(d.get("sections", {}))
                else:
                    st.error("Could not load that record.")
    else:
        st.error(f"DB list error: {r.status_code}")
        st.code(r.text)
except Exception as e:
    st.error("Could not reach backend for DB list.")
    st.exception(e)

st.divider()
st.subheader("RAG Evidence Search (from saved FDA labels)")

user_q = st.text_input(
    "Ask a question",
    placeholder="e.g., What are the main warnings? Any dosage guidance for kids?"
)
top_k = st.slider("Top-k evidence chunks", min_value=3, max_value=10, value=5)

if st.button("Retrieve evidence"):
    if not user_q.strip():
        st.warning("Type a question first.")
    else:
        try:
            resp = requests.get(
                f"{BACKEND_URL}/rag/search",
                params={"q": user_q, "k": int(top_k)},
                timeout=30
            )
            if resp.status_code != 200:
                st.error(f"Backend error: {resp.status_code}")
                st.code(resp.text)
            else:
                data = resp.json()
                matches = data.get("matches", [])

                if not matches:
                    st.info("No matches yet. Try saving/chunking/embedding more labels first.")
                else:
                    st.success(f"Found {len(matches)} evidence chunks")

                    for i, m in enumerate(matches, start=1):
                        section = m.get("section", "unknown_section")
                        dist = m.get("distance", None)
                        label_id = m.get("label_id", "?")
                        chunk_index = m.get("chunk_index", "?")

                        title = f"{i}) Label {label_id} • {section} • chunk {chunk_index}"
                        if dist is not None:
                            title += f" • distance {float(dist):.4f}"

                        with st.expander(title):
                            st.write(m.get("content", ""))
        except Exception as e:
            st.error("Request failed.")
            st.code(str(e))
