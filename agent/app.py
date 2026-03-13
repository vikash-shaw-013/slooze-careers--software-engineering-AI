"""Streamlit UI for the PDF Summarization & Question Answering Agent."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `agent.*` imports work
# regardless of which directory Streamlit is launched from.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from agent.config import Config
from agent.rag_pipeline import RAGPipeline


# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF AI Agent — Summarize & Ask",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & Global ───────────────────────────────────────── */
:root {
    --accent: #6C63FF;
    --accent-light: #A29BFE;
    --accent-glow: rgba(108, 99, 255, 0.25);
    --bg-dark: #0E1117;
    --bg-card: #161B22;
    --bg-card-hover: #1C2333;
    --border: #30363D;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --success: #3FB950;
    --warning: #D29922;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.5rem;
}

/* ── File Uploader ───────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.3s ease;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* ── Card style for expanders ────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: background 0.2s ease;
}

.streamlit-expanderHeader:hover {
    background: var(--bg-card-hover) !important;
}

/* ── Chat Messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid var(--border);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    animation: fadeIn 0.35s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

[data-testid="stMetricValue"] {
    color: var(--accent-light) !important;
    font-weight: 700;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #5A52D5 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 12px var(--accent-glow);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px var(--accent-glow);
}

/* ── Status badge ────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
}

.status-ready {
    background: rgba(63, 185, 80, 0.15);
    color: var(--success);
    border: 1px solid rgba(63, 185, 80, 0.3);
}

.status-waiting {
    background: rgba(210, 153, 34, 0.15);
    color: var(--warning);
    border: 1px solid rgba(210, 153, 34, 0.3);
}

/* ── Source chunk cards ──────────────────────────────────── */
.source-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    line-height: 1.5;
    transition: border-color 0.2s ease;
}

.source-card:hover {
    border-color: var(--accent);
}

.source-score {
    color: var(--accent-light);
    font-weight: 600;
    font-size: 0.78rem;
}

/* ── Hero banner ─────────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 50%, #E0DDFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-sub {
    color: var(--text-secondary);
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}

/* ── Feature pills ───────────────────────────────────────── */
.feature-pills {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}

.pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    transition: all 0.2s ease;
}

.pill:hover {
    border-color: var(--accent);
    color: var(--accent-light);
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ────────────────────────────────────────────────
def init_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None


init_state()
pipeline: RAGPipeline = st.session_state.pipeline


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📄 PDF AI Agent")
    st.caption("Summarize & ask questions about any PDF")

    st.divider()

    # Upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Drag & drop or click to browse. Max 200 MB.",
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
        with st.spinner("🔄 Processing document…"):
            try:
                stats = pipeline.ingest_bytes(uploaded_file.read())
                st.session_state.doc_stats = stats
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = []  # reset chat for new doc
                st.success("✅ Document ingested!", icon="📄")
            except Exception as e:
                st.error(f"❌ {e}")

    # Document status
    st.divider()
    if pipeline.is_ready:
        st.markdown(
            '<span class="status-badge status-ready">● Document Ready</span>',
            unsafe_allow_html=True,
        )
        stats = st.session_state.doc_stats or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", stats.get("pages", "—"))
        c2.metric("Chunks", stats.get("chunks", "—"))
        c3.metric("Chars", f"{stats.get('characters', 0):,}")
    else:
        st.markdown(
            '<span class="status-badge status-waiting">◌ Awaiting PDF</span>',
            unsafe_allow_html=True,
        )

    # Settings
    st.divider()
    with st.expander("⚙️ Settings", expanded=False):
        st.caption(f"**LLM Provider:** {Config.LLM_PROVIDER.title()}")
        st.caption(f"**Embedding Model:** {Config.EMBEDDING_MODEL}")
        st.caption(f"**Chunk Size:** {Config.CHUNK_SIZE}")
        st.caption(f"**Overlap:** {Config.CHUNK_OVERLAP}")
        st.caption(f"**Top-K Results:** {Config.TOP_K}")

    # Summarize button
    st.divider()
    summarize_btn = st.button(
        "✨ Summarize Document",
        use_container_width=True,
        disabled=not pipeline.is_ready,
    )

    # Config warnings
    issues = Config.validate()
    if issues:
        for issue in issues:
            st.warning(issue, icon="⚠️")


# ── Main Area ────────────────────────────────────────────────────

if not pipeline.is_ready:
    # Hero / landing state
    st.markdown(
        """
    <div class="hero-container">
        <div class="hero-title">PDF Intelligence Agent</div>
        <div class="hero-sub">
            Upload a PDF in the sidebar to unlock AI-powered
            summarization and context-aware question answering.
        </div>
        <div class="feature-pills">
            <span class="pill">📑 Text Extraction</span>
            <span class="pill">🧩 Smart Chunking</span>
            <span class="pill">🔢 Vector Embeddings</span>
            <span class="pill">🔍 Semantic Search</span>
            <span class="pill">🤖 RAG Answers</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Architecture explainer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1 · Ingest")
        st.markdown(
            "Upload your PDF and the agent extracts text page-by-page, "
            "splits it into overlapping chunks, and generates vector embeddings."
        )
    with col2:
        st.markdown("### 2 · Retrieve")
        st.markdown(
            "When you ask a question, the system finds the most relevant "
            "chunks using cosine similarity search powered by FAISS."
        )
    with col3:
        st.markdown("### 3 · Generate")
        st.markdown(
            "The retrieved context is fed to an LLM which generates a "
            "grounded, citation-aware answer based solely on your document."
        )
else:
    # ── Summarize action ─────────────────────────────────────────
    if summarize_btn:
        with st.spinner("✨ Generating summary…"):
            try:
                summary = pipeline.summarize()
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"📋 **Document Summary**\n\n{summary}"}
                )
            except Exception as e:
                st.error(f"Summarization failed: {e}")

    # ── Chat display ─────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources if present
            if "sources" in msg:
                with st.expander("📚 Retrieved Context Chunks", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        score_pct = src["score"] * 100
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-score">Chunk #{i} · '
                            f"Relevance: {score_pct:.1f}%</span><br/>"
                            f'{src["chunk"]}</div>',
                            unsafe_allow_html=True,
                        )

    # ── Chat input ───────────────────────────────────────────────
    if question := st.chat_input("Ask a question about the document…"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching & generating answer…"):
                try:
                    result = pipeline.query(question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    # Show sources
                    with st.expander("📚 Retrieved Context Chunks", expanded=False):
                        for i, src in enumerate(sources, 1):
                            score_pct = src["score"] * 100
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="source-score">Chunk #{i} · '
                                f"Relevance: {score_pct:.1f}%</span><br/>"
                                f'{src["chunk"]}</div>',
                                unsafe_allow_html=True,
                            )

                    # Save to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
