# 📄 PDF Summarization & Question Answering AI Agent

An AI-powered system that reads PDF documents, summarizes their content, and answers questions using **Retrieval-Augmented Generation (RAG)** — exposed via a polished **Streamlit** web interface.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-00ADD8)
![Gemini](https://img.shields.io/badge/Google_Gemini-LLM-4285F4?logo=google&logoColor=white)

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                            │
│            (Upload · Summarize · Chat Q&A · Sources)             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  RAG Pipeline   │  ← Orchestrator
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  ┌─────────┐   ┌───────────┐   ┌───────────┐
  │  Ingest  │   │ Retrieve  │   │ Generate  │
  └────┬─────┘   └─────┬─────┘   └─────┬─────┘
       │               │               │
  ┌────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
  │PDF Extract│   │  FAISS    │   │  Gemini/  │
  │→ Chunk   │   │  Vector   │   │  OpenAI   │
  │→ Embed   │   │  Store    │   │  LLM      │
  └──────────┘   └───────────┘   └───────────┘
```

### Data Flow

1. **Ingest** — PDF → `pdfplumber` text extraction → `RecursiveCharacterTextSplitter` chunking → `sentence-transformers` embedding → FAISS index
2. **Retrieve** — User query → embed query → cosine similarity search → top-K chunks
3. **Generate** — Retrieved chunks + question → LLM prompt → grounded answer

---

## 📂 Project Structure

```
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── agent/
    ├── __init__.py
    ├── app.py              # Streamlit entry point
    ├── config.py            # Environment & settings
    ├── pdf_extractor.py     # PDF → raw text
    ├── chunker.py           # Text → overlapping chunks
    ├── embeddings.py        # Chunks → vector embeddings
    ├── vector_store.py      # FAISS index management
    ├── llm.py               # LLM abstraction (Gemini/OpenAI)
    └── rag_pipeline.py      # Orchestrates the full RAG flow
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+**
- A **Google Gemini API key** (free) from [AI Studio](https://aistudio.google.com/apikey)  
  *— or an OpenAI API key*

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-rag-agent.git
cd pdf-rag-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
copy .env.example .env
# Edit .env and set your API key:
# GOOGLE_API_KEY=your_key_here
```

### 5. Run the Application

```bash
streamlit run agent/app.py
```

The app will open at **http://localhost:8501**.

---

## 🎯 How to Use

1. **Upload** a PDF document via the sidebar
2. The system automatically extracts, chunks, and indexes the document
3. Click **✨ Summarize Document** for a comprehensive summary
4. **Ask questions** in the chat input — the agent retrieves relevant passages and generates grounded answers
5. Expand **📚 Retrieved Context Chunks** to see which parts of the document were used

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `pdfplumber` | PDF text extraction with layout awareness |
| `langchain-text-splitters` | Recursive character-based text chunking |
| `sentence-transformers` | Local embedding generation (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Fast approximate nearest-neighbour search |
| `google-generativeai` | Google Gemini LLM API |
| `openai` | OpenAI LLM API (alternative provider) |
| `streamlit` | Interactive web UI framework |
| `python-dotenv` | Environment variable management |
| `numpy` | Numerical array operations |

---

## 🏛️ Design Decisions & Trade-offs

### Embedding Model: `all-MiniLM-L6-v2`
- **Decision**: Use a local sentence-transformers model instead of cloud embedding APIs.
- **Trade-off**: Runs entirely offline (no API cost for embeddings), but requires ~80 MB model download on first run. The 384-dimensional model offers an excellent speed/quality balance.

### Vector Store: FAISS (In-Memory)
- **Decision**: Use FAISS `IndexFlatIP` (flat inner-product index) rather than a persistent database like Chroma or Pinecone.
- **Trade-off**: Zero setup overhead and blazing-fast similarity search. However, the index lives in memory per session — fine for single-document Q&A, but not persisted across restarts. For production use, swap to `IndexIVFFlat` or a managed vector DB.

### Chunking Strategy: Recursive Character Splitting
- **Decision**: 500-char chunks with 100-char overlap, splitting on paragraph → sentence → word boundaries.
- **Trade-off**: Small chunks improve retrieval precision (a chunk is more likely to be on a single topic), but may miss broader context. The overlap mitigates information loss at chunk boundaries.

### LLM: Google Gemini (Default)
- **Decision**: Default to Gemini's free tier (`gemini-2.0-flash`) with OpenAI as an alternative.
- **Trade-off**: Free and powerful, but requires an API key and internet connection. The abstraction layer makes it trivial to switch providers.

### Summarization: Strategic Chunk Selection
- **Decision**: For long documents, select chunks from the beginning, middle, and end rather than sending the entire text.
- **Trade-off**: Avoids token-limit issues and reduces cost, but may miss isolated details. For most documents, the beginning/middle/end heuristic captures the narrative arc well.


