import streamlit as st
from ingestion import run_ingestion, load_models
from query import run_query
import config
import shutil
from pathlib import Path

st.set_page_config(
    page_title="Advanced RAG",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #112240 100%);
        color: #e8eaf0;
    }
    [data-testid="stSidebar"] {
        background: #0d1b2a !important;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* Hide ALL avatars */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        gap: 0 !important;
    }

    /* Chat input — blue theme, no white */
    [data-testid="stBottom"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 8px 0 !important;
    }
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInputContainer"] > div,
    [data-testid="stChatInputContainer"] > div > div {
        background: #0d2137 !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] {
        background: #0d2137 !important;
        border: none !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #0d2137 !important;
        color: #e8f4fd !important;
        font-size: 17px !important;
        min-height: 52px !important;
        caret-color: #2196f3 !important;
        border: none !important;
        padding: 14px 16px !important;
        line-height: 1.5 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #5ba3d9 !important;
        font-size: 16px !important;
    }
    [data-testid="stChatInput"] button {
        background: #2196f3 !important;
        border-radius: 10px !important;
        border: none !important;
        margin: 6px !important;
    }

    /* User bubble — RIGHT */
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 8px 0;
        clear: both;
    }
    .user-bubble-inner {
        background: #1565c0;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        font-size: 15px;
        line-height: 1.5;
        word-wrap: break-word;
    }

    /* Assistant bubble — LEFT */
    .assistant-bubble {
        display: flex;
        justify-content: flex-start;
        margin: 8px 0;
        clear: both;
    }
    .assistant-bubble-inner {
        background: #1a2f4a;
        color: #e8eaf0;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        border: 1px solid #2196f3;
        max-width: 85%;
        font-size: 15px;
        line-height: 1.6;
        word-wrap: break-word;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1a2f4a !important;
        border: 2px dashed #2196f3 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] * { color: #e8eaf0 !important; }
    [data-testid="stFileUploaderDropzone"] { background: #1a2f4a !important; border: none !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #2196f3 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* All buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #2196f3) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2196f3, #42a5f5) !important;
        transform: translateY(-1px) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #1a2f4a !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #1a2f4a !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    [data-testid="stMetricValue"] { color: #2196f3 !important; }
    [data-testid="stMetricLabel"] { color: #90a4ae !important; }

    hr { border-color: #1e3a5f !important; }
    h1, h2, h3 { color: #e8eaf0 !important; }
    p, li { color: #b0bec5; }
    code { background: #1a2f4a !important; color: #2196f3 !important; border-radius: 4px; padding: 2px 6px; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d1b2a; }
    ::-webkit-scrollbar-thumb { background: #2196f3; border-radius: 3px; }

    .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-right: 6px; }
    .badge-text  { background: #1565c0; color: #90caf9; }
    .badge-table { background: #1b5e20; color: #a5d6a7; }
    .badge-image { background: #4a148c; color: #ce93d8; }

    /* Ready banner */
    .ready-banner {
        text-align: center;
        padding: 60px 20px 30px;
    }
    .ready-banner h2 {
        font-size: 2rem !important;
        background: linear-gradient(135deg, #2196f3, #42a5f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px !important;
    }
    .ready-banner p {
        color: #546e7a;
        font-size: 1rem;
    }
    
</style>
""", unsafe_allow_html=True)

# ─── Load models ──────────────────────────────────────────────────────────
embedder = load_models()

# ─── Initialize uploader key ──────────────────────────────────────────────
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Advanced RAG")
    st.markdown("---")
    st.markdown("### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        if st.button("⚡ Process Documents"):
            run_ingestion(uploaded_files)

    st.markdown("---")


    if st.button("🗑️ Clear All"):
        try:
            from chromadb import PersistentClient
            chroma_client = PersistentClient(path=config.CHROMA_DIR)
            for c in chroma_client.list_collections():
                chroma_client.delete_collection(c.name)
        except:
            pass
        # Delete all data folders
        for folder in [config.UPLOAD_DIR, config.IMAGES_DIR,
                       config.MARKDOWN_DIR, config.CHROMA_DIR]:
            shutil.rmtree(folder, ignore_errors=True)
            Path(folder).mkdir(parents=True, exist_ok=True)
        # Clear ChromaDB system cache
        try:
            import chromadb
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except:
            pass
        # Reset uploader widget
        uploader_key = st.session_state.uploader_key + 1
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.uploader_key = uploader_key
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='font-size:11px;color:#546e7a;text-align:center'>Docling · ChromaDB · Groq · BGE</p>", unsafe_allow_html=True)

# ─── Helper: render message ───────────────────────────────────────────────
def render_message(role, content, sources=None):
    if role == "user":
        st.markdown(f"""
        <div class='user-bubble'>
            <div class='user-bubble-inner'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='assistant-bubble'>
            <div class='assistant-bubble-inner'>{content}</div>
        </div>
        """, unsafe_allow_html=True)
        if sources:
            with st.expander("📚 Sources"):
                for i, (score, doc) in enumerate(sources):
                    meta = doc.metadata
                    chunk_type = meta.get("chunk_type", "text")
                    st.markdown(f"""
                    <span class='badge badge-{chunk_type}'>{chunk_type.upper()}</span>
                    <strong>{meta.get('source_file','unknown')}</strong> —
                    section: <code>{meta.get('h1','') or meta.get('h2','N/A')}</code> |
                    score: <code>{score:.3f}</code>
                    """, unsafe_allow_html=True)
                    st.caption(doc.page_content[:300] + "...")
                    if chunk_type == "image" and meta.get("image_path"):
                        st.image(meta["image_path"])
                    st.divider()

# ─── Main area ────────────────────────────────────────────────────────────
if "chroma_collection" not in st.session_state:
    # Welcome screen
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px 40px;'>
        <h1 style='font-size:3rem; background: linear-gradient(135deg, #2196f3, #42a5f5);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:10px;'>
        Advanced RAG</h1>
        <p style='color:#546e7a; font-size:1.1rem;'>
        Upload your documents and ask questions</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#1a2f4a;border:1px solid #1e3a5f;border-radius:12px;
        padding:24px;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:8px'>📄</div>
            <h3 style='color:#e8eaf0;margin:0 0 8px'>Multi-format</h3>
            <p style='color:#546e7a;font-size:13px;margin:0'>
            PDF, DOCX with OCR for scanned documents</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#1a2f4a;border:1px solid #1e3a5f;border-radius:12px;
        padding:24px;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:8px'>🔎</div>
            <h3 style='color:#e8eaf0;margin:0 0 8px'>Hybrid Search</h3>
            <p style='color:#546e7a;font-size:13px;margin:0'>
            BM25 keyword + BGE semantic search with reranking</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#1a2f4a;border:1px solid #1e3a5f;border-radius:12px;
        padding:24px;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:8px'>🤖</div>
            <h3 style='color:#e8eaf0;margin:0 0 8px'>Groq LLM</h3>
            <p style='color:#546e7a;font-size:13px;margin:0'>
            Llama 3.3 70B answers strictly from your documents</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload documents in the sidebar to get started")

else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Ready banner when no messages yet
    if not st.session_state.messages:
        st.markdown("""
        <div class='ready-banner'>
            <h2>✦ Your documents are ready</h2>
            <p>Ask questions here — summaries, specific facts, tables, comparisons, key findings</p>
        </div>
        """, unsafe_allow_html=True)

    # Render all messages
    for message in st.session_state.messages:
        render_message(
            message["role"],
            message["content"],
            message.get("sources")
        )

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        render_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("🔍 Searching..."):
            answer, scored_docs, standalone = run_query(prompt, embedder)

        if answer is None:
            msg = "⚠️ Answer not found in the provided documents. Try rephrasing your question."
            render_message("assistant", msg)
            st.session_state.messages.append({
                "role": "assistant", "content": msg, "sources": []
            })
        else:
            render_message("assistant", answer, scored_docs)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": scored_docs
            })