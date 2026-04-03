import streamlit as st
from ingestion import run_ingestion, load_models
from query import run_query, filter_display_sources
import config
import shutil
from pathlib import Path

st.set_page_config(page_title="Advanced RAG", page_icon="🔍", layout="wide")

# ── Styling ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#0a0e1a 0%,#0d1b2a 50%,#112240 100%); color:#e8eaf0; }
    [data-testid="stSidebar"] { background:#0d1b2a !important; border-right:1px solid #1e3a5f; }
    [data-testid="stHeader"]  { background:transparent; }

    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] { display:none !important; width:0 !important; min-width:0 !important; }
    [data-testid="stChatMessage"] { background:transparent !important; border:none !important; padding:0 !important; gap:0 !important; }

    [data-testid="stBottom"] > div { background:transparent !important; border:none !important; box-shadow:none !important; padding:8px 0 !important; }
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInputContainer"] > div,
    [data-testid="stChatInputContainer"] > div > div { background:#0d2137 !important; border:none !important; box-shadow:none !important; }
    [data-testid="stChatInput"] { background:#0d2137 !important; border:none !important; border-radius:16px !important; box-shadow:none !important; }
    [data-testid="stChatInput"] textarea { background:#0d2137 !important; color:#e8f4fd !important; font-size:17px !important; min-height:52px !important; caret-color:#2196f3 !important; border:none !important; padding:14px 16px !important; line-height:1.5 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color:#5ba3d9 !important; font-size:16px !important; }
    [data-testid="stChatInput"] button { background:#2196f3 !important; border-radius:10px !important; border:none !important; margin:6px !important; }

    .user-bubble { display:flex; justify-content:flex-end; margin:8px 0; clear:both; }
    .user-bubble-inner { background:#1565c0; color:#fff; padding:12px 18px; border-radius:18px 18px 4px 18px; max-width:70%; font-size:15px; line-height:1.5; word-wrap:break-word; }

    .assistant-bubble { display:flex; justify-content:flex-start; margin:8px 0; clear:both; }
    .assistant-bubble-inner { background:#1a2f4a; color:#e8eaf0; padding:14px 18px; border-radius:18px 18px 18px 4px; border:1px solid #2196f3; max-width:85%; font-size:15px; line-height:1.6; word-wrap:break-word; }

    [data-testid="stFileUploader"] { background:#1a2f4a !important; border:2px dashed #2196f3 !important; border-radius:12px !important; }
    [data-testid="stFileUploader"] * { color:#e8eaf0 !important; }
    [data-testid="stFileUploaderDropzone"] { background:#1a2f4a !important; border:none !important; }
    [data-testid="stFileUploaderDropzone"] button { background:#2196f3 !important; color:white !important; border:none !important; border-radius:8px !important; }

    .stButton > button { background:linear-gradient(135deg,#1565c0,#2196f3) !important; color:white !important; border:none !important; border-radius:10px !important; padding:10px 16px !important; font-weight:600 !important; font-size:15px !important; width:100% !important; }
    .stButton > button:hover { background:linear-gradient(135deg,#2196f3,#42a5f5) !important; transform:translateY(-1px) !important; }

    [data-testid="stExpander"] { background:#1a2f4a !important; border:1px solid #1e3a5f !important; border-radius:8px !important; }

    hr { border-color:#1e3a5f !important; }
    h1,h2,h3 { color:#e8eaf0 !important; }
    p,li { color:#b0bec5; }
    code { background:#1a2f4a !important; color:#2196f3 !important; border-radius:4px; padding:2px 6px; }

    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:#0d1b2a; }
    ::-webkit-scrollbar-thumb { background:#2196f3; border-radius:3px; }

    .badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; margin-right:6px; }
    .badge-text  { background:#1565c0; color:#90caf9; }
    .badge-table { background:#1b5e20; color:#a5d6a7; }
    .badge-image { background:#4a148c; color:#ce93d8; }

    .ready-banner { text-align:center; padding:60px 20px 30px; }
    .ready-banner h2 { font-size:2rem !important; background:linear-gradient(135deg,#2196f3,#42a5f5); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px !important; }
    .ready-banner p  { color:#546e7a; font-size:1rem; }

    /*
     * Author credit bar — pinned to the absolute bottom of the sidebar.
     * Uses position:fixed so it stays put regardless of sidebar scroll height.
     * Everything (name + both links) is on a single flex row.
     */
    .author-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 244px;           /* matches Streamlit's default sidebar width */
        background: #0d1b2a;
        border-top: 1px solid #1e3a5f;
        padding: 8px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 999;
        box-sizing: border-box;
    }
    .author-bar .name {
        color: #90caf9;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }
    .author-bar a {
        color: #5ba3d9;
        text-decoration: none;
        font-size: 11px;
        display: flex;
        align-items: center;
        gap: 3px;
        white-space: nowrap;
    }
    .author-bar a:hover { color: #42a5f5; }
    .author-bar svg { flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ── Session initialisation ─────────────────────────────────────────────────

embedder = load_models()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Advanced RAG")
    st.markdown("---")
    st.markdown("### 📂 Upload Documents")
    st.caption("Supported: PDF · DOCX · TXT")

    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt"],
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
            cc = PersistentClient(path=config.CHROMA_DIR)
            for c in cc.list_collections():
                cc.delete_collection(c.name)
        except Exception:
            pass
        for folder in [config.UPLOAD_DIR, config.IMAGES_DIR,
                       config.MARKDOWN_DIR, config.CHROMA_DIR]:
            shutil.rmtree(folder, ignore_errors=True)
            Path(folder).mkdir(parents=True, exist_ok=True)
        try:
            import chromadb
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        uploader_key = st.session_state.uploader_key + 1
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.uploader_key = uploader_key
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:11px;color:#546e7a;text-align:center'>Docling · ChromaDB · Groq · BGE</p>",
        unsafe_allow_html=True
    )

    # Author credit — fixed to the absolute bottom of the sidebar, single row
    st.markdown("""
    <div class="author-bar">
        <span class="name">Charu Nethra Sekar</span>
        <a href="https://www.linkedin.com/in/charu-nethra-sekar" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="#0A66C2">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
            </svg>LinkedIn
        </a>
        <a href="https://github.com/charunethra" target="_blank">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="#e8eaf0">
                <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
            </svg>GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)


# ── Message renderer ───────────────────────────────────────────────────────

def render_message(role, content, sources=None):
    if role == "user":
        st.markdown(
            f"<div class='user-bubble'><div class='user-bubble-inner'>{content}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='assistant-bubble'><div class='assistant-bubble-inner'>{content}</div></div>",
            unsafe_allow_html=True
        )
        if sources:
            display_sources = filter_display_sources(sources)
            if display_sources:
                with st.expander(f"📚 {len(display_sources)} source(s) used"):
                    for score, doc in display_sources:
                        meta      = doc.metadata
                        ctype     = meta.get("chunk_type", "text")
                        ref_label = meta.get("figure_label", "") or meta.get("table_label", "")
                        section   = meta.get("h1") or meta.get("h2") or "—"
                        extra     = f" · {ref_label}" if ref_label else ""
                        st.markdown(
                            f"<span class='badge badge-{ctype}'>{ctype.upper()}</span>"
                            f"<strong>{meta.get('source_file','?')}</strong>{extra} — "
                            f"section: <code>{section}</code> | score: <code>{score:.2f}</code>",
                            unsafe_allow_html=True
                        )
                        st.caption(doc.page_content[:280] + "…")
                        if ctype == "image":
                            img_path = meta.get("image_path", "")
                            if img_path and Path(img_path).exists():
                                st.image(img_path)
                        st.divider()


# ── Welcome screen ─────────────────────────────────────────────────────────

def show_welcome():
    st.markdown("""
    <div style='text-align:center;padding:80px 20px 40px;'>
        <h1 style='font-size:3rem;background:linear-gradient(135deg,#2196f3,#42a5f5);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px;'>
        Advanced RAG</h1>
        <p style='color:#546e7a;font-size:1.1rem;'>Upload your documents and ask questions</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    def card(icon, title, body):
        return (
            f"<div style='background:#1a2f4a;border:1px solid #1e3a5f;border-radius:12px;"
            f"padding:24px;text-align:center;'>"
            f"<div style='font-size:2rem;margin-bottom:8px'>{icon}</div>"
            f"<h3 style='color:#e8eaf0;margin:0 0 8px'>{title}</h3>"
            f"<p style='color:#546e7a;font-size:13px;margin:0'>{body}</p></div>"
        )
    with c1: st.markdown(card("📄","Multi-format","PDF, DOCX, TXT — scanned PDFs via OCR"), unsafe_allow_html=True)
    with c2: st.markdown(card("🔎","Hybrid Search","BM25 keyword + BGE semantic + reranking"), unsafe_allow_html=True)
    with c3: st.markdown(card("🤖","Groq LLM","Llama 3.3 70B answers strictly from your docs"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload documents in the sidebar to get started")


# ── Main chat area ─────────────────────────────────────────────────────────

if "chroma_collection" not in st.session_state:
    show_welcome()
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("""
        <div class='ready-banner'>
            <h2>✦ Your documents are ready</h2>
            <p>Ask anything — summaries, facts, tables, figures, comparisons</p>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("sources"))

    if prompt := st.chat_input("Type your question here..."):
        render_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("🔍 Searching…"):
            answer, scored_docs, standalone = run_query(prompt, embedder)

        if answer is None:
            msg = (
                "⚠️ No relevant answer found in the uploaded documents. "
                "Try rephrasing, or check that the document was processed correctly."
            )
            render_message("assistant", msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "sources": []})
        else:
            render_message("assistant", answer, scored_docs)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": scored_docs}
            )