import streamlit as st
from ingestion import run_ingestion, load_models
from query import run_query

# ─── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced RAG",
    page_icon="🔍",
    layout="wide"
)

# ─── Load models at startup ───────────────────────────────────────────────
embedder = load_models()

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            run_ingestion(uploaded_files)

    if st.button("🗑️ Clear All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("Advanced RAG System")

# ─── Main chat UI ─────────────────────────────────────────────────────────
st.title("🔍 Advanced RAG")
st.caption("Upload documents and ask questions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if "chroma_collection" not in st.session_state:
        st.warning("Please upload and process documents first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, scored_docs, standalone = run_query(prompt, embedder)

            if answer is None:
                st.warning("Answer not found in the provided documents.")
            else:
                st.markdown(answer)

                # Sources expander
                with st.expander("📚 View Sources"):
                    for i, (score, doc) in enumerate(scored_docs):
                        meta = doc.metadata
                        st.markdown(f"**Source {i+1}** — `{meta.get('source_file', 'unknown')}` | section: `{meta.get('h1', '') or meta.get('h2', 'N/A')}`")
                        st.caption(f"Relevance score: {score:.3f} | Type: {meta.get('chunk_type', 'text')}")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

                        # Show image if chunk is image type
                        if meta.get("chunk_type") == "image" and meta.get("image_path"):
                            st.image(meta["image_path"])

                st.session_state.messages.append({"role": "assistant", "content": answer})

