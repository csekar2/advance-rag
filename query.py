import re
import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import groq
import config


# ── Model loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_query_models():
    return CrossEncoder(config.RERANKER_MODEL)


# ── Roman numeral conversion ───────────────────────────────────────────────

def _to_arabic(raw: str) -> str:
    try:
        import roman
        return str(roman.fromRoman(raw.upper()))
    except Exception:
        return raw


# ── Standalone question rewriting ─────────────────────────────────────────
# Only rewrites when the question contains back-references (it, they, this…).
# Self-contained questions pass through unchanged.

_REFERENCE_PATTERNS = re.compile(
    r'\b(it|they|them|their|this|that|these|those|he|she|'
    r'the same|mentioned|above|previous|earlier|last|before)\b',
    re.IGNORECASE
)

def _needs_rewrite(query: str, chat_history: list) -> bool:
    return bool(chat_history) and bool(_REFERENCE_PATTERNS.search(query))

def make_standalone_question(query: str, chat_history: list) -> str:
    if not _needs_rewrite(query, chat_history):
        return query
    client       = groq.Groq(api_key=config.GROQ_API_KEY)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-6:]])
    response = client.chat.completions.create(
        model=config.GROQ_STANDALONE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a question rewriter. The question contains pronouns or references "
                    "that depend on the chat history. Rewrite it as a fully self-contained "
                    "question — resolve pronouns only, do NOT change the topic or add any "
                    "information not implied by the original. Return ONLY the rewritten question."
                )
            },
            {"role": "user", "content": f"Chat history:\n{history_text}\n\nQuestion: {query}"}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────
# Merges BM25 and dense results by combining rank positions.

def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    scores = {}; contents = {}; metadatas = {}
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        scores[key]    = scores.get(key, 0) + 1 / (rank + k)
        contents[key]  = doc.page_content
        metadatas[key] = doc.metadata
    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        scores[key]    = scores.get(key, 0) + 1 / (rank + k)
        contents[key]  = doc.page_content
        metadatas[key] = doc.metadata
    top = sorted(scores, key=lambda x: scores[x], reverse=True)[:config.RETRIEVAL_TOP_K]
    return [Document(page_content=contents[k], metadata=metadatas[k]) for k in top]


# ── Query normalisation ────────────────────────────────────────────────────
# Detects figure/table references in the query and expands them in both
# directions (Roman ↔ Arabic) so retrieval matches whichever form the
# document uses. Returns the expanded query string.

def _normalise_query(query: str) -> str:
    additions = []
    for m in re.finditer(
        r'\b(fig(?:ure)?|tab(?:le)?)\.?\s*([IVXLCDM\d]+)\b', query, re.IGNORECASE
    ):
        label  = "Figure" if m.group(1).lower().startswith("fig") else "Table"
        raw    = m.group(2)
        arabic = _to_arabic(raw)
        if arabic != raw:
            additions.append(f"{label} {arabic}")
        else:
            try:
                import roman
                additions.append(f"{label} {roman.toRoman(int(raw))}")
            except Exception:
                pass
    return (query + " " + " ".join(additions)).strip() if additions else query


# ── Direct table/figure lookup ─────────────────────────────────────────────
# When the user directly asks about a specific table or figure (e.g. "what's
# in Table 4" or "show me Figure III"), we do a targeted metadata lookup in
# ChromaDB — filtering by chunk_type and matching the stored label — in
# addition to the normal hybrid search. This handles cases where Roman
# numerals in the document title don't surface through BM25 keyword matching.

def _extract_direct_ref(query: str):
    """
    Return list of (chunk_type, normalised_label) pairs explicitly mentioned
    in the query. e.g. "Table IV" → [("table","Table 4")],
    "Figure 3 and Table II" → [("image","Figure 3"),("table","Table 2")]
    """
    refs = []
    for m in re.finditer(
        r'\b(fig(?:ure)?|tab(?:le)?)\.?\s*([IVXLCDM\d]+)\b', query, re.IGNORECASE
    ):
        kind   = m.group(1).lower()
        arabic = _to_arabic(m.group(2))
        if kind.startswith("fig"):
            refs.append(("image", f"Figure {arabic}"))
        else:
            refs.append(("table", f"Table {arabic}"))
    return refs

def _direct_lookup(query: str, collection, embedder):
    """
    For each figure/table explicitly named in the query, query ChromaDB
    filtered to that chunk_type and verify the stored label matches.
    Returns a list of matching Document objects (deduplicated).
    """
    refs  = _extract_direct_ref(query)
    seen  = set()
    extra = []
    for chunk_type, label in refs:
        label_field = "figure_label" if chunk_type == "image" else "table_label"
        try:
            results = collection.query(
                query_embeddings=[embedder.encode(label).tolist()],
                n_results=config.RETRIEVAL_TOP_K,
                where={"chunk_type": {"$eq": chunk_type}}
            )
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                stored = meta.get(label_field, "")
                # Match: stored label equals the normalised label
                if stored == label and text not in seen:
                    extra.append(Document(page_content=text, metadata=meta))
                    seen.add(text)
        except Exception:
            pass
    return extra


# ── Cross-reference chunk expansion ───────────────────────────────────────
# When a retrieved text chunk says "see Figure 3" or "as in Table II",
# fetches those referenced chunks from ChromaDB and appends them to context.

def _expand_with_cross_refs(docs, collection, embedder):
    seen  = {d.page_content for d in docs}
    extra = []
    for doc in list(docs):
        meta = doc.metadata
        for label in [r.strip() for r in meta.get("figure_refs", "").split(",") if r.strip()]:
            _fetch_ref_chunks(label, "image", "figure_label", collection, embedder, seen, extra)
        for label in [r.strip() for r in meta.get("table_refs", "").split(",") if r.strip()]:
            _fetch_ref_chunks(label, "table", "table_label", collection, embedder, seen, extra)
    return docs + extra

def _fetch_ref_chunks(label, chunk_type, label_field, collection, embedder, seen, extra):
    try:
        results = collection.query(
            query_embeddings=[embedder.encode(label).tolist()],
            n_results=3,
            where={"chunk_type": {"$eq": chunk_type}}
        )
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            stored = meta.get(label_field, "")
            if stored == label and text not in seen:
                extra.append(Document(page_content=text, metadata=meta))
                seen.add(text)
    except Exception:
        pass


# ── Hybrid retrieval ───────────────────────────────────────────────────────
# 1. Expand query with Roman/Arabic alternates
# 2. BM25 + ChromaDB dense retrieval → merge with RRF
# 3. Prepend any directly named table/figure chunks
# 4. Append cross-referenced chunks from text

def hybrid_retrieve(standalone_question: str, embedder):
    expanded_q   = _normalise_query(standalone_question)
    query_vector = embedder.encode(expanded_q).tolist()

    bm25_results = st.session_state.bm25_retriever.invoke(expanded_q)

    collection = st.session_state.chroma_collection
    dense_raw  = collection.query(
        query_embeddings=[query_vector], n_results=config.RETRIEVAL_TOP_K
    )
    dense_results = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(dense_raw["documents"][0], dense_raw["metadatas"][0])
    ]

    fused = reciprocal_rank_fusion(bm25_results, dense_results)

    # Prepend directly named table/figure chunks so they appear at the top
    direct = _direct_lookup(standalone_question, collection, embedder)
    seen_in_fused = {d.page_content for d in fused}
    direct_new    = [d for d in direct if d.page_content not in seen_in_fused]
    fused = direct_new + fused

    return _expand_with_cross_refs(fused, collection, embedder)


# ── Reranking ──────────────────────────────────────────────────────────────

def rerank(question: str, docs, reranker):
    pairs       = [[question, doc.page_content] for doc in docs]
    scores      = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return scored_docs[:config.RERANKER_TOP_K]


# ── Answer generation ──────────────────────────────────────────────────────

def generate_answer(question: str, scored_docs):
    if scored_docs[0][0] < config.RELEVANCE_THRESHOLD:
        return None, scored_docs

    client  = groq.Groq(api_key=config.GROQ_API_KEY)
    context = ""
    for i, (score, doc) in enumerate(scored_docs):
        meta    = doc.metadata
        section = meta.get("h1") or meta.get("h2") or meta.get("h3") or ""
        ref_lbl = meta.get("figure_label", "") or meta.get("table_label", "")
        label   = (
            f"[Source {i+1} — {meta.get('source_file','?')}"
            + (f" | {section}" if section else "")
            + (f" | {ref_lbl}" if ref_lbl else "")
            + f" | {meta.get('chunk_type','text')}]"
        )
        context += f"\n{label}\n{doc.page_content}\n"

    response = client.chat.completions.create(
        model=config.GROQ_ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise QA assistant. Answer ONLY from the provided context. "
                    "Do not use external knowledge. Give direct, concise answers when the "
                    "information is present. If the answer is not in the context, say: "
                    "'The provided documents do not contain enough information to answer this.'"
                )
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip(), scored_docs


# ── Source display filtering ───────────────────────────────────────────────
# Shows only sources that scored above DISPLAY_SCORE_THRESHOLD.
# Always shows at least one source regardless of score.

def filter_display_sources(scored_docs):
    filtered = [(s, d) for s, d in scored_docs if s >= config.DISPLAY_SCORE_THRESHOLD]
    if not filtered:
        filtered = scored_docs[:1]
    return filtered[:config.MAX_DISPLAY_SOURCES]


# ── Main query pipeline ────────────────────────────────────────────────────

def run_query(user_question: str, embedder):
    chat_history = st.session_state.get("messages", [])
    standalone   = make_standalone_question(user_question, chat_history)
    docs         = hybrid_retrieve(standalone, embedder)
    reranker     = load_query_models()
    scored_docs  = rerank(standalone, docs, reranker)
    answer, scored_docs = generate_answer(standalone, scored_docs)
    return answer, scored_docs, standalone
