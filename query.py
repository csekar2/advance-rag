"""
query.py
--------
Handles everything related to answering a user question:
  1. Standalone rewriting — resolves pronouns in follow-up questions.
  2. Query normalisation — expands figure/table Roman↔Arabic numeral variants.
  3. Direct lookup      — exact metadata match for named tables/figures.
  4. Hybrid retrieval   — BM25 (keyword) + ChromaDB dense, merged via RRF.
  5. Cross-ref expansion — fetches figure/table chunks referenced in text chunks.
  6. Reranking          — cross-encoder scores each (question, chunk) pair.
  7. Answer generation  — LLM synthesises an answer from the top-K chunks.
  8. Error handling     — user-friendly messages for all API failures.
"""

import re
import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import groq
import config


# ---------------------------------------------------------------------------
# Model loading
# The cross-encoder reranker is cached so it is only loaded once per
# Streamlit server process.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_query_models():
    return CrossEncoder(config.RERANKER_MODEL)


# ---------------------------------------------------------------------------
# Roman numeral conversion  (shared with ingestion.py)
# ---------------------------------------------------------------------------

def _to_arabic(raw: str) -> str:
    try:
        import roman
        return str(roman.fromRoman(raw.upper()))
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# Standalone question rewriting
# Follow-up questions like "What does it show?" need rewriting before retrieval
# ---------------------------------------------------------------------------

_REFERENCE_PATTERNS = re.compile(
    r'\b(it|they|them|their|this|that|these|those|he|she|'
    r'the same|mentioned|above|previous|earlier|last|before)\b',
    re.IGNORECASE
)


def _needs_rewrite(query: str, chat_history: list) -> bool:
    """Return True only when there is prior history AND the question has a pronoun reference."""
    return bool(chat_history) and bool(_REFERENCE_PATTERNS.search(query))


def make_standalone_question(query: str, chat_history: list):
    """
    Returns (standalone_question, error_message).
    On success, error_message is None.
    On API failure, standalone_question is None and error_message is a
    user-friendly string starting with ❌.
    """
    if not _needs_rewrite(query, chat_history):
        return query, None

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-6:]])
    try:
        client   = groq.Groq(api_key=config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=config.GROQ_STANDALONE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a question rewriter. The question contains pronouns or "
                        "references that depend on the chat history. Rewrite it as a fully "
                        "self-contained question — resolve pronouns only, do NOT change the "
                        "topic or add any information not implied by the original. "
                        "Return ONLY the rewritten question."
                    )
                },
                {"role": "user", "content": f"Chat history:\n{history_text}\n\nQuestion: {query}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip(), None

    except groq.RateLimitError:
        # Rewriter rate-limited — fall back to the original question silently
        return query, None

    except groq.AuthenticationError:
        return None, "❌ Groq API key is invalid or missing. Please check your .env file."

    except groq.APIStatusError as e:
        if "quota" in str(e).lower() or "exceeded" in str(e).lower():
            return query, None   # quota hit on lightweight rewriter — degrade gracefully
        return None, f"❌ Groq API error during question rewriting: {e}"

    except Exception:
        return query, None   # any unexpected error — use the original question


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# Merges the ranked lists from BM25 and dense retrieval into a single
# unified ranking.  A chunk that appears high in both lists receives a
# higher combined score than one that only appears in one list.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Query normalisation
# When the user writes "Table V" or "Fig. III", the query is expanded to
# also include "Table 5" / "Figure 3" (and vice-versa for Arabic→Roman).
# This ensures BM25 and dense retrieval can match regardless of whether
# the document stores the label in Roman or Arabic form.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Direct table / figure lookup
# When the user explicitly names a table or figure ("what's in Table 4?"),
# a targeted ChromaDB query filtered by chunk_type and matching the stored
# label guarantees the chunk is found even if BM25 keyword matching misses.
# Results are prepended ahead of the hybrid-search results so the reranker
# always sees them.
# ---------------------------------------------------------------------------

def _extract_direct_ref(query: str):
    """
    Parse all figure/table references from the query.
    Returns a list of (chunk_type, normalised_label) tuples.
    Example: "Table IV and Figure 2" → [("table","Table 4"), ("image","Figure 2")]
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
    For each explicitly named figure/table, do a metadata-filtered ChromaDB
    query and return any chunk whose stored label matches exactly.
    """
    refs = _extract_direct_ref(query)
    seen, extra = set(), []
    for chunk_type, label in refs:
        label_field = "figure_label" if chunk_type == "image" else "table_label"
        try:
            results = collection.query(
                query_embeddings=[embedder.encode(label).tolist()],
                n_results=config.RETRIEVAL_TOP_K,
                where={"chunk_type": {"$eq": chunk_type}}
            )
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                if meta.get(label_field, "") == label and text not in seen:
                    extra.append(Document(page_content=text, metadata=meta))
                    seen.add(text)
        except Exception:
            pass
    return extra


# ---------------------------------------------------------------------------
# Cross-reference expansion
# Retrieved text chunks often contain phrases like "see Figure 3" or
# "as shown in Table II".  For each such reference we fetch the actual
# figure / table chunk from ChromaDB and add it to the candidate pool so
# the LLM has the full content, not just the text that mentions it.
# ---------------------------------------------------------------------------

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
    """Fetch up to 3 chunks of the given type whose label matches exactly."""
    try:
        results = collection.query(
            query_embeddings=[embedder.encode(label).tolist()],
            n_results=3,
            where={"chunk_type": {"$eq": chunk_type}}
        )
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta.get(label_field, "") == label and text not in seen:
                extra.append(Document(page_content=text, metadata=meta))
                seen.add(text)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hybrid retrieval
# Combines BM25 (keyword) and ChromaDB dense (semantic) retrieval:
#   1. Expand the query with Roman/Arabic numeral alternates.
#   2. Run BM25 and dense retrieval in parallel.
#   3. Merge results with Reciprocal Rank Fusion.
#   4. Prepend any directly named table/figure chunks (exact metadata match).
#   5. Append cross-referenced chunks found inside retrieved text chunks.
# The single flat index contains chunks from ALL loaded documents, so a
# general question naturally draws from whichever files are most relevant.
# ---------------------------------------------------------------------------

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

    direct        = _direct_lookup(standalone_question, collection, embedder)
    seen_in_fused = {d.page_content for d in fused}
    fused         = [d for d in direct if d.page_content not in seen_in_fused] + fused

    return _expand_with_cross_refs(fused, collection, embedder)


# ---------------------------------------------------------------------------
# Reranking
# The cross-encoder scores every (question, chunk) pair more accurately than
# the embedding cosine similarity used during retrieval.  Only the top-K
# chunks survive and are passed to the LLM.
# ---------------------------------------------------------------------------

def rerank(question: str, docs, reranker):
    pairs       = [[question, doc.page_content] for doc in docs]
    scores      = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return scored_docs[:config.RERANKER_TOP_K]


# ---------------------------------------------------------------------------
# Answer generation
# Generate an answer strictly from the top reranked chunks
# ---------------------------------------------------------------------------

def generate_answer(question: str, scored_docs):
    if scored_docs[0][0] < config.RELEVANCE_THRESHOLD:
        return None, scored_docs

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

    try:
        client   = groq.Groq(api_key=config.GROQ_API_KEY)
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

    except groq.RateLimitError:
        return (
            "⏳ The Groq API rate limit has been reached. "
            "Please wait a moment and try your question again.",
            scored_docs
        )

    except groq.AuthenticationError:
        return (
            "❌ Groq API authentication failed. "
            "Please check that your GROQ_API_KEY in the .env file is correct.",
            scored_docs
        )

    except groq.APIStatusError as e:
        if "quota" in str(e).lower() or "exceeded" in str(e).lower():
            return (
                "🚫 Your Groq API daily quota has been exhausted. "
                "Quota resets at midnight UTC. Upgrade at console.groq.com for a higher limit.",
                scored_docs
            )
        return (f"❌ Unexpected Groq error: {e}", scored_docs)

    except Exception as e:
        return (f"❌ Unexpected error: {e}", scored_docs)


# ---------------------------------------------------------------------------
# Source display filtering
# Shows only the source cards whose reranker logit exceeds the display
# threshold, capped at MAX_DISPLAY_SOURCES.  At least one card is always
# shown so the user can see where the answer came from.
# ---------------------------------------------------------------------------

def filter_display_sources(scored_docs):
    filtered = [(s, d) for s, d in scored_docs if s >= config.DISPLAY_SCORE_THRESHOLD]
    if not filtered:
        filtered = scored_docs[:1]
    return filtered[:config.MAX_DISPLAY_SOURCES]


# ---------------------------------------------------------------------------
# Main query pipeline
# Orchestrates all steps in order and returns the final answer.
# ---------------------------------------------------------------------------

def run_query(user_question: str, embedder):
    chat_history = st.session_state.get("messages", [])

    standalone, rewrite_err = make_standalone_question(user_question, chat_history)
    if standalone is None:
        return rewrite_err, [], user_question

    docs        = hybrid_retrieve(standalone, embedder)
    reranker    = load_query_models()
    scored_docs = rerank(standalone, docs, reranker)
    answer, scored_docs = generate_answer(standalone, scored_docs)
    return answer, scored_docs, standalone
