import streamlit as st
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from spellchecker import SpellChecker
from langchain_core.documents import Document
import groq
import config

# ─── Load models once at startup ──────────────────────────────────────────
@st.cache_resource
def load_query_models():
    reranker = CrossEncoder(config.RERANKER_MODEL)
    return reranker

# ─── Step 1: Spell correction ─────────────────────────────────────────────
def spell_correct(query):
    spell = SpellChecker()
    spell.word_frequency.load_words([
        "RAG", "LLM", "ChromaDB", "BGE", "Docling",
        "Groq", "Llama", "OCR", "BERT", "embeddings"
    ])
    words = query.split()
    corrected = []
    for word in words:
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return " ".join(corrected)

# ─── Step 2: Standalone question ──────────────────────────────────────────
def make_standalone_question(query, chat_history):
    if not chat_history:
        return query

    client = groq.Groq(api_key=config.GROQ_API_KEY)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-10:]])

    response = client.chat.completions.create(
        model=config.GROQ_STANDALONE_MODEL,
        messages=[
            {"role": "system", "content": "Rewrite the question as a fully self-contained question using the chat history. Fix grammar. Do not answer it. Return only the rewritten question."},
            {"role": "user", "content": f"Chat history:\n{history_text}\n\nQuestion: {query}"}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────
def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    scores = {}
    contents = {}
    metadatas = {}

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        contents[key] = doc.page_content
        metadatas[key] = doc.metadata

    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        contents[key] = doc.page_content
        metadatas[key] = doc.metadata

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    top_keys = sorted_keys[:config.RETRIEVAL_TOP_K]
    return [Document(page_content=contents[k], metadata=metadatas[k]) for k in top_keys]

# ─── Step 3: Hybrid Retrieval ─────────────────────────────────────────────
def hybrid_retrieve(standalone_question, embedder):
    query_vector = embedder.encode(standalone_question).tolist()

    # BM25 search
    bm25 = st.session_state.bm25_retriever
    bm25_results = bm25.invoke(standalone_question)

    # Dense search via ChromaDB
    collection = st.session_state.chroma_collection
    dense_raw = collection.query(
        query_embeddings=[query_vector],
        n_results=config.RETRIEVAL_TOP_K
    )
    dense_results = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(
            dense_raw["documents"][0],
            dense_raw["metadatas"][0]
        )
    ]

    # Fuse with RRF
    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    return fused

# ─── Step 4: Reranking ────────────────────────────────────────────────────
def rerank(question, docs, reranker):
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return scored_docs[:config.RERANKER_TOP_K]

# ─── Step 5: Relevance gate + LLM answer ─────────────────────────────────
def generate_answer(question, scored_docs):
    top_score = scored_docs[0][0]

    if top_score < config.RELEVANCE_THRESHOLD:
        return None, scored_docs

    client = groq.Groq(api_key=config.GROQ_API_KEY)

    context = ""
    for i, (score, doc) in enumerate(scored_docs):
        meta = doc.metadata
        source = f"[Source {i+1} - {meta.get('source_file','unknown')} - section: {meta.get('h1','') or meta.get('h2','')}]"
        context += f"\n{source}\n{doc.page_content}\n"

    response = client.chat.completions.create(
        model=config.GROQ_ANSWER_MODEL,
        messages=[
            {"role": "system", "content": "Answer ONLY from the provided context. Do not use external knowledge. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip(), scored_docs

# ─── Master query function ────────────────────────────────────────────────
def run_query(user_question, embedder):
    # Step 1: Spell correct
    corrected = spell_correct(user_question)

    # Step 2: Standalone question
    chat_history = st.session_state.get("messages", [])
    standalone = make_standalone_question(corrected, chat_history)

    # Step 3: Hybrid retrieve
    docs = hybrid_retrieve(standalone, embedder)

    # Step 4: Rerank
    reranker = load_query_models()
    scored_docs = rerank(standalone, docs, reranker)

    # Step 5: Generate answer
    answer, scored_docs = generate_answer(standalone, scored_docs)

    return answer, scored_docs, standalone

