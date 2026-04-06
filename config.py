"""
config.py
Central configuration for the Advanced RAG application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys  (loaded from .env — never hard-code these)
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
GROQ_STANDALONE_MODEL = "llama-3.1-8b-instant"    # lightweight rewriter for follow-up questions
GROQ_ANSWER_MODEL     = "llama-3.3-70b-versatile"  # main answering model
GEMINI_MODEL          = "models/gemini-2.5-flash"   # used to describe extracted images
EMBEDDING_MODEL       = "BAAI/bge-base-en-v1.5"    # local sentence embedding model
RERANKER_MODEL        = "BAAI/bge-reranker-base"    # local cross-encoder reranker

# ---------------------------------------------------------------------------
# Chunking
# Chunk sizes are in characters because RecursiveCharacterTextSplitter uses characters.
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# Number of chunks kept at each retrieval stage
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 20
RERANKER_TOP_K  = 5

# ---------------------------------------------------------------------------
# Relevance gate
# Minimum reranker score required before generating an answer
# ---------------------------------------------------------------------------
RELEVANCE_THRESHOLD = -5.0

# ---------------------------------------------------------------------------
# Source panel display
# Controls how many source cards are shown in the UI.
# ---------------------------------------------------------------------------
DISPLAY_SCORE_THRESHOLD = 0.0
MAX_DISPLAY_SOURCES     = 3

# ---------------------------------------------------------------------------
# Gemini rate-limit retry
# Retry settings for Gemini image description on rate limits
# ---------------------------------------------------------------------------
GEMINI_MAX_RETRIES        = 3
GEMINI_RETRY_WAIT_SECONDS = 62   # slightly over 60 s to clear the per-minute window

# ---------------------------------------------------------------------------
# Local storage paths  (session subfolders are created inside each of these)
# ---------------------------------------------------------------------------
UPLOAD_DIR   = "data/uploads"
IMAGES_DIR   = "data/images"
MARKDOWN_DIR = "data/markdown"
CHROMA_DIR   = "data/chroma"
