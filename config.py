import os
from dotenv import load_dotenv

load_dotenv()

# API keys loaded from .env file
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model names — change these to swap providers without touching other files
GROQ_STANDALONE_MODEL = "llama-3.1-8b-instant"   # lightweight, used only for question rewriting
GROQ_ANSWER_MODEL     = "llama-3.3-70b-versatile" # main answering model
GEMINI_MODEL          = "models/gemini-2.5-flash"  # used to describe extracted images
EMBEDDING_MODEL       = "BAAI/bge-base-en-v1.5"   # sentence embedding model for dense retrieval
RERANKER_MODEL        = "BAAI/bge-reranker-base"   # cross-encoder for reranking retrieved chunks

# Chunking — CHUNK_SIZE is in characters (matches RecursiveCharacterTextSplitter)
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 150

# Retrieval — how many candidates to fetch before reranking, and how many to keep after
RETRIEVAL_TOP_K = 20
RERANKER_TOP_K  = 5

# Relevance gate — bge-reranker-base outputs raw logits (not probabilities).
# Positive = relevant, large negative = irrelevant.
# Answers are suppressed only when the top reranker score falls below this value.
# Raise toward 0.0 if you see hallucinated answers on weak context.
RELEVANCE_THRESHOLD = -5.0

# Source display — logit threshold above which a source is shown in the UI.
# Sources scoring below this are hidden to keep the panel clean.
# At least one source is always shown regardless of score.
DISPLAY_SCORE_THRESHOLD = 0.0
MAX_DISPLAY_SOURCES     = 3

# Local storage paths
UPLOAD_DIR   = "data/uploads"
IMAGES_DIR   = "data/images"
MARKDOWN_DIR = "data/markdown"
CHROMA_DIR   = "data/chroma"