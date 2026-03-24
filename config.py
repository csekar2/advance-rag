import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
GROQ_STANDALONE_MODEL = "llama-3.1-8b-instant"
GROQ_ANSWER_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "models/gemini-1.5-flash"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
RETRIEVAL_TOP_K = 20
RERANKER_TOP_K = 5
RELEVANCE_THRESHOLD = 0.35

# Local paths
UPLOAD_DIR = "data/uploads"
IMAGES_DIR = "data/images"
MARKDOWN_DIR = "data/markdown"
CHROMA_DIR = "data/chroma"
