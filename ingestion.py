import os
import uuid
import streamlit as st
from pathlib import Path
import google.genai as genai
import pymupdf
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import config


# ─── Step 1: imports, load models, session setup ──────────────────────────────────────
# ─── Load models once at startup ───────────────────────────────────────────
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return embedder

# ─── Session setup ─────────────────────────────────────────────────────────
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# ─── Directory setup ───────────────────────────────────────────────────────
def get_session_dirs(session_id):
    upload_dir = Path(config.UPLOAD_DIR) / session_id
    images_dir = Path(config.IMAGES_DIR) / session_id
    markdown_dir = Path(config.MARKDOWN_DIR) / session_id
    for d in [upload_dir, images_dir, markdown_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return upload_dir, images_dir, markdown_dir

# ─── Step 2: Docling Parsing ───────────────────────────────────────────────
def is_scanned(file_path):
    doc = pymupdf.open(str(file_path))
    text = doc[0].get_text()
    doc.close()
    return len(text.strip()) == 0

def parse_document(file_path, images_dir):
    if is_scanned(file_path):
        ocr_options = TesseractCliOcrOptions(lang=["eng"])
        pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
    else:
        converter = DocumentConverter()

    result = converter.convert(str(file_path))
    markdown_text = result.document.export_to_markdown()
    return markdown_text

# ─── Step 3: Gemini Image Enrichment ──────────────────────────────────────
def enrich_images(markdown_text, images_dir, filename):
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    lines = markdown_text.split("\n")
    enriched_lines = []

    for line in lines:
        if line.startswith("![") and "](images/" in line:
            try:
                img_path = line.split("](")[1].rstrip(")")
                full_img_path = images_dir / img_path.split("/")[-1]
                if full_img_path.exists():
                    img_data = full_img_path.read_bytes()
                    caption = line.split("![")[1].split("]")[0]
                    prompt = f"Describe this image. Extract all data, values, axis labels, legends, titles. Context: {caption}. Be specific with all numbers and labels."
                    response = client.models.generate_content(
                        model=config.GEMINI_MODEL,
                        contents=[{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": img_data}}]}]
                    )
                    description = response.text
                    enriched_lines.append(f"<!-- IMAGE_START: {img_path} -->")
                    enriched_lines.append(description)
                    enriched_lines.append("<!-- IMAGE_END -->")
                else:
                    enriched_lines.append(line)
            except Exception as e:
                enriched_lines.append(line)
        else:
            enriched_lines.append(line)

    return "\n".join(enriched_lines)

# ─── Save enriched markdown ────────────────────────────────────────────────
def save_markdown(markdown_text, markdown_dir, filename):
    md_path = markdown_dir / f"{Path(filename).stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path

# ─── Step 4: Chunking ─────────────────────────────────────────────────────
def chunk_document(markdown_text, filename, session_id):
    headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    header_chunks = header_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for i, block in enumerate(header_chunks):
        content = block.page_content
        metadata = block.metadata
        metadata["source_file"] = filename
        metadata["session_id"] = session_id

        # Route by content type
        if "<!-- IMAGE_START:" in content:
            chunks = [{"text": content, "chunk_type": "image", "metadata": {**metadata, "chunk_type": "image"}}]
        elif content.count("|") > 6:
            chunks = [{"text": content, "chunk_type": "table", "metadata": {**metadata, "chunk_type": "table"}}]
        else:
            if len(content) > config.CHUNK_SIZE:
                splits = text_splitter.split_text(content)
            else:
                splits = [content]
            chunks = [{"text": s, "chunk_type": "text", "metadata": {**metadata, "chunk_type": "text"}} for s in splits]

        for j, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = f"{i}_{j}"
            all_chunks.append(chunk)

    return all_chunks

# ─── Step 5: Embed + BM25 + ChromaDB ──────────────────────────────────────
def index_chunks(chunks, session_id, embedder):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # BM25
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = config.RETRIEVAL_TOP_K
    st.session_state.bm25_retriever = bm25

    # Embeddings
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    # ChromaDB
    chroma_client = PersistentClient(path=config.CHROMA_DIR)
    collection_name = f"session_{session_id}"
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    collection = chroma_client.create_collection(collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[f"chunk_{i}" for i in range(len(texts))]
    )
    st.session_state.chroma_collection = collection
    return len(chunks)

# ─── Master ingestion function ─────────────────────────────────────────────
def run_ingestion(uploaded_files):
    session_id = get_session_id()
    upload_dir, images_dir, markdown_dir = get_session_dirs(session_id)
    embedder = load_models()
    all_chunks = []

    for uploaded_file in uploaded_files:
        with st.status(f"Processing {uploaded_file.name}...") as status:
            # Save file
            file_path = upload_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getvalue())
            status.update(label=f"Parsing {uploaded_file.name}...")

            # Parse
            markdown_text = parse_document(file_path, images_dir)
            status.update(label=f"Enriching images in {uploaded_file.name}...")

            # Enrich images
            markdown_text = enrich_images(markdown_text, images_dir, uploaded_file.name)

            # Save markdown
            save_markdown(markdown_text, markdown_dir, uploaded_file.name)
            status.update(label=f"Chunking {uploaded_file.name}...")

            # Chunk
            chunks = chunk_document(markdown_text, uploaded_file.name, session_id)
            all_chunks.extend(chunks)
            status.update(label=f"Done processing {uploaded_file.name} → {len(chunks)} chunks", state="complete")

    # Index everything
    with st.status("Building search indexes...") as status:
        total = index_chunks(all_chunks, session_id, embedder)
        status.update(label=f"Indexed {total} chunks. Ready to query!", state="complete")

