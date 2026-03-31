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
import chromadb


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
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    if is_scanned(file_path):
        ocr_options = TesseractCliOcrOptions(lang=["eng"])
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(file_path))
    doc = result.document

    # Save images using pictures iterator
    for pic in doc.pictures:
        try:
            if pic.image and pic.image.pil_image:
                ref = str(pic.self_ref).replace("/", "_").replace("#", "")
                img_path = images_dir / f"{ref}.png"
                pic.image.pil_image.save(str(img_path), format="PNG")
        except Exception as e:
            print(f"Image save error: {e}")

    markdown_text = doc.export_to_markdown()
    return markdown_text
 

# ─── Step 3: Gemini Image Enrichment ──────────────────────────────────────
def enrich_images(markdown_text, images_dir, filename):
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    # Get all saved images in order
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        return markdown_text
    
    image_index = 0
    enriched_lines = []
    
    for line in markdown_text.split("\n"):
        if line.strip() == "<!-- image -->" and image_index < len(image_files):
            img_path = image_files[image_index]
            try:
                img_data = img_path.read_bytes()
                prompt = "Describe this image in detail. Extract all data, values, axis labels, legends, titles, and text visible. Be specific with all numbers and labels."
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=[{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {
                                "mime_type": "image/png",
                                "data": img_data
                            }}
                        ]
                    }]
                )
                description = response.text
                enriched_lines.append(f"<!-- IMAGE_START: {img_path.name} -->")
                enriched_lines.append(description)
                enriched_lines.append("<!-- IMAGE_END -->")
                image_index += 1
            except Exception as e:
                print(f"Gemini error: {e}")
                enriched_lines.append(line)
                image_index += 1
        else:
            enriched_lines.append(line)
    
    success_count = sum(1 for l in enriched_lines if "IMAGE_START" in l)
    print(f"Successfully enriched {success_count} out of {len(image_files)} images")
    return "\n".join(enriched_lines)

# ─── Save enriched markdown ────────────────────────────────────────────────
def save_markdown(markdown_text, markdown_dir, filename):
    md_path = markdown_dir / f"{Path(filename).stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path

# ─── Step 4: Chunking ─────────────────────────────────────────────────────
def chunk_document(markdown_text, filename, session_id):
    # Handle ALL heading levels — works for any document type
    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4")
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False  # Keep headers in content for context
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    header_chunks = header_splitter.split_text(markdown_text)
    
    # If no headers found at all (plain text doc), treat whole doc as one block
    if not header_chunks:
        header_chunks_text = text_splitter.split_text(markdown_text)
        return [{
            "text": t,
            "chunk_type": "text",
            "metadata": {
                "source_file": filename,
                "session_id": session_id,
                "chunk_type": "text",
                "h1": "", "h2": "", "h3": "", "h4": "",
                "chunk_index": f"0_{i}"
            }
        } for i, t in enumerate(header_chunks_text)]

    all_chunks = []

    for i, block in enumerate(header_chunks):
        content = block.page_content
        metadata = block.metadata.copy()
        
        # Always add file info
        metadata["source_file"] = filename
        metadata["session_id"] = session_id

        # Get the most specific heading available for context
        heading = (
            metadata.get("h1") or
            metadata.get("h2") or
            metadata.get("h3") or
            metadata.get("h4") or ""
        )

        # Skip empty blocks
        if not content.strip():
            continue

        # ── Route by content type ──────────────────────────────────────
        
        # IMAGE block
        if "<!-- IMAGE_START:" in content:
            # Extract image path from comment
            try:
                img_path = content.split("<!-- IMAGE_START:")[1].split("-->")[0].strip()
            except:
                img_path = ""
            
            chunk_meta = {
                **metadata,
                "chunk_type": "image",
                "image_path": str(Path(config.IMAGES_DIR) / session_id / img_path) if img_path else "",
                "chunk_index": f"{i}_0"
            }
            all_chunks.append({
                "text": f"{heading}\n\n{content}".strip() if heading else content,
                "chunk_type": "image",
                "metadata": chunk_meta
            })

        # TABLE block — detected by pipe characters
        elif content.count("|") > 3 or content.count("\n|") > 1:
            # Split large tables into row-aware chunks
            lines = content.split("\n")
            header_rows = []
            data_rows = []
            found_separator = False
            
            for line in lines:
                if "|" in line and "---" in line:
                    found_separator = True
                    header_rows.append(line)
                elif not found_separator:
                    header_rows.append(line)
                else:
                    data_rows.append(line)
            
            table_header = "\n".join(header_rows)
            current_chunk_rows = []
            current_tokens = len(table_header.split())
            chunk_num = 0
            
            for row in data_rows:
                row_tokens = len(row.split())
                if current_tokens + row_tokens > config.CHUNK_SIZE and current_chunk_rows:
                    # Flush current chunk
                    chunk_text = f"{heading}\n\n{table_header}\n" + "\n".join(current_chunk_rows)
                    all_chunks.append({
                        "text": chunk_text.strip(),
                        "chunk_type": "table",
                        "metadata": {
                            **metadata,
                            "chunk_type": "table",
                            "chunk_index": f"{i}_{chunk_num}"
                        }
                    })
                    chunk_num += 1
                    current_chunk_rows = [row]
                    current_tokens = len(table_header.split()) + row_tokens
                else:
                    current_chunk_rows.append(row)
                    current_tokens += row_tokens
            
            # Flush remaining rows
            if current_chunk_rows:
                chunk_text = f"{heading}\n\n{table_header}\n" + "\n".join(current_chunk_rows)
                all_chunks.append({
                    "text": chunk_text.strip(),
                    "chunk_type": "table",
                    "metadata": {
                        **metadata,
                        "chunk_type": "table",
                        "chunk_index": f"{i}_{chunk_num}"
                    }
                })

        # TEXT block
        else:
            # Prepend heading to every text chunk for context
            content_with_heading = f"{heading}\n\n{content}".strip() if heading else content
            
            if len(content_with_heading.split()) > config.CHUNK_SIZE:
                splits = text_splitter.split_text(content_with_heading)
            else:
                splits = [content_with_heading]
            
            for j, split_text in enumerate(splits):
                if not split_text.strip():
                    continue
                all_chunks.append({
                    "text": split_text,
                    "chunk_type": "text",
                    "metadata": {
                        **metadata,
                        "chunk_type": "text",
                        "chunk_index": f"{i}_{j}"
                    }
                })

    print(f"Total chunks created: {len(all_chunks)}")
    text_count = sum(1 for c in all_chunks if c['chunk_type'] == 'text')
    table_count = sum(1 for c in all_chunks if c['chunk_type'] == 'table')
    image_count = sum(1 for c in all_chunks if c['chunk_type'] == 'image')
    print(f"Text: {text_count} | Tables: {table_count} | Images: {image_count}")
    
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
    
    chromadb.api.client.SharedSystemClient.clear_system_cache()
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
        status.update(label=f"Ready to query!", state="complete")

