"""
ingestion.py
------------
Handles everything related to document intake:
  1. Startup cleanup  — wipes leftover data from previous sessions on restart.
  2. Document parsing — PDF (with optional OCR), DOCX, and TXT via Docling.
  3. Image enrichment — Gemini describes each extracted image; rate-limit retries included.
  4. Chunking        — splits markdown into text / table / image chunks with full context.
  5. Indexing        — builds a BM25 (keyword) and ChromaDB (dense vector) index together.
  6. Sync            — incrementally adds new files and removes deleted files without
                       re-processing documents that were already indexed.
  7. Full reset      — wipes everything for the Restart button.
"""

import re
import time
import uuid
import shutil
import streamlit as st
from pathlib import Path
import google.genai as genai
from google.genai import errors as genai_errors
import pymupdf
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import config
import chromadb


# ---------------------------------------------------------------------------
# Model loading
# SentenceTransformer is cached so it is only downloaded and loaded once
# per Streamlit server process, regardless of how many sessions run.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    return SentenceTransformer(config.EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Startup disk cleanup
# Called once at the very beginning of each new session.
# Removes every file and folder that was created by previous sessions so
# stale data never leaks into a fresh run.
# ---------------------------------------------------------------------------

def wipe_all_data():
    """Delete all session data from disk and reset ChromaDB's in-memory cache."""
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    try:
        client = PersistentClient(path=config.CHROMA_DIR)
        for col in client.list_collections():
            client.delete_collection(col.name)
    except Exception:
        pass
    for folder in [config.UPLOAD_DIR, config.IMAGES_DIR,
                   config.MARKDOWN_DIR, config.CHROMA_DIR]:
        shutil.rmtree(folder, ignore_errors=True)
        Path(folder).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Session helpers
# Each browser session gets a unique UUID so that concurrent users are
# fully isolated from each other on disk and in ChromaDB.
# ---------------------------------------------------------------------------

def get_session_id():
    """
    Returns the session-unique ID, creating one if this is the first call.
    On first call in a new session, wipe_all_data() is run to clear any
    leftover files from a previous run of the application.
    """
    if "session_id" not in st.session_state:
        wipe_all_data()
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_session_dirs(session_id):
    """Create and return the three per-session data directories."""
    upload_dir   = Path(config.UPLOAD_DIR)   / session_id
    images_dir   = Path(config.IMAGES_DIR)   / session_id
    markdown_dir = Path(config.MARKDOWN_DIR) / session_id
    for d in [upload_dir, images_dir, markdown_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return upload_dir, images_dir, markdown_dir


# ---------------------------------------------------------------------------
# Roman numeral helper
# Academic documents frequently label figures and tables with Roman numerals
# (e.g. "Table V", "Fig. III"). This converts them to Arabic so that both
# "Table 5" and "Table V" index and retrieve identically.
# ---------------------------------------------------------------------------

def _to_arabic(raw: str) -> str:
    try:
        import roman
        return str(roman.fromRoman(raw.upper()))
    except Exception:
        return raw  # already Arabic or not a recognised Roman numeral


# ---------------------------------------------------------------------------
# Document parsing  (PDF, DOCX, TXT)
# Each file's extracted images are saved into their own per-file subfolder
# (images/<session>/<file_stem>/) so that multiple documents never overwrite
# each other's images.
# For PDFs with no selectable text (scanned), Tesseract OCR is enabled.
# ---------------------------------------------------------------------------

def _is_scanned_pdf(file_path: Path) -> bool:
    """Return True if the first page of a PDF has no extractable text."""
    try:
        doc  = pymupdf.open(str(file_path))
        text = doc[0].get_text()
        doc.close()
        return len(text.strip()) == 0
    except Exception:
        return False


def parse_document(file_path: Path, images_dir: Path) -> str:
    """
    Convert a PDF, DOCX, or TXT file to a Markdown string.
    Extracted images are saved as PNGs in a per-file subfolder.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="replace")

    file_images_dir = images_dir / file_path.stem
    file_images_dir.mkdir(parents=True, exist_ok=True)

    pdf_options                         = PdfPipelineOptions()
    pdf_options.images_scale            = 2.0
    pdf_options.generate_page_images    = False
    pdf_options.generate_picture_images = True

    if suffix == ".pdf" and _is_scanned_pdf(file_path):
        pdf_options.do_ocr      = True
        pdf_options.ocr_options = TesseractCliOcrOptions(lang=["eng"])

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF:  PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.DOCX: WordFormatOption(),
        }
    )
    result = converter.convert(str(file_path))
    doc    = result.document

    for pic in doc.pictures:
        try:
            if pic.image and pic.image.pil_image:
                ref      = str(pic.self_ref).replace("/", "_").replace("#", "")
                img_path = file_images_dir / f"{ref}.png"
                pic.image.pil_image.save(str(img_path), format="PNG")
        except Exception as e:
            print(f"[ingestion] Image save error: {e}")

    return doc.export_to_markdown()


# ---------------------------------------------------------------------------
# Caption extraction
# Look around an image/table placeholder and extract a nearby caption.
# ---------------------------------------------------------------------------

def _extract_caption(lines: list, idx: int, kind: str = "figure"):
    if kind == "figure":
        pattern = re.compile(r'\b(?:fig(?:ure)?\.?)\s*([IVXLCDM\d]+)[\.:\s]*(.*)', re.IGNORECASE)
        prefix  = "Figure"
    else:
        pattern = re.compile(r'\b(?:tab(?:le)?\.?)\s*([IVXLCDM\d]+)[\.:\s]*(.*)', re.IGNORECASE)
        prefix  = "Table"

    window = lines[max(0, idx - 6): idx + 4]
    for line in reversed(window):
        m = pattern.search(line.strip())
        if m:
            arabic = _to_arabic(m.group(1).strip())
            return f"{prefix} {arabic}", m.group(2).strip()
    return "", ""


# ---------------------------------------------------------------------------
# Gemini image enrichment
# Describe one image with Gemini and retry on rate limits.
# ---------------------------------------------------------------------------

def _call_gemini_with_retry(client, img_path: Path):
    """
    Ask Gemini to describe a single image.
    Retries on HTTP 429 (rate limit) up to GEMINI_MAX_RETRIES times.
    Returns the description string, or None when quota is exhausted.
    """
    prompt = (
        "Describe this image in full detail. Extract all data, values, "
        "axis labels, legends, titles, and any visible text. "
        "Be precise with numbers and labels."
    )
    for attempt in range(config.GEMINI_MAX_RETRIES):
        try:
            img_data = img_path.read_bytes()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": img_data}}
                    ]
                }]
            )
            return response.text
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < config.GEMINI_MAX_RETRIES - 1:
                    wait = config.GEMINI_RETRY_WAIT_SECONDS
                    print(f"[Gemini] Rate limit — waiting {wait}s (attempt {attempt+1})…")
                    time.sleep(wait)
                else:
                    print(f"[Gemini] Quota exhausted for {img_path.name}.")
                    return None
            else:
                print(f"[Gemini] Error on {img_path.name}: {e}")
                return None
        except Exception as e:
            print(f"[Gemini] Unexpected error on {img_path.name}: {e}")
            return None
    return None


def enrich_images(markdown_text: str, images_dir: Path, filename: str) -> str:
    """
    Replace image placeholders with searchable figure blocks and Gemini descriptions.
    """
    client       = genai.Client(api_key=config.GEMINI_API_KEY)
    file_img_dir = images_dir / Path(filename).stem
    image_files  = sorted(file_img_dir.glob("*.png"))

    if not image_files:
        return markdown_text

    image_index    = 0
    enriched_lines = []
    lines          = markdown_text.split("\n")

    for line_idx, line in enumerate(lines):
        if re.search(r'<!--\s*image\s*-->', line, re.IGNORECASE) and image_index < len(image_files):
            img_path              = image_files[image_index]
            figure_label, caption = _extract_caption(lines, line_idx, kind="figure")
            description           = _call_gemini_with_retry(client, img_path)

            header = f"<!-- IMAGE_START: {img_path.name}"
            if figure_label:
                header += f" | label={figure_label}"
            if caption:
                header += f" | caption={caption}"
            header += " -->"
            enriched_lines.append(header)

            if description is None:
                enriched_lines.append(
                    "[Image description unavailable — Gemini quota reached. "
                    "Re-process this file later to enrich this image.]"
                )
            else:
                if figure_label:
                    raw_num = figure_label.split()[-1]
                    try:
                        import roman
                        roman_str = roman.toRoman(int(raw_num))
                        enriched_lines.append(f"**{figure_label}** (Fig. {roman_str}): {caption}")
                    except Exception:
                        enriched_lines.append(f"**{figure_label}**: {caption}")
                enriched_lines.append(description)

            enriched_lines.append("<!-- IMAGE_END -->")
            image_index += 1
        else:
            enriched_lines.append(line)

    ok = sum(1 for l in enriched_lines if "IMAGE_START" in l)
    print(f"[{filename}] Enriched {ok}/{len(image_files)} images")
    return "\n".join(enriched_lines)


def save_markdown(markdown_text: str, markdown_dir: Path, filename: str) -> Path:
    """Save the enriched markdown to the session's markdown directory."""
    md_path = markdown_dir / f"{Path(filename).stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Identify markdown table rows.
# ---------------------------------------------------------------------------

def _is_table_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and s.startswith("|") and s.endswith("|")


def _collect_full_table(lines: list, start: int):
    """Collect all contiguous table rows from `start`. Returns (rows, next_index)."""
    rows, i = [], start
    while i < len(lines) and _is_table_line(lines[i]):
        rows.append(lines[i])
        i += 1
    return rows, i


def _collect_full_image_block(lines: list, start: int):
    """Collect lines from IMAGE_START to IMAGE_END inclusive. Returns (block, next_index)."""
    block, i = [], start
    while i < len(lines):
        block.append(lines[i])
        if "<!-- IMAGE_END -->" in lines[i]:
            i += 1
            break
        i += 1
    return block, i


def _parse_image_header(header_line: str):
    """Extract (filename, figure_label, caption) from an IMAGE_START comment."""
    img_filename = figure_label = caption = ""
    m = re.search(r'IMAGE_START:\s*([^\s|>]+)', header_line)
    if m:
        img_filename = m.group(1).strip()
    lm = re.search(r'label=([^|>]+)', header_line)
    if lm:
        figure_label = lm.group(1).strip()
    cm = re.search(r'caption=([^|>]+)', header_line)
    if cm:
        caption = cm.group(1).strip()
    return img_filename, figure_label, caption


# ---------------------------------------------------------------------------
# Cross-reference detection
# When a text chunk contains "see Figure 3" or "refer to Table II", those
# labels are stored in chunk metadata so the query pipeline can later fetch
# the actual figure/table chunks and include them in the LLM context.
# ---------------------------------------------------------------------------

def _detect_cross_refs(text: str):
    """
    Return (fig_refs_str, table_refs_str) — comma-separated normalised labels
    of every figure and table referenced in the text.
    """
    fig_labels, table_labels = [], []
    for m in re.finditer(r'\bfig(?:ure)?\.?\s*([IVXLCDM\d]+)\b', text, re.IGNORECASE):
        fig_labels.append(f"Figure {_to_arabic(m.group(1))}")
    for m in re.finditer(r'\btab(?:le)?\.?\s*([IVXLCDM\d]+)\b', text, re.IGNORECASE):
        table_labels.append(f"Table {_to_arabic(m.group(1))}")
    return (
        ", ".join(dict.fromkeys(fig_labels)),
        ", ".join(dict.fromkeys(table_labels))
    )


def _detect_table_label(context_lines: list):
    """
    Look for a table caption in lines immediately around the table block.
    Returns (normalised_label, caption_text) or ("", "").
    Handles both Roman (Table V) and Arabic (Table 5) numbering.
    """
    pattern = re.compile(r'\b(?:tab(?:le)?\.?)\s*([IVXLCDM\d]+)[\.:\s]*(.*)', re.IGNORECASE)
    for line in context_lines:
        m = pattern.search(line.strip())
        if m:
            return f"Table {_to_arabic(m.group(1))}", m.group(2).strip()
    return "", ""


# ---------------------------------------------------------------------------
# Chunking
# Splits each markdown section into retrieval-sized chunks while guaranteeing:
#   - Tables are kept as complete atomic units (never split mid-row).
#   - Image blocks (IMAGE_START…IMAGE_END) are kept together.
#   - The section heading is prepended to every sub-chunk for retrieval quality.
#   - All size comparisons use character counts matching RecursiveCharacterTextSplitter.
#   - Every chunk carries source_file in its metadata for targeted deletion.
# ---------------------------------------------------------------------------

def chunk_document(markdown_text: str, filename: str, session_id: str) -> list:
    """
    Parse the enriched markdown and produce a flat list of chunk dicts.
    Each dict has keys: 'text', 'chunk_type', 'metadata'.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3"),("####","h4")],
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    header_chunks = header_splitter.split_text(markdown_text)

    # Preserve content before the first heading (abstract, preamble, author list etc.)
    # MarkdownHeaderTextSplitter silently drops anything before the first # heading.
    first_header_pos = -1
    for marker in ["# ", "## ", "### ", "#### "]:
        pos = markdown_text.find(f"\n{marker}")
        if pos != -1 and (first_header_pos == -1 or pos < first_header_pos):
            first_header_pos = pos
    if first_header_pos > 50:
        pre_content = markdown_text[:first_header_pos].strip()
        if pre_content:
            from langchain_core.documents import Document as LCDoc
            header_chunks = [LCDoc(
                page_content=pre_content,
                metadata={"h1": "", "h2": "", "h3": "", "h4": ""}
            )] + header_chunks

    if not header_chunks:
        return [{
            "text": t, "chunk_type": "text",
            "metadata": {
                "source_file": filename, "session_id": session_id,
                "chunk_type": "text", "h1": "", "h2": "", "h3": "", "h4": "",
                "figure_refs": "", "table_refs": "", "chunk_index": f"0_{i}"
            }
        } for i, t in enumerate(text_splitter.split_text(markdown_text))]

    all_chunks = []

    for block_idx, block in enumerate(header_chunks):
        content  = block.page_content
        metadata = {**block.metadata, "source_file": filename, "session_id": session_id}
        heading  = (
            metadata.get("h1") or metadata.get("h2") or
            metadata.get("h3") or metadata.get("h4") or ""
        )
        if not content.strip():
            continue

        lines, i, chunk_num = content.split("\n"), 0, 0

        while i < len(lines):
            line = lines[i]

            # --- Image block: collect IMAGE_START…IMAGE_END as one unit ---
            if "<!-- IMAGE_START:" in line:
                block_lines, i           = _collect_full_image_block(lines, i)
                block_text               = "\n".join(block_lines)
                img_fn, fig_lbl, caption = _parse_image_header(line)
                label_prefix = f"{fig_lbl}: {caption}\n\n" if fig_lbl else ""
                chunk_text   = (
                    f"{heading}\n\n{label_prefix}{block_text}".strip()
                    if heading else f"{label_prefix}{block_text}".strip()
                )
                all_chunks.append({
                    "text": chunk_text, "chunk_type": "image",
                    "metadata": {
                        **metadata,
                        "chunk_type": "image", "figure_label": fig_lbl,
                        "figure_caption": caption, "figure_refs": "", "table_refs": "",
                        "image_path": str(
                            Path(config.IMAGES_DIR) / session_id / Path(filename).stem / img_fn
                        ) if img_fn else "",
                        "chunk_index": f"{block_idx}_{chunk_num}"
                    }
                })
                chunk_num += 1

            # --- Table block: collect all contiguous rows as one unit ---
            elif _is_table_line(line):
                table_lines, i = _collect_full_table(lines, i)
                ctx_before     = lines[max(0, i - len(table_lines) - 3): i - len(table_lines)]
                ctx_after      = lines[i: min(len(lines), i + 3)]
                tbl_lbl, tbl_cap = _detect_table_label(ctx_before + ctx_after)
                table_text   = "\n".join(table_lines)
                label_prefix = f"{tbl_lbl}: {tbl_cap}\n\n" if tbl_lbl else ""
                chunk_text   = (
                    f"{heading}\n\n{label_prefix}{table_text}".strip()
                    if heading else f"{label_prefix}{table_text}".strip()
                )
                base_meta = {
                    **metadata,
                    "chunk_type": "table", "table_label": tbl_lbl,
                    "table_caption": tbl_cap, "figure_refs": "", "table_refs": ""
                }

                if len(chunk_text) <= config.CHUNK_SIZE:
                    # Table fits in one chunk — emit as-is
                    all_chunks.append({
                        "text": chunk_text, "chunk_type": "table",
                        "metadata": {**base_meta, "chunk_index": f"{block_idx}_{chunk_num}"}
                    })
                    chunk_num += 1
                else:
                    # Large table: repeat header rows in every sub-chunk so
                    # each sub-chunk is self-contained and searchable
                    hdr_rows, sep_rows, data_rows = [], [], []
                    found_sep = False
                    for tl in table_lines:
                        if not found_sep and re.match(r'\|[\s\-:|]+\|', tl):
                            sep_rows.append(tl)
                            found_sep = True
                        elif not found_sep:
                            hdr_rows.append(tl)
                        else:
                            data_rows.append(tl)
                    hdr_text, current_rows = "\n".join(hdr_rows + sep_rows), []
                    for row in data_rows:
                        candidate = (
                            f"{heading}\n\n{label_prefix}{hdr_text}\n"
                            + "\n".join(current_rows + [row])
                        )
                        if len(candidate) > config.CHUNK_SIZE and current_rows:
                            ct = (f"{heading}\n\n{label_prefix}{hdr_text}\n"
                                  + "\n".join(current_rows))
                            all_chunks.append({
                                "text": ct.strip(), "chunk_type": "table",
                                "metadata": {**base_meta, "chunk_index": f"{block_idx}_{chunk_num}"}
                            })
                            chunk_num    += 1
                            current_rows  = [row]
                        else:
                            current_rows.append(row)
                    if current_rows:
                        ct = (f"{heading}\n\n{label_prefix}{hdr_text}\n"
                              + "\n".join(current_rows))
                        all_chunks.append({
                            "text": ct.strip(), "chunk_type": "table",
                            "metadata": {**base_meta, "chunk_index": f"{block_idx}_{chunk_num}"}
                        })
                        chunk_num += 1

            # --- Text block: accumulate until next table or image ---
            else:
                text_run = []
                while (
                    i < len(lines)
                    and not _is_table_line(lines[i])
                    and "<!-- IMAGE_START:" not in lines[i]
                ):
                    text_run.append(lines[i])
                    i += 1
                text_content = "\n".join(text_run).strip()
                if not text_content:
                    continue
                fig_refs, tbl_refs = _detect_cross_refs(text_content)
                full_text = f"{heading}\n\n{text_content}".strip() if heading else text_content
                splits = (
                    text_splitter.split_text(full_text)
                    if len(full_text) > config.CHUNK_SIZE else [full_text]
                )
                for split_text in splits:
                    if not split_text.strip():
                        continue
                    all_chunks.append({
                        "text": split_text, "chunk_type": "text",
                        "metadata": {
                            **metadata,
                            "chunk_type": "text", "figure_refs": fig_refs,
                            "table_refs": tbl_refs, "chunk_index": f"{block_idx}_{chunk_num}"
                        }
                    })
                    chunk_num += 1

    print(
        f"[{filename}] {len(all_chunks)} chunks — "
        f"text={sum(1 for c in all_chunks if c['chunk_type']=='text')} "
        f"table={sum(1 for c in all_chunks if c['chunk_type']=='table')} "
        f"image={sum(1 for c in all_chunks if c['chunk_type']=='image')}"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Indexing
# Builds two complementary indexes from the supplied chunk list:
#   BM25     — keyword-based sparse retrieval, stored in session state (RAM).
#   ChromaDB — dense vector retrieval, persisted to disk for durability.
# Both indexes always contain chunks from ALL currently loaded documents.
# ---------------------------------------------------------------------------

def index_chunks(chunks: list, session_id: str, embedder) -> int:
    """Embed all chunks and write them to BM25 and ChromaDB. Returns chunk count."""
    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    bm25   = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = config.RETRIEVAL_TOP_K
    st.session_state.bm25_retriever = bm25

    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    chroma_client   = PersistentClient(path=config.CHROMA_DIR)
    collection_name = f"session_{session_id}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(collection_name)
    collection.add(
        documents=texts, embeddings=embeddings.tolist(),
        metadatas=metadatas, ids=[f"chunk_{i}" for i in range(len(texts))]
    )
    st.session_state.chroma_collection = collection
    return len(chunks)


# ---------------------------------------------------------------------------
# Per-document disk cleanup
# Removes exactly the files belonging to one document:
#   - The original uploaded file
#   - Its enriched markdown file
#   - Its per-file image subfolder (and the subfolder itself)
# Called when a document is removed from the uploader widget.
# ---------------------------------------------------------------------------

def _delete_file_data(session_id: str, filename: str):
    """Delete all on-disk artefacts for a single document."""
    stem         = Path(filename).stem
    upload_file  = Path(config.UPLOAD_DIR)   / session_id / filename
    markdown_file = Path(config.MARKDOWN_DIR) / session_id / f"{stem}.md"
    images_subdir = Path(config.IMAGES_DIR)   / session_id / stem

    for f in [upload_file, markdown_file]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass

    # Remove the image subfolder entirely (not just its contents)
    shutil.rmtree(images_subdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Document sync  (core entry point called from app.py)
# Compares the current uploader file list against the already-indexed set.
#   • New files      → parsed, enriched, chunked; index rebuilt with all chunks.
#   • Removed files  → disk data deleted, chunks dropped; index rebuilt without them.
#   • Unchanged files → their existing chunks are reused with NO re-processing.
# This ensures that adding a 3rd document never re-parses documents 1 and 2,
# and that removing document 2 keeps documents 1 and 3 intact.
# ---------------------------------------------------------------------------

def sync_documents(current_files, embedder, session_id, upload_dir, images_dir, markdown_dir):
    """
    Incrementally keep the index in sync with the current file list.
    Returns the total chunk count after the operation.
    """
    current_names   = {f.name for f in current_files}
    existing_chunks = st.session_state.get("_ingested_chunks", [])
    existing_names  = {c["metadata"]["source_file"] for c in existing_chunks}

    # Step 1: handle removals — delete disk data and drop chunks
    removed = existing_names - current_names
    if removed:
        for name in removed:
            _delete_file_data(session_id, name)
        existing_chunks = [
            c for c in existing_chunks
            if c["metadata"]["source_file"] not in removed
        ]
        st.session_state["_ingested_chunks"] = existing_chunks

    # Step 2: handle additions — process only files not yet indexed
    to_add     = [f for f in current_files if f.name not in existing_names]
    new_chunks = []

    for uploaded_file in to_add:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            st.warning(f"⚠️ Unsupported type '{suffix}' — skipping {uploaded_file.name}.")
            continue

        with st.status(f"Processing {uploaded_file.name}...") as status:
            file_path = upload_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getvalue())

            status.update(label=f"Parsing {uploaded_file.name}...")
            markdown_text = parse_document(file_path, images_dir)

            if suffix != ".txt":
                status.update(label=f"Enriching images in {uploaded_file.name}...")
                markdown_text = enrich_images(markdown_text, images_dir, uploaded_file.name)

            save_markdown(markdown_text, markdown_dir, uploaded_file.name)

            status.update(label=f"Chunking {uploaded_file.name}...")
            chunks = chunk_document(markdown_text, uploaded_file.name, session_id)
            new_chunks.extend(chunks)
            status.update(
                label=f"✓ {uploaded_file.name} — {len(chunks)} chunks",
                state="complete"
            )

    # Step 3: merge surviving old chunks with any newly produced chunks
    all_chunks = existing_chunks + new_chunks

    if not all_chunks:
        # Every document was removed — clear indexes entirely
        st.session_state.pop("bm25_retriever",   None)
        st.session_state.pop("chroma_collection", None)
        st.session_state["_ingested_chunks"] = []
        return 0

    # Rebuild both indexes only when something actually changed
    if removed or new_chunks:
        with st.status("Rebuilding search index...") as status:
            total = index_chunks(all_chunks, session_id, embedder)
            st.session_state["_ingested_chunks"] = all_chunks
            status.update(label=f"Ready — {total} chunks indexed!", state="complete")
        return total

    return len(all_chunks)


# ---------------------------------------------------------------------------
# Full reset
# Used by the Restart button. Wipes all disk data, clears ChromaDB's cache,
# and resets session state so the app returns to the welcome screen.
# ---------------------------------------------------------------------------

def full_reset():
    """Wipe everything — disk, memory, and session state."""
    wipe_all_data()
    uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.session_state.clear()
    st.session_state["uploader_key"] = uploader_key