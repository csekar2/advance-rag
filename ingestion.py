import os
import re
import uuid
import streamlit as st
from pathlib import Path
import google.genai as genai
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


# ── Model loading ──────────────────────────────────────────────────────────
# Cached so the embedding model is only downloaded and loaded once per session.

@st.cache_resource
def load_models():
    return SentenceTransformer(config.EMBEDDING_MODEL)


# ── Session helpers ────────────────────────────────────────────────────────
# Each browser session gets a unique ID so uploaded files and indexes
# are isolated between concurrent users.

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_session_dirs(session_id):
    upload_dir   = Path(config.UPLOAD_DIR)   / session_id
    images_dir   = Path(config.IMAGES_DIR)   / session_id
    markdown_dir = Path(config.MARKDOWN_DIR) / session_id
    for d in [upload_dir, images_dir, markdown_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return upload_dir, images_dir, markdown_dir


# ── Roman numeral conversion ───────────────────────────────────────────────
# Captions in academic documents often use Roman numerals (Table V, Fig. III).
# This helper normalises them to Arabic so retrieval is consistent.

def _to_arabic(raw: str) -> str:
    try:
        import roman
        return str(roman.fromRoman(raw.upper()))
    except Exception:
        return raw  # already Arabic or unrecognised — return unchanged


# ── Document parsing ───────────────────────────────────────────────────────
# Supports PDF (including scanned via OCR), DOCX, and plain TXT.
# Each file's extracted images are saved into their own subfolder so that
# uploading multiple PDFs in one session never mixes their images.

def _is_scanned_pdf(file_path: Path) -> bool:
    try:
        doc  = pymupdf.open(str(file_path))
        text = doc[0].get_text()
        doc.close()
        return len(text.strip()) == 0
    except Exception:
        return False

def parse_document(file_path: Path, images_dir: Path) -> str:
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
            print(f"Image save error: {e}")

    return doc.export_to_markdown()


# ── Caption extraction ─────────────────────────────────────────────────────
# Scans lines around a placeholder to find captions like "Fig. IV." or
# "Table 3:" and returns a normalised label ("Figure 4") and caption text.

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


# ── Gemini image enrichment ────────────────────────────────────────────────
# Replaces Docling's <!-- image --> placeholders with detailed AI descriptions.
# Both Arabic and Roman forms of the figure number are embedded in the text
# so BM25 keyword search matches either form.

def enrich_images(markdown_text: str, images_dir: Path, filename: str) -> str:
    client       = genai.Client(api_key=config.GEMINI_API_KEY)
    file_img_dir = images_dir / Path(filename).stem
    image_files  = sorted(file_img_dir.glob("*.png"))

    if not image_files:
        return markdown_text

    image_index    = 0
    enriched_lines = []
    lines          = markdown_text.split("\n")

    for line_idx, line in enumerate(lines):
        if line.strip() == "<!-- image -->" and image_index < len(image_files):
            img_path              = image_files[image_index]
            figure_label, caption = _extract_caption(lines, line_idx, kind="figure")
            try:
                img_data = img_path.read_bytes()
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=[{
                        "parts": [
                            {"text": (
                                "Describe this image in full detail. Extract all data, values, "
                                "axis labels, legends, titles, and any visible text. "
                                "Be precise with numbers and labels."
                            )},
                            {"inline_data": {"mime_type": "image/png", "data": img_data}}
                        ]
                    }]
                )
                header = f"<!-- IMAGE_START: {img_path.name}"
                if figure_label:
                    header += f" | label={figure_label}"
                if caption:
                    header += f" | caption={caption}"
                header += " -->"
                enriched_lines.append(header)

                if figure_label:
                    raw_num = figure_label.split()[-1]
                    try:
                        import roman
                        roman_str = roman.toRoman(int(raw_num))
                        enriched_lines.append(f"**{figure_label}** (Fig. {roman_str}): {caption}")
                    except Exception:
                        enriched_lines.append(f"**{figure_label}**: {caption}")

                enriched_lines.append(response.text)
                enriched_lines.append("<!-- IMAGE_END -->")
                image_index += 1
            except Exception as e:
                print(f"Gemini error on {img_path.name}: {e}")
                enriched_lines.append(line)
                image_index += 1
        else:
            enriched_lines.append(line)

    ok = sum(1 for l in enriched_lines if "IMAGE_START" in l)
    print(f"[{filename}] Enriched {ok}/{len(image_files)} images")
    return "\n".join(enriched_lines)


def save_markdown(markdown_text: str, markdown_dir: Path, filename: str) -> Path:
    md_path = markdown_dir / f"{Path(filename).stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    return md_path


# ── Markdown line classifiers ──────────────────────────────────────────────

def _is_table_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and s.startswith("|") and s.endswith("|")

def _collect_full_table(lines: list, start: int):
    rows, i = [], start
    while i < len(lines) and _is_table_line(lines[i]):
        rows.append(lines[i])
        i += 1
    return rows, i

def _collect_full_image_block(lines: list, start: int):
    block, i = [], start
    while i < len(lines):
        block.append(lines[i])
        if "<!-- IMAGE_END -->" in lines[i]:
            i += 1
            break
        i += 1
    return block, i

def _parse_image_header(header_line: str):
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


# ── Cross-reference detection ──────────────────────────────────────────────
# Finds all figure and table references inside a text block (e.g. "see Fig. 3"
# or "as shown in Table IV") and stores them as normalised labels in metadata.
# The query pipeline later uses these to fetch the referenced chunks.

def _detect_cross_refs(text: str):
    fig_labels, table_labels = [], []
    for m in re.finditer(r'\bfig(?:ure)?\.?\s*([IVXLCDM\d]+)\b', text, re.IGNORECASE):
        fig_labels.append(f"Figure {_to_arabic(m.group(1))}")
    for m in re.finditer(r'\btab(?:le)?\.?\s*([IVXLCDM\d]+)\b', text, re.IGNORECASE):
        table_labels.append(f"Table {_to_arabic(m.group(1))}")
    return (
        ", ".join(dict.fromkeys(fig_labels)),
        ", ".join(dict.fromkeys(table_labels))
    )


# ── Table caption detection ────────────────────────────────────────────────
# Looks at lines immediately before and after a table block for a caption
# like "TABLE V. Summary of RAG methods" and normalises the number.

def _detect_table_label(context_lines: list):
    pattern = re.compile(r'\b(?:tab(?:le)?\.?)\s*([IVXLCDM\d]+)[\.:\s]*(.*)', re.IGNORECASE)
    for line in context_lines:
        m = pattern.search(line.strip())
        if m:
            return f"Table {_to_arabic(m.group(1))}", m.group(2).strip()
    return "", ""


# ── Chunking ───────────────────────────────────────────────────────────────
# Splits each markdown section into retrieval-sized chunks while preserving:
#   - Tables as atomic units (never split mid-row)
#   - Image blocks as atomic units (IMAGE_START … IMAGE_END together)
#   - Heading context prepended to every sub-chunk for retrieval quality
# All size comparisons use character counts matching RecursiveCharacterTextSplitter.

def chunk_document(markdown_text: str, filename: str, session_id: str) -> list:
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

            # Image block — collect until IMAGE_END, emit as one chunk
            if "<!-- IMAGE_START:" in line:
                block_lines, i    = _collect_full_image_block(lines, i)
                block_text        = "\n".join(block_lines)
                img_fn, fig_lbl, caption = _parse_image_header(line)
                label_prefix      = f"{fig_lbl}: {caption}\n\n" if fig_lbl else ""
                chunk_text        = (
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

            # Table block — collect all contiguous rows, detect caption, emit atomically
            elif _is_table_line(line):
                table_lines, i = _collect_full_table(lines, i)
                ctx_before     = lines[max(0, i - len(table_lines) - 3): i - len(table_lines)]
                ctx_after      = lines[i: min(len(lines), i + 3)]
                tbl_lbl, tbl_cap = _detect_table_label(ctx_before + ctx_after)
                table_text     = "\n".join(table_lines)
                label_prefix   = f"{tbl_lbl}: {tbl_cap}\n\n" if tbl_lbl else ""
                chunk_text     = (
                    f"{heading}\n\n{label_prefix}{table_text}".strip()
                    if heading else f"{label_prefix}{table_text}".strip()
                )
                base_meta = {
                    **metadata,
                    "chunk_type": "table", "table_label": tbl_lbl,
                    "table_caption": tbl_cap, "figure_refs": "", "table_refs": ""
                }

                if len(chunk_text) <= config.CHUNK_SIZE:
                    all_chunks.append({
                        "text": chunk_text, "chunk_type": "table",
                        "metadata": {**base_meta, "chunk_index": f"{block_idx}_{chunk_num}"}
                    })
                    chunk_num += 1
                else:
                    # Large table: repeat the header row(s) in every sub-chunk
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
                            ct = f"{heading}\n\n{label_prefix}{hdr_text}\n" + "\n".join(current_rows)
                            all_chunks.append({
                                "text": ct.strip(), "chunk_type": "table",
                                "metadata": {**base_meta, "chunk_index": f"{block_idx}_{chunk_num}"}
                            })
                            chunk_num    += 1
                            current_rows  = [row]
                        else:
                            current_rows.append(row)
                    if current_rows:
                        ct = f"{heading}\n\n{label_prefix}{hdr_text}\n" + "\n".join(current_rows)
                        all_chunks.append({
                            "text": ct.strip(), "chunk_type": "table",
                            "metadata": {**base_meta, "chunk_index": f"{block_idx}_{chunk_num}"}
                        })
                        chunk_num += 1

            # Text block — accumulate until next table or image, then split if needed
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


# ── Indexing ───────────────────────────────────────────────────────────────
# Builds two complementary indexes from all chunks:
#   BM25  — keyword-based sparse retrieval (stored in session state)
#   ChromaDB — dense vector retrieval with embeddings (persisted to disk)

def index_chunks(chunks: list, session_id: str, embedder) -> int:
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


# ── Master ingestion entry point ───────────────────────────────────────────
# Skips files that were already processed in this session (no duplicates).
# Merges old and new chunks before re-indexing so both indexes always
# cover the full corpus.

def run_ingestion(uploaded_files):
    session_id                           = get_session_id()
    upload_dir, images_dir, markdown_dir = get_session_dirs(session_id)
    embedder                             = load_models()

    existing_chunks = st.session_state.get("_ingested_chunks", [])
    already_named   = {c["metadata"]["source_file"] for c in existing_chunks}
    new_chunks      = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name in already_named:
            st.info(f"⏭️ {uploaded_file.name} already processed — skipping.")
            continue

        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            st.warning(f"⚠️ Unsupported file type '{suffix}' — skipping {uploaded_file.name}.")
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
            status.update(label=f"✓ {uploaded_file.name} → {len(chunks)} chunks", state="complete")

    all_chunks = existing_chunks + new_chunks
    if not all_chunks:
        st.warning("No new files to process.")
        return

    with st.status("Building search indexes...") as status:
        total = index_chunks(all_chunks, session_id, embedder)
        st.session_state["_ingested_chunks"] = all_chunks
        status.update(label=f"Ready — {total} total chunks indexed!", state="complete")