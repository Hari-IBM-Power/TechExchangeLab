"""
PDF ingestion, chunking, embedding generation, and ChromaDB storage pipeline.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None
    embedding_functions = None

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_DB_PATH = "./db"
DEFAULT_COLLECTION_NAME = "pdf_documents"
DEFAULT_MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 10


class PDFEmbeddingPipelineError(Exception):
    """Raised for PDF embedding pipeline failures."""


def sanitize_filename(filename: str) -> str:
    """Create a safe filename/id component."""
    base = Path(filename).stem
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
    return cleaned or "document"


def _split_markdown_sections(markdown_text: str) -> List[Tuple[str, str]]:
    """Split markdown into heading-aware sections."""
    lines = markdown_text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_header = "Introduction"
    buffer: List[str] = []

    for line in lines:
        if re.match(r"^#{1,6}\s+", line.strip()):
            if buffer:
                sections.append((current_header, "\n".join(buffer).strip()))
                buffer = []
            current_header = line.strip().lstrip("#").strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_header, "\n".join(buffer).strip()))

    return [(header, body) for header, body in sections if body]


def _split_paragraph_sentences(paragraph: str) -> List[str]:
    """Split paragraph into sentences without cutting mid-sentence aggressively."""
    normalized = re.sub(r"\s+", " ", paragraph).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _build_contextualized_text(document_name: str, section_header: str, chunk_text: str) -> str:
    """Add document and section context to chunk text."""
    return (
        f"Document: {document_name}\n"
        f"Section: {section_header}\n\n"
        f"{chunk_text}"
    )


def convert_pdf_to_markdown(source_dir: str, output_dir: str) -> List[str]:
    """
    Convert all PDFs in a directory to markdown files and return markdown paths.
    """
    _ensure_dependencies()

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists() or not source_path.is_dir():
        raise PDFEmbeddingPipelineError(f"Source directory does not exist: {source_dir}")

    output_path.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    processed_files: List[str] = []

    pdf_files = sorted(source_path.glob("*.pdf"))
    if not pdf_files:
        raise PDFEmbeddingPipelineError(f"No PDF files found in source directory: {source_dir}")

    total_files = len(pdf_files)
    for index, pdf_file in enumerate(pdf_files, start=1):
        print(f"Processing file {index} of {total_files}: {pdf_file.name}")
        try:
            result = converter.convert(str(pdf_file))
            markdown = result.document.export_to_markdown()
            markdown_path = output_path / f"{pdf_file.stem}.md"
            markdown_path.write_text(markdown, encoding="utf-8")
            processed_files.append(str(markdown_path))
            print("  ✅ Converted to markdown")
        except Exception as exc:
            print(f"  ❌ Failed to convert {pdf_file.name}: {exc}")

    return processed_files


def ensure_collection(client: Any, collection_name: str, embedding_function: Any) -> Tuple[Any, str]:
    """
    Get or create a Chroma collection and return the collection with status.
    """
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        return collection, "existing"
    except Exception:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        return collection, "created"

def chunk_document(document_path: str, tokenizer: Any, max_tokens: int = DEFAULT_MAX_TOKENS) -> List[Dict[str, Any]]:
    """
    Load markdown document, apply heading-aware chunking, and add context.
    """
    document = Path(document_path)
    if not document.exists():
        raise PDFEmbeddingPipelineError(f"Document path does not exist: {document_path}")

    markdown_text = document.read_text(encoding="utf-8")
    sections = _split_markdown_sections(markdown_text)
    chunks: List[Dict[str, Any]] = []
    document_name = document.stem

    for section_header, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current_parts: List[str] = []
        current_token_count = 0

        for paragraph in paragraphs:
            paragraph_token_count = len(tokenizer.encode(paragraph, add_special_tokens=False))

            if paragraph_token_count > max_tokens:
                for sentence in _split_paragraph_sentences(paragraph):
                    sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
                    if current_parts and current_token_count + sentence_tokens > max_tokens:
                        chunk_text = "\n\n".join(current_parts).strip()
                        contextualized = _build_contextualized_text(document_name, section_header, chunk_text)
                        chunks.append({
                            "text": chunk_text,
                            "contextualized_text": contextualized,
                            "section_header": section_header,
                            "token_count": current_token_count,
                        })
                        current_parts = [sentence]
                        current_token_count = sentence_tokens
                    else:
                        current_parts.append(sentence)
                        current_token_count += sentence_tokens
                continue

            if current_parts and current_token_count + paragraph_token_count > max_tokens:
                chunk_text = "\n\n".join(current_parts).strip()
                contextualized = _build_contextualized_text(document_name, section_header, chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "contextualized_text": contextualized,
                    "section_header": section_header,
                    "token_count": current_token_count,
                })
                current_parts = [paragraph]
                current_token_count = paragraph_token_count
            else:
                current_parts.append(paragraph)
                current_token_count += paragraph_token_count

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            contextualized = _build_contextualized_text(document_name, section_header, chunk_text)
            chunks.append({
                "text": chunk_text,
                "contextualized_text": contextualized,
                "section_header": section_header,
                "token_count": current_token_count,
            })

    return chunks


def _ensure_dependencies() -> None:
    """Validate optional dependencies are installed before runtime usage."""
    missing = []
    if DocumentConverter is None:
        missing.append("docling")
    if chromadb is None or embedding_functions is None:
        missing.append("chromadb")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")
    if AutoTokenizer is None:
        missing.append("transformers")

    if missing:
        raise PDFEmbeddingPipelineError(
            "Missing required dependencies for PDF embedding pipeline: "
            + ", ".join(missing)
            + ". Install them from requirements.txt."
        )

# Made with Bob


def insert_embeddings(
    chunks: List[Dict[str, Any]],
    collection: Any,
    document_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Insert contextualized chunks into ChromaDB in batches.
    """
    inserted_count = 0
    ids_batch: List[str] = []
    docs_batch: List[str] = []
    meta_batch: List[Dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        ids_batch.append(f"{document_name}_chunk{index}")
        docs_batch.append(chunk["contextualized_text"])
        meta_batch.append({
            "filename": document_name,
            "chunk_index": index,
            "token_count": chunk["token_count"],
            "section_header": chunk["section_header"],
            "original_text": chunk["text"],
        })

        if len(ids_batch) == batch_size:
            collection.add(ids=ids_batch, documents=docs_batch, metadatas=meta_batch)
            inserted_count += len(ids_batch)
            print(f"  💾 Inserted batch of {len(ids_batch)} embeddings")
            ids_batch, docs_batch, meta_batch = [], [], []

    if ids_batch:
        collection.add(ids=ids_batch, documents=docs_batch, metadatas=meta_batch)
        inserted_count += len(ids_batch)
        print(f"  💾 Inserted final batch of {len(ids_batch)} embeddings")

    return {
        "inserted_count": inserted_count,
        "batch_size": batch_size,
    }


def _build_embedding_function(model_name: str) -> Any:
    """Create Chroma sentence-transformer embedding function."""
    if embedding_functions is None:
        raise PDFEmbeddingPipelineError("chromadb embedding functions are unavailable")
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


def _estimate_average_tokens(chunks: List[Dict[str, Any]]) -> float:
    """Compute average token count for chunks."""
    if not chunks:
        return 0.0
    total_tokens = sum(chunk["token_count"] for chunk in chunks)
    return total_tokens / len(chunks)


def process_pdf_file(
    pdf_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
    output_dir: Optional[str] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Process one PDF file end-to-end and store chunk embeddings in ChromaDB.
    """
    _ensure_dependencies()

    source_pdf = Path(pdf_path)
    if not source_pdf.exists() or source_pdf.suffix.lower() != ".pdf":
        raise PDFEmbeddingPipelineError(f"Invalid PDF file path: {pdf_path}")

    markdown_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="pdf_md_"))
    markdown_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    try:
        result = converter.convert(str(source_pdf))
        markdown = result.document.export_to_markdown()
    except Exception as exc:
        raise PDFEmbeddingPipelineError(f"Failed to convert PDF to markdown: {exc}") from exc

    markdown_path = markdown_dir / f"{source_pdf.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ = SentenceTransformer(model_name)
    chunks = chunk_document(str(markdown_path), tokenizer=tokenizer, max_tokens=max_tokens)

    if not chunks:
        raise PDFEmbeddingPipelineError(f"No chunks generated for PDF: {source_pdf.name}")

    print(f"  📊 Generated {len(chunks)} chunks (avg {int(_estimate_average_tokens(chunks))} tokens/chunk)")

    client = chromadb.PersistentClient(path=db_path)
    embedding_function = _build_embedding_function(model_name)
    collection, status = ensure_collection(client, collection_name, embedding_function)

    insertion_stats = insert_embeddings(
        chunks=chunks,
        collection=collection,
        document_name=sanitize_filename(source_pdf.name),
        batch_size=batch_size,
    )

    return {
        "document_name": source_pdf.name,
        "markdown_path": str(markdown_path),
        "collection_name": collection_name,
        "collection_status": status,
        "chunk_count": len(chunks),
        "average_tokens_per_chunk": _estimate_average_tokens(chunks),
        "inserted_count": insertion_stats["inserted_count"],
        "db_path": db_path,
        "embedding_model": model_name,
    }
