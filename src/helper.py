# src/helper.py
"""
Helper utilities for the PUC (Premier University) RAG chatbot.

Responsibilities:
- Load PDF documents from a folder (DirectoryLoader + PyPDFLoader).
- Optionally reduce documents to minimal metadata (source).
- Split documents into chunks suitable for embedding.
- Provide a helper to obtain HuggingFace embeddings.
- Annotate each chunk with detected topic and extracted entities (using src/topics.py and src/entities.py).
- Provide a single pipeline function `prepare_documents_for_rag` that runs the full flow.

Notes / changes made:
- Use relative imports for topics/entities so imports are stable when running as package.
- Added logging instead of bare prints.
- Added safe error handling for PDF loading.
- Parameterized chunk size / overlap and embedding model name.
- Added `annotate` flag to optionally skip topic/entity annotation (useful during initial indexing).
- Kept code dependency-light and readable.
"""

from typing import List, Optional
import logging
# New community imports to avoid deprecation warnings:
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Relative imports to other src modules
from .topics import detect_topic
from .entities import extract_entities

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_pdf_file(path: str) -> List[Document]:
    """
    Load all PDF files from the given directory into LangChain Document objects.

    Args:
        path: path to the directory containing PDFs (e.g. "./data").

    Returns:
        List[Document]: list of Document objects. Returns an empty list on error or if no PDFs found.
    """
    try:
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            logger.warning("No PDF files found in directory: %s", path)
        else:
            logger.info("Loaded %d PDF documents from %s", len(documents), path)
        return documents
    except Exception as exc:
        logger.exception("Failed to load PDFs from %s: %s", path, exc)
        return []


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Reduce documents to minimal metadata for RAG.
    Keeps 'source' metadata (if available) and original page_content.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        List[Document]: new list of Documents with metadata containing only 'source'.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    logger.info("Converted %d documents to minimal docs", len(minimal_docs))
    return minimal_docs


def text_split(
    extracted_data: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 20
) -> List[Document]:
    """
    Split documents into text chunks for embedding and retrieval.

    Args:
        extracted_data: list of Document objects to split.
        chunk_size: number of characters per chunk (default 500).
        chunk_overlap: overlap between chunks (default 20).

    Returns:
        List[Document]: list of chunked Document objects.
    """
    if not extracted_data:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(extracted_data)
    logger.info("Split into %d text chunks (chunk_size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initialize and return a HuggingFaceEmbeddings instance.

    Args:
        model_name: the HF model name to use for embeddings (default: all-MiniLM-L6-v2).

    Returns:
        HuggingFaceEmbeddings instance.
    """
    logger.info("Initializing HuggingFace embeddings model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def annotate_docs_with_topics_entities(docs: List[Document]) -> List[Document]:
    """
    Annotate each document chunk with:
      - 'topic' detected using src.topics.detect_topic
      - 'entities' list using src.entities.extract_entities

    Adds 'topic' and 'entities' keys to metadata.

    Args:
        docs: list of Document objects (typically chunks).

    Returns:
        List[Document]: new list of Documents with metadata['topic'] and metadata['entities'].
    """
    annotated: List[Document] = []
    for doc in docs:
        try:
            topic = detect_topic(doc.page_content)
        except Exception:
            topic = None
        try:
            entities = extract_entities(doc.page_content)
        except Exception:
            entities = []

        metadata = dict(doc.metadata) if doc.metadata else {}
        metadata["topic"] = topic
        metadata["entities"] = entities
        annotated.append(Document(page_content=doc.page_content, metadata=metadata))

    logger.info("Annotated %d chunks with topics and entities", len(annotated))
    return annotated


def prepare_documents_for_rag(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 20,
    annotate: bool = True
) -> List[Document]:
    """
    Full pipeline to prepare documents for RAG indexing:
      1. Load PDFs from directory
      2. Convert to minimal docs
      3. Split into chunks
      4. (Optional) Annotate with topics and entities

    Args:
        pdf_path: directory containing PDF files.
        chunk_size: chunk size for splitting.
        chunk_overlap: overlap size for splitting.
        annotate: if True, run topic/entity annotation on each chunk.

    Returns:
        List[Document]: list of prepared Document chunks (ready for embedding/indexing).
    """
    docs = load_pdf_file(pdf_path)
    if not docs:
        return []

    minimal = filter_to_minimal_docs(docs)
    chunks = text_split(minimal, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if annotate:
        chunks = annotate_docs_with_topics_entities(chunks)

    logger.info("Prepared %d document chunks for RAG (annotate=%s)", len(chunks), annotate)
    return chunks
