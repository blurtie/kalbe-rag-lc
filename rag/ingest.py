"""Indexing with LangChain: load PDFs -> label pages -> split -> embed into Chroma.

Same steps as the "RAG From Scratch" notebook (Part 2: Indexing), plus page metadata:
  1. PyMuPDFLoader: one Document per PDF page (metadata["page"] is 0-based, we store 1-based).
  2. Label each page main / references / appendix from papers.yaml; drop "references".
  3. RecursiveCharacterTextSplitter (sizes counted in bge-m3 tokens). Splitting is done per
     page, so every chunk belongs to exactly ONE page -> exact page citations.
  4. HuggingFaceEmbeddings(bge-m3) -> Chroma (persisted to data/chroma).
  5. Chunks are also written to data/chunks.jsonl so BM25 can be rebuilt at query time.
"""
import json
import shutil
from collections import Counter

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import load_config, load_manifest, resolve


def page_kind(page_no: int, paper: dict) -> str:
    """main / references / appendix for a 1-based page number. Works in both orders:
    main -> references -> appendix, and main -> appendix -> references (P5)."""
    starts = [(paper.get("references_start_page"), "references"),
              (paper.get("appendix_start_page"), "appendix")]
    kind = "main"
    for start, label in sorted((s, k) for s, k in starts if s is not None):
        if page_no >= start:
            kind = label
    return kind


def make_splitter(cfg: dict) -> RecursiveCharacterTextSplitter:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["embedding"]["model"])
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )


def make_embeddings(cfg: dict):
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    device = cfg["embedding"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, embedding on CPU (slower).")
        device = "cpu"
    return HuggingFaceEmbeddings(
        model_name=cfg["embedding"]["model"],
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # unit vectors -> cosine similarity
    )


def load_and_split(cfg: dict) -> list:
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    keep = set(cfg["chunking"]["index_kinds"])
    splitter = make_splitter(cfg)
    chunks = []

    print(f"{'ID':<4} {'pages':>5} {'main':>5} {'appx':>5} {'refs(dropped)':>14} {'chunks':>7}")
    for paper in load_manifest(cfg):
        pdf_path = raw_dir / f"{paper['id']}.pdf"
        if not pdf_path.exists():
            print(f"{paper['id']:<4} MISSING {pdf_path}")
            continue
        pages = PyMuPDFLoader(str(pdf_path)).load()
        kinds = Counter()
        n_before = len(chunks)
        for page_doc in pages:
            page_no = page_doc.metadata["page"] + 1          # loader is 0-based
            kind = page_kind(page_no, paper)
            kinds[kind] += 1
            if kind not in keep:
                continue
            for i, piece in enumerate(splitter.split_text(page_doc.page_content)):
                chunks.append({
                    "chunk_id": f"{paper['id']}-p{page_no:02d}-{i}",
                    "doc_id": paper["id"],
                    "title": paper["title"],
                    "page": page_no,
                    "kind": kind,
                    "text": piece,
                })
        print(f"{paper['id']:<4} {len(pages):>5} {kinds['main']:>5} {kinds['appendix']:>5} "
              f"{kinds['references']:>14} {len(chunks) - n_before:>7}")
    return chunks


def ingest(cfg: dict) -> None:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    chunks = load_and_split(cfg)

    chunks_file = resolve(cfg["paths"]["chunks_file"])
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with chunks_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    chroma_dir = resolve(cfg["paths"]["chroma_dir"])
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)  # rebuild from scratch so the index always matches chunks.jsonl
    docs = [Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"},
                     id=c["chunk_id"]) for c in chunks]
    Chroma.from_documents(
        documents=docs,
        embedding=make_embeddings(cfg),
        ids=[c["chunk_id"] for c in chunks],
        collection_name="papers",
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"\n{len(chunks)} chunks -> {chunks_file} and Chroma at {chroma_dir}")


if __name__ == "__main__":
    ingest(load_config())
