"""Retrieval + generation with LangChain (notebook Parts 3-4), plus citations.

Retrievers
  dense  : Chroma.as_retriever(k)                       -- semantic similarity (bge-m3)
  bm25   : BM25Retriever.from_documents(chunks, k)      -- keyword match
  hybrid : EnsembleRetriever([bm25, dense])             -- weighted Reciprocal Rank Fusion (c=60)

Generation
  Retrieved chunks are numbered [C1]..[Cn] in the prompt; the LLM must cite them. After the
  answer, each [Cn] is mapped back to the chunk's paper + page, so every reference shown to
  the user comes from evidence that was actually retrieved. Numbers that were never given
  to the LLM are reported as invalid citations.
"""
import json
import re
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import resolve

SYSTEM_PROMPT = """You answer questions about a fixed collection of research papers.
Rules:
1. Use ONLY the information in the numbered context passages. Do not add facts from outside knowledge.
2. Cite the passage(s) supporting each claim right after it, e.g. [C1] or [C2][C4].
3. If the passages do not contain enough information to answer, say so plainly instead of guessing.
4. Answer in the same language as the question. Be concise."""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context passages:\n\n{context}\n\nQuestion: {question}"),
])

CLOSED_BOOK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer the question about research papers from your own knowledge. "
               "If you do not know, say so. Answer in the same language as the question. Be concise."),
    ("human", "Question: {question}"),
])

CITATION_RE = re.compile(r"\[C(\d+)\]")
WORD_RE = re.compile(r"\w+")


def words_lowercase(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


BM25_TOKENIZERS = {"whitespace": str.split, "words_lowercase": words_lowercase}
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_chunk_docs(cfg: dict) -> list[Document]:
    path = resolve(cfg["paths"]["chunks_file"])
    docs = []
    # Iterate the file object (splits on "\n" only). str.splitlines() would also split on
    # characters like "\x0c" or "" that PDF text contains INSIDE a JSON string.
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    for line in lines:
        c = json.loads(line)
        docs.append(Document(page_content=c["text"],
                             metadata={k: v for k, v in c.items() if k != "text"}))
    return docs


def make_llm(cfg: dict):
    from langchain_ollama import ChatOllama

    g = cfg["generation"]
    return ChatOllama(model=g["model"], temperature=g["temperature"], num_ctx=g["num_ctx"],
                      reasoning=g["reasoning"], base_url=g["host"])


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[C{i}] {d.metadata['doc_id']}: {d.metadata['title']} | p.{d.metadata['page']} | {d.metadata['kind']}\n"
        f"{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def to_hit(rank: int, doc: Document) -> dict:
    """Plain-dict view of a retrieved chunk (used for printing, saving and evaluation)."""
    m = doc.metadata
    return {"rank": rank, "chunk": {"chunk_id": m["chunk_id"], "doc_id": m["doc_id"],
                                    "title": m["title"], "pages": [m["page"]], "kind": m["kind"],
                                    "text": doc.page_content}}


class RagPipeline:
    def __init__(self, cfg: dict, embeddings=None, llm=None):
        from langchain_chroma import Chroma
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        from rag.ingest import make_embeddings

        self.cfg = cfg
        r = cfg["retrieval"]
        cand_k = max(r["candidate_k"], r["top_k"])

        self.vectorstore = Chroma(
            collection_name="papers",
            embedding_function=embeddings or make_embeddings(cfg),
            persist_directory=str(resolve(cfg["paths"]["chroma_dir"])),
        )
        self.retrievers = {
            "dense": self.vectorstore.as_retriever(search_kwargs={"k": cand_k}),
            "bm25": BM25Retriever.from_documents(
                load_chunk_docs(cfg), k=cand_k,
                preprocess_func=BM25_TOKENIZERS[r.get("bm25_tokenizer", "whitespace")]),
        }
        self.retrievers["hybrid"] = EnsembleRetriever(
            retrievers=[self.retrievers["bm25"], self.retrievers["dense"]],
            weights=r["hybrid_weights"],
            id_key="chunk_id",  # fuse by chunk id, not by text content
        )
        self.llm = llm or make_llm(cfg)

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, question: str, method: str | None = None, top_k: int | None = None) -> list[Document]:
        method = method or self.cfg["retrieval"]["method"]
        top_k = top_k or self.cfg["retrieval"]["top_k"]
        return self.retrievers[method].invoke(question)[:top_k]

    def search(self, question: str, method: str | None = None, top_k: int | None = None) -> list[dict]:
        docs = self.retrieve(question, method, top_k)
        return [to_hit(i, d) for i, d in enumerate(docs, start=1)]

    # ---------------------------------------------------------------- generation
    def _run(self, prompt, inputs: dict) -> dict:
        chain = prompt | self.llm | StrOutputParser()
        t0 = time.time()
        content = THINK_RE.sub("", chain.invoke(inputs)).strip()
        return {"content": content, "seconds": round(time.time() - t0, 1)}

    def answer(self, question: str, docs: list[Document]) -> dict:
        context = format_context(docs)
        result = self._run(RAG_PROMPT, {"context": context, "question": question})
        cited = sorted({int(n) for n in CITATION_RE.findall(result["content"])})
        valid = [n for n in cited if 1 <= n <= len(docs)]
        result.update({
            "context": context,
            "cited_passages": valid,
            "invalid_citations": [n for n in cited if n not in valid],
            "sources": [{"label": f"C{n}", "chunk_id": docs[n - 1].metadata["chunk_id"],
                         "doc_id": docs[n - 1].metadata["doc_id"],
                         "title": docs[n - 1].metadata["title"],
                         "pages": [docs[n - 1].metadata["page"]]} for n in valid],
        })
        return result

    def answer_closed_book(self, question: str) -> dict:
        """Same LLM without retrieved context: what does the model already 'know'?"""
        return self._run(CLOSED_BOOK_PROMPT, {"question": question})
