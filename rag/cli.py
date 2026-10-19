"""Command-line entry point.

    python -m rag.cli ingest                         PDFs -> chunks -> Chroma
    python -m rag.cli search "question"              retrieval only, no LLM
    python -m rag.cli ask "question" --save          retrieval + answer, saved to outputs/
    python -m rag.cli ask "question" --closed-book   LLM only, no retrieval (baseline)
"""
import argparse
import json
import re
import time

from rag.config import load_config, resolve


def print_hits(hits: list[dict], preview_chars: int = 300) -> None:
    for h in hits:
        c = h["chunk"]
        print(f"\n#{h['rank']} [{c['chunk_id']}] {c['doc_id']} p.{c['pages'][0]} | {c['kind']}")
        print("   " + c["text"][:preview_chars].replace("\n", " ") + " ...")


def save_run(cfg: dict, record: dict) -> None:
    out_dir = resolve(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"\W+", "_", record["question"].lower())[:40].strip("_")
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    for name in ("search", "ask"):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--method", choices=["dense", "bm25", "hybrid"])
        p.add_argument("--k", type=int)
        if name == "ask":
            p.add_argument("--save", action="store_true")
            p.add_argument("--closed-book", action="store_true")
    args = parser.parse_args()
    cfg = load_config()

    if args.command == "ingest":
        from rag.ingest import ingest
        ingest(cfg)
        return

    from rag.pipeline import RagPipeline, to_hit
    rag = RagPipeline(cfg)
    record = {"question": args.question, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "llm": cfg["generation"]["model"], "embedding": cfg["embedding"]["model"]}

    if args.command == "ask" and args.closed_book:
        result = rag.answer_closed_book(args.question)
        print(f"\n=== CLOSED-BOOK ANSWER ({result['seconds']}s) ===\n{result['content']}")
        record.update({"mode": "closed_book", **result})
        if args.save:
            save_run(cfg, record)
        return

    method = args.method or cfg["retrieval"]["method"]
    docs = rag.retrieve(args.question, method=method, top_k=args.k)
    hits = [to_hit(i, d) for i, d in enumerate(docs, start=1)]
    print(f"=== RETRIEVED ({method}, top {len(hits)}) ===")
    print_hits(hits)
    record.update({"mode": "rag", "method": method, "retrieved": hits})

    if args.command == "ask":
        result = rag.answer(args.question, docs)
        print(f"\n=== ANSWER ({result['seconds']}s) ===\n{result['content']}")
        print("\n=== SOURCES ===")
        for s in result["sources"]:
            print(f"[{s['label']}] {s['doc_id']} p.{s['pages'][0]} | {s['chunk_id']}")
        if not result["sources"]:
            print("(no valid citations in the answer)")
        if result["invalid_citations"]:
            print(f"[warn] answer cites passages that were not provided: {result['invalid_citations']}")
        record.update(result)
        if args.save:
            save_run(cfg, record)


if __name__ == "__main__":
    main()
