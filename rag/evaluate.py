"""Evaluation on data/eval/questions.yaml.

Three steps, kept separate on purpose so retrieval and generation are judged apart:

  python -m rag.evaluate retrieval
      No LLM. For dense / bm25 / hybrid: hit@1, hit@3, hit@5, MRR, and how many of the
      top-5 chunks are appendix chunks from a paper that is NOT the gold paper (tests H1).
      A retrieved chunk is a "hit" if it comes from a gold doc AND shares a page with it.

  python -m rag.evaluate generate [--method dense]
      Runs RAG and closed-book answers for every question and writes a CSV with empty
      label columns. Judging correctness is done BY HAND (open the CSV in Excel):
          rag_correct    y / n / p(artial)   for "none" questions: y = correctly said "not in the papers"
          rag_faithful   y / n               every claim supported by the passages it cites?
          closed_correct y / n / p           closed-book answer correct?

  python -m rag.evaluate summarize outputs/eval/answers_XXXX.csv
      Reads your labels and classifies each answerable question:
          correct              RAG answer correct
          retrieval_failure    wrong AND no gold passage was given to the LLM
          retrieval_failure_partial  wrong; cross-paper question where only SOME needed papers were retrieved
          generation_failure   wrong although a gold passage WAS given to the LLM
      plus the closed-book vs RAG table (tests H2).
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict

import yaml

from rag.config import load_config, resolve

QUESTIONS_FILE = "data/eval/questions.yaml"
EVAL_DIR = "outputs/eval"
METHODS = ("dense", "bm25", "hybrid")
LABEL_COLUMNS = ("rag_correct", "rag_faithful", "closed_correct", "notes")


# ---------------------------------------------------------------- helpers
def load_questions() -> list[dict]:
    data = yaml.safe_load(resolve(QUESTIONS_FILE).read_text(encoding="utf-8"))
    questions = []
    for q in data["questions"]:
        gold = q.get("gold") or []
        if any(not g.get("pages") for g in gold):
            print(f"[skip] {q['id']}: gold entry without pages - fill them in first")
            continue
        questions.append({**q, "gold": gold})
    return questions


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["doc_id"] == g["doc"] and set(chunk["pages"]) & set(g["pages"]) for g in gold)


def first_hit_rank(hits: list[dict], gold: list[dict]) -> int | None:
    for h in hits:
        if is_hit(h["chunk"], gold):
            return h["rank"]
    return None


def context_coverage(hits: list[dict], gold: list[dict]) -> str:
    """Did the LLM get the evidence? y = every gold paper hit, partial = only some
    (matters for cross-paper questions), n = none, "" = unanswerable question."""
    if not gold:
        return ""
    covered = {g["doc"] for g in gold if any(is_hit(h["chunk"], [g]) for h in hits)}
    if len(covered) == len({g["doc"] for g in gold}):
        return "y"
    return "partial" if covered else "n"


def out_path(prefix: str, ext: str):
    path = resolve(EVAL_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.{ext}"


# ---------------------------------------------------------------- 1. retrieval
def eval_retrieval(cfg: dict) -> None:
    from rag.pipeline import RagPipeline

    questions = [q for q in load_questions() if q["gold"]]  # "none" questions have no gold
    if not questions:
        raise SystemExit("No answerable questions with gold pages yet.")
    retriever = RagPipeline(cfg, llm=object())  # retrieval only: no LLM needed
    rows, totals = [], defaultdict(lambda: defaultdict(float))

    for q in questions:
        gold_docs = {g["doc"] for g in q["gold"]}
        for method in METHODS:
            hits = retriever.search(q["question"], method=method, top_k=10)
            rank = first_hit_rank(hits, q["gold"])
            top5 = [h["chunk"] for h in hits[:5]]
            off_appendix = sum(c["kind"] == "appendix" and c["doc_id"] not in gold_docs for c in top5)
            docs_found = {c["doc_id"] for c in top5 if is_hit(c, q["gold"])}
            row = {
                "id": q["id"], "type": q.get("type", ""), "lang": q.get("lang", ""), "method": method,
                "first_hit_rank": rank or "",
                "hit@1": int(rank is not None and rank <= 1),
                "hit@3": int(rank is not None and rank <= 3),
                "hit@5": int(rank is not None and rank <= 5),
                "rr": round(1 / rank, 3) if rank else 0.0,
                "all_gold_docs@5": int(docs_found == gold_docs),
                "offdoc_appendix@5": off_appendix,
                "top5": " ".join(c["chunk_id"] for c in top5),
            }
            rows.append(row)
            for key in ("hit@1", "hit@3", "hit@5", "rr", "all_gold_docs@5", "offdoc_appendix@5"):
                totals[method][key] += row[key]

    n = len(questions)
    print(f"\nRetrieval over {n} answerable questions (top-10 searched, metrics at k shown)")
    print(f"{'method':<8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR@10':>7} {'allDocs@5':>10} {'offDocAppx@5':>13}")
    summary = {}
    for method in METHODS:
        t = totals[method]
        summary[method] = {k: round(v / n, 3) for k, v in t.items()}
        s = summary[method]
        print(f"{method:<8} {s['hit@1']:>6.2f} {s['hit@3']:>6.2f} {s['hit@5']:>6.2f} {s['rr']:>7.2f} "
              f"{s['all_gold_docs@5']:>10.2f} {s['offdoc_appendix@5']:>13.2f}")
    print("offDocAppx@5 = average number of top-5 chunks that are appendix chunks of a NON-gold paper")

    # hit@5 as raw counts per question language: BM25 cannot match Indonesian words to English text
    langs = sorted({r["lang"] for r in rows if r["lang"]})
    if langs:
        print("\nhit@5 per question language (count / n):")
        for method in METHODS:
            parts = []
            for lang in langs:
                sel = [r for r in rows if r["method"] == method and r["lang"] == lang]
                parts.append(f"{lang}: {sum(r['hit@5'] for r in sel)}/{len(sel)}")
            print(f"  {method:<8} " + "   ".join(parts))

    csv_path = out_path("retrieval", "csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({"n_questions": n, "config": cfg["retrieval"],
                                     "summary": summary}, indent=2), encoding="utf-8")
    print(f"\n[saved] {csv_path}\n[saved] {json_path}")


# ---------------------------------------------------------------- 2. generation
def eval_generate(cfg: dict, method: str | None) -> None:
    from rag.pipeline import RagPipeline, to_hit

    questions = load_questions()
    pipeline = RagPipeline(cfg)
    method = method or cfg["retrieval"]["method"]
    rows, full = [], []

    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']} ...", flush=True)
        docs = pipeline.retrieve(q["question"], method=method)
        hits = [to_hit(i, d) for i, d in enumerate(docs, start=1)]
        rag = pipeline.answer(q["question"], docs)
        closed = pipeline.answer_closed_book(q["question"])
        gold_in_context = context_coverage(hits, q["gold"])
        rows.append({
            "id": q["id"], "type": q.get("type", ""), "question": q["question"],
            "reference_answer": q.get("answer", ""),
            "gold": "; ".join(f"{g['doc']} p{','.join(map(str, g['pages']))}" for g in q["gold"]),
            "gold_in_context": gold_in_context,
            "retrieved": " ".join(h["chunk"]["chunk_id"] for h in hits),
            "rag_answer": rag["content"],
            "cited": " ".join(s["chunk_id"] for s in rag["sources"]),
            "invalid_citations": " ".join(map(str, rag["invalid_citations"])),
            "closed_book_answer": closed["content"],
            **{col: "" for col in LABEL_COLUMNS},
        })
        full.append({"question": q, "method": method, "retrieved": hits, "rag": rag, "closed_book": closed})

    csv_path = out_path(f"answers_{method}", "csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(".json").write_text(
        json.dumps({"llm": cfg["generation"]["model"], "method": method, "runs": full},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {csv_path}\nNow fill in the columns {', '.join(LABEL_COLUMNS[:3])} by hand, "
          f"then run:  python -m rag.evaluate summarize \"{csv_path}\"")


# ---------------------------------------------------------------- 3. summarize labels
def summarize(csv_file: str) -> None:
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    unlabeled = [r["id"] for r in rows if not r["rag_correct"].strip()]
    if unlabeled:
        print(f"[warn] not labeled yet (skipped): {', '.join(unlabeled)}")
    rows = [r for r in rows if r["rag_correct"].strip()]

    def yes(value: str) -> bool:
        return value.strip().lower() == "y"

    categories, closed_vs_rag = Counter(), Counter()
    print(f"\n{'id':<6} {'type':<7} {'goldCtx':<8} {'rag':<4} {'faith':<6} {'closed':<7} category")
    for r in rows:
        if r["type"] == "none":
            category = "abstained_ok" if yes(r["rag_correct"]) else "answered_unanswerable"
        elif yes(r["rag_correct"]):
            category = "correct"
        elif r["gold_in_context"] == "y":
            category = "generation_failure"
        elif r["gold_in_context"] == "partial":
            category = "retrieval_failure_partial"  # cross-paper: some needed papers missing
        else:
            category = "retrieval_failure"
        categories[category] += 1
        if r["type"] != "none" and r["closed_correct"].strip():
            closed_vs_rag[(yes(r["closed_correct"]), yes(r["rag_correct"]))] += 1
        print(f"{r['id']:<6} {r['type']:<7} {r['gold_in_context']:<8} {r['rag_correct']:<4} "
              f"{r['rag_faithful']:<6} {r['closed_correct']:<7} {category}")

    print("\nCategories:", dict(categories))
    unfaithful = [r["id"] for r in rows if r["rag_faithful"].strip().lower() == "n"]
    print("Unfaithful answers (claims not supported by cited passages):", unfaithful or "none")
    if closed_vs_rag:
        print("\nClosed-book vs RAG (answerable questions with both labels):")
        print(f"  closed wrong, RAG right : {closed_vs_rag[(False, True)]}   <- RAG clearly helped")
        print(f"  closed right, RAG right : {closed_vs_rag[(True, True)]}   <- model may already know it (H2)")
        print(f"  closed right, RAG wrong : {closed_vs_rag[(True, False)]}   <- retrieved context hurt")
        print(f"  closed wrong, RAG wrong : {closed_vs_rag[(False, False)]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag.evaluate")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("retrieval")
    gen = sub.add_parser("generate")
    gen.add_argument("--method", choices=METHODS)
    summ = sub.add_parser("summarize")
    summ.add_argument("csv_file")
    args = parser.parse_args()

    if args.command == "summarize":
        summarize(args.csv_file)
        return
    cfg = load_config()
    if args.command == "retrieval":
        eval_retrieval(cfg)
    else:
        eval_generate(cfg, args.method)


if __name__ == "__main__":
    main()
