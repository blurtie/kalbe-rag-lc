# Evaluation outputs

Files are named `<kind>_<method>_<YYYYMMDD-HHMMSS>`. The numbers in the main README come from the **final** files.

| File | What it is | Used in README |
|---|---|---|
| `retrieval_20261019-192102.*` | Retrieval metrics, final index (P7 page boundaries fixed), `bm25_tokenizer: words_lowercase` | **Yes** — section 5 |
| `retrieval_20261019-193636.*` | Same, with `bm25_tokenizer: whitespace` (LangChain default) | **Yes** — section 8 |
| `answers_hybrid_20261019-192446.*` | Answers + closed-book answers, final index; JSON has retrieved chunks, full prompt and sources | Raw run |
| `answers_hybrid_lc_labeled.csv` | The run above, labelled by hand (`rag_correct`, `rag_faithful`, `closed_correct`) | **Yes** — section 5 |
| `retrieval_20261019-185517.*`, `answers_hybrid_20261019-190101.*` | Earlier run, before P7's page boundaries were corrected | No (kept for transparency; metrics were identical, only q02's answer wording changed) |

The notebook's section 8 prints `[saved] retrieval_20261019-194008.*`: that run had identical numbers to `192102`, so it was not committed.

Summary of the labelled file: `python -m rag.evaluate summarize outputs/eval/answers_hybrid_lc_labeled.csv`
