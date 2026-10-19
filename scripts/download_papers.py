"""Download the PDFs listed in papers.yaml into data/raw/.

Usage (from repo root):
    python scripts/download_papers.py

- Skips files that already exist (safe to re-run).
- Checks that each download really is a PDF (starts with b"%PDF"), because some
  sites return an HTML error page with status 200.
- Waits between requests to be polite to arXiv / PMLR / ACL servers.
"""
from pathlib import Path
import time

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "papers.yaml"
RAW_DIR = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "kalbe-rag-assessment/0.1 (student project; contact via GitHub blurtie)"}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    papers = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["papers"]

    for paper in papers:
        out_path = RAW_DIR / f"{paper['id']}.pdf"
        if out_path.exists():
            print(f"[skip] {paper['id']} already downloaded")
            continue

        print(f"[get ] {paper['id']} <- {paper['url']}")
        resp = requests.get(paper["url"], headers=HEADERS, timeout=60)
        resp.raise_for_status()

        if not resp.content.startswith(b"%PDF"):
            print(f"[FAIL] {paper['id']}: response is not a PDF (first bytes: {resp.content[:40]!r})")
            continue

        out_path.write_bytes(resp.content)
        print(f"[ ok ] {paper['id']}: {len(resp.content) / 1e6:.1f} MB")
        time.sleep(2)


if __name__ == "__main__":
    main()
