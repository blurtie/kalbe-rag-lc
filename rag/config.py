"""Load config.yaml (+ optional .env) and resolve paths relative to the repo root."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def load_config(path: str | None = None) -> dict:
    _load_dotenv()
    cfg = yaml.safe_load((Path(path) if path else ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["generation"]["host"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return cfg


def resolve(relative: str) -> Path:
    return ROOT / relative


def load_manifest(cfg: dict) -> list[dict]:
    return yaml.safe_load(resolve(cfg["manifest"]).read_text(encoding="utf-8"))["papers"]
