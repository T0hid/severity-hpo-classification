"""Repository hygiene: no keys or tokens in any tracked file, and the entry point loads."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [r"sk-or-v1-[0-9a-f]{20,}", r"sk-[A-Za-z0-9]{32,}", r"ghp_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
            r"AKIA[0-9A-Z]{16}", r"NCBI_API_KEY\s*=\s*[0-9a-f]{20,}",
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]


def text_files():
    skip = {".git", "inputs", "unpacked", "outputs", "__pycache__"}
    for p in ROOT.rglob("*"):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip) and p.suffix in {
                ".py", ".md", ".txt", ".json", ".toml", ".cfg", ".cff", ".yml", ".yaml", ".example", ""}:
            yield p


def test_no_secrets():
    hits = []
    for p in text_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in PATTERNS:
            for m in re.finditer(pat, text):
                if p.name == "test_repository.py":
                    continue
                hits.append(f"{p.relative_to(ROOT)}: {m.group(0)[:20]}")
    assert not hits, hits


def test_entry_point_help():
    out = subprocess.run([sys.executable, str(ROOT / "run.py"), "--help"], capture_output=True, text=True, check=True)
    for cmd in ("verify", "classify", "baseline", "figures", "demo"):
        assert cmd in out.stdout
