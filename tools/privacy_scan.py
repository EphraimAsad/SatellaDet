from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".csv", ".ipynb"
}
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "__pycache__", ".pytest_cache"}
BINARY_BLOCKLIST = {".pt", ".pth", ".ckpt", ".onnx", ".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx"}

PATTERNS = {
    "windows user path": re.compile(r"[A-Za-z]:[\\\\/]Users[\\\\/][^\\\\/\\s]+", re.I),
    "unix home path": re.compile(r"/(?:home|Users)/[^/\\s]+", re.I),
    "email address": re.compile(r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", re.I),
    "OpenAI-style secret": re.compile(r"\\bsk-[A-Za-z0-9_-]{20,}\\b"),
    "Google-style API key": re.compile(r"\\bAIza[0-9A-Za-z_-]{20,}\\b"),
    "GitHub token": re.compile(r"\\bgh[pousr]_[A-Za-z0-9_]{20,}\\b"),
    "private key header": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}


def main():
    ap = argparse.ArgumentParser(description="Heuristic privacy/secrets scan for a SatellaDet release tree.")
    ap.add_argument("path", nargs="?", default=".")
    args = ap.parse_args()
    root = Path(args.path).resolve()
    findings = []

    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in BINARY_BLOCKLIST:
            findings.append((p, "blocked artifact type", p.suffix.lower()))
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in {"LICENSE", "NOTICE", ".gitignore", "MANIFEST.in"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append((p, label, match.group(0)[:120]))

    if findings:
        print("Potential release-hygiene findings:")
        for path, label, value in findings:
            print(f"- {path.relative_to(root)}: {label}: {value}")
        return 1

    print("PASS: no blocked artifacts or token/path patterns detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
