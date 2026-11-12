"""Entry point for the make.fun cutest-picture-01 entry.

Writes the artwork (stored in this repository as base64 in art.b64) to
the make.fun output directory as entry.png. Standard library only, fully
deterministic: the same bytes are written on every run.
"""
import base64
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    # art.b64 holds the PNG bytes, base64-encoded so the repo stays plain text.
    data = base64.b64decode((HERE / "art.b64").read_text().strip())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "entry.png").write_bytes(data)


if __name__ == "__main__":
    main()
