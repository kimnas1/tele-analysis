from __future__ import annotations

import hashlib
import os
import time
from typing import Any


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_cell(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "\\n")

