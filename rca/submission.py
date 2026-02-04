from __future__ import annotations

import csv
import re
from typing import Dict, List, Tuple


_ID_SUFFIX_RE = re.compile(r"^(?P<base>ID_[A-Z0-9]+)_(?P<k>[1-4])$")


def load_template(path: str) -> Tuple[List[dict], List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty submission template: {path}")
    fieldnames = list(rows[0].keys())
    return rows, fieldnames


def write_submission(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def row_id_index(rows: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        rid = str(row.get("ID") or "").strip()
        if rid:
            out[rid] = idx
    return out


def base_slot_index(rows: List[dict]) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {}
    for idx, row in enumerate(rows):
        rid = str(row.get("ID") or "").strip()
        m = _ID_SUFFIX_RE.match(rid)
        if not m:
            continue
        base = m.group("base")
        k = int(m.group("k"))
        out.setdefault(base, {})[k] = idx
    return out


def force_placeholder_else(rows: List[dict], *, keep_column: str) -> None:
    for row in rows:
        for col in list(row.keys()):
            if col == "ID":
                continue
            if col == keep_column:
                row[col] = "placeholder" # Fresh start for our column too
                continue
            row[col] = "placeholder"

