from __future__ import annotations

import csv
import re
from typing import Dict, Iterable, List, Tuple


_ID_SUFFIX_RE = re.compile(r"^(?P<base>ID_[A-Z0-9]+)_(?P<k>[1-4])$")


def base_id(qid: str) -> str:
    qid = (qid or "").strip()
    m = _ID_SUFFIX_RE.match(qid)
    if m:
        return m.group("base")
    return qid


def load_questions_csv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "ID" not in r.fieldnames or "question" not in r.fieldnames:
            raise ValueError(f"Expected columns ID,question in {path}; got {r.fieldnames}")
        for row in r:
            qid = base_id(row.get("ID") or "")
            q = row.get("question") or ""
            if qid:
                out[qid] = q
    return out


def write_questions_csv(path: str, items: Iterable[Tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "question"])
        w.writeheader()
        for qid, q in items:
            w.writerow({"ID": qid, "question": q})

