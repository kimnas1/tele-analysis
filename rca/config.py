from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PipelineConfig:
    raw: Dict[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.raw.get("schema_version", "v1"))


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object at top-level.")
    return PipelineConfig(raw=raw)

