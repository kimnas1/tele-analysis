"""
Base classes and utilities for multi-stage reasoning strategies.

This module provides the abstract base class for reasoning strategies
and common data structures used across all implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import re
import time


@dataclass
class IntermediateStep:
    """A single step in the reasoning process."""
    step_name: str
    input_data: Dict[str, Any]
    output: str
    llm_calls: int = 1
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "input_data": self.input_data,
            "output": self.output,
            "llm_calls": self.llm_calls,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ReasoningResult:
    """Result from a reasoning strategy execution."""
    final_answer: str                         # The extracted option key (A-I)
    raw_output: str                           # Full final reasoning output
    intermediate_steps: List[IntermediateStep] = field(default_factory=list)
    num_llm_calls: int = 0                    # Total LLM invocations
    total_duration_ms: float = 0.0            # Total execution time
    strategy_name: str = ""                   # Name of strategy used
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "raw_output": self.raw_output,
            "intermediate_steps": [s.to_dict() for s in self.intermediate_steps],
            "num_llm_calls": self.num_llm_calls,
            "total_duration_ms": self.total_duration_ms,
            "strategy_name": self.strategy_name,
        }


# =============================================================================
# Answer Extraction Utilities
# =============================================================================

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
BOX_RE_ESCAPED = re.compile(r"\\\\boxed\{([^}]*)\}")

# Multiple option format patterns for robust parsing
OPTION_PATTERNS = [
    re.compile(r"^\s*([A-Za-z])\s*:\s*(.+)$"),           # A: description
    re.compile(r"^\s*\(([A-Za-z])\)\s*(.+)$"),           # (A) description
    re.compile(r"^\s*\[([A-Za-z])\]\s*(.+)$"),           # [A] description
    re.compile(r"^\s*([A-Za-z])\)\s*(.+)$"),             # A) description
    re.compile(r"^\s*([A-Za-z])\.\s+(.+)$"),             # A. description
    re.compile(r"^\s*Option\s+([A-Za-z])\s*:\s*(.+)$", re.IGNORECASE),  # Option A: description
    re.compile(r"^\s*(\d+)\s*:\s*(.+)$"),                # 1: description (numeric)
    re.compile(r"^\s*\((\d+)\)\s*(.+)$"),                # (1) description (numeric)
    re.compile(r"^\s*(\d+)\)\s*(.+)$"),                  # 1) description (numeric)
    re.compile(r"^\s*(\d+)\.\s+(.+)$"),                  # 1. description (numeric)
    re.compile(r"^\s*(YES|NO|TRUE|FALSE)\s*:\s*(.+)$", re.IGNORECASE),  # YES: / NO:
]


def extract_box_value(text: str) -> Optional[str]:
    """Extract value from \\boxed{...} in text."""
    matches = list(BOX_RE.finditer(text))
    if not matches:
        matches = list(BOX_RE_ESCAPED.finditer(text))
    if not matches:
        return None
    # Return last non-empty match
    for m in reversed(matches):
        v = m.group(1).strip()
        if v:
            return v
    return ""


def normalize_to_keys(raw: Optional[str], keys: Sequence[str]) -> str:
    """
    Normalize model output to one of the option keys.
    For RCA_OOD, keys are typically A-I.
    """
    fallback = keys[5] if len(keys) > 5 else keys[0]
    if not keys:
        return "A"
    if raw is None:
        return fallback

    v = raw.strip()
    if not v:
        return fallback

    # If already a valid key (case-insensitive)
    v_up = v.upper()
    for k in keys:
        if v_up == str(k).upper():
            return str(k)

    keys_are_digits = all(str(k).isdigit() for k in keys)

    # Digits:
    # - If keys are digits, keep digits (do NOT map to letters).
    # - If keys are letters, allow 1-based indexing (1->A, 2->B, ...) as a convenience.
    if re.fullmatch(r"\d+", v):
        if keys_are_digits and v in keys:
            return v
        idx = int(v)
        if 1 <= idx <= len(keys) and not keys_are_digits:
            return str(keys[idx - 1])

    # Letters: if keys are digits, allow A->1, B->2, ... (1-based index).
    if re.fullmatch(r"[A-Z]", v_up) and keys_are_digits:
        idx = ord(v_up) - ord("A") + 1
        if 1 <= idx <= len(keys):
            return str(keys[idx - 1])

    # Try to find any key token inside the string (prefer later matches).
    keys_sorted = sorted([str(k) for k in keys], key=len, reverse=True)
    for k in keys_sorted:
        pat = rf"(?<![A-Za-z0-9_]){re.escape(str(k))}(?![A-Za-z0-9_])"
        if re.search(pat, v, flags=re.IGNORECASE):
            return str(k)

    return fallback


def extract_option_keys(question: str) -> List[str]:
    """
    Extract option keys (A, B, C, ... or 1, 2, 3, ...) from question text.
    
    Supports multiple formats:
    - A: description
    - (A) description
    - A) description
    - A. description
    - Option A: description
    - 1: description (converts to A, B, C, ...)
    - (1) description
    """
    letter_keys: List[str] = []
    numeric_keys: List[int] = []
    word_keys: List[str] = []
    
    for line in question.splitlines():
        for pattern in OPTION_PATTERNS:
            m = pattern.match(line)
            if m:
                key = m.group(1)
                # Check if numeric
                if key.isdigit():
                    numeric_keys.append(int(key))
                elif key.upper() in {"YES", "NO", "TRUE", "FALSE"}:
                    word_keys.append(key.upper())
                else:
                    letter_keys.append(key.upper())
                break  # Only match first pattern per line
    
    # Prefer explicit option lists:
    # 1) Letters (A, B, C, ...)
    # 2) Numbers (1, 2, 3, ...) as literal keys
    # 3) Word keys (YES/NO/TRUE/FALSE)
    if letter_keys:
        keys = letter_keys
    elif numeric_keys:
        keys = [str(n) for n in sorted(set(numeric_keys))]
    else:
        keys = word_keys
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
    
    # If nothing found, try to infer an RCA-style label set (C1..C8) before defaulting.
    if not unique_keys:
        c_labels = sorted(set(re.findall(r"\bC[1-8]\b", question.upper())))
        if len(c_labels) >= 6:
            unique_keys = [f"C{i}" for i in range(1, 9)]
        else:
            unique_keys = list("ABCDEFGHI")
    
    return unique_keys


def extract_answer(text: str, keys: Sequence[str]) -> str:
    """
    Extract and normalize answer from model output.
    
    This function tries multiple extraction strategies in order:
    1. Look for \\boxed{X} format (preferred)
    2. Look for explicit answer statements: "answer is X", "option X", "choose X"
    3. Look for concluding statements: "Therefore X", "So X is", "root cause is X"
    4. Fall back to last mentioned valid key in the text
    """
    if not keys:
        return "A"

    # 1) Preferred: boxed format.
    raw = extract_box_value(text)
    if raw:
        return normalize_to_keys(raw, keys)

    # 2) Try a "FinalAnswer:" line if present.
    m = re.search(r"^\s*FinalAnswer\s*:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        # Strip trailing punctuation (e.g., "C4." or "3)")
        cand = re.sub(r"[^\w]+$", "", cand)
        if cand:
            return normalize_to_keys(cand, keys)

    # 3) Find the last mentioned key token in the output (robust for multi-char keys).
    last_pos = -1
    last_key: Optional[str] = None
    for k in sorted([str(x) for x in keys], key=len, reverse=True):
        pat = rf"(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])"
        for mm in re.finditer(pat, text, flags=re.IGNORECASE):
            if mm.start() >= last_pos:
                last_pos = mm.start()
                last_key = k
    if last_key:
        return normalize_to_keys(last_key, keys)

    # 4) Ultimate fallback to F (keys[5])
    return keys[5] if len(keys) > 5 else keys[0]


# =============================================================================
# Abstract Base Class
# =============================================================================

class ReasoningStrategy(ABC):
    """
    Abstract base class for multi-stage reasoning strategies.
    
    Each strategy implements a different approach to solving RCA problems:
    - CoT: Single-pass chain-of-thought
    - ToT: Tree-of-thought with search
    - Reflexion: Draft → Critique → Revise loop
    - SelfRefine: Generate → Feedback → Refine loop
    - PEV: Plan → Execute → Verify pipeline
    - CoVe: Chain-of-Verification with verification questions
    """
    
    name: str = "base"
    
    def __init__(self, llm: Any, verbose: bool = False):
        """
        Initialize the strategy.
        
        Args:
            llm: LLM wrapper instance for generating text
            verbose: If True, print intermediate steps
        """
        self.llm = llm
        self.verbose = verbose
        self._call_count = 0
        self._start_time = 0.0
    
    def _log(self, msg: str) -> None:
        """Print verbose log message."""
        if self.verbose:
            print(f"  [{self.name}] {msg}")
    
    def _reset_counters(self) -> None:
        """Reset call counter and timer."""
        self._call_count = 0
        self._start_time = time.time()
    
    def _generate(self, prompt: str, step_name: str = "") -> str:
        """
        Generate text and track call count.
        
        Args:
            prompt: The prompt to send to the LLM
            step_name: Name of this step for logging
            
        Returns:
            Generated text
        """
        self._call_count += 1
        if step_name:
            self._log(f"Step '{step_name}' - LLM call #{self._call_count}")
        result = self.llm.generate(prompt)
        return result
    
    def _finalize_result(
        self, 
        final_output: str, 
        keys: Sequence[str],
        intermediate_steps: List[IntermediateStep]
    ) -> ReasoningResult:
        """Create final ReasoningResult with timing and answer extraction."""
        duration = (time.time() - self._start_time) * 1000  # ms
        answer = extract_answer(final_output, keys)
        
        return ReasoningResult(
            final_answer=answer,
            raw_output=final_output,
            intermediate_steps=intermediate_steps,
            num_llm_calls=self._call_count,
            total_duration_ms=duration,
            strategy_name=self.name,
        )
    
    @abstractmethod
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute the reasoning strategy for a given question.
        
        Args:
            question: The RCA question text (including data tables)
            keys: Valid option keys (e.g., ['A', 'B', 'C', ...])
            
        Returns:
            ReasoningResult with final answer and intermediate steps
        """
        pass
