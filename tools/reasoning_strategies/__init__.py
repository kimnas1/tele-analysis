"""
Reasoning strategies package.

Provides multi-stage reasoning strategies for RCA problem solving:
- CoT: Chain-of-Thought (single-pass baseline)
- CoT2: Chain-of-Thought (structured variant)
- ToT: Tree-of-Thought with BFS search
- Reflexion: Draft → Critique → Revise loop
- SelfRefine: Generate → Feedback → Refine loop
- PEV: Plan → Execute → Verify pipeline
- CoVe: Chain-of-Verification with verification questions
"""

from .base import (
    ReasoningStrategy,
    ReasoningResult,
    IntermediateStep,
    extract_answer,
    extract_option_keys,
)
from .llm_wrapper import (
    BaseLLM,
    HuggingFaceLLM,
    LLMWrapper,  # backward compat alias
    create_llm,
    load_model_and_create_wrapper,  # backward compat alias
    load_hf_model_and_create_wrapper,
)

# Strategy imports - these will be available after implementation
try:
    from .cot import CoTStrategy
except ImportError:
    CoTStrategy = None

try:
    from .cot2 import CoT2Strategy
except ImportError:
    CoT2Strategy = None

try:
    from .self_refine import SelfRefineStrategy
except ImportError:
    SelfRefineStrategy = None

try:
    from .reflexion import ReflexionStrategy
except ImportError:
    ReflexionStrategy = None

try:
    from .cove import CoVeStrategy
except ImportError:
    CoVeStrategy = None

try:
    from .pev import PlanExecuteVerifyStrategy
except ImportError:
    PlanExecuteVerifyStrategy = None

try:
    from .tot import ToTStrategy
except ImportError:
    ToTStrategy = None


# Registry of available strategies
STRATEGY_REGISTRY = {
    "cot": CoTStrategy,
    "cot2": CoT2Strategy,
    "self_refine": SelfRefineStrategy,
    "reflexion": ReflexionStrategy,
    "cove": CoVeStrategy,
    "pev": PlanExecuteVerifyStrategy,
    "tot": ToTStrategy,
}


def get_strategy(name: str, llm: LLMWrapper, **kwargs) -> ReasoningStrategy:
    """
    Get a reasoning strategy by name.
    
    Args:
        name: Strategy name (cot, tot, reflexion, self_refine, pev, cove)
        llm: LLM wrapper instance
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Configured strategy instance
        
    Raises:
        ValueError: If strategy name is unknown or not implemented
    """
    name_lower = name.lower().replace("-", "_")
    
    if name_lower not in STRATEGY_REGISTRY:
        available = [k for k, v in STRATEGY_REGISTRY.items() if v is not None]
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available strategies: {available}"
        )
    
    strategy_class = STRATEGY_REGISTRY[name_lower]
    
    if strategy_class is None:
        raise ValueError(
            f"Strategy '{name}' is not yet implemented. "
            f"Please check the reasoning_strategies package."
        )
    
    return strategy_class(llm=llm, **kwargs)


def list_strategies() -> dict:
    """
    List all available strategies with their implementation status.
    
    Returns:
        Dict mapping strategy name to implementation status
    """
    return {
        name: "implemented" if cls is not None else "not_implemented"
        for name, cls in STRATEGY_REGISTRY.items()
    }


__all__ = [
    # Base classes
    "ReasoningStrategy",
    "ReasoningResult", 
    "IntermediateStep",
    # LLM backends
    "BaseLLM",
    "HuggingFaceLLM",
    "LLMWrapper",  # backward compat
    "create_llm",
    "load_model_and_create_wrapper",  # backward compat
    "load_hf_model_and_create_wrapper",
    # Utilities
    "extract_answer",
    "extract_option_keys",
    "get_strategy",
    "list_strategies",
    "STRATEGY_REGISTRY",
    # Strategies
    "CoTStrategy",
    "CoT2Strategy",
    "SelfRefineStrategy",
    "ReflexionStrategy",
    "CoVeStrategy",
    "PlanExecuteVerifyStrategy",
    "ToTStrategy",
]
