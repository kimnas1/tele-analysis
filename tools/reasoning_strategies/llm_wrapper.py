"""
Unified LLM interface supporting local inference:
- HuggingFace Transformers (local GPU)

This module handles loading and managing local LLM models for inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import time
import os

# Type hints only - no runtime import
if TYPE_CHECKING:
    import torch


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseLLM(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_time_ms = 0.0
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from prompt."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_call_ms": self.total_time_ms / max(1, self.total_calls),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_time_ms = 0.0


# =============================================================================
# HuggingFace Local Backend
# =============================================================================

class HuggingFaceLLM(BaseLLM):
    """
    LLM backend using HuggingFace Transformers (local GPU).
    
    Requires torch and transformers installed.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[Any] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        verbose: bool = False,
    ):
        super().__init__(verbose)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        
        # Ensure padding is set up
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _build_chat_input(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Build chat-formatted input using tokenizer's chat template."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        
        # Fallback
        if system_message:
            return f"{system_message}\n\n{prompt}"
        return prompt
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text using local HuggingFace model."""
        import torch
        
        with torch.inference_mode():
            start_time = time.time()
            
            chat_input = self._build_chat_input(prompt, system_message)
            inputs = self.tokenizer(chat_input, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            
            gen_kwargs = {
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": temperature or self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "num_return_sequences": 1,
                "num_beams": 1,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = output_ids[0, input_len:]
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            duration = (time.time() - start_time) * 1000
            self.total_calls += 1
            self.total_input_tokens += input_len
            self.total_output_tokens += len(new_tokens)
            self.total_time_ms += duration
            
            if self.verbose:
                print(f"  [HF] Call #{self.total_calls}: {input_len} in, {len(new_tokens)} out, {duration:.0f}ms")
            
            return output_text


# =============================================================================
# Factory Functions
# =============================================================================

def create_llm(
    backend: str = "hf",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 600,
    temperature: float = 0.3,
    verbose: bool = False,
    **kwargs,
) -> BaseLLM:
    """
    Create an LLM instance with specified backend.
    
    Args:
        backend: "hf" (huggingface)
        model: Model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
        api_key: (Ignored, for backward compatibility)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        verbose: Print generation details
        **kwargs: Additional backend-specific arguments
    """
    backend = backend.lower()
    
    if backend in ("hf", "huggingface", "local"):
        return load_hf_model_and_create_wrapper(
            model_name=model or "Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
            **kwargs,
        )
    else:
        # Fallback to HF if unknown, as Cerebras is removed
        if verbose:
            print(f"[create_llm] Unknown backend '{backend}', defaulting to 'hf'")
        return load_hf_model_and_create_wrapper(
            model_name=model or "Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose,
            **kwargs,
        )


def load_hf_model_and_create_wrapper(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dtype: str = "bfloat16",
    max_new_tokens: int = 500,
    temperature: float = 0.2,
    verbose: bool = False,
    **kwargs,
) -> HuggingFaceLLM:
    """Load HuggingFace model and create wrapper."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(f"This function requires torch and transformers. Error: {e}")
    
    if verbose:
        print(f"Loading model: {model_name}")
        print(f"Using dtype: {dtype}")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    device = next(model.parameters()).device
    if verbose:
        print(f"Model loaded on device: {device}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {mem_alloc:.1f}GB / {mem_total:.1f}GB")
    
    return HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        verbose=verbose,
    )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

LLMWrapper = HuggingFaceLLM
load_model_and_create_wrapper = load_hf_model_and_create_wrapper
