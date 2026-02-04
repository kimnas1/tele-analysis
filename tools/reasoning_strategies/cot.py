"""
Chain-of-Thought (CoT) Strategy.

This is the baseline single-pass strategy that prompts the model
to think step-by-step before giving the final answer.

Reference: Wei et al., 2022 - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
https://arxiv.org/abs/2201.11903
"""

from typing import Sequence

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class CoTStrategy(ReasoningStrategy):
    """
    Chain-of-Thought: Single-pass step-by-step reasoning.
    
    This is the simplest strategy and serves as a baseline.
    The model is prompted to think step-by-step before answering.
    
    LLM calls per question: 1
    """
    
    name = "cot"
    
    def __init__(self, llm, verbose: bool = False):
        super().__init__(llm, verbose)
    
    def _build_prompt(self, question: str, keys: Sequence[str]) -> str:
        """Build CoT prompt."""
        keys_str = ", ".join(keys)
        
        return f"""You are analyzing a 5G RCA (Root Cause Analysis) problem.

You will be given a multiple-choice question with options labeled: {keys_str}.

Instructions:
1. Think step-by-step through the problem.
2. Analyze the data provided in the tables carefully.
3. Eliminate options that don't match the evidence.
4. Your final answer MUST be exactly one of: {keys_str}
5. Output the final answer by writing exactly: \\boxed{{<ONE_OPTION_KEY>}}

Reasoning style: Chain-of-Thought
Think step-by-step and eliminate inconsistent options. Keep reasoning concise but thorough.

Question:
{question}

Now analyze step-by-step and provide your answer:"""
    
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute CoT reasoning.
        
        Args:
            question: The RCA question text
            keys: Valid option keys (e.g., ['A', 'B', 'C', ...])
            
        Returns:
            ReasoningResult with final answer
        """
        self._reset_counters()
        
        prompt = self._build_prompt(question, keys)
        self._log("Generating CoT response...")
        
        output = self._generate(prompt, step_name="cot_reasoning")
        
        step = IntermediateStep(
            step_name="cot_reasoning",
            input_data={"prompt_length": len(prompt)},
            output=output[:500] + "..." if len(output) > 500 else output,
            llm_calls=1,
        )
        
        return self._finalize_result(output, keys, [step])
