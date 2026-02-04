"""
Chain-of-Thought (CoT) Strategy - Variant 2.

This is a lighter / more structured CoT prompt than the baseline `cot` strategy.
The main goal is to reduce ambiguous outputs and increase the likelihood of an
explicit, parseable final key (e.g., "CANDIDATE: X" and a single \\boxed{X}).
"""

from typing import Sequence

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class CoT2Strategy(ReasoningStrategy):
    """
    CoT2: Structured single-pass reasoning.

    LLM calls per question: 1
    """

    name = "cot2"

    def __init__(self, llm, verbose: bool = False):
        super().__init__(llm, verbose)

    def _build_prompt(self, question: str, keys: Sequence[str]) -> str:
        keys_str = ", ".join(str(k) for k in keys)
        return f"""You are analyzing a multiple-choice problem.

Valid option keys: {keys_str}

Strict requirements:
- Decide the single best option key from: {keys_str}
- First line MUST be: CANDIDATE: <ONE_OPTION_KEY>
- Then provide a concise, evidence-grounded rationale (do not invent facts).
- End with EXACTLY one final answer formatted as: \\boxed{{<ONE_OPTION_KEY>}}
- The CANDIDATE key must match the boxed key.
- Do not output any other \\boxed{{...}} anywhere.

Recommended structure:
CANDIDATE: X
Evidence:
- <quote/mention key observations from the prompt>
Rejections:
- <briefly reject 2-5 other options based on evidence or missing evidence>
Decision:
- <why the evidence supports X>
\\boxed{{X}}

Question:
{question}
"""

    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        self._reset_counters()
        prompt = self._build_prompt(question, keys)
        self._log("Generating CoT2 response...")

        output = self._generate(prompt, step_name="cot2_reasoning")

        step = IntermediateStep(
            step_name="cot2_reasoning",
            input_data={"prompt_length": len(prompt)},
            output=output[:500] + "..." if len(output) > 500 else output,
            llm_calls=1,
        )

        return self._finalize_result(output, keys, [step])

