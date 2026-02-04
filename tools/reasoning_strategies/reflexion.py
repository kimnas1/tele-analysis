"""
Reflexion Strategy.

Implements the Reflexion architecture where the model:
1. Generates an initial answer with reasoning
2. Critiques its own answer (identifying what's missing/wrong)
3. Revises the answer based on the critique
4. Maintains episodic memory of reflections across iterations

Reference: Shinn et al., 2023 - "Reflexion: Language Agents with Verbal Reinforcement Learning"
https://arxiv.org/abs/2303.11366
https://github.com/noahshinn/reflexion
"""

from typing import List, Sequence

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class ReflexionStrategy(ReasoningStrategy):
    """
    Reflexion: Draft → Critique → Revise with episodic memory.
    
    Key difference from Self-Refine:
    - Reflexion maintains episodic memory of past reflections
    - Critique focuses on what's missing/wrong (gaps)
    - Can inform future attempts with accumulated reflections
    
    Flow:
    1. Generate initial answer
    2. Critique: Identify what's missing/wrong
    3. Revise: Improve based on critique + memory
    4. Repeat until no issues found or max_iterations
    
    LLM calls per question: 1 + 2 * iterations (typically 5-7)
    """
    
    name = "reflexion"
    
    def __init__(
        self,
        llm,
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize Reflexion strategy.
        
        Args:
            llm: LLM wrapper instance
            max_iterations: Maximum reflection iterations
            verbose: Print intermediate steps
        """
        super().__init__(llm, verbose)
        self.max_iterations = max_iterations
    
    def _build_initial_prompt(self, question: str, keys: Sequence[str]) -> str:
        """Build prompt for initial answer generation."""
        keys_str = ", ".join(keys)
        
        return f"""You are analyzing a 5G RCA (Root Cause Analysis) problem.

Your answer must be one of: {keys_str}

Analyze the following question:
{question}

Think step-by-step:
1. Identify key data points from the tables
2. Check each potential root cause against the evidence
3. Eliminate causes that don't fit the data
4. Select the most likely cause

Provide your analysis and conclude with \\boxed{{X}} where X is your answer."""
    
    def _build_critique_prompt(
        self,
        question: str,
        current_answer: str,
        past_reflections: List[str],
        keys: Sequence[str],
    ) -> str:
        """Build prompt for self-critique."""
        keys_str = ", ".join(keys)
        
        reflections_section = ""
        if past_reflections:
            reflections_section = "\n\nPast Reflections (issues found previously):\n"
            for i, ref in enumerate(past_reflections, 1):
                reflections_section += f"{i}. {ref}\n"
        
        return f"""You are critically evaluating an RCA analysis.

Original Question:
{question}

Current Answer/Analysis:
{current_answer}

Valid options: {keys_str}
{reflections_section}

Critically evaluate this analysis by checking for:

1. MISSING DATA: What important data from the tables was NOT considered?
2. MISREAD VALUES: Were any values incorrectly read or interpreted?
3. FLAWED LOGIC: Are there logical gaps or unjustified conclusions?
4. OVERLOOKED ALTERNATIVES: Were other plausible causes properly ruled out?
5. UNSUPPORTED CLAIMS: Are there claims not supported by the table data?

If you find NO significant issues, respond with exactly: "NO_ISSUES_FOUND"

Otherwise, provide a specific reflection on what is MISSING or WRONG. Be concrete:
- Which specific data point was missed?
- Which value was misread (and what is correct)?
- Which logical step is flawed?
- Which alternative cause wasn't properly eliminated?"""
    
    def _build_revise_prompt(
        self,
        question: str,
        current_answer: str,
        critique: str,
        all_reflections: List[str],
        keys: Sequence[str],
    ) -> str:
        """Build prompt for revision based on critique."""
        keys_str = ", ".join(keys)
        
        memory_section = ""
        if len(all_reflections) > 1:
            memory_section = "\n\nAll reflections so far (learn from these):\n"
            for i, ref in enumerate(all_reflections, 1):
                memory_section += f"Reflection {i}: {ref[:200]}...\n" if len(ref) > 200 else f"Reflection {i}: {ref}\n"
        
        return f"""You are revising an RCA analysis based on self-critique.

Original Question:
{question}

Your Previous Analysis:
{current_answer}

Critique of Your Analysis:
{critique}
{memory_section}

Valid options: {keys_str}

Based on the critique, provide a REVISED and IMPROVED analysis:
1. Address each issue identified in the critique
2. Re-examine the data mentioned
3. Strengthen weak points in your reasoning
4. Consider alternatives that weren't properly evaluated
5. Ensure your conclusion is fully supported by evidence

Provide your revised analysis and conclude with \\boxed{{X}} where X is your answer."""
    
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute Reflexion reasoning loop with episodic memory.
        
        Args:
            question: The RCA question text
            keys: Valid option keys
            
        Returns:
            ReasoningResult with final answer and reflection history
        """
        self._reset_counters()
        intermediate_steps = []
        reflections: List[str] = []  # Episodic memory
        
        # Step 1: Generate initial answer
        self._log("Generating initial answer...")
        prompt = self._build_initial_prompt(question, keys)
        current_answer = self._generate(prompt, step_name="initial_answer")
        
        intermediate_steps.append(IntermediateStep(
            step_name="initial_answer",
            input_data={"iteration": 0},
            output=current_answer[:500] + "..." if len(current_answer) > 500 else current_answer,
            llm_calls=1,
        ))
        
        # Reflection loop
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"Iteration {iteration}/{self.max_iterations}: Critiquing...")
            
            # Step 2: Critique
            critique_prompt = self._build_critique_prompt(
                question, current_answer, reflections, keys
            )
            critique = self._generate(critique_prompt, step_name=f"critique_{iteration}")
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"critique_iter_{iteration}",
                input_data={"iteration": iteration, "num_past_reflections": len(reflections)},
                output=critique[:300] + "..." if len(critique) > 300 else critique,
                llm_calls=1,
            ))
            
            # Check if no issues found
            if "NO_ISSUES_FOUND" in critique.upper():
                self._log(f"No issues found after {iteration} reflection(s)")
                break
            
            # Add to episodic memory
            reflections.append(critique)
            
            # Step 3: Revise with accumulated memory
            self._log(f"Iteration {iteration}/{self.max_iterations}: Revising...")
            revise_prompt = self._build_revise_prompt(
                question, current_answer, critique, reflections, keys
            )
            current_answer = self._generate(revise_prompt, step_name=f"revise_{iteration}")
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"revise_iter_{iteration}",
                input_data={"iteration": iteration, "num_reflections": len(reflections)},
                output=current_answer[:500] + "..." if len(current_answer) > 500 else current_answer,
                llm_calls=1,
            ))
        
        return self._finalize_result(current_answer, keys, intermediate_steps)
