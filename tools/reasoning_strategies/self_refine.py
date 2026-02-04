"""
Self-Refine Strategy.

Implements iterative self-refinement where the model:
1. Generates an initial output
2. Generates feedback on its own output
3. Refines the output based on feedback
4. Repeats until feedback is positive or max iterations reached

Reference: Madaan et al., 2023 - "Self-Refine: Iterative Refinement with Self-Feedback"
https://arxiv.org/abs/2303.17651
https://github.com/madaan/self-refine
"""

from typing import Sequence, Tuple

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class SelfRefineStrategy(ReasoningStrategy):
    """
    Self-Refine: Iterative refinement with self-feedback.
    
    Flow:
    1. Generate initial output
    2. Self-feedback: What could be improved?
    3. Refine: Generate improved version
    4. Repeat until feedback is positive or max_iterations
    
    LLM calls per question: 1 + 2 * iterations (typically 5-7)
    """
    
    name = "self_refine"
    
    def __init__(
        self,
        llm,
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize Self-Refine strategy.
        
        Args:
            llm: LLM wrapper instance
            max_iterations: Maximum refinement iterations
            verbose: Print intermediate steps
        """
        super().__init__(llm, verbose)
        self.max_iterations = max_iterations
    
    def _build_initial_prompt(self, question: str, keys: Sequence[str]) -> str:
        """Build prompt for initial generation."""
        keys_str = ", ".join(keys)
        
        return f"""You are analyzing a 5G RCA (Root Cause Analysis) problem.

Your answer must be one of: {keys_str}

Analyze the following question carefully:
{question}

Provide your analysis and conclude with \\boxed{{X}} where X is your answer."""
    
    def _build_feedback_prompt(
        self, 
        question: str, 
        current_output: str,
        keys: Sequence[str],
    ) -> str:
        """Build prompt for generating self-feedback."""
        keys_str = ", ".join(keys)
        
        return f"""You are reviewing an analysis of a 5G RCA problem.

Original Question:
{question}

Current Analysis:
{current_output}

Valid answer options: {keys_str}

Critically evaluate this analysis:
1. Data Accuracy: Are the values correctly read from the tables?
2. Calculations: Are any computations correct?
3. Logic: Is the reasoning sound and consistent?
4. Conclusion: Is the final answer well-supported by evidence?

If the analysis is GOOD and needs no changes, respond with exactly: "ANALYSIS_SATISFACTORY"

If there are issues, explain specifically what needs to be fixed. Be concrete about:
- Which data values might be wrong
- Which calculations should be rechecked
- Which logical steps are flawed
- What alternative conclusions should be considered"""
    
    def _build_refine_prompt(
        self,
        question: str,
        current_output: str,
        feedback: str,
        keys: Sequence[str],
    ) -> str:
        """Build prompt for refinement based on feedback."""
        keys_str = ", ".join(keys)
        
        return f"""You are refining an analysis of a 5G RCA problem based on feedback.

Original Question:
{question}

Your Previous Analysis:
{current_output}

Feedback on Your Analysis:
{feedback}

Valid answer options: {keys_str}

Based on this feedback, provide an IMPROVED analysis:
1. Address each issue raised in the feedback
2. Re-check the data values mentioned
3. Correct any calculation errors
4. Strengthen the logical reasoning
5. Ensure your conclusion is well-supported

Provide your refined analysis and conclude with \\boxed{{X}} where X is your answer."""
    
    def _generate_initial(self, question: str, keys: Sequence[str]) -> str:
        """Generate initial output."""
        prompt = self._build_initial_prompt(question, keys)
        return self._generate(prompt, step_name="initial_generation")
    
    def _get_feedback(self, question: str, output: str, keys: Sequence[str]) -> Tuple[str, bool]:
        """
        Generate feedback on current output.
        
        Returns:
            Tuple of (feedback_text, is_satisfactory)
        """
        prompt = self._build_feedback_prompt(question, output, keys)
        feedback = self._generate(prompt, step_name="self_feedback")
        
        is_satisfactory = "ANALYSIS_SATISFACTORY" in feedback.upper()
        return feedback, is_satisfactory
    
    def _refine(
        self,
        question: str,
        output: str,
        feedback: str,
        keys: Sequence[str],
    ) -> str:
        """Refine output based on feedback."""
        prompt = self._build_refine_prompt(question, output, feedback, keys)
        return self._generate(prompt, step_name="refinement")
    
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute Self-Refine reasoning loop.
        
        Args:
            question: The RCA question text
            keys: Valid option keys
            
        Returns:
            ReasoningResult with final answer and iteration history
        """
        self._reset_counters()
        intermediate_steps = []
        
        # Step 1: Initial generation
        self._log("Generating initial output...")
        current_output = self._generate_initial(question, keys)
        
        intermediate_steps.append(IntermediateStep(
            step_name="initial_generation",
            input_data={"iteration": 0},
            output=current_output[:500] + "..." if len(current_output) > 500 else current_output,
            llm_calls=1,
        ))
        
        # Iterative refinement loop
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"Iteration {iteration}/{self.max_iterations}: Getting feedback...")
            
            # Step 2: Get feedback
            feedback, is_satisfactory = self._get_feedback(question, current_output, keys)
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"feedback_iter_{iteration}",
                input_data={"iteration": iteration, "is_satisfactory": is_satisfactory},
                output=feedback[:300] + "..." if len(feedback) > 300 else feedback,
                llm_calls=1,
            ))
            
            # Stop if satisfactory
            if is_satisfactory:
                self._log(f"Analysis satisfactory after {iteration} iteration(s)")
                break
            
            # Step 3: Refine
            self._log(f"Iteration {iteration}/{self.max_iterations}: Refining...")
            current_output = self._refine(question, current_output, feedback, keys)
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"refine_iter_{iteration}",
                input_data={"iteration": iteration},
                output=current_output[:500] + "..." if len(current_output) > 500 else current_output,
                llm_calls=1,
            ))
        
        return self._finalize_result(current_output, keys, intermediate_steps)
