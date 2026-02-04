"""
Tree-of-Thought (ToT) Strategy with proper BFS/Beam Search.

Implements the ToT architecture with true BFS search:
1. Maintain a FRONTIER of multiple states (beam width)
2. Expand each state with N hypotheses
3. Evaluate all candidates
4. Select top-k to form next frontier
5. Repeat for max_depth levels

Reference: Yao et al., 2023 - "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
https://arxiv.org/abs/2305.10601
https://github.com/princeton-nlp/tree-of-thought-llm
"""

import itertools
import re
from typing import List, Sequence, Tuple

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class ToTStrategy(ReasoningStrategy):
    """
    Tree-of-Thought: True BFS/Beam search through hypothesis space.
    
    Key difference from greedy: maintains a FRONTIER of multiple states,
    expands ALL of them, and selects top-k across the combined candidates.
    
    Flow (per depth level):
    1. For EACH state in frontier: generate N candidate thoughts
    2. Flatten all candidates into one list
    3. Evaluate all candidates
    4. Select top-k candidates to form new frontier
    5. Repeat until max_depth
    6. Synthesize final answer from best state
    
    LLM calls: sum over depths of (frontier_size * n_generate + total_candidates * n_evaluate) + 1
    With defaults (3 gen, 2 eval, 3 select, 2 depth): ~30-40 calls
    """
    
    name = "tot"
    
    def __init__(
        self,
        llm,
        n_generate: int = 3,
        n_evaluate: int = 2,
        n_select: int = 3,  # Beam width - now actually used!
        max_depth: int = 2,
        verbose: bool = False,
    ):
        """
        Initialize ToT strategy.
        
        Args:
            llm: LLM wrapper instance
            n_generate: Number of hypotheses to generate per state
            n_evaluate: Number of evaluation samples per hypothesis
            n_select: Beam width - number of states to keep in frontier
            max_depth: Maximum tree depth
            verbose: Print intermediate steps
        """
        super().__init__(llm, verbose)
        self.n_generate = n_generate
        self.n_evaluate = n_evaluate
        self.n_select = n_select  # Beam width
        self.max_depth = max_depth
    
    def _build_propose_prompt(
        self,
        question: str,
        current_state: str,
        previous_in_batch: List[str],
        keys: Sequence[str],
    ) -> str:
        """Build prompt to propose a new hypothesis from current state."""
        keys_str = ", ".join(keys)
        
        prev_section = ""
        if previous_in_batch:
            prev_section = "\n\nOther hypotheses already generated (propose a DIFFERENT one):\n"
            for i, t in enumerate(previous_in_batch, 1):
                prev_section += f"- {t[:100]}...\n" if len(t) > 100 else f"- {t}\n"
        
        state_section = ""
        if current_state:
            state_section = f"\n\nCurrent analysis state:\n{current_state}"
        
        return f"""You are exploring hypotheses for an RCA problem.

Question:
{question}

Valid answer options: {keys_str}
{state_section}
{prev_section}

Generate ONE distinct hypothesis for the root cause.
- Consider a different angle or data point than already proposed
- Be specific about which data supports this hypothesis
- Focus on a single potential root cause

Provide your hypothesis:"""
    
    def _build_evaluate_prompt(
        self,
        question: str,
        hypothesis: str,
        keys: Sequence[str],
    ) -> str:
        """Build prompt to evaluate a hypothesis."""
        keys_str = ", ".join(keys)
        
        return f"""You are evaluating an RCA hypothesis.

Question:
{question}

Hypothesis being evaluated:
{hypothesis}

Valid answer options: {keys_str}

Evaluate this hypothesis on a scale of 1-10:

Criteria:
1. DATA SUPPORT (1-10): Is the hypothesis supported by table data?
2. LOGIC (1-10): Is the reasoning sound and consistent?
3. COMPLETENESS (1-10): Does it consider all relevant evidence?
4. PLAUSIBILITY (1-10): Is this a likely root cause?

Output format:
Reasoning: [brief reasoning]
Score: X"""
    
    def _build_synthesis_prompt(
        self,
        question: str,
        best_states: List[Tuple[str, float]],
        keys: Sequence[str],
    ) -> str:
        """Build prompt to synthesize final answer from best states."""
        keys_str = ", ".join(keys)
        
        states_section = "\n".join([
            f"State {i+1} (score: {score:.1f}):\n{state[:300]}...\n"
            for i, (state, score) in enumerate(best_states[:3])
        ])
        
        return f"""You are finalizing an RCA analysis based on explored hypotheses.

Question:
{question}

Best hypotheses from search (ranked by score):
{states_section}

Valid answer options: {keys_str}

Based on the highest-scored hypothesis, provide:
1. A clear explanation of why this is the most likely root cause
2. Supporting evidence from the data
3. Your final answer

Conclude with \\boxed{{X}} where X is your answer."""
    
    def _parse_score(self, text: str) -> float:
        """Parse score from evaluation text."""
        patterns = [
            r'[Ss]core:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s*out of\s*10',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    score = float(match.group(1))
                    return min(10.0, max(1.0, score))
                except ValueError:
                    continue
        
        return 5.0  # Default mid-score
    
    def _generate_from_state(
        self,
        question: str,
        state: str,
        keys: Sequence[str],
    ) -> List[str]:
        """Generate N hypotheses from a single state."""
        hypotheses = []
        
        for _ in range(self.n_generate):
            prompt = self._build_propose_prompt(
                question, state, hypotheses, keys
            )
            hypothesis = self._generate(prompt, step_name="propose")
            # Append state context to hypothesis for continuity
            full_hypothesis = f"{state}\n---\n{hypothesis}" if state else hypothesis
            hypotheses.append(full_hypothesis)
        
        return hypotheses
    
    def _evaluate_hypothesis(
        self,
        question: str,
        hypothesis: str,
        keys: Sequence[str],
    ) -> float:
        """Evaluate a single hypothesis with multiple samples."""
        scores = []
        
        for _ in range(self.n_evaluate):
            prompt = self._build_evaluate_prompt(question, hypothesis, keys)
            evaluation = self._generate(prompt, step_name="evaluate")
            score = self._parse_score(evaluation)
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute Tree-of-Thought BFS/Beam search.
        
        This is TRUE beam search: maintains frontier of n_select states,
        expands ALL of them, evaluates all candidates, selects top-k.
        
        Args:
            question: The RCA question text
            keys: Valid option keys
            
        Returns:
            ReasoningResult with final answer and exploration trace
        """
        self._reset_counters()
        intermediate_steps = []
        
        # Initialize frontier with empty state
        # frontier: List[Tuple[state_string, score]]
        frontier: List[Tuple[str, float]] = [("", 0.0)]
        
        for depth in range(self.max_depth):
            self._log(f"Depth {depth + 1}/{self.max_depth}: Frontier size = {len(frontier)}")
            
            # Step 1: Generate candidates from ALL states in frontier
            all_candidates: List[str] = []
            
            for state, _ in frontier:
                self._log(f"  Expanding state...")
                new_hypotheses = self._generate_from_state(question, state, keys)
                all_candidates.extend(new_hypotheses)
            
            self._log(f"  Generated {len(all_candidates)} total candidates")
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"generate_depth_{depth}",
                input_data={"depth": depth, "frontier_size": len(frontier), "n_candidates": len(all_candidates)},
                output=f"Generated {len(all_candidates)} candidates from {len(frontier)} frontier states",
                llm_calls=len(frontier) * self.n_generate,
            ))
            
            # Step 2: Evaluate ALL candidates
            scored_candidates: List[Tuple[str, float]] = []
            
            for candidate in all_candidates:
                score = self._evaluate_hypothesis(question, candidate, keys)
                scored_candidates.append((candidate, score))
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"evaluate_depth_{depth}",
                input_data={"depth": depth, "n_evaluated": len(scored_candidates)},
                output=f"Scores: {[f'{s:.1f}' for _, s in sorted(scored_candidates, key=lambda x: -x[1])[:5]]}",
                llm_calls=len(all_candidates) * self.n_evaluate,
            ))
            
            # Step 3: Select top-k to form NEW frontier (true beam search!)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            frontier = scored_candidates[:self.n_select]
            
            self._log(f"  Selected top {len(frontier)} states. Best score: {frontier[0][1]:.1f}")
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"select_depth_{depth}",
                input_data={"depth": depth, "beam_width": len(frontier)},
                output=f"New frontier: {len(frontier)} states, scores: {[f'{s:.1f}' for _, s in frontier]}",
                llm_calls=0,
            ))
        
        # Step 4: Synthesize final answer from best state(s)
        self._log("Synthesizing final answer from best states...")
        synthesis_prompt = self._build_synthesis_prompt(question, frontier, keys)
        final_output = self._generate(synthesis_prompt, step_name="synthesis")
        
        intermediate_steps.append(IntermediateStep(
            step_name="synthesis",
            input_data={
                "best_score": frontier[0][1] if frontier else 0,
                "frontier_size": len(frontier),
            },
            output=final_output[:500] + "..." if len(final_output) > 500 else final_output,
            llm_calls=1,
        ))
        
        return self._finalize_result(final_output, keys, intermediate_steps)
