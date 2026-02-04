"""
Plan-Execute-Verify (PEV) Strategy.

Implements a pipeline approach where:
1. Planner: Creates a step-by-step analysis plan
2. Executor: Executes each step of the plan
3. Verifier: Checks results for consistency and correctness
4. Optional re-planning if verification fails

Reference: LangGraph Plan-and-Execute tutorial
https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/
"""

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class PlanExecuteVerifyStrategy(ReasoningStrategy):
    """
    Plan-Execute-Verify: Explicit planning with verification.
    
    Key advantages:
    - Explicit long-term planning (helps with complex problems)
    - Systematic execution of analysis steps
    - Verification catches errors before finalizing
    - Can re-plan if verification fails
    
    Flow:
    1. Planner: Create step-by-step analysis plan
    2. Executor: Execute each step
    3. Verifier: Check for contradictions, finalize or replan
    
    LLM calls: 1 (plan) + n_steps (execute) + 1 (verify) + replan_attempts * (same)
    Typically ~5-15 calls
    """
    
    name = "pev"
    
    def __init__(
        self,
        llm,
        max_replan_attempts: int = 2,
        verify_plan_first: bool = True,
        require_grounded_quotes: bool = True,
        min_grounded_quotes: int = 2,
        verbose: bool = False,
    ):
        """
        Initialize PEV strategy.
        
        Args:
            llm: LLM wrapper instance
            max_replan_attempts: Maximum replanning attempts
            verbose: Print intermediate steps
        """
        super().__init__(llm, verbose)
        self.max_replan_attempts = max_replan_attempts
        self.verify_plan_first = verify_plan_first
        self.require_grounded_quotes = require_grounded_quotes
        self.min_grounded_quotes = min_grounded_quotes

    def _strip_boxed(self, text: str) -> str:
        """Remove any \\boxed{...} occurrences to avoid anchoring later verifier prompts."""
        return re.sub(r"\\\\boxed\{[^}]*\}|\\boxed\{[^}]*\}", "<BOX_REMOVED>", text)

    def _extract_verdict(self, text: str, field: str) -> Optional[str]:
        """
        Extract verdict tokens like:
          VERDICT: PASS
          PLAN_VERDICT: FAIL
        """
        m = re.search(rf"^\s*{re.escape(field)}\s*:\s*(PASS|FAIL)\b", text, flags=re.MULTILINE | re.IGNORECASE)
        if not m:
            return None
        return m.group(1).upper()

    def _extract_recommendation(self, text: str) -> Optional[str]:
        m = re.search(
            r"^\s*RECOMMEND\s*:\s*(ACCEPT|REPLAN|REEXECUTE)\b",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if not m:
            return None
        return m.group(1).upper()

    def _extract_backticked_quotes(self, text: str) -> List[str]:
        # collect up to ~10 quotes for grounding checks
        qs = re.findall(r"`([^`]+)`", text)
        out: List[str] = []
        for q in qs:
            q = q.strip()
            if q:
                out.append(q)
        return out[:10]

    def _grounding_ok(self, question: str, quotes: List[str]) -> bool:
        if not self.require_grounded_quotes:
            return True
        if len(quotes) < int(self.min_grounded_quotes):
            return False
        return all(q in question for q in quotes)
    
    def _build_planner_prompt(self, question: str, keys: Sequence[str]) -> str:
        """Build prompt for the planner."""
        keys_str = ", ".join(keys)
        
        return f"""You are the PLANNER for an RCA (Root Cause Analysis) problem.

Your task is to create a step-by-step analysis plan.

Question to analyze:
{question}

Valid answer options: {keys_str}

Create a systematic plan with 3-5 concrete steps. Each step should:
1. Be specific and actionable
2. Reference what data to check
3. Lead toward identifying the root cause

Format your plan as a numbered list:
1. [First step: what to check and how]
2. [Second step: what to check and how]
3. [Third step: what to check and how]
...

Each step should check specific data against root cause criteria.
Do NOT include the final answer in the plan - that comes after execution."""
    
    def _build_executor_prompt(
        self,
        question: str,
        step: str,
        step_num: int,
        previous_results: Dict[int, str],
    ) -> str:
        """Build prompt for executing a single step."""
        prev_section = ""
        if previous_results:
            prev_section = "\n\nResults from previous steps:\n"
            for i, result in sorted(previous_results.items()):
                prev_section += f"Step {i}: {result[:200]}...\n" if len(result) > 200 else f"Step {i}: {result}\n"
        
        return f"""You are the EXECUTOR for an RCA analysis. Execute ONE specific step.

Original Data/Question:
{question}
{prev_section}

YOUR TASK - Execute Step {step_num}:
{step}

Instructions:
1. Look at the specific data mentioned in this step
2. Extract the relevant values from the tables
3. Apply the check or calculation specified
4. Report your findings concisely

Be specific and cite actual data values. Report what you find."""
    
    def _build_verifier_prompt(
        self,
        question: str,
        plan: List[str],
        execution_results: Dict[int, str],
        keys: Sequence[str],
    ) -> str:
        """Build prompt for the final verifier (quality + grounding checks)."""
        keys_str = ", ".join(keys)
        
        results_section = "\n".join([
            f"Step {i}: {plan[i-1]}\nResult: {result}\n"
            for i, result in sorted(execution_results.items())
        ])
        
        return f"""You are the VERIFIER for an RCA analysis.

Original Question:
{question}

Analysis Plan and Execution Results:
{results_section}

Valid answer options: {keys_str}

Your tasks:
1. CHECK PLAN QUALITY: Is the plan appropriate and does the execution follow it?
2. CHECK GROUNDING: Reject claims that are not supported by the provided tables.
3. CHECK CONSISTENCY: Identify contradictions or missing critical checks.

Output format (STRICT):
VERDICT: PASS|FAIL
RECOMMEND: ACCEPT|REPLAN|REEXECUTE
FINAL: \\boxed{{X}}   (X must be one of: {keys_str})
EVIDENCE_QUOTES:
- `paste an exact substring from the question/tables`
- `paste another exact substring from the question/tables`
ISSUES:
- (optional) list concrete issues if FAIL

Rules:
- The EVIDENCE_QUOTES must be copied verbatim from the question text.
- If you cannot provide grounded quotes, set VERDICT: FAIL.
- If the plan missed an obvious check, set RECOMMEND: REPLAN.
""".strip()

    def _build_plan_verifier_prompt(self, question: str, plan_steps: List[str], keys: Sequence[str]) -> str:
        keys_str = ", ".join(keys)
        plan_section = "\n".join([f"{i}. {s}" for i, s in enumerate(plan_steps, 1)]) or "(empty)"
        return f"""You are the PLAN VERIFIER for an RCA analysis.

Original Question:
{question}

Proposed Plan:
{plan_section}

Valid answer options: {keys_str}

Your job:
- Decide if this plan is actionable, complete enough, and clearly grounded in the data provided.
- If it is missing key checks or is vague, mark it FAIL and propose a fixed plan.

Output format (STRICT):
PLAN_VERDICT: PASS|FAIL
ISSUES:
- (optional) list concrete issues
FIXED_PLAN:
1. ...
2. ...
3. ...
""".strip()
    
    def _build_replan_prompt(
        self,
        question: str,
        previous_plan: List[str],
        execution_results: Dict[int, str],
        verification_feedback: str,
        keys: Sequence[str],
    ) -> str:
        """Build prompt for replanning after verification failure."""
        keys_str = ", ".join(keys)
        
        prev_results = "\n".join([
            f"Step {i}: {execution_results.get(i, 'N/A')[:100]}..."
            for i in range(1, len(previous_plan) + 1)
        ])
        
        return f"""You are RE-PLANNING an RCA analysis after verification found issues.

Original Question:
{question}

Previous Plan Results (abbreviated):
{prev_results}

Verification Feedback:
{verification_feedback}

Valid options: {keys_str}

Create an IMPROVED plan that addresses the verification issues.
Focus on the gaps or errors identified by the verifier.

Format as a numbered list of 2-4 focused steps:
1. [Step to address the issues]
2. [Additional analysis needed]
..."""
    
    def _parse_plan_steps(self, text: str) -> List[str]:
        """Parse numbered plan steps from text."""
        steps = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', text, re.DOTALL)
        steps = [s.strip() for s in steps if s.strip()]
        
        if not steps:
            # Fallback: split by newlines
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    clean = re.sub(r'^\d+[.)\s]+', '', line).strip()
                    if clean and len(clean) > 10:
                        steps.append(clean)
        
        return steps[:5]  # Max 5 steps
    
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute Plan-Execute-Verify pipeline.
        
        Args:
            question: The RCA question text
            keys: Valid option keys
            
        Returns:
            ReasoningResult with final answer and execution trace
        """
        self._reset_counters()
        intermediate_steps = []
        
        for attempt in range(self.max_replan_attempts + 1):
            is_replan = attempt > 0
            prefix = f"[Attempt {attempt + 1}] " if is_replan else ""
            
            # Step 1: Plan
            self._log(f"{prefix}Creating analysis plan...")
            if not is_replan:
                plan_prompt = self._build_planner_prompt(question, keys)
            else:
                plan_prompt = replan_prompt  # Set from previous iteration
            
            plan_response = self._generate(plan_prompt, step_name=f"plan_{attempt}")
            plan_steps = self._parse_plan_steps(plan_response)
            
            self._log(f"{prefix}Plan has {len(plan_steps)} steps")
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"planning_attempt_{attempt}",
                input_data={"is_replan": is_replan, "n_steps": len(plan_steps)},
                output=plan_response[:400] + "..." if len(plan_response) > 400 else plan_response,
                llm_calls=1,
            ))
            
            if not plan_steps:
                self._log(f"{prefix}Warning: No plan steps parsed, using fallback")
                plan_steps = ["Analyze the data tables to identify the root cause"]

            # Optional: verify the plan before spending tokens on execution.
            if self.verify_plan_first:
                self._log(f"{prefix}Verifying plan quality...")
                plan_verify_prompt = self._build_plan_verifier_prompt(question, plan_steps, keys)
                plan_verify_out = self._generate(plan_verify_prompt, step_name=f"plan_verify_{attempt}")

                intermediate_steps.append(IntermediateStep(
                    step_name=f"plan_verification_attempt_{attempt}",
                    input_data={"n_steps": len(plan_steps)},
                    output=plan_verify_out[:500] + "..." if len(plan_verify_out) > 500 else plan_verify_out,
                    llm_calls=1,
                ))

                plan_verdict = self._extract_verdict(plan_verify_out, "PLAN_VERDICT")
                if plan_verdict == "FAIL":
                    fixed = self._parse_plan_steps(plan_verify_out)
                    if fixed:
                        self._log(f"{prefix}Plan verifier proposed a fixed plan ({len(fixed)} steps). Using it.")
                        plan_steps = fixed
                    else:
                        self._log(f"{prefix}Plan failed and no fixed plan parsed; replanning...")
                        if attempt < self.max_replan_attempts:
                            replan_prompt = self._build_replan_prompt(
                                question, plan_steps, {}, plan_verify_out, keys
                            )
                            continue
            
            # Step 2: Execute each step
            execution_results: Dict[int, str] = {}
            
            for i, step in enumerate(plan_steps, 1):
                self._log(f"{prefix}Executing step {i}/{len(plan_steps)}...")
                
                exec_prompt = self._build_executor_prompt(
                    question, step, i, execution_results
                )
                result = self._generate(exec_prompt, step_name=f"execute_{attempt}_{i}")
                execution_results[i] = self._strip_boxed(result)
                
                intermediate_steps.append(IntermediateStep(
                    step_name=f"execute_attempt_{attempt}_step_{i}",
                    input_data={"step": step[:100]},
                    output=result[:300] + "..." if len(result) > 300 else result,
                    llm_calls=1,
                ))
            
            # Step 3: Verify
            self._log(f"{prefix}Verifying results...")
            verify_prompt = self._build_verifier_prompt(
                question, plan_steps, execution_results, keys
            )
            verification = self._generate(verify_prompt, step_name=f"verify_{attempt}")
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"verification_attempt_{attempt}",
                input_data={"n_results": len(execution_results)},
                output=verification[:500] + "..." if len(verification) > 500 else verification,
                llm_calls=1,
            ))
            
            from .base import extract_box_value

            verdict = self._extract_verdict(verification, "VERDICT")
            recommendation = self._extract_recommendation(verification)
            boxed_value = extract_box_value(verification)
            has_valid_boxed = boxed_value is not None and boxed_value.upper() in keys
            quotes = self._extract_backticked_quotes(verification)
            grounded_ok = self._grounding_ok(question, quotes)

            if verdict == "PASS" and has_valid_boxed and grounded_ok and recommendation == "ACCEPT":
                self._log(f"{prefix}Verifier PASS with grounded evidence, answer={boxed_value}")
                return self._finalize_result(verification, keys, intermediate_steps)
            
            if attempt < self.max_replan_attempts:
                self._log(
                    f"{prefix}Replanning needed (verdict={verdict}, boxed={has_valid_boxed}, grounded={grounded_ok}, rec={recommendation})..."
                )
                replan_prompt = self._build_replan_prompt(
                    question, plan_steps, execution_results, verification, keys
                )
            else:
                self._log(f"{prefix}Max replan attempts reached, using last verification")
                # Extract best guess from verification
                return self._finalize_result(verification, keys, intermediate_steps)
        
        # Should not reach here, but return last result if we do
        return self._finalize_result(verification, keys, intermediate_steps)
