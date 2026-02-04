"""
Chain-of-Verification (CoVe) Strategy.

Implements the CoVe architecture to reduce hallucinations:
1. Generate baseline response
2. Generate verification questions about key claims
3. Answer verification questions using source data
4. Check consistency between baseline and verified answers
5. Generate final verified response

Reference: Dhuliawala et al., 2023 - "Chain-of-Verification Reduces Hallucination in Large Language Models"
https://arxiv.org/abs/2309.11495
https://github.com/ritun16/chain-of-verification
"""

import re
from typing import List, Sequence, Tuple

from .base import ReasoningStrategy, ReasoningResult, IntermediateStep


class CoVeStrategy(ReasoningStrategy):
    """
    Chain-of-Verification: Verify claims before finalizing.
    
    Key idea: Generate questions to verify the key claims in the baseline
    answer, then use those verified facts to produce a more accurate final answer.
    
    Flow:
    1. Generate baseline response
    2. Generate verification questions about key claims
    3. Answer verification questions (using source data)
    4. Check consistency with baseline
    5. Generate final verified answer
    
    LLM calls per question: 1 + 1 + n_questions + 1 = ~6 calls
    """
    
    name = "cove"
    
    def __init__(
        self,
        llm,
        n_verification_questions: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize CoVe strategy.
        
        Args:
            llm: LLM wrapper instance
            n_verification_questions: Number of verification questions to generate
            verbose: Print intermediate steps
        """
        super().__init__(llm, verbose)
        self.n_questions = n_verification_questions
    
    def _build_baseline_prompt(self, question: str, keys: Sequence[str]) -> str:
        """Build prompt for baseline response."""
        keys_str = ", ".join(keys)
        
        return f"""You are analyzing a 5G RCA (Root Cause Analysis) problem.

Your answer must be one of: {keys_str}

Analyze the following question:
{question}

Provide your initial analysis, citing specific data from the tables.
Conclude with \\boxed{{X}} where X is your answer."""
    
    def _build_verification_questions_prompt(
        self,
        question: str,
        baseline: str,
        keys: Sequence[str],
    ) -> str:
        """Build prompt to generate verification questions."""
        return f"""You need to verify the accuracy of an RCA analysis.

Original Question:
{question}

Baseline Analysis:
{baseline}

Generate exactly {self.n_questions} verification questions to check the KEY CLAIMS in this analysis.

Focus on:
1. DATA ACCURACY: Questions that verify specific values cited from tables
2. CALCULATION CORRECTNESS: Questions that check any computed metrics
3. LOGICAL VALIDITY: Questions that test the reasoning chain

Format your questions as a numbered list:
1. [First verification question]
2. [Second verification question]
3. [Third verification question]

Each question should be answerable by looking at the original data tables."""
    
    def _build_answer_verification_prompt(
        self,
        question: str,
        verification_question: str,
    ) -> str:
        """Build prompt to answer a single verification question."""
        return f"""You need to answer a verification question using ONLY the data provided.

Original Data/Question:
{question}

Verification Question:
{verification_question}

Answer this verification question by:
1. Finding the relevant data in the tables
2. Citing the specific values you find
3. Providing a clear, factual answer

Be precise and cite specific values. If the data is not available, say "DATA NOT FOUND"."""
    
    def _build_final_verified_prompt(
        self,
        question: str,
        baseline: str,
        verification_qa: List[Tuple[str, str]],
        keys: Sequence[str],
    ) -> str:
        """Build prompt for final verified answer."""
        keys_str = ", ".join(keys)
        
        qa_section = "\n".join([
            f"Q: {q}\nA: {a}\n"
            for q, a in verification_qa
        ])
        
        return f"""You are finalizing an RCA analysis after verification.

Original Question:
{question}

Your Baseline Analysis:
{baseline}

Verification Results:
{qa_section}

Valid options: {keys_str}

Based on the verification results:
1. Check if your baseline analysis is consistent with the verified facts
2. If verification revealed errors, CORRECT them
3. If verification confirmed your analysis, strengthen your confidence
4. Provide your FINAL answer with clear reasoning

Important: Your final answer must be supported by the verified facts.

Provide your final analysis and conclude with \\boxed{{X}} where X is your answer."""
    
    def _parse_verification_questions(self, text: str) -> List[str]:
        """Parse numbered verification questions from text."""
        # Match numbered list items
        questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', text, re.DOTALL)
        
        # Clean up
        questions = [q.strip() for q in questions if q.strip()]
        
        # Fallback: split by newlines if regex fails
        if not questions:
            for line in text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove leading numbers
                    clean = re.sub(r'^\d+[.)\s]+', '', line).strip()
                    if clean and '?' in clean:
                        questions.append(clean)
        
        return questions[:self.n_questions]
    
    def solve(self, question: str, keys: Sequence[str]) -> ReasoningResult:
        """
        Execute Chain-of-Verification reasoning.
        
        Args:
            question: The RCA question text
            keys: Valid option keys
            
        Returns:
            ReasoningResult with final answer and verification history
        """
        self._reset_counters()
        intermediate_steps = []
        
        # Step 1: Generate baseline response
        self._log("Generating baseline response...")
        baseline_prompt = self._build_baseline_prompt(question, keys)
        baseline = self._generate(baseline_prompt, step_name="baseline")
        
        intermediate_steps.append(IntermediateStep(
            step_name="baseline_response",
            input_data={},
            output=baseline[:500] + "..." if len(baseline) > 500 else baseline,
            llm_calls=1,
        ))
        
        # Step 2: Generate verification questions
        self._log("Generating verification questions...")
        vq_prompt = self._build_verification_questions_prompt(question, baseline, keys)
        vq_response = self._generate(vq_prompt, step_name="gen_verification_questions")
        
        verification_questions = self._parse_verification_questions(vq_response)
        self._log(f"Generated {len(verification_questions)} verification questions")
        
        intermediate_steps.append(IntermediateStep(
            step_name="generate_verification_questions",
            input_data={"n_questions": len(verification_questions)},
            output=vq_response[:400] + "..." if len(vq_response) > 400 else vq_response,
            llm_calls=1,
        ))
        
        # Step 3: Answer each verification question
        verification_qa: List[Tuple[str, str]] = []
        
        for i, vq in enumerate(verification_questions, 1):
            self._log(f"Answering verification question {i}/{len(verification_questions)}...")
            
            answer_prompt = self._build_answer_verification_prompt(question, vq)
            answer = self._generate(answer_prompt, step_name=f"answer_vq_{i}")
            
            verification_qa.append((vq, answer))
            
            intermediate_steps.append(IntermediateStep(
                step_name=f"answer_verification_{i}",
                input_data={"question": vq[:100]},
                output=answer[:200] + "..." if len(answer) > 200 else answer,
                llm_calls=1,
            ))
        
        # Step 4: Generate final verified answer
        self._log("Generating final verified answer...")
        final_prompt = self._build_final_verified_prompt(
            question, baseline, verification_qa, keys
        )
        final_output = self._generate(final_prompt, step_name="final_verified")
        
        intermediate_steps.append(IntermediateStep(
            step_name="final_verified_answer",
            input_data={"n_verifications": len(verification_qa)},
            output=final_output[:500] + "..." if len(final_output) > 500 else final_output,
            llm_calls=1,
        ))
        
        return self._finalize_result(final_output, keys, intermediate_steps)
