# agents/clinical_agent.py
from models.deepseek_model import DeepSeekModel
from retrieval.medical_retriever import MedicalRetriever
from tools.tool_manager import ToolManager
from collections import Counter
import re


class ClinicalAgent:
    def __init__(self, n_samples: int = 5):
        self.deepseek = DeepSeekModel()
        self.tools = ToolManager()
        self.n_samples = n_samples  # number of self-consistency samples

        # Answer frequency tracker for smarter fallback (seeded equally)
        self._answer_counts = Counter({"A": 1, "B": 1, "C": 1, "D": 1, "E": 1})

        # Load retriever once
        try:
            self.retriever = MedicalRetriever(top_k=3)
        except Exception as e:
            print("Retriever initialization failed:", e)
            self.retriever = None

    # ---------------------------
    # ANSWER EXTRACTION
    # ---------------------------
    def extract_answer(self, text: str):
        """
        Extract a single A-E letter from model output.
        Priority:
          1. 'Answer: X' or 'Final answer: X' pattern
          2. Standalone letter on its own line
          3. First standalone word-boundary letter A-E
        """
        # Pattern 1 – explicit label
        labeled = re.search(
            r"(?:final\s+)?answer\s*[:\-]\s*\**([A-E])\**",
            text,
            re.IGNORECASE,
        )
        if labeled:
            return labeled.group(1).upper()

        # Pattern 2 – letter alone on a line
        line_match = re.search(r"^\s*\**([A-E])\**\s*$", text, re.MULTILINE)
        if line_match:
            return line_match.group(1).upper()

        # Pattern 3 – first word-boundary occurrence
        wb_match = re.search(r"\b([A-E])\b", text)
        if wb_match:
            return wb_match.group(1).upper()

        return None

    # ---------------------------
    # TOOL USAGE  (LLM-driven)
    # ---------------------------
    def call_tools_if_needed(self, question: str) -> dict:
        """
        Use a lightweight LLM call to detect drug names, then look them up.
        Falls back to keyword matching for known high-value pairs.
        """
        tools_used = {}

        # LLM-based drug detection
        try:
            detection_prompt = (
                "List every drug name (generic or brand) mentioned in the question below. "
                "Return ONLY a comma-separated list, or the single word 'none' if there are no drugs.\n\n"
                f"Question: {question}"
            )
            drugs_text = self.deepseek.generate(detection_prompt, max_tokens=48)
            detected = [
                d.strip().lower()
                for d in drugs_text.split(",")
                if d.strip().lower() not in ("none", "")
            ]
        except Exception:
            detected = []

        # Cap lookups to avoid blowing the context window
        for drug in detected[:3]:
            try:
                result = self.tools.run("lookup_drug", drug)
                if result:
                    tools_used[f"drug_info_{drug}"] = result
            except Exception as e:
                print(f"Drug lookup failed for {drug}:", e)

        # Keyword override for well-known interactions
        q = question.lower()
        if "warfarin" in q and "aspirin" in q:
            key = "interaction_warfarin_aspirin"
            if key not in tools_used:
                try:
                    tools_used[key] = self.tools.run(
                        "check_interaction", "warfarin", "aspirin"
                    )
                except Exception as e:
                    print("Interaction check failed:", e)

        return tools_used

    # ---------------------------
    # RETRIEVAL
    # ---------------------------
    def retrieve_context(self, question: str) -> str:
        """
        Return top-2 document snippets (<=800 chars each) separated by a divider.
        """
        try:
            if self.retriever is None:
                return ""
            docs = self.retriever.retrieve(question)
            if not docs:
                return ""
            snippets = [d[:800] for d in docs[:2]]
            return "\n\n---\n\n".join(snippets)
        except Exception as e:
            print("Retrieval failed:", e)
            return ""

    # ---------------------------
    # CHAIN-OF-THOUGHT PASSES
    # ---------------------------
    def _reasoning_pass(self, question: str, context: str, tools_used: dict) -> str:
        """Pass 1: generate explicit reasoning over all options."""
        prompt = f"""You are a clinical pharmacology expert answering a multiple-choice question.

Retrieved context:
{context if context else "No context retrieved."}

Tool outputs:
{tools_used if tools_used else "No tools used."}

Question:
{question}

Think step by step:
1. Identify the core clinical concept being tested.
2. For each option (A-E), briefly state whether it is correct or incorrect and why.
3. State which single option is most defensible based on clinical evidence.

Reasoning:"""

        try:
            return self.deepseek.generate(prompt, max_tokens=600)
        except Exception as e:
            print("Reasoning pass failed:", e)
            return ""

    def _answer_pass(self, reasoning: str) -> str:
        """Pass 2: extract a single committed letter given the reasoning trace."""
        prompt = (
            "Based on the following clinical reasoning, output ONLY the single letter "
            "(A, B, C, D, or E) that represents the correct answer. No explanation.\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            "Final answer (one letter only):"
        )
        try:
            return self.deepseek.generate(prompt, max_tokens=16)
        except Exception as e:
            print("Answer pass failed:", e)
            return ""

    # ---------------------------
    # SINGLE CoT SAMPLE
    # ---------------------------
    def _single_cot_sample(self, question: str, context: str, tools_used: dict):
        """Run one full CoT sample. Returns (answer_letter, reasoning_text)."""
        reasoning = self._reasoning_pass(question, context, tools_used)
        raw_answer = self._answer_pass(reasoning)
        answer = self.extract_answer(raw_answer)
        return answer, reasoning

    # ---------------------------
    # SELF-CONSISTENCY VOTING
    # ---------------------------
    def _self_consistency_vote(self, question: str, context: str, tools_used: dict):
        """
        Run n_samples CoT passes and return the majority-vote answer
        plus all reasoning traces for transparency.
        """
        votes = []
        traces = []

        for i in range(self.n_samples):
            answer, reasoning = self._single_cot_sample(question, context, tools_used)
            traces.append({"sample": i + 1, "reasoning": reasoning, "answer": answer})
            if answer:
                votes.append(answer)

        if votes:
            winner = Counter(votes).most_common(1)[0][0]
        else:
            # Smarter fallback: pick historically most-common answer in this run
            winner = self._answer_counts.most_common(1)[0][0]

        return winner, votes, traces

    # ---------------------------
    # MAIN SOLVER
    # ---------------------------
    def solve(self, question: str) -> dict:
        # Step 1: Retrieve context
        context = self.retrieve_context(question)

        # Step 2: Call tools (LLM-driven)
        tools_used = self.call_tools_if_needed(question)

        # Step 3: Self-consistency with CoT
        final_answer, votes, traces = self._self_consistency_vote(
            question, context, tools_used
        )

        # Update frequency tracker for smarter future fallbacks
        if final_answer:
            self._answer_counts[final_answer] += 1

        return {
            "question": question,
            "context": context,
            "tools_used": tools_used,
            "reasoning_trace": traces,
            "votes": votes,
            "final_answer": final_answer,
        }
