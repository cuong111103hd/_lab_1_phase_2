"""
llm_runtime.py — Real OpenAI-backed runtime replacing mock_runtime.py

Implements:
  - actor_answer()   : calls GPT to answer the question
  - evaluator()      : calls GPT to judge the answer (structured JSON)
  - reflector()      : calls GPT to generate a reflection lesson
"""
from __future__ import annotations

import json
import os
import time

from openai import OpenAI
from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _chat(system: str, user: str, temperature: float = 0.0) -> tuple[str, int, int]:
    """Returns (content, prompt_tokens, completion_tokens)."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    prompt_tok = usage.prompt_tokens if usage else 0
    completion_tok = usage.completion_tokens if usage else 0
    return content.strip(), prompt_tok, completion_tok


def _build_context_str(example: QAExample) -> str:
    parts = []
    for chunk in example.context:
        parts.append(f"[{chunk.title}]\n{chunk.text}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# Public API (mirrors mock_runtime.py signature)
# ─────────────────────────────────────────────

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    """Return (answer_text, prompt_tokens, completion_tokens)."""
    context_str = _build_context_str(example)

    reflection_section = ""
    if reflection_memory:
        reflection_section = "\n\nREFLECTION MEMORY (lessons from previous failed attempts):\n"
        for idx, mem in enumerate(reflection_memory, 1):
            reflection_section += f"- Attempt {idx} lesson: {mem}\n"

    user_prompt = (
        f"QUESTION: {example.question}\n\n"
        f"CONTEXT:\n{context_str}"
        f"{reflection_section}\n\n"
        f"Attempt #{attempt_id}. Provide ONLY the final answer."
    )

    answer, pt, ct = _chat(ACTOR_SYSTEM, user_prompt, temperature=0.0)
    # Strip quotes that models sometimes add
    answer = answer.strip('"\'').strip()
    return answer, pt, ct


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    """Return (JudgeResult, prompt_tokens, completion_tokens).

    Falls back to exact-match normalization if LLM returns invalid JSON.
    """
    # Fast path: exact match
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return (
            JudgeResult(
                score=1,
                reason="Exact match after normalization.",
                missing_evidence=[],
                spurious_claims=[],
            ),
            0,
            0,
        )

    user_prompt = (
        f"QUESTION: {example.question}\n"
        f"GOLD ANSWER: {example.gold_answer}\n"
        f"PREDICTED ANSWER: {answer}\n\n"
        "Evaluate and respond with JSON only."
    )

    raw, pt, ct = _chat(EVALUATOR_SYSTEM, user_prompt, temperature=0.0)

    # Parse JSON robustly
    try:
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.rstrip("`").strip()
        data = json.loads(clean)
        judge = JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", ""),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except Exception:
        # Fallback: score 0
        judge = JudgeResult(
            score=0,
            reason=f"LLM evaluator parse error. Raw: {raw[:200]}",
            missing_evidence=[],
            spurious_claims=[answer],
        )
    return judge, pt, ct


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int]:
    """Return (ReflectionEntry, prompt_tokens, completion_tokens)."""
    user_prompt = (
        f"QUESTION: {example.question}\n"
        f"GOLD ANSWER: {example.gold_answer}\n"
        f"WRONG ANSWER (attempt {attempt_id}): {judge.spurious_claims[0] if judge.spurious_claims else 'unknown'}\n"
        f"EVALUATOR FEEDBACK: {judge.reason}\n"
        f"MISSING EVIDENCE: {', '.join(judge.missing_evidence) if judge.missing_evidence else 'none'}\n\n"
        "Reflect on the failure and respond with JSON only."
    )

    raw, pt, ct = _chat(REFLECTOR_SYSTEM, user_prompt, temperature=0.2)

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.rstrip("`").strip()
        data = json.loads(clean)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", ""),
            next_strategy=data.get("next_strategy", "Re-read context carefully."),
        )
    except Exception:
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Parse error; re-read context and complete all reasoning hops.",
            next_strategy="Re-read context carefully and identify all intermediate entities before answering.",
        )
    return entry, pt, ct
