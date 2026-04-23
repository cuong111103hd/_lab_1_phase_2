"""
agents.py — ReAct and Reflexion agents using real LLM runtime.

Bonus features implemented:
  - reflection_memory  : accumulated lessons passed to Actor on each retry
  - adaptive_max_attempts : stops early as soon as score == 1
  - structured_evaluator  : evaluator returns structured JSON via LLM
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .llm_runtime import actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _classify_failure(judge_reason: str, spurious: list[str]) -> str:
    """Heuristically map failure reason to a failure mode label."""
    reason_lower = judge_reason.lower()
    if "hop" in reason_lower or "incomplete" in reason_lower or "first hop" in reason_lower:
        return "incomplete_multi_hop"
    if "drift" in reason_lower or "confus" in reason_lower or "wrong entity" in reason_lower:
        return "entity_drift"
    if "loop" in reason_lower or "repeat" in reason_lower or "same" in reason_lower:
        return "looping"
    if "overfit" in reason_lower or "memoriz" in reason_lower:
        return "reflection_overfit"
    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        # Bonus: reflection_memory — accumulated lessons across attempts
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []

        final_answer = ""
        final_score = 0
        final_reason = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for attempt_id in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()

            # Actor
            answer, act_pt, act_ct = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            total_prompt_tokens += act_pt
            total_completion_tokens += act_ct

            # Evaluator (structured JSON — Bonus: structured_evaluator)
            judge, eval_pt, eval_ct = evaluator(example, answer)
            total_prompt_tokens += eval_pt
            total_completion_tokens += eval_ct

            latency_ms = int((time.perf_counter() - t0) * 1000)
            token_total = act_pt + act_ct + eval_pt + eval_ct  # real token count

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_total,
                latency_ms=latency_ms,
            )
            traces.append(trace)

            final_answer = answer
            final_score = judge.score
            final_reason = judge.reason

            # Bonus: adaptive_max_attempts — stop early on success
            if judge.score == 1:
                break

            # Reflexion loop: only for reflexion agent with remaining attempts
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                entry, ref_pt, ref_ct = reflector(example, attempt_id, judge)
                total_prompt_tokens += ref_pt
                total_completion_tokens += ref_ct

                # Bonus: reflection_memory — append strategy for next Actor call
                reflection_memory.append(entry.next_strategy)
                reflections.append(entry)

                # Attach reflection to the trace
                traces[-1] = AttemptTrace(
                    attempt_id=trace.attempt_id,
                    answer=trace.answer,
                    score=trace.score,
                    reason=trace.reason,
                    reflection=entry,
                    token_estimate=trace.token_estimate + ref_pt + ref_ct,
                    latency_ms=trace.latency_ms,
                )

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)

        if final_score == 1:
            failure_mode = "none"
        else:
            failure_mode = _classify_failure(
                final_reason,
                [final_answer] if final_answer else [],
            )

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
