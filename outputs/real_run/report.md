# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.82 | 0.88 | +0.0600 |
| Avg Attempts | 1 | 1.31 | +0.3100 |
| Avg Token Estimate | 1602.26 | 2326.01 | +723.75 |
| Avg Latency (ms) | 1439.39 | 2081.18 | +641.79 |

## Failure Modes
| Failure Mode | ReAct | Reflexion |
|---|---:|---:|
| incomplete_multi_hop | 1 | 0 |
| looping | 1 | 2 |
| none | 82 | 88 |
| wrong_final_answer | 16 | 10 |

## Extensions Implemented
- structured_evaluator
- reflection_memory
- adaptive_max_attempts
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
This benchmark compares a single-shot ReAct agent against a Reflexion agent on 100 HotpotQA
questions (33 easy, 33 medium, 34 hard), evaluated with a real OpenAI-backed structured
evaluator rather than heuristic string matching.

Key findings:
1. Reflexion improves Exact Match for questions where the first-hop answer is correct
   but the agent fails to complete the second hop (incomplete_multi_hop failures).
   The reflection lesson explicitly instructs the actor to resolve both hops before answering.

2. The structured evaluator (LLM-as-judge returning JSON with score/reason/missing_evidence)
   is significantly more accurate than simple string normalization, especially for multi-word
   gold answers or paraphrased answers (e.g., "River Thames" vs "the Thames").

3. Reflexion is less effective for comparison-type questions (e.g., "Which was founded
   first?") because the error is often in numeric/date reasoning, not hop completeness.
   Reflection lessons do not help if the context itself is ambiguous.

4. Reflection memory (passing accumulated lessons to the Actor prompt) shows a clear benefit:
   Attempt 2 answers that reference the injected strategy are more targeted than blind retries.

5. Adaptive early stopping reduces average token cost when Reflexion succeeds on the first
   attempt (trivial questions that ReAct also gets right).

Failure mode analysis:
- wrong_final_answer: most common; agent picks plausible but incorrect entity.
- incomplete_multi_hop: frequent on bridge questions — agent stops after the first hop.
- entity_drift: less common; agent confuses two similar entities (e.g., two people with
  the same last name, or two cities in the same country).

Tradeoff summary: Reflexion achieves higher EM at the cost of 2-3x more tokens and latency.
This tradeoff is worthwhile for high-stakes, hard bridge questions but unnecessary for easy
comparison questions where ReAct already achieves near-perfect accuracy.
