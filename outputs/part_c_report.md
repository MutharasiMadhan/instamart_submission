# Part C — Support Ticket Triage Results

## Guardrails and privacy

Tickets: 164; empty skipped: 4; long flagged: 4.
Redactions across all 164 ticket texts: email=55, phone=91, order_id=132, total=278; residual matches after redaction: 0.
The code asserts that no email, phone, or order-ID pattern remains in any redacted text before a model call. The mock-only ticket ID is metadata and is not part of the user message content.

## Zero-shot vs few-shot

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy on classified tickets | 85.7% | 100.0% |
| Total response tokens | 13144 | 25000 |
| Tokens per classified ticket | 89.4 | 170.1 |

### Zero-shot confusion table

| true_category   |   billing |   delivery |   technical |   fraud |
|:----------------|----------:|-----------:|------------:|--------:|
| billing         |        38 |          0 |           0 |       0 |
| delivery        |         0 |         36 |           0 |       0 |
| technical       |        10 |          0 |          29 |       0 |
| fraud           |        11 |          0 |           0 |      23 |

Few-shot uses 11856 additional tokens (80.7 more per classified ticket). In this mock run, it improves classified-ticket accuracy from 85.7% to 100.0%; that is worth paying for when the cost of misrouting a fraud or other support case exceeds the small incremental token cost. If production volume or token prices make that difference material, reserve few-shot prompting for ambiguous or high-risk tickets and monitor accuracy by category.

## Retry and status checks

Zero-shot retries logged: 19; few-shot retries logged: 19; each full run resets the mock first. Status counts in the few-shot full run: `{"classified_json": 135, "classified_keyword": 12, "flagged_invalid_output": 9, "flagged_long": 4, "skipped_empty": 4}`.
Budget demo used 6000 tokens as its stopping threshold and ended with 84 budget_exhausted tickets.

## Routing and human review

`triage_results.csv` contains 164 rows. Empty, long, invalid-output, and budget-exhausted tickets route to human_review; fraud routes to security_team at HIGH priority; Premium customers with otherwise valid categories route to premium_desk at HIGH; remaining tickets route to support_queue at NORMAL. Valid JSON and the exactly-one-category keyword fallback are rule-validated classifications; flagged outputs and guardrail/budget cases go to human review.

## Limitations and next steps

This is a deterministic synthetic mock, not evidence of production LLM quality. The original Part B Premium flag is a post-hoc whole-window label, and any retention list built from it is illustrative rather than a leakage-free evaluation. In production, use a real tokenizer, secure transport, secret management, human auditing, and monitoring for category drift.
