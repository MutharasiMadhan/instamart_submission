# Part D — Policy RAG Assistant and Refund Agent

## Retrieval evaluation

| Chunking | Expected document in top 3 | Expected fact in answer | Unanswerable fallback |
|---|---:|---:|---:|
| 40 words, overlap 10 | 12/12 | 12/12 | 2/2 |
| 40 words, overlap 0 | 12/12 | 10/12 | 2/2 |

With zero overlap, the delivery-time question “What delivery time makes a completed order eligible for a full refund?” fails the expected-fact check because the “more than 45 minutes” condition is split from the sentence fragment selected by the mock. The payment-failure question also fails the fact check because the relevant answer sentence is split across chunks.

## Agent scenarios

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 957 |
| S2 | REFUND | 135.00 | 5 | 957 |
| S3 | ESCALATE | 1999.00 | 5 | 944 |
| S4 | DENIED | 0.00 | 4 | 653 |
| S5 | DENIED | 0.00 | 3 | 373 |
| S6 | NOT_FOUND | 0.00 | 2 | 96 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3072 |

## Safety and resume demos

- Token limit: **FALLBACK_TOKEN_LIMIT**, 3 calls, 390 tokens.
- Timeout: **FALLBACK_TIMEOUT**, 0 calls, 0 tokens.
- Crash/resume: **REFUND**, refund INR 128.25; tools: `get_order → search_policy → calculate_refund → issue_refund`.

The full seven-scenario table, implementation explanation, TF-IDF description, and ReAct versus plan-and-execute comparison are in the project README. All model calls use the local deterministic mock; no API key or network request is used.
