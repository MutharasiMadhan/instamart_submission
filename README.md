# Instamart Customer Intelligence — Parts A–D

## Overview

This repository is a reproducible, offline capstone project covering raw-data auditing and cleaning, SQL/pandas analysis, RFM customer segmentation, lapse prediction, privacy-preserving support-ticket triage, retrieval-grounded policy Q&A, and a guarded tool-using refund agent. All currency amounts are in Indian rupees (INR). The supplied `generate_data.py` and `mock_llm.py` are kept at the repository root. The mock LLM is deterministic local test scaffolding: no API key or real network call is used.

The project commits its generated inputs, cleaned CSVs, SQLite database, source code, SQL queries, trained model, evaluation outputs, JSONL ledgers/runs, and resumable state checkpoints so results can be inspected without first regenerating them.

## Reproduce

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Regenerate the deterministic synthetic inputs (seed 42)
python generate_data.py

# Part A
python audit_data.py
python clean_data.py
python validate_data.py
python run_queries.py
python analysis.py

# Parts B–D
python part_b.py
python part_c.py
python part_d.py
```

`python validate_data.py` should print `ALL CHECKPOINTS PASSED`. The Part B script writes segment assignments, model artifacts and reports under `outputs/`; Part C writes triage outputs; Part D writes policy evaluations, agent runs, refund ledger entries and checkpoints under `state/`. The `.gitignore` excludes virtual environments, `.env`, Python bytecode/cache and OS metadata. Do not add credentials or secrets.

## Part A

### Raw-data audit

`audit_data.py` reports dimensions, data types, missing counts and percentages for raw tables, and separately accounts for blank strings in ticket text. The deterministic raw data has 1,200 customers, 40 products, 13,084 raw order rows and 164 support tickets.

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with `pending` status | 656 |
| Distinct raw `order_id` | 12,309 |
| Unparseable raw `order_amount` | 120 |
| Negative raw `delivery_minutes` | 106 |
| Blank raw `delivery_minutes` | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and validation

- Customer city values are trimmed/title-cased; 476 of 1,200 raw city values are standardized. Final counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191 and Mumbai 184.
- Gender is trimmed and lowercased; `male`/`m` and `female`/`f` are mapped to canonical labels, and blank/unrecognized values become `unknown`. There are 950 standardized values; final counts are female 610, male 560 and unknown 30.
- The 6 ages outside 10–100 are set missing. The 61 blank plus 6 invalid ages (67 total) are imputed using the median for each standardized city. Clean ages have no missing values and range from 18 to 60. Signup dates are parsed and stored as `YYYY-MM-DD`.
- Order timestamps are parsed. Exact duplicate rows are removed (13,084 → 12,965), then the latest record per order is retained, removing 656 older records and leaving 12,309 unique orders: 11,110 completed and 1,199 cancelled, with no pending orders.
- There are 112 missing/unparseable amounts after order deduplication; these are imputed using `round(price_inr * quantity * (1 - discount_pct / 100), 2)`. Negative delivery durations are set missing (105 remain after deduplication). The 669 missing completed-order delivery values are imputed by product-category median. All 1,199 cancelled orders retain missing delivery time.
- The 40-product catalogue is retained and its prices are numeric. Clean CSVs are stored in `data/clean/`; SQLite tables `customers`, `products` and `orders` are stored in `data/instamart.db`.

Completed-order revenue is **INR 2,998,668.80**. `validate_data.py` checks the reference counts, city/gender/age rules, unique order IDs, final status counts, amount completeness, delivery-time missingness, product count, SQLite tables and revenue; success prints `ALL CHECKPOINTS PASSED`.

### SQL and pandas analysis

Seven SQL queries are in `sql/queries.sql` and are executed by `run_queries.py` against SQLite:

| Query | Result |
|---|---|
| Q1 | 30 customers |
| Q2 | 30 city/category cells; total revenue INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45 |
| Q3 | 11 products priced above catalogue average |
| Q4 | 25 customers with no completed order |
| Q5 | 550 customers above average completed spend; average customer spend INR 2,552.06 |
| Q6 | 1,175 ranked customers; 8 rows where `RANK` differs from `ROW_NUMBER`; ties share `RANK`, while row numbers remain unique |
| Q7 | 183 daily revenue rows; final cumulative revenue INR 2,998,668.80 |

`analysis.py` cross-checks Q2 with `pandas.pivot_table(..., margins=True)`, computes city aggregates, distribution statistics, Tukey fences/outliers and correlations. Completed order amount mean is INR 269.91, median INR 135.00, skewness about 4.61; Q1 is INR 64.80, Q3 INR 297.00, upper fence INR 645.30, with 897 values above and none below. The Bengaluru completed amount sum is INR 523,643.40 (mean INR 276.47 across 1,894 orders); Mumbai is INR 479,058.50 (mean INR 297.00 across 1,613 orders). The largest absolute correlation is order amount vs discount percentage at about −0.095; order amount vs quantity is about +0.087. These small correlations do not establish causation.

## Part B

Run `python part_b.py`. Outputs include `customer_segments.csv`, `kmeans_metrics.csv`, `segment_profiles.csv`, `lapse_model.joblib`, `retention_list.csv`, feature-importance/coefficient CSVs, summary JSON and report Markdown.

### RFM and K-means

RFM is calculated as of `2025-10-01 00:00:00` from completed orders. There are 1,175 customers with at least one completed order; recency median/min/max is 19/0/180 days, frequency mean/max is 9.46/33, and monetary mean/max is INR 2,552.06/INR 14,766.55.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because it has the highest silhouette score and the K=3-to-K=4 inertia reduction is substantial, while four groups are actionable for campaigns. Inertia always decreases as K increases because adding clusters lets observations sit closer to a centroid; the lower inertia at K=8 alone does not account for complexity or business usefulness. PCA on scaled RFM gives explained-variance ratios 0.63427875 and 0.21803762 (total about 0.852).

### Segment profiles

Segment IDs are arbitrary labels, not ranks or an inherent order. The Premium flag is true only for the segment with the highest mean monetary value.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Name | Recommended action |
|---:|---:|---:|---:|---:|---|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Value Seekers | Use relevant bundles and low-friction second-order incentives. |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium High-Value (**Premium**) | Offer a personal loyalty benefit and early access; protect the relationship. |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Frequent Loyalists | Use replenishment reminders and a subscription or streak reward. |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Cooling Customers | Send a targeted win-back offer and ask what changed. |

### Lapse definition and leakage controls

Cutoff: `2025-09-01`. The cohort contains customers with at least one completed order before the cutoff; `lapsed=1` means no completed order from September 1 through September 30 inclusive. The cohort has 1,156 customers, of whom 443 are lapsed (38.32%); the stratified `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training and 232 test rows. Features include pre-cutoff recency, frequency, monetary, average discount, cancellation rate, average completed delivery time, favourite category, city, age, gender and tenure. **No orders dated on or after 2025-09-01 enter the features, and Task 3 segment labels are not features.** Encoders and scalers are fitted on training rows only and applied to test rows; no encoder, scaler or resampler saw test rows. Feature checks confirm no missing values. The training-fitted scaler's frequency mean differs from the full-cohort frequency mean because the former is computed from only the training partition.

### Model tuning and evaluation

Baseline (always predict not lapsed): test accuracy 0.616; lapsed precision/recall/F1 = 0/0/0 (`zero_division=0`). Both GridSearchCV runs use `cv=5`, `scoring="f1"`; logistic regression searches `C ∈ {0.01, 0.1, 1, 10, 100}` with `max_iter=1000`; random forest searches `max_depth ∈ {3, 5, 8}` and `min_samples_leaf ∈ {5, 10, 20}` with 100 trees and `random_state=42` (45 fits).

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=100 | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | max_depth=8, min_samples_leaf=10 | 0.683 | `[[114, 29], [31, 58]]` | 0.667 | 0.652 | **0.659** |

The random forest is selected for its higher test F1 and precision with recall equal to logistic regression. Missing a lapsing customer risks future revenue, while wrongly targeting a loyal customer wastes campaign budget and can encourage discount-seeking; the trade-off should be monitored after deployment.

Training-only oversampling retains original training rows and duplicates lapsed rows until classes are 570/570. Logistic regression before oversampling: precision 0.644, recall 0.652, F1 0.648. After oversampling: precision 0.589, recall 0.854, F1 0.697. Recall changes most, catching more likely lapsers but increasing unnecessary outreach.

Random-forest top five features are frequency (0.28086), monetary (0.26961), recency (0.11367), average delivery minutes (0.06171) and cancellation rate (0.05274). Logistic coefficient signs: frequency −1.69733 (higher frequency associates with lower lapse log-odds), monetary +0.28343 (higher monetary associates with higher lapse log-odds conditional on other features), cancellation rate −0.12382 (negative conditional association in this fitted model). The saved model is reloaded with `joblib.load`; predictions match exactly. The Premium-segment test retention list contains 10 customers, 2 of whom actually lapsed. **The retention list is illustrative and not a leakage-free evaluation because its Premium flag is based on whole-window RFM, which includes the September label window.**

## Part C

Run `python part_c.py`. The pipeline processes all 164 tickets in file order using the supplied local `mock_llm.py` interface.

### Guardrails, privacy and validation

Before model calls, 4 empty/whitespace tickets are `skipped_empty` and 4 tickets estimated at 734 tokens are `flagged_long` (maximum input 400). Redaction patterns are applied in this order: email ` [\w.+-]+@[\w-]+\.[\w.]+ ` → `[EMAIL]`; phone `\b[6-9]\d{9}\b` → `[PHONE]`; order ID `ORD-\d{6}` → `[ORDER_ID]`. Across 164 tickets, substitutions are email 55, phone 91 and order ID 132 (278 total); rechecking the redacted text finds 0 remaining matches. The user message content is exactly the redacted text, with no extra metadata. Each full run resets the mock. The retry policy allows at most 3 retries with `BASE_DELAY=0.05` and jitter multiplier `random.uniform(0.9, 1.15)`; each full run recovered 19 transient errors.

Zero-shot and few-shot prompts both contain `CLASSIFY`, the four categories (`billing`, `delivery`, `technical`, `fraud`) and require JSON-only output of `{"category": "...", "reason": "..."}`. The zero-shot prompt has no `EXAMPLES:` section; the few-shot prompt has labelled examples including fraud. Valid JSON is accepted only for an allowed category; non-JSON responses are accepted only if `re.findall(r"[a-z]+", text.lower())` finds exactly one allowed category. All other outputs are flagged for human review, never guessed.

### Results, cost and routes

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets in run | 164 | 164 |
| Classified tickets | 147 | 147 |
| Correct among classified | 126/147 | 147/147 |
| Accuracy | 85.7% | 100.0% |
| Total response tokens | 13,144 | 25,000 |
| Tokens per classified ticket | 89.4 | 170.1 |
| Transient errors recovered | 19 | 19 |

Few-shot uses 11,856 additional tokens overall (about 80.7 more per classified ticket) and improves accuracy from 85.7% to 100% in this deterministic run. That trade-off is useful when incorrect routing, especially of fraud tickets, is more costly than the extra tokens; production teams should compare incremental token cost with misrouting and review costs.

Zero-shot confusion matrix (rows=true category, columns=predicted category):

| True \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

For the full few-shot run, status counts are 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long` and 4 `skipped_empty`. The zero-shot budget demo uses `TOKEN_BUDGET=6000`; the running usage reaches 6,076 tokens and 84 tickets end as `budget_exhausted`. The final `outputs/triage_results.csv` has 164 rows and columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. `skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` route to `human_review` with NORMAL priority; fraud routes to `security_team` with HIGH priority; otherwise Premium customers route to `premium_desk` with HIGH priority; all remaining cases route to `support_queue` with NORMAL priority. Customers absent from the segment file are treated as not Premium. Only `classified_json` and `classified_keyword` categories are rule-validated automated classifications; flagged/guardrail/budget cases go to human review.

## Part D

Run `python part_d.py`. It uses `data/policies/`, `data/instamart.db` and the same offline `mock_llm.py`.

### Chunking, TF-IDF and evaluation

Each policy document is split into 40-word chunks with 10-word overlap (30-word step), yielding 15 chunks: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer()` is fitted on chunk text; `retrieve(query, k=3)` transforms the query with that fitted vectorizer and ranks chunks using cosine similarity. TF measures how often a term appears in a chunk; IDF gives more weight to terms that occur in fewer chunks. TF-IDF is lexical, not semantic, so equivalent phrasings that use different words may rank poorly.

| Chunking | Expected document among top 3 (12 answerable) | `expected_fact` substring in answer | Unanswerable fallback (2 questions) |
|---|---:|---:|---:|
| 40 words, overlap 10 | 12/12 | 12/12 | 2/2 |
| 40 words, overlap 0 | 12/12 | 10/12 | 2/2 |

With zero overlap, “What delivery time makes a completed order eligible for a full refund?” fails the fact check because the “more than 45 minutes” condition is split at a chunk boundary; the mock selects the fragment about receiving a full refund without the condition. A payment-failure question also loses its expected fact when its answer sentence is split. With overlap, all 12 answerable questions retrieve the expected document and contain the expected fact, and both unanswerable questions receive the fallback beginning `I could not find`.

`answer(question)` retrieves three chunks and sends a `POLICY ASSISTANT` system message plus a user message formatted as `CONTEXT:`, each chunk prefixed by `[source file]`, and `QUESTION:`. The answer function returns the model text and sources used.

### Tools, refunds and ReAct scenarios

`TOOLS_SPEC` documents five tools and JSON-style parameters; `TOOL_MAP` dispatches tool names to functions. `get_order` reads SQLite; `search_policy` reuses retrieval; `calculate_refund` returns the full order amount only for completed orders delivered in more than 45 minutes (45 minutes or less returns zero); refunds above INR 1,000 are escalated and not automatically issued. `issue_refund` and `escalate_to_human` append timestamped JSON lines to `outputs/refund_ledger.jsonl`. All simulated requests are treated as within the policy's seven-day window. Tool functions independently enforce order status and refund/escalation constraints.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 957 |
| S2 | REFUND | 135.00 | 5 | 957 |
| S3 | ESCALATE | 1,999.00 | 5 | 944 |
| S4 | DENIED | 0.00 | 4 | 653 |
| S5 | DENIED | 0.00 | 3 | 373 |
| S6 | NOT_FOUND | 0.00 | 2 | 96 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,072 |

S7 repeatedly requests a policy-search tool and reaches the six-call iteration limit. Safety demos: S1 with `token_limit=200` ends `FALLBACK_TOKEN_LIMIT` after 3 model calls and 390 accumulated tokens (the tool requested by the over-budget response is not executed); S1 with `time_limit_seconds=0` ends `FALLBACK_TIMEOUT` before any model call (0 calls, 0 tokens). Each fallback includes completed steps and a short human-handoff summary.

### Checkpoint and resume

The crash/resume demo simulates a crash after the second tool execution, after its checkpoint has been written. `resume_agent(run_id)` loads the saved messages and continues without repeating a completed tool. The complete tool sequence is exactly `get_order → search_policy → calculate_refund → issue_refund` (4 executions); the final status is `REFUND` for INR 128.25. Checkpoints in `state/` contain run ID, scenario ID, messages, accumulated tokens, step number, executed-tool list and run settings.

### ReAct versus plan-and-execute (under 250 words)

Measured ReAct runs used 5 calls and 957 tokens for S1, 3 calls and 373 tokens for S5, and 2 calls and 96 tokens for S6. Plan-and-execute can request a whole tool plan once and may reduce repeated prompt context, although the initial plan can be longer and exact token savings depend on serialization. ReAct sees each observation before selecting the next action, so it can stop immediately for `NOT_FOUND` or deny a cancelled order rather than blindly following a plan that assumed a completed order. Plan-and-execute can also handle surprises if the executor validates each step and replans or stops safely. For refunds, ReAct is chosen with independent tool guards, iteration/token/time limits and checkpoints: extra calls buy observation-driven decisions while financial side effects remain bounded.

## Limitations and Next Steps

All data are synthetic and the model is a deterministic offline mock, so the metrics demonstrate reproducibility and guardrail behavior rather than production LLM quality. The retention list is illustrative and not a leakage-free evaluation because its Premium flag is a post-hoc whole-window label; the model features themselves use only pre-cutoff orders and do not use segment labels. Part C's few-shot performance is specific to the mock's programmed response behavior. Part D's retrieval evaluation uses only 14 questions (12 answerable, 2 unanswerable), and the seven-day policy window is assumed rather than validated against a real request timestamp. Before production, use a reviewed model endpoint only after privacy approval, implement secure secret management, validate tool arguments and order state independently, add idempotency keys for ledger side effects, use a production tokenizer, and monitor category drift, retries, human-review load and retrieval quality.
