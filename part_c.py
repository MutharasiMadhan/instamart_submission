"""Part C: privacy-guarded, resilient support ticket triage using mock_llm."""
from __future__ import annotations
import json, math, random, re, time
from pathlib import Path
from collections import Counter
import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parent
RAW_TICKETS = ROOT / "data/raw/tickets.csv"
SEGMENTS = ROOT / "outputs/customer_segments.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CALL_AUDIT = []  # records every attempted call at the model boundary, including retries
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED = {"billing", "delivery", "technical", "fraud"}
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = '''CLASSIFY the customer's support issue into exactly one allowed category: billing, delivery, technical, fraud. Return JSON only in the form {"category": "...", "reason": "..."}. The reason should be concise and based only on the supplied message. If uncertain, choose the best-supported allowed category.'''
FEW_SHOT_PROMPT = '''CLASSIFY the customer's support issue into exactly one allowed category: billing, delivery, technical, fraud. Return JSON only in the form {"category": "...", "reason": "..."}. The reason should be concise and based only on the supplied message.
EXAMPLES:
Message: I was charged twice for an order.
Reply: {"category": "billing", "reason": "Duplicate charge."}
Message: The rider marked my order delivered but I never received it.
Reply: {"category": "delivery", "reason": "Package not received."}
Message: Someone hacked my account and used my wallet.
Reply: {"category": "fraud", "reason": "Unauthorized account activity."}
Message: The app crashes when I open the cart.
Reply: {"category": "technical", "reason": "App malfunction."}'''


def redact(text):
    out = text or ""
    counts = {}
    for name, pattern, replacement in PATTERNS:
        out, n = pattern.subn(replacement, out)
        counts[name] = n
    return out, counts


def assert_redacted(text):
    for name, pattern, _ in PATTERNS:
        assert pattern.search(text) is None, f"Unredacted {name} detected before model call"


def validate_reply(reply):
    try:
        parsed = json.loads(reply)
        if isinstance(parsed, dict) and parsed.get("category") in ALLOWED:
            return parsed["category"], "classified_json"
        # Valid JSON with a disallowed category must be flagged, not keyword parsed.
        return "", "flagged_invalid_output"
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(reply).lower())
        hits = [w for w in words if w in ALLOWED]
        if len(hits) == 1:
            return hits[0], "classified_keyword"
        return "", "flagged_invalid_output"


def retry_call(prompt, content, ticket_id, retry_log):
    for attempt in range(4):  # initial attempt + maximum 3 retries
        try:
            assert_redacted(content)
            CALL_AUDIT.append({"ticket_id": ticket_id, "user_content": content})
            return mock_llm.chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ])
        except mock_llm.TransientLLMError:
            if attempt >= 3:
                raise
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            line = f"retry ticket_id={ticket_id} retry_number={attempt} wait_seconds={wait:.4f}"
            retry_log.append(line)
            print(line)
            time.sleep(wait)
    raise RuntimeError("unreachable retry state")


def run_pipeline(prompt, *, token_budget=None, output_name=None, collect_detail=False):
    tickets = pd.read_csv(RAW_TICKETS, dtype=str, keep_default_na=False)
    segments = pd.read_csv(SEGMENTS, dtype={"customer_id": str})
    segment_map = segments.set_index("customer_id")["segment_name"].to_dict()
    premium_map = segments.set_index("customer_id")["is_premium"].astype(str).str.lower().isin(["true", "1", "yes"]).to_dict()

    global CALL_AUDIT
    mock_llm.reset()  # required immediately before every full run
    CALL_AUDIT = []
    # Deterministic jitter makes the run reproducible while staying in the required range.
    random.seed(42)
    running_tokens = 0
    retry_log, rows, raw_replies = [], [], []
    redaction_totals = Counter()
    for _, ticket in tickets.iterrows():
        tid, cid, raw_text = ticket["ticket_id"], ticket["customer_id"], ticket["text"]
        status, category, tokens_used, reply = "", "", 0, ""
        # Input guardrails are applied before any model call and in file order.
        if not raw_text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            redacted, counts = redact(raw_text)
            redaction_totals.update(counts)
            assert_redacted(redacted)
            if token_budget is not None and running_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                # This exact redacted string is the complete user content; no metadata is appended.
                response = retry_call(prompt, redacted, tid, retry_log)
                reply = response.choices[0].message.content or ""
                tokens_used = int(response.usage.total_tokens)
                running_tokens += tokens_used
                category, status = validate_reply(reply)
                raw_replies.append({"ticket_id": tid, "reply": reply, "status": status, "category": category})
        segment_name = segment_map.get(cid, "")
        is_premium = premium_map.get(cid, False)
        if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        rows.append({"ticket_id": tid, "customer_id": cid, "status": status,
                     "category": category, "segment_name": segment_name,
                     "priority": priority, "route": route, "tokens_used": tokens_used,
                     "true_category": ticket["true_category"], "_reply": reply})

    df = pd.DataFrame(rows)
    if output_name:
        df[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]].to_csv(OUT / output_name, index=False)
    eligible = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = (eligible.category == eligible.true_category).sum()
    accuracy = float(correct / len(eligible)) if len(eligible) else float("nan")
    summary = {
        "tickets": len(df), "status_counts": {k: int(v) for k, v in df.status.value_counts().items()},
        "classified": len(eligible), "correct": int(correct), "accuracy": accuracy,
        "total_tokens": int(running_tokens), "tokens_per_classified_ticket": float(running_tokens / len(eligible)) if len(eligible) else 0,
        "redactions": {k: int(redaction_totals[k]) for k, _, _ in PATTERNS},
        "redactions_total": int(sum(redaction_totals.values())), "redaction_leftovers": 0,
        "retry_count": len(retry_log), "retry_log": retry_log,
        "premium_rows_in_final": int((df.route == "premium_desk").sum()),
    }
    # Guardrail proof: model call IDs must exclude empty and over-limit inputs.
    ticket_lookup = tickets.set_index("ticket_id")
    for called_id in {entry["ticket_id"] for entry in CALL_AUDIT}:
        original = ticket_lookup.loc[called_id, "text"]
        assert original.strip() and mock_llm.est_tokens(original) <= MAX_INPUT_TOKENS
    # Privacy proof at the actual mock boundary: all submitted user content is clean.
    for entry in CALL_AUDIT:
        assert_redacted(entry["user_content"])
        # The user message content is exactly the redacted ticket text, with no metadata.
        assert entry["user_content"] == redact(ticket_lookup.loc[entry["ticket_id"], "text"])[0]
    if collect_detail:
        summary["dataframe"] = df
        summary["raw_replies"] = raw_replies
    return summary


def main():
    # Count redactions over all tickets, including guardrailed tickets, for the report.
    tickets = pd.read_csv(RAW_TICKETS, dtype=str, keep_default_na=False)
    all_counts = Counter()
    for txt in tickets.text:
        redacted, counts = redact(txt)
        all_counts.update(counts)
        assert_redacted(redacted)
    assert len(tickets) == 164
    assert sum(all_counts.values()) == 278, f"Unexpected redaction total: {sum(all_counts.values())}"

    zero = run_pipeline(ZERO_SHOT_PROMPT, collect_detail=True)
    few = run_pipeline(FEW_SHOT_PROMPT, collect_detail=True)
    budget = run_pipeline(ZERO_SHOT_PROMPT, token_budget=6000, output_name="triage_budget_demo.csv")
    if budget["status_counts"].get("budget_exhausted", 0) == 0:
        budget = run_pipeline(ZERO_SHOT_PROMPT, token_budget=5000, output_name="triage_budget_demo.csv")
        budget["budget_used"] = 5000
    else:
        budget["budget_used"] = 6000
    final = run_pipeline(FEW_SHOT_PROMPT, token_budget=100000, output_name="triage_results.csv")
    assert zero["status_counts"].get("classified_json") == 135
    assert zero["status_counts"].get("classified_keyword") == 12
    assert zero["status_counts"].get("flagged_invalid_output") == 9
    assert zero["classified"] == 147 and abs(zero["accuracy"] - 126 / 147) < 1e-12
    assert few["classified"] == 147 and few["accuracy"] == 1.0
    assert zero["retry_count"] == few["retry_count"] == 19
    final_csv = pd.read_csv(OUT / "triage_results.csv", keep_default_na=False)
    assert len(final_csv) == 164 and list(final_csv.columns) == ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    assert budget["status_counts"].get("budget_exhausted", 0) > 0

    # Confusion table for zero-shot, using classified tickets only.
    zdf = zero["dataframe"]
    classified = zdf[zdf.status.isin(["classified_json", "classified_keyword"])]
    confusion = pd.crosstab(classified.true_category, classified.category, dropna=False).reindex(index=["billing", "delivery", "technical", "fraud"], columns=["billing", "delivery", "technical", "fraud"], fill_value=0)
    confusion.to_csv(OUT / "zero_shot_confusion.csv")
    report = [
        "# Part C — Support Ticket Triage Results", "",
        "## Guardrails and privacy", "",
        f"Tickets: {len(tickets)}; empty skipped: {(tickets.text.str.strip() == '').sum()}; long flagged: {(tickets.text.map(mock_llm.est_tokens) > MAX_INPUT_TOKENS).sum()}.",
        f"Redactions across all 164 ticket texts: email={all_counts['email']}, phone={all_counts['phone']}, order_id={all_counts['order_id']}, total={sum(all_counts.values())}; residual matches after redaction: 0.",
        "The code asserts that no email, phone, or order-ID pattern remains in any redacted text before a model call. The mock-only ticket ID is metadata and is not part of the user message content.", "",
        "## Zero-shot vs few-shot", "",
        "| Metric | Zero-shot | Few-shot |", "|---|---:|---:|",
        f"| Classified tickets | {zero['classified']} | {few['classified']} |",
        f"| Accuracy on classified tickets | {zero['accuracy']:.1%} | {few['accuracy']:.1%} |",
        f"| Total response tokens | {zero['total_tokens']} | {few['total_tokens']} |",
        f"| Tokens per classified ticket | {zero['tokens_per_classified_ticket']:.1f} | {few['tokens_per_classified_ticket']:.1f} |",
        "", "### Zero-shot confusion table", "", confusion.to_markdown(), "",
        f"Few-shot uses {few['total_tokens']-zero['total_tokens']} additional tokens ({few['tokens_per_classified_ticket']-zero['tokens_per_classified_ticket']:.1f} more per classified ticket). In this mock run, it improves classified-ticket accuracy from {zero['accuracy']:.1%} to {few['accuracy']:.1%}; that is worth paying for when the cost of misrouting a fraud or other support case exceeds the small incremental token cost. If production volume or token prices make that difference material, reserve few-shot prompting for ambiguous or high-risk tickets and monitor accuracy by category.", "",
        "## Retry and status checks", "",
        f"Zero-shot retries logged: {zero['retry_count']}; few-shot retries logged: {few['retry_count']}; each full run resets the mock first. Status counts in the few-shot full run: `{json.dumps(few['status_counts'], sort_keys=True)}`.",
        f"Budget demo used {budget['budget_used']} tokens as its stopping threshold and ended with {budget['status_counts'].get('budget_exhausted', 0)} budget_exhausted tickets.", "",
        "## Routing and human review", "",
        "`triage_results.csv` contains 164 rows. Empty, long, invalid-output, and budget-exhausted tickets route to human_review; fraud routes to security_team at HIGH priority; Premium customers with otherwise valid categories route to premium_desk at HIGH; remaining tickets route to support_queue at NORMAL. Valid JSON and the exactly-one-category keyword fallback are rule-validated classifications; flagged outputs and guardrail/budget cases go to human review.", "",
        "## Limitations and next steps", "",
        "This is a deterministic synthetic mock, not evidence of production LLM quality. The original Part B Premium flag is a post-hoc whole-window label, and any retention list built from it is illustrative rather than a leakage-free evaluation. In production, use a real tokenizer, secure transport, secret management, human auditing, and monitoring for category drift.", ""
    ]
    (OUT / "part_c_report.md").write_text("\n".join(report), encoding="utf-8")
    summary = {"redactions_all_tickets": {k: int(all_counts[k]) for k, _, _ in PATTERNS},
               "zero_shot": {k:v for k,v in zero.items() if k not in {"dataframe", "raw_replies", "retry_log"}},
               "few_shot": {k:v for k,v in few.items() if k not in {"dataframe", "raw_replies", "retry_log"}},
               "budget_demo": {k:v for k,v in budget.items() if k not in {"dataframe", "raw_replies", "retry_log"}},
               "final_few_shot": {k:v for k,v in final.items() if k not in {"dataframe", "raw_replies", "retry_log"}}}
    (OUT / "part_c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT / 'triage_results.csv'} and {OUT / 'triage_budget_demo.csv'}")

if __name__ == "__main__":
    main()
