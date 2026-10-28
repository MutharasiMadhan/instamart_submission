"""Part D: retrieval-grounded policy assistant and resumable refund agent.
All LLM interactions use the deterministic, offline mock_llm.py.
"""
from __future__ import annotations
import csv, json, os, re, sqlite3, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "instamart.db"
OUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
OUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)


def make_chunks(overlap: int = 10) -> list[dict[str, str]]:
    """Split policy files into 40-word chunks with configurable overlap."""
    if not 0 <= overlap < 40:
        raise ValueError("overlap must be from 0 through 39")
    chunks: list[dict[str, str]] = []
    step = 40 - overlap
    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            part = words[start:start + 40]
            chunks.append({"source": path.name, "text": " ".join(part)})
            if start + 40 >= len(words):
                break
            start += step
    return chunks


class PolicyRetriever:
    def __init__(self, overlap: int = 10):
        self.chunks = make_chunks(overlap=overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix).ravel()
        # Stable tie-break by original chunk order.
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [{"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in order]


RETRIEVER = PolicyRetriever(overlap=10)


def answer(question: str, retriever: PolicyRetriever | None = None) -> dict[str, Any]:
    r = retriever or RETRIEVER
    hits = r.retrieve(question, k=3)
    context = "\n".join(f"[{hit['source']}]\n{hit['text']}" for hit in hits)
    user_content = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    system = ("You are a POLICY ASSISTANT. Answer only from the supplied CONTEXT. "
              "If the answer is not stated in the context, reply exactly: "
              "I could not find that information in the policy documents.")
    response = mock_llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user_content}])
    return {"answer": response.choices[0].message.content or "", "sources": list(dict.fromkeys(h["source"] for h in hits)), "retrieved": hits,
            "tokens_used": response.usage.total_tokens}


def evaluate_retrieval(retriever: PolicyRetriever) -> dict[str, Any]:
    eval_df = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)
    rows = []
    for row in eval_df.to_dict("records"):
        result = answer(row["question"], retriever=retriever)
        if row["expected_doc"] == "NONE":
            rows.append({"question": row["question"], "expected_doc": "NONE", "doc_hit": "", "fact_hit": "",
                         "fallback": result["answer"].startswith("I could not find"), "answer": result["answer"],
                         "sources": ";".join(result["sources"])})
        else:
            rows.append({"question": row["question"], "expected_doc": row["expected_doc"],
                         "doc_hit": row["expected_doc"] in [h["source"] for h in result["retrieved"]],
                         "fact_hit": row["expected_fact"] in result["answer"], "fallback": "",
                         "answer": result["answer"], "sources": ";".join(result["sources"])})
    result_df = pd.DataFrame(rows)
    answerable = result_df[result_df.expected_doc != "NONE"]
    unanswerable = result_df[result_df.expected_doc == "NONE"]
    return {"details": result_df, "doc_hits": int(answerable.doc_hit.sum()), "fact_hits": int(answerable.fact_hit.sum()),
            "fallback_hits": int(unanswerable.fallback.sum()), "answerable": len(answerable), "unanswerable": len(unanswerable)}


# Tool definitions use JSON-schema-style parameter specifications.
TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up a single order by its order id in the Instamart SQLite database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the most relevant policy chunks for a policy query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate the late-delivery refund amount from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["integer", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Record an automatically approved refund of up to INR 1000 in the refund ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Record a refund above INR 1000 for approval by a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query: str) -> dict[str, Any]:
    return {"results": RETRIEVER.retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: int | None) -> dict[str, float]:
    amount = float(order_amount) if delivery_minutes is not None and float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def _ledger(tool_name: str, order_id: str, refund_inr: float) -> dict[str, float]:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool_name": tool_name,
              "order_id": order_id, "refund_inr": round(float(refund_inr), 2)}
    with (OUT_DIR / "refund_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, Any]:
    order = get_order(order_id)
    if "error" in order:
        return {"error": "order not found"}
    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}
    if float(refund_inr) > 1000:
        return {"error": "refunds above INR 1000 must be escalated to a human"}
    if float(refund_inr) <= 0:
        return {"error": "zero-value refunds are not issued"}
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, Any]:
    order = get_order(order_id)
    if "error" in order:
        return {"error": "order not found"}
    if order["status"] != "completed":
        return {"error": "only completed orders can be considered for refund"}
    if float(refund_inr) <= 1000:
        return {"error": "refunds of INR 1000 or less are not escalated by the amount rule"}
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


class SimulatedCrash(RuntimeError):
    pass


def _checkpoint(run_id: str, scenario_id: str, messages: list[dict[str, Any]], tokens_used: int,
                llm_calls: int, steps: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    payload = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages,
               "tokens_used": tokens_used, "step_number": len(steps), "llm_calls": llm_calls,
               "tools_executed": [s["tool_name"] for s in steps], "steps": steps, "settings": settings}
    path = STATE_DIR / f"{run_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _tool_calls_from(message: Any) -> list[dict[str, Any]]:
    result = []
    for call in (getattr(message, "tool_calls", None) or []):
        name = getattr(call, "name", None) or call.get("name")
        args = getattr(call, "arguments", None) if not isinstance(call, dict) else call.get("arguments")
        if isinstance(args, str):
            args = json.loads(args)
        result.append({"id": getattr(call, "id", None) if not isinstance(call, dict) else call.get("id"),
                       "name": name, "arguments": args or {}})
    return result


def _continue_agent(run_id: str, state: dict[str, Any], crash_after_tool: int | None = None) -> dict[str, Any]:
    settings = state["settings"]
    max_iterations = int(settings["max_iterations"])
    token_limit = int(settings["token_limit"])
    time_limit_seconds = float(settings["time_limit_seconds"])
    messages = state["messages"]
    tokens_used = int(state["tokens_used"])
    llm_calls = int(state["llm_calls"])
    steps = state["steps"]
    run_start = float(settings.get("started_monotonic", time.monotonic()))
    while llm_calls < max_iterations:
        elapsed = time.monotonic() - run_start
        if elapsed >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "FALLBACK_TIMEOUT: handing this case to a human agent; no further automated action was taken.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "run_id": run_id}
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)
        if tokens_used > token_limit:
            state.update(messages=messages, tokens_used=tokens_used, llm_calls=llm_calls, steps=steps)
            _checkpoint(run_id, state["scenario_id"], messages, tokens_used, llm_calls, steps, settings)
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "FALLBACK_TOKEN_LIMIT: handing this case to a human agent because the token budget was exceeded.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "run_id": run_id}
        msg = response.choices[0].message
        calls = _tool_calls_from(msg)
        if not calls:
            final_answer = msg.content or ""
            status = final_answer.split(":", 1)[0].strip() if ":" in final_answer else "COMPLETED"
            state.update(messages=messages, tokens_used=tokens_used, llm_calls=llm_calls, steps=steps)
            _checkpoint(run_id, state["scenario_id"], messages, tokens_used, llm_calls, steps, settings)
            return {"status": status, "final_answer": final_answer, "steps": steps, "llm_calls": llm_calls,
                    "tokens_used": tokens_used, "run_id": run_id}
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            name, args = call["name"], call["arguments"]
            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                try:
                    observation = TOOL_MAP[name](**args)
                except Exception as exc:
                    observation = {"error": f"tool execution failed: {type(exc).__name__}: {exc}"}
            step = {"step": len(steps) + 1, "tool_name": name, "arguments": args, "observation": observation}
            steps.append(step)
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": name,
                             "content": json.dumps(observation, ensure_ascii=False, default=str)})
            state.update(messages=messages, tokens_used=tokens_used, llm_calls=llm_calls, steps=steps)
            _checkpoint(run_id, state["scenario_id"], messages, tokens_used, llm_calls, steps, settings)
            if crash_after_tool is not None and len(steps) == crash_after_tool:
                raise SimulatedCrash(f"simulated crash after tool execution {crash_after_tool}; checkpoint saved")
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "FALLBACK_MAX_ITERATIONS: handing this case to a human agent after reaching the model-call limit.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "run_id": run_id}


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6, token_limit: int = 6000,
              time_limit_seconds: float = 10, run_id: str | None = None,
              crash_after_tool: int | None = None) -> dict[str, Any]:
    run_id = run_id or f"{scenario_id.lower()}_{uuid.uuid4().hex[:8]}"
    state = {"run_id": run_id, "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": "You are an AGENT. Use tools to inspect orders, consult policy and follow refund rules. Do not invent observations."},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [],
             "settings": {"max_iterations": max_iterations, "token_limit": token_limit,
                          "time_limit_seconds": time_limit_seconds, "started_monotonic": time.monotonic()}}
    _checkpoint(run_id, scenario_id, state["messages"], 0, 0, [], state["settings"])
    return _continue_agent(run_id, state, crash_after_tool=crash_after_tool)


def resume_agent(run_id: str) -> dict[str, Any]:
    path = STATE_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for run id {run_id}")
    state = json.loads(path.read_text(encoding="utf-8"))
    return _continue_agent(run_id, state)


def _refund_amount(result: dict[str, Any]) -> float:
    for step in result["steps"]:
        if step["tool_name"] == "calculate_refund":
            return float(step["observation"].get("refund_inr", 0.0))
    return 0.0


def run_all() -> dict[str, Any]:
    # Build and persist the chunk inventory and retrieval evaluation.
    chunks_df = pd.DataFrame(RETRIEVER.chunks)
    chunks_df.insert(0, "chunk_id", range(len(chunks_df)))
    chunks_df.to_csv(OUT_DIR / "policy_chunks.csv", index=False)
    retrieval = evaluate_retrieval(RETRIEVER)
    retrieval["details"].to_csv(OUT_DIR / "policy_eval_results.csv", index=False)
    zero_retriever = PolicyRetriever(overlap=0)
    zero = evaluate_retrieval(zero_retriever)
    zero["details"].to_csv(OUT_DIR / "policy_eval_zero_overlap.csv", index=False)

    # Fresh ledger for a reproducible batch run.
    ledger = OUT_DIR / "refund_ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
    runs = []
    with (OUT_DIR / "agent_runs.jsonl").open("w", encoding="utf-8") as f:
        for row in scenarios.to_dict("records"):
            mock_llm.reset()
            result = run_agent(row["message"], row["scenario_id"], run_id=f"scenario_{row['scenario_id'].lower()}")
            record = {"scenario_id": row["scenario_id"], "status": result["status"], "final_answer": result["final_answer"],
                      "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"], "steps": result["steps"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            result["expected_outcome"] = row["expected_outcome"]
            result["expected_refund_inr"] = float(row["expected_refund_inr"])
            result["refund_inr"] = _refund_amount(result)
            runs.append(result)

        # Token-limit and timeout safety demonstrations.
        mock_llm.reset()
        token_demo = run_agent(scenarios.iloc[0]["message"], "S1_TOKEN_LIMIT", token_limit=200,
                               run_id="demo_s1_token_limit")
        f.write(json.dumps({"scenario_id": "S1_TOKEN_LIMIT", **{k: token_demo[k] for k in ("status", "final_answer", "llm_calls", "tokens_used", "steps")}}, ensure_ascii=False) + "\n")
        mock_llm.reset()
        timeout_demo = run_agent(scenarios.iloc[0]["message"], "S1_TIMEOUT", time_limit_seconds=0,
                                 run_id="demo_s1_timeout")
        f.write(json.dumps({"scenario_id": "S1_TIMEOUT", **{k: timeout_demo[k] for k in ("status", "final_answer", "llm_calls", "tokens_used", "steps")}}, ensure_ascii=False) + "\n")

        # Crash after the second observation; checkpoint is already durable.
        mock_llm.reset()
        resume_run_id = "demo_s1_crash_resume"
        try:
            run_agent(scenarios.iloc[0]["message"], "S1_CRASH_RESUME", run_id=resume_run_id, crash_after_tool=2)
        except SimulatedCrash:
            pass
        resumed = resume_agent(resume_run_id)
        f.write(json.dumps({"scenario_id": "S1_CRASH_RESUME", **{k: resumed[k] for k in ("status", "final_answer", "llm_calls", "tokens_used", "steps")}}, ensure_ascii=False) + "\n")

    # Summaries and assertions for the assignment acceptance criteria.
    answerable = retrieval["answerable"]
    unanswerable = retrieval["unanswerable"]
    main_table = []
    for row, result in zip(scenarios.to_dict("records"), runs):
        main_table.append({"scenario_id": row["scenario_id"], "status": result["status"],
                           "refund_inr": result["refund_inr"], "llm_calls": result["llm_calls"],
                           "tokens_used": result["tokens_used"], "expected_outcome": row["expected_outcome"],
                           "expected_refund_inr": float(row["expected_refund_inr"])})
    summary = {
        "chunk_counts_by_source": chunks_df.groupby("source").size().to_dict(),
        "chunk_count": len(chunks_df),
        "retrieval_overlap_10": {k: retrieval[k] for k in ("doc_hits", "fact_hits", "fallback_hits", "answerable", "unanswerable")},
        "retrieval_overlap_0": {k: zero[k] for k in ("doc_hits", "fact_hits", "fallback_hits", "answerable", "unanswerable")},
        "scenarios": main_table,
        "token_limit_demo": {k: token_demo[k] for k in ("status", "llm_calls", "tokens_used")},
        "timeout_demo": {k: timeout_demo[k] for k in ("status", "llm_calls", "tokens_used")},
        "crash_resume": {"status": resumed["status"], "refund_inr": _refund_amount(resumed),
                         "tools_in_order": [s["tool_name"] for s in resumed["steps"]], "llm_calls": resumed["llm_calls"],
                         "tokens_used": resumed["tokens_used"]},
    }
    (OUT_DIR / "part_d_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Core hard assertions.
    assert len(chunks_df) == 15, f"expected 15 chunks, got {len(chunks_df)}"
    assert summary["chunk_counts_by_source"] == {"delivery_policy.txt": 3, "escalation_policy.txt": 3, "payment_policy.txt": 3, "privacy_policy.txt": 2, "refund_policy.txt": 4}
    assert (retrieval["doc_hits"], retrieval["fact_hits"], retrieval["fallback_hits"]) == (12, 12, 2), summary["retrieval_overlap_10"]
    assert zero["doc_hits"] < 12 or zero["fact_hits"] < 12, "zero-overlap evaluation should fail at least one answerable question"
    for result in runs:
        assert (result["status"] == result["expected_outcome"] or (result["expected_outcome"] == "FALLBACK" and result["status"] == "FALLBACK_MAX_ITERATIONS")), (result["run_id"], result["status"], result["expected_outcome"])
        assert abs(result["refund_inr"] - result["expected_refund_inr"]) < 0.01, result
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT", token_demo
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT", timeout_demo
    assert resumed["status"] == "REFUND" and abs(_refund_amount(resumed) - 128.25) < 0.01
    assert [s["tool_name"] for s in resumed["steps"]] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    return summary


def main() -> None:
    summary = run_all()
    print("Part D complete")
    print("Chunks:", summary["chunk_counts_by_source"])
    print("Retrieval (overlap=10):", summary["retrieval_overlap_10"])
    print("Retrieval (overlap=0):", summary["retrieval_overlap_0"])
    print("Scenarios:")
    for row in summary["scenarios"]:
        print(row)
    print("Token demo:", summary["token_limit_demo"])
    print("Timeout demo:", summary["timeout_demo"])
    print("Crash/resume:", summary["crash_resume"])

if __name__ == "__main__":
    main()
