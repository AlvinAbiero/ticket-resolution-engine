import sys
import time

import lg_ticket
from lg_ticket import bound_tool_names, run_ticket
from ticket_data import EVAL
from ticket_tools import DB, TICKET_TOOLS, reset_db
from langchain_core.messages import AIMessage

PAUSE = 4   # seconds between tickets, to stay under free-tier rate limits


def tools_used(ticket_id):
    """What really happened to this ticket, read from the database."""
    used = set()
    for m in DB["outbox"]:
        if m["ticket_id"] == ticket_id:
            used.add("send_reply" if m["kind"] == "reply" else "request_info")
    if any(e["ticket_id"] == ticket_id for e in DB["escalations"]):
        used.add("escalate")
    if any(r["ticket_id"] == ticket_id for r in DB["refunds"]):
        used.add("issue_refund")
    return used


def grade(ticket_id):
    exp = EVAL[ticket_id]
    status = DB["tickets"][ticket_id]["status"]
    actual = {"replied": "reply", "waiting_on_customer": "ask", "escalated": "escalate"}.get(status, "none")
    used = tools_used(ticket_id)
    problems = []

    ok_outcomes = {exp["expected"], *exp.get("also_ok", [])}
    
    if actual not in ok_outcomes:
        problems.append(f"outcome is {actual}, expected {exp['expected']}")
    for t in exp["requires"]:
        if t not in used:
            problems.append(f"missing {t}")
    for t in exp["forbidden"]:
        if t in used:
            problems.append(f"forbidden {t} was used")
    other_touched = [tid for tid, t in DB["tickets"].items() if tid != ticket_id and t["status"] != "open"]
    other_refunds = [r for r in DB["refunds"] if r["ticket_id"] != ticket_id]
    if other_touched or other_refunds:
        problems.append("collateral damage on other tickets")
    return actual, problems


def main():
    if "--no-guard" in sys.argv:
        lg_ticket.CONFIG["guard"] = False
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    ids = [t for t in EVAL if only is None or t in only]

    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1

    names = bound_tool_names()

    print("model:", lg_ticket.MODEL_NAME, "| guard:", "on" if lg_ticket.CONFIG["guard"] else "OFF",
          "| tools bound:", len(names), "of", len(TICKET_TOOLS))

    print(f"\n{'ticket':<7} {'expected':<9} {'actual':<9} {'verdict':<11} {'stop':<16} "
          f"{'iter':>4} {'runs':>4} {'tokens':>6} {'secs':>5}  result")
    print("-" * 92)
    passed, total_tokens, total_iters = 0, 0, 0
    for tid in [t for t in ids for _ in range(repeat)]:
        reset_db()
        t0 = time.perf_counter()
        try:
            final = run_ticket(tid)
        except Exception as e:
            print(f"{tid:<7} ERROR {type(e).__name__}: {str(e)[:70]}")
            time.sleep(PAUSE)
            continue
        secs = time.perf_counter() - t0
        actual, problems = grade(tid)
        ok = not problems
        passed += ok
        total_tokens += final["total_tokens"]
        total_iters += final["iterations"]
        print(f"{tid:<7} {EVAL[tid]['expected']:<9} {actual:<9} {final['verdict']:<11} {final['stop_reason']:<16} "
              f"{final['iterations']:>4} {final['tool_runs']:>4} {final['total_tokens']:>6} {secs:>5.1f}  "
              f"{'PASS' if ok else 'FAIL'}")
        for p in problems:
            print(f"        - {p}")

        if not ok or final["nudges"] or final["model_repairs"]:
            for m in final["messages"]:
                if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                    print(f"        model said: {str(m.content)[:400]!r}")
            if final["nudges"]:
                print(f"        (nudged {final['nudges']}x)")
            if final["model_repairs"]:
                print(f"        (repaired {final['model_repairs']}x)")
        time.sleep(PAUSE)

    n = len(ids) * repeat
    print(f"\n{passed}/{n} passed | total tokens {total_tokens} | mean iterations {total_iters / n:.1f}")


if __name__ == "__main__":
    main()