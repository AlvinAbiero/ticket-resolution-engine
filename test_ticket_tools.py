from ticket_data import CUSTOMERS, EVAL, KB
from ticket_tools import DB, TICKET_TOOLS, call_tool, reset_db

passed = failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print("PASS", name)
    else:
        failed += 1
        print("FAIL", name)

reset_db()
print(len(DB["tickets"]), "tickets,", len(KB), "KB articles\n")

r = call_tool("search_kb", {"query": "reset my password"})
check("kb: password query ranks KB-1 first", bool(r["results"]) and r["results"][0]["id"] == "KB-1")
r = call_tool("search_kb", {"query": "refund policy"})
check("kb: refund query finds KB-3", any(a["id"] == "KB-3" for a in r["results"]))
r = call_tool("search_kb", {"query": "zzzz qqqq"})
check("kb: nonsense query returns nothing", r["results"] == [])

check("get_ticket: unknown id is an error", "error" in call_tool("get_ticket", {"ticket_id": "T-999"}))
r = call_tool("get_ticket", {"ticket_id": "X1"})
check("validation: malformed id names the field", "error" in r and "ticket_id" in r["error"])
check("get_customer: plan is returned", call_tool("get_customer", {"customer_id": "C-101"}).get("plan") == "pro")

r = call_tool("issue_refund", {"ticket_id": "T-004", "amount": 300})
check("refund: above the limit is refused", "error" in r and DB["refunds"] == [])
r = call_tool("issue_refund", {"ticket_id": "T-003", "amount": 49})
check("refund: within the limit works", r.get("status") == "refunded" and len(DB["refunds"]) == 1)
check("refund: ticket stays open so a reply can follow", DB["tickets"]["T-003"]["status"] == "open")
check("refund: a second refund on the same ticket is refused",
      "error" in call_tool("issue_refund", {"ticket_id": "T-003", "amount": 49}))

check("reply: too-short message is rejected",
      "error" in call_tool("send_reply", {"ticket_id": "T-001", "message": "ok"}))
msg = "Use the Forgot password link on the login page to reset it."
r = call_tool("send_reply", {"ticket_id": "T-001", "message": msg})
check("reply: valid reply is sent and the ticket moves on",
      r.get("status") == "sent" and DB["tickets"]["T-001"]["status"] == "replied" and len(DB["outbox"]) == 1)
check("reply: a second reply to the same ticket is refused",
      "error" in call_tool("send_reply", {"ticket_id": "T-001", "message": msg}))

r = call_tool("escalate", {"ticket_id": "T-005", "reason": "Lost 2FA device, identity check needed", "priority": "high"})
check("escalate: ticket is marked escalated",
      r.get("status") == "escalated" and DB["tickets"]["T-005"]["status"] == "escalated")
check("escalate: cannot reply after escalating",
      "error" in call_tool("send_reply", {"ticket_id": "T-005", "message": "We are looking into your request now."}))

call_tool("request_info", {"ticket_id": "T-006", "question": "What exactly isn't working, and what do you see?"})
check("request_info: ticket waits on the customer", DB["tickets"]["T-006"]["status"] == "waiting_on_customer")

reset_db()
check("reset_db: statuses, outbox and refunds are back to the start",
      all(t["status"] == "open" for t in DB["tickets"].values()) and DB["outbox"] == [] and DB["refunds"] == [])

check("data: every EVAL ticket exists", set(EVAL) == set(DB["tickets"]))
check("data: every ticket's customer exists", all(t["customer_id"] in CUSTOMERS for t in DB["tickets"].values()))
used = {n for e in EVAL.values() for n in e["requires"] + e["forbidden"]}
check("data: EVAL only names real tools", used <= set(TICKET_TOOLS))
check("data: expected outcomes are valid", all(e["expected"] in {"reply", "ask", "escalate"} for e in EVAL.values()))

print(f"\n{passed} passed, {failed} failed")