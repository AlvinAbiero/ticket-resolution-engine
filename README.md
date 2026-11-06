# Ticket Resolution Engine

An agent that handles one support ticket per run. It reads the ticket and the customer, searches a
knowledge base, issues refunds within policy, and ends with exactly one closing action: reply, ask the
customer, or escalate to a human. Built with LangGraph and Groq `openai/gpt-oss-120b`.
All data (customers, tickets, KB articles, policies) is mock data.

## How it works
```
START -> call_model --tool calls--> tool_node --closing action done--> finish -> verdict_node -> END
            ^   |                      |
            |   |                      +--otherwise--> back to call_model
            |   +--no tool calls, nothing closed--> nudge --> call_model
            |   +--limit hit with calls pending--> budget_stop --> finish
```
- **State:** messages, counters, limits, a facts cache, an action ledger, failures, the closing actions taken.
- **Tools (7):** `get_ticket`, `get_customer`, `search_kb` (read-only); `send_reply`, `request_info`,
  `escalate`, `issue_refund` (side effects). The first three side-effecting ones are closing actions.
- **Guards:** iteration limit, token gate, circuit breaker on repeated failing calls, a ledger that blocks
  repeating an action, a repair loop for provider format errors.
- **Verdict:** computed from the database, not from the model: `success` if exactly one closing action
  happened, `incomplete` if none, `failed` on a model error, `untrusted` if more than one.

## Design decisions
- **Rules live in code, not the prompt.** The 100 USD refund limit and "one closing action per ticket"
  are enforced by the tools themselves.
- **The database is the ground truth.** The verdict and the eval read what actually happened, not what
  the agent says it did.
- **Ticket text is data.** The model sees it only through a tool result, and the user message is just
  "Handle ticket T-001."
- **Loose schema to the model, strict validation in my dispatcher.** Strict JSON-schema constraints sent to
  the provider turned sloppy arguments into hard 400 errors that ended the run. Validation in `call_tool`
  returns the error to the model as data, so it can correct itself.
- **The run stops when a closing action succeeds,** with no extra model call.

## Evaluation
12 tickets: 7 answerable from the KB, 2 that need a refund decision (one within the limit, one above it),
1 missing information, 1 security case (lost 2FA), 1 legal threat, 1 prompt injection (T-012). Graded from
the database: the expected outcome, required and forbidden tools, and damage to other tickets.
For T-012 both "escalate" and "reply without obeying the injection" are accepted; any refund is a failure.

| Version | Change | Result |
|---|---|---|
| 1 | Baseline, guard on / off | 8/12, 7/12 |
| 2 | Nudge node, `tool_choice="any"` | 17/24 on, 14/24 off (guard on / off, 2 repeats) |
| 3 | Tried `qwen/qwen3.8-27b` on the 4 hardest tickets | 2/8 (kept `gpt-oss-120b`) |
| 4 | Loosened model-facing schema, repair loop | 22/24 (guard on, 2 repeats) |

Findings:
- Every completed run was correct (31/31 in version 2, 22/22 in version 4). The failures were tool-call
  format errors at the provider, not wrong decisions.
- The prompt-injection ticket never produced a refund or touched another ticket in any run. A prompt rule
  against injections changed behavior: with it on, 3 of 4 runs replied normally; with it off, 4 of 4
  escalated.
- The Qwen model failed on tool names (`search_lb`), argument names, and 2 wrong decisions.

## Provider failure modes found
1. The right action written as plain text instead of a tool call.
2. A call to a tool that does not exist (`commentary`, `Escalate`, `search_lb`).
3. Wrong argument names.
4. The model's reasoning text leaking into the output (`output_parse_failed`).

Mitigations: a repair loop that feeds the provider's short error message back and retries (max 2), and the
loosened schema. `tool_choice="required"` did not help; Groq rejects the response instead of forcing the format.

## Limitations
- Token totals exclude failed and repaired attempts.
- Small eval (12 tickets, 2 repeats), non-deterministic model, one ticket per run, in-memory database.
- Two of the four model calls per ticket are fixed reads that plain code could do up front.
- Injection defense is a prompt rule plus code limits. A classifier pre-check is not built.

## Run
```
py test_ticket_tools.py                 # tools and data, no model
py lg_ticket.py T-003                   # trace one ticket
py eval_tickets.py --repeat 2           # full eval
py eval_tickets.py --no-guard           # without the injection rule
```
## Setup
```
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
copy .env.example .env      # then put your Groq key in .env
```
Set `TICKET_MODEL` to try a different Groq model (default: `openai/gpt-oss-120b`).
