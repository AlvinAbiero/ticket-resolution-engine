import copy
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ticket_data import CUSTOMERS, KB, TICKETS_SEED

REFUND_LIMIT = 100.0
STOP = {"the", "and", "for", "with", "you", "your", "can", "how", "are", "not", "was",
        "has", "this", "that", "from", "have", "what", "where", "when", "does", "did",
        "get", "want"}

DB = {}

def reset_db():
    DB["tickets"] = copy.deepcopy(TICKETS_SEED)
    DB["outbox"] = []
    DB["refunds"] = []
    DB["escalations"] = []

reset_db()

TICKET_ID = r"^T-\d{3}$"
CUSTOMER_ID = r"^C-\d{3}$"

class GetTicketArgs(BaseModel):
    ticket_id: str = Field(pattern=TICKET_ID, description="Ticket id, for example T-004")


class GetCustomerArgs(BaseModel):
    customer_id: str = Field(pattern=CUSTOMER_ID, description="Customer id, for example C-103")


class SearchKbArgs(BaseModel):
    query: str = Field(min_length=3, description="Keywords describing the customer's problem")


class SendReplyArgs(BaseModel):
    ticket_id: str = Field(pattern=TICKET_ID, description="Ticket id, for example T-004")
    message: str = Field(min_length=20, description="The reply to send to the customer")


class RequestInfoArgs(BaseModel):
    ticket_id: str = Field(pattern=TICKET_ID, description="Ticket id, for example T-004")
    question: str = Field(min_length=10, description="The question to ask the customer")


class EscalateArgs(BaseModel):
    ticket_id: str = Field(pattern=TICKET_ID, description="Ticket id, for example T-004")
    reason: str = Field(min_length=10, description="Why a human needs to handle this ticket")
    priority: Literal["normal", "high", "urgent"] = Field(default="normal", description="How urgent the escalation is")


class IssueRefundArgs(BaseModel):
    ticket_id: str = Field(pattern=TICKET_ID, description="Ticket id, for example T-004")
    amount: float = Field(gt=0, description="Refund amount in USD")

def _words(text):
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOP}

def _require_open(ticket_id):
    t = DB["tickets"].get(ticket_id)
    if t is None:
        return {"error": f"ticket {ticket_id} not found"}
    if t["status"] != "open":
        return {"error": f"ticket {ticket_id} is not open (status: {t['status']})"}
    return None

def get_ticket(ticket_id):
    t = DB["tickets"].get(ticket_id)
    if t is None:
        return {"error": f"ticket {ticket_id} not found"}
    return {"ticket_id": ticket_id, **t}

def get_customer(customer_id):
    c = CUSTOMERS.get(customer_id)
    if c is None:
        return {"error": f"customer {customer_id} not found"}
    return {"customer_id": customer_id, **c}

def search_kb(query):
    q = _words(query)
    scored = []
    for a in KB:
        text = a["title"] + " " + a["body"] + " " + " ".join(a["keywords"])
        score = len(q & _words(text))
        if score:
            scored.append((score, a))
    scored.sort(key=lambda s: -s[0])
    return {"results": [{"id": a["id"], "title": a["title"], "body": a["body"]}
                        for _, a in scored[:3]]}

def send_reply(ticket_id, message):
    err = _require_open(ticket_id)
    if err:
        return err
    DB["outbox"].append({"ticket_id": ticket_id, "kind": "reply", "message": message})
    DB["tickets"][ticket_id]["status"] = "replied"
    return {"status": "sent", "ticket_id": ticket_id}


def request_info(ticket_id, question):
    err = _require_open(ticket_id)
    if err:
        return err
    DB["outbox"].append({"ticket_id": ticket_id, "kind": "question", "message": question})
    DB["tickets"][ticket_id]["status"] = "waiting_on_customer"
    return {"status": "asked", "ticket_id": ticket_id}

def escalate(ticket_id, reason, priority="normal"):
    err = _require_open(ticket_id)
    if err:
        return err
    DB["escalations"].append({"ticket_id": ticket_id, "reason": reason, "priority": priority})
    DB["tickets"][ticket_id]["status"] = "escalated"
    return {"status": "escalated", "ticket_id": ticket_id, "priority": priority}


def issue_refund(ticket_id, amount):
    err = _require_open(ticket_id)
    if err:
        return err
    if amount > REFUND_LIMIT:
        return {"error": f"amount {amount:.2f} exceeds the {REFUND_LIMIT:.2f} limit; "
                         "refunds above it need human approval, so escalate instead"}
    if any(r["ticket_id"] == ticket_id for r in DB["refunds"]):
        return {"error": f"a refund was already issued for ticket {ticket_id}"}
    DB["refunds"].append({"ticket_id": ticket_id, "amount": amount})
    return {"status": "refunded", "ticket_id": ticket_id, "amount": amount}

TICKET_TOOLS = {
    "get_ticket": {"func": get_ticket, "args_model": GetTicketArgs, "side_effect": False,
                   "description": "Read a support ticket by id (subject, body, status, customer_id)."},
    "get_customer": {"func": get_customer, "args_model": GetCustomerArgs, "side_effect": False,
                     "description": "Read a customer record by id (name, plan, signup date)."},
    "search_kb": {"func": search_kb, "args_model": SearchKbArgs, 
    "side_effect": False,
                  "description": "Search the knowledge base and return up to 3 matching articles."},
    "send_reply": {"func": send_reply, "args_model": SendReplyArgs, "side_effect": True,
                   "description": "Send a final reply to the customer. Closes the ticket for the agent."},
    "request_info": {"func": request_info, "args_model": RequestInfoArgs, "side_effect": True,
                     "description": "Ask the customer a question when information is missing. Closes the ticket for the agent."},
    "escalate": {"func": escalate, "args_model": EscalateArgs, "side_effect": True,
                 "description": "Hand the ticket to a human with a reason and priority. Closes the ticket for the agent."},
    "issue_refund": {"func": issue_refund, "args_model": IssueRefundArgs, "side_effect": True,
                     "description": "Refund an amount in USD against a ticket. Limited to 100 USD."},
}

SIDE_EFFECTING = {n for n, t in TICKET_TOOLS.items() if t["side_effect"]}
TERMINAL = {"send_reply", "request_info", "escalate"}


def call_tool(name, args):
    """The dispatcher: validate the model's arguments, then run the tool. Errors come back as data."""
    tool = TICKET_TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name}"}
    try:
        parsed = tool["args_model"](**args)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in x['loc'])}: {x['msg']}" for x in e.errors())
        return {"error": f"invalid arguments: {problems}"}
    except TypeError:
        return {"error": "invalid arguments: expected an object of named arguments"}
    return tool["func"](**parsed.model_dump())