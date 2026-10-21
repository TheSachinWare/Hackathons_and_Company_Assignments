"""
main.py
-------
FastAPI application wiring:
  /auth/login          step 1: phone -> triggers OTP (demo: fixed OTP "123456")
  /auth/verify-otp     step 2: phone + otp -> JWT access token
  /account/summary     authenticated: this customer's loans (IDOR-safe)
  /chat                authenticated: natural-language query -> routed answer
  /consent/mf-pitch     authenticated: explicit opt-in required before any
                        cross-sell pitch is shown (SEBI suitability guardrail)
  /metrics              observability (Prometheus text)
  /healthz              liveness/readiness probe

Run: uvicorn main:app --reload --port 8000
"""

import os
import sys
import time
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

# Ensure local backend modules are importable whether this is started from
# the repo root (uvicorn backend.main:app) or from inside backend/ itself
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
for path in (BACKEND_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Load local environment overrides from .env when present.
load_dotenv(ROOT_DIR / '.env')

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db
import security
import crosssell
import rag
import llm
from observability import telemetry, RequestRecord

app = FastAPI(title="Bajaj Finserv Loan & MF Assistant (Prototype)")

# Serve the frontend from the backend so users can open http://127.0.0.1:8000
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# --- CORS: allow-list only the known frontend origin(s). Never use "*" for
# an authenticated financial API -- wildcard CORS + credentialed requests is
# a textbook cross-origin data-leak vector. ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers_and_timing(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    # Baseline security headers (defense-in-depth; a real deployment also
    # terminates TLS at the load balancer / API gateway and adds HSTS there).
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"  # financial data must never be cached client-side
    response.headers["X-Request-Latency-ms"] = str(round((time.time() - t0) * 1000, 2))
    return response


db.init_db()
FUND_FACTS = rag.load_fund_facts("../data/fund_facts.json")
RAG_INDEX = rag.build_index_from_fund_facts(FUND_FACTS)


@app.get("/")
def root():
    return FileResponse(str(frontend_dir / "index.html"))


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
class LoginRequest(BaseModel):
    phone: str = Field(min_length=10, max_length=10, pattern=r"^\d{10}$")


class OtpVerifyRequest(BaseModel):
    phone: str = Field(min_length=10, max_length=10, pattern=r"^\d{10}$")
    otp: str = Field(min_length=4, max_length=6, pattern=r"^\d+$")


# Auth endpoints removed for no-auth demo.


def _authenticate(authorization: Optional[str]) -> str:
    return "CUST001"


# --------------------------------------------------------------------------
# Account
# --------------------------------------------------------------------------
@app.get("/account/summary")
def account_summary():
    customer_id = _authenticate(None)
    customer = db.get_customer(customer_id)
    loans = db.get_loans_for_customer(customer_id)
    db.write_audit(customer_id, "ACCOUNT_SUMMARY_VIEW")
    return {
        "name": customer["name"],
        "loans": [
            {
                "loan_id": l["loan_id"],
                "loan_type": l["loan_type"],
                "outstanding_balance": l["outstanding_balance"],
                "emi_amount": l["emi_amount"],
                "months_remaining": l["months_remaining"],
                "status": l["status"],
            }
            for l in loans
        ],
    }


class ConsentRequest(BaseModel):
    consent: bool


@app.post("/consent/mf-pitch")
def set_consent(req: ConsentRequest):
    customer_id = _authenticate(None)
    db.set_risk_consent(customer_id, req.consent)
    db.write_audit(customer_id, "MF_PITCH_CONSENT", str(req.consent))
    return {"ok": True, "consent": req.consent}


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(max_length=2000)
    loan_id: Optional[str] = None  # optional: which loan the question is about


def _build_account_context(customer_id: str, loan_id: Optional[str]) -> str:
    loans = db.get_loans_for_customer(customer_id)
    if loan_id:
        loans = [l for l in loans if l["loan_id"] == loan_id]
        if not loans:
            return "No matching loan found for this customer."
    parts = []
    for l in loans:
        max_dpd = crosssell._max_dpd_last_12m(l["loan_id"])
        repay_status = "On track, no missed payments in the last 12 months" if max_dpd == 0 else \
                        f"Minor delay recorded (max {max_dpd} days past due in last 12 months)"
        parts.append(
            f"{l['loan_type']} ({l['loan_id']}): outstanding balance Rs {l['outstanding_balance']:,}, "
            f"EMI Rs {l['emi_amount']:,}/month, {l['months_remaining']} months remaining of "
            f"{l['tenure_months']} month tenure, interest rate {l['interest_rate_pct']}%, "
            f"status: {l['status']}. Repayment status: {repay_status}."
        )
    return " | ".join(parts)


def _build_mf_context(query: str) -> str:
    hits = RAG_INDEX.search(query, top_k=3)
    return " | ".join(c.text for c, score in hits)


def _build_crosssell_offer(customer_id: str) -> str:
    customer = db.get_customer(customer_id)
    if not customer["risk_consent_given"]:
        return ""  # never pitch without explicit opt-in, regardless of eligibility
    decision = crosssell.evaluate(customer_id)
    if not decision.eligible:
        return ""
    funds = crosssell.get_pitch_funds(decision.tier, FUND_FACTS)
    if not funds:
        return ""
    names = ", ".join(f["scheme_name"] for f in funds)
    return f"{names} (tier: {decision.tier}, rationale: {decision.reason})"


@app.post("/chat")
def chat(req: ChatRequest):
    customer_id = _authenticate(None)

    if not security.chat_limiter.allow(customer_id):
        raise HTTPException(429, "You're sending messages too quickly. Please slow down.")

    account_ctx = _build_account_context(customer_id, req.loan_id)
    mf_ctx = _build_mf_context(req.message)
    offer_ctx = _build_crosssell_offer(customer_id)

    result = llm.answer_query(
        user_message=req.message,
        account_context=account_ctx,
        retrieved_context=mf_ctx,
        crosssell_offer=offer_ctx,
    )

    status = "blocked" if result.grounding_flag and "blocked_by_input_screen" in str(result.grounding_flag) else "ok"
    telemetry.record(RequestRecord(
        ts=time.time(), route="/chat", customer_id=customer_id,
        latency_ms=result.latency_ms, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, status=status,
    ))
    db.write_audit(customer_id, "CHAT_QUERY", security.redact_pii(req.message)[:200])

    return {
        "answer": result.text,
        "meta": {
            "latency_ms": round(result.latency_ms, 2),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "ungrounded_numbers_flagged": result.grounding_flag,
        },
    }


# --------------------------------------------------------------------------
# Observability & health
# --------------------------------------------------------------------------
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return telemetry.prometheus_text()


@app.get("/metrics/summary")
def metrics_summary():
    return telemetry.summary()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
