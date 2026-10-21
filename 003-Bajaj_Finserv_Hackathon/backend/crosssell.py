"""
crosssell.py
------------
Deterministic, auditable rule engine that decides:
  (a) is this customer even ELIGIBLE to be shown a mutual fund pitch, and
  (b) if so, which risk TIER of fund to suggest.

Design principle: the LLM never decides eligibility. Cross-selling a
regulated financial product is exactly the kind of decision that should be
deterministic, logged, and reviewable by compliance -- not left to a
probabilistic model that could be prompt-injected into "always recommend
the highest-fee fund". The LLM's only job downstream is to phrase the
already-decided recommendation naturally and answer follow-up questions
about it, grounded in fund_facts.json / the factsheet.

COMPLIANCE / FAIRNESS GUARDRAILS (flagged explicitly because this is the
single biggest "red-team the business logic, not just the code" risk in
this feature):
  1. We must not use credit/repayment data to target financially STRESSED
     customers with investment products -- that's a dark pattern and a
     potential SEBI/RBI conduct-risk issue. Any DPD > 30 in the trailing
     12 months hard-disqualifies the customer from any pitch, full stop.
  2. Creditworthiness (repayment behaviour) is NOT the same as investment
     RISK APPETITE or investor SUITABILITY under SEBI's mutual-fund
     distribution norms. This engine only decides "is it appropriate to
     surface an offer at all" and a coarse *starting* risk tier; it must
     never be treated as a KYC risk-profiling substitute. Before any actual
     transaction, the customer must complete a real risk-profiling
     questionnaire and give explicit consent (risk_consent_given).
  3. No auto-invest / no silent SIP creation. The assistant can only pitch
     and inform; a human-reviewable, explicit customer action is required
     to invest. This keeps "agency" of the AI system bounded.
  4. Every eligibility decision is written to the audit log with the inputs
     that drove it, so a compliance reviewer can reconstruct why a pitch
     was (or wasn't) shown.
"""

from dataclasses import dataclass
from typing import Optional

import db


@dataclass
class CrossSellDecision:
    eligible: bool
    tier: str                 # TIER_EQUITY_GROWTH / TIER_HYBRID_BALANCED / TIER_CONSERVATIVE_DEBT / TIER_NONE
    reason: str                # human-auditable explanation
    requires_consent_first: bool = True


def _max_dpd_last_12m(loan_id: str) -> int:
    history = db.get_repayment_history(loan_id, months=12)
    return max((row["dpd"] for row in history), default=0)


def evaluate(customer_id: str) -> CrossSellDecision:
    customer = db.get_customer(customer_id)
    if customer is None:
        return CrossSellDecision(False, "TIER_NONE", "unknown customer")

    loans = db.get_loans_for_customer(customer_id)
    active_loans = [l for l in loans if l["status"] == "ACTIVE"]
    if not active_loans:
        return CrossSellDecision(False, "TIER_NONE", "no active loan relationship")

    worst_dpd = max((_max_dpd_last_12m(l["loan_id"]) for l in active_loans), default=0)

    # --- Hard disqualifiers (financial-wellness / conduct-risk guardrail) ---
    if worst_dpd > 30:
        db.write_audit(customer_id, "CROSSSELL_EVAL",
                        f"ineligible: dpd={worst_dpd} in trailing 12m")
        return CrossSellDecision(False, "TIER_NONE",
                                    f"trailing-12m delinquency (max DPD {worst_dpd}) exceeds threshold")

    if customer["relationship_months"] < 6:
        db.write_audit(customer_id, "CROSSSELL_EVAL", "ineligible: relationship < 6 months")
        return CrossSellDecision(False, "TIER_NONE", "relationship tenure below 6 months")

    if customer["credit_score"] < 700:
        db.write_audit(customer_id, "CROSSSELL_EVAL",
                        f"ineligible: credit_score={customer['credit_score']}")
        return CrossSellDecision(False, "TIER_NONE", "credit score below eligibility threshold")

    # EMI-to-income headroom check: don't pitch investments to someone whose
    # EMI burden leaves little free cash flow, even if their score is fine
    total_emi = sum(l["emi_amount"] for l in active_loans)
    income = customer["monthly_income"] or 0
    emi_to_income = (total_emi / income) if income else 1.0
    if emi_to_income > 0.5:
        db.write_audit(customer_id, "CROSSSELL_EVAL",
                        f"ineligible: emi_to_income={emi_to_income:.2f}")
        return CrossSellDecision(False, "TIER_NONE", "insufficient free cash flow (EMI/income > 50%)")

    # --- Eligible: pick a starting tier (a *conversation starter*, not a
    # final suitability match -- see guardrail #2 above) ---
    score = customer["credit_score"]
    minor_blip = 0 < worst_dpd <= 30

    if score >= 780 and not minor_blip and customer["relationship_months"] >= 18:
        tier = "TIER_EQUITY_GROWTH"
        reason = "strong score, clean repayment record, long relationship tenure"
    elif score >= 700 and not minor_blip:
        tier = "TIER_HYBRID_BALANCED"
        reason = "good score and clean recent repayment record; balanced risk starting point"
    else:
        tier = "TIER_CONSERVATIVE_DEBT"
        reason = "eligible but with a minor recent repayment blip or shorter track record; low-volatility starting point"

    db.write_audit(customer_id, "CROSSSELL_EVAL", f"eligible tier={tier} reason={reason}")
    return CrossSellDecision(True, tier, reason)


def get_pitch_funds(tier: str, fund_facts: dict, limit: int = 2) -> list[dict]:
    names = fund_facts.get("cross_sell_tiers", {}).get(tier, [])[:limit]
    by_name = {f["scheme_name"]: f for f in fund_facts["funds"]}
    return [by_name[n] for n in names if n in by_name]
