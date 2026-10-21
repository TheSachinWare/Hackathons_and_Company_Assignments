"""
security.py
------------
Central security controls for the assistant. Every one of these exists to close
a specific threat identified in the README threat model (see "Security" section).

Covers:
  - Password/OTP hashing (never store plaintext secrets)
  - JWT issuance & verification (short TTL, explicit algorithm allow-list -> blocks
    the classic "alg: none" / algorithm-confusion JWT attack)
  - Per-customer authorization guard (blocks IDOR: customer A must never be able
    to fetch customer B's loan/account data by editing an ID in the request)
  - Sliding-window rate limiter (in-memory here; Redis in production) to blunt
    brute force / credential stuffing / LLM-cost-exhaustion DoS
  - Prompt-injection / jailbreak heuristic screen on user input
  - Output-side grounding guard: strips content the model produced that is not
    traceable to retrieved context or the structured fund_facts.json (defense
    against hallucinated numbers reaching the customer)
  - PII/financial-data redaction helper for logs (never log raw account numbers,
    PAN, phone, OTP)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import base64
import json
import secrets
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Secrets management
# ---------------------------------------------------------------------------
# In production these MUST come from a secrets manager (AWS Secrets Manager /
# Azure Key Vault / HashiCorp Vault), injected as env vars into the container
# at runtime -- never committed to source control, never baked into the image.
JWT_SECRET = os.environ.get("JWT_SECRET", None)
if not JWT_SECRET:
    # Fail loudly in prod. For the hackathon demo we generate an ephemeral key
    # so the app is runnable out of the box, but this line is exactly the kind
    # of thing a red-teamer should flag: it must never survive into a real
    # deployment (silent fallback to a weak/default secret is a classic
    # "insecure default" vulnerability, e.g. CWE-798 hardcoded credentials).
    JWT_SECRET = secrets.token_hex(32)

JWT_ALG = "HS256"  # explicit allow-list of exactly one algorithm
JWT_TTL_SECONDS = 15 * 60  # short-lived access token -> limits blast radius of token theft
ALLOWED_ALGS = {"HS256"}


# ---------------------------------------------------------------------------
# Password / OTP hashing
# ---------------------------------------------------------------------------
def hash_secret(secret: str, salt: Optional[bytes] = None) -> str:
    """PBKDF2-HMAC-SHA256 with per-user random salt (never reuse a global salt)."""
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, 200_000)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def verify_secret(secret: str, stored: str) -> bool:
    try:
        salt_b64, dk_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, 200_000)
    # constant-time compare -> avoid timing side-channel on OTP/password check
    return hmac.compare_digest(dk, expected)


# ---------------------------------------------------------------------------
# Minimal, dependency-free JWT (HS256 only). In production, swap for the
# maintained `pyjwt` library -- this hand-rolled version is kept intentionally
# small so the whole auth flow is auditable in one file for the hackathon demo.
# ---------------------------------------------------------------------------
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def issue_jwt(customer_id: str, extra_claims: Optional[dict] = None) -> str:
    header = {"alg": JWT_ALG, "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": customer_id,
        "iat": now,
        "exp": now + JWT_TTL_SECONDS,
        "jti": secrets.token_hex(8),  # unique token id -> enables server-side revocation list
    }
    if extra_claims:
        payload.update(extra_claims)
    h = _b64url(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{h}.{p}".encode()
    sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"


class AuthError(Exception):
    pass


def verify_jwt(token: str) -> dict:
    try:
        h_b64, p_b64, s_b64 = token.split(".")
    except ValueError:
        raise AuthError("malformed token")

    header = json.loads(_b64url_decode(h_b64))
    if header.get("alg") not in ALLOWED_ALGS:
        # Blocks the classic "change alg to none / HS256<->RS256 confusion" attack
        raise AuthError("disallowed algorithm")

    signing_input = f"{h_b64}.{p_b64}".encode()
    expected_sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(s_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise AuthError("bad signature")

    payload = json.loads(_b64url_decode(p_b64))
    if payload.get("exp", 0) < int(time.time()):
        raise AuthError("token expired")
    if payload.get("jti") in REVOKED_JTIS:
        raise AuthError("token revoked")
    return payload


REVOKED_JTIS: set[str] = set()  # server-side logout / incident revocation list


# ---------------------------------------------------------------------------
# Authorization guard -> blocks IDOR
# ---------------------------------------------------------------------------
def assert_owns_resource(token_customer_id: str, requested_customer_id: str) -> None:
    """
    Every account/loan lookup MUST call this. The customer_id used to fetch data
    is taken ONLY from the verified JWT `sub` claim -- never from a client-supplied
    field in the request body/query string. This function exists as a second,
    explicit checkpoint so an accidental future refactor (e.g. someone wiring a
    'customer_id' query param straight into the DB layer) fails loudly instead
    of silently leaking another customer's loan data.
    """
    if token_customer_id != requested_customer_id:
        raise PermissionError("IDOR blocked: token identity does not match requested resource")


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, in-memory). Swap for Redis (INCR + EXPIRE, or
# a token-bucket via redis-cell) once you run more than one API replica.
# ---------------------------------------------------------------------------
@dataclass
class RateLimiter:
    max_requests: int
    window_seconds: int
    _hits: dict = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        hits = [t for t in self._hits.get(key, []) if t > window_start]
        if len(hits) >= self.max_requests:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True


login_limiter = RateLimiter(max_requests=5, window_seconds=60)      # anti brute-force
otp_limiter = RateLimiter(max_requests=5, window_seconds=300)       # anti OTP guessing
chat_limiter = RateLimiter(max_requests=20, window_seconds=60)      # anti LLM-cost DoS


# ---------------------------------------------------------------------------
# Prompt-injection / jailbreak heuristic screen (defense-in-depth layer #1;
# the real defense is architectural -- see llm.py -- but a cheap pre-filter
# catches the loud, obvious attempts and lets us log/alert on them).
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all|any|the)?\s*(?:previous|prior|above)?\s*(?:instructions|prompt|rules)",
    r"disregard\s+(?:all|any|the)?\s*(?:previous|prior|above)?\s*(?:instructions|prompt|rules)",
    r"system prompt",
    r"you are now",
    r"reveal (your|the) (prompt|instructions|system message)",
    r"act as (an? )?(unrestricted|jailbroken|dan)",
    r"print (your|the) (api key|secret|token)",
    r"\bsudo\b",
    r"drop table",
    r"select \* from",
    r"';--",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def screen_user_input(text: str) -> tuple[bool, Optional[str]]:
    """Returns (is_suspicious, matched_pattern)."""
    m = _INJECTION_RE.search(text)
    if m:
        return True, m.group(0)
    if len(text) > 4000:
        return True, "input too long"
    return False, None


# ---------------------------------------------------------------------------
# Output-side grounding guard
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b")


def flag_ungrounded_numbers(answer: str, context_text: str) -> list[str]:
    """
    Cheap heuristic: every standalone number the model outputs should appear
    somewhere in the retrieved context (or fund_facts.json, merged into
    context_text by the caller). Numbers that don't appear anywhere in the
    grounding context are flagged for the caller to strip or to trigger a
    "let me confirm that" fallback rather than showing a possibly hallucinated
    figure (e.g. AUM, expense ratio, EMI amount) to a customer.
    """
    answer_nums = set(_NUMBER_RE.findall(answer))
    context_nums = set(_NUMBER_RE.findall(context_text))
    return sorted(n for n in answer_nums if n not in context_nums)


# ---------------------------------------------------------------------------
# PII redaction for logs / observability sinks
# ---------------------------------------------------------------------------
_PATTERNS_TO_REDACT = {
    "phone": re.compile(r"\b[6-9]\d{9}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "account_like": re.compile(r"\b\d{9,18}\b"),
    "otp": re.compile(r"\botp\D{0,3}(\d{4,6})\b", re.IGNORECASE),
}


def redact_pii(text: str) -> str:
    redacted = text
    for name, pattern in _PATTERNS_TO_REDACT.items():
        redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    return redacted
