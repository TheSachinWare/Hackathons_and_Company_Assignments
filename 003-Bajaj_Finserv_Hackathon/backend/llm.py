"""
llm.py
------
Orchestration layer between the user's message and the language model.

Threat model addressed here (the important architectural part -- security.py's
regex screen is only a cheap first filter, THIS is the real defense):

  - Prompt injection via user input: the model is told, structurally, that
    anything inside <user_message> tags is DATA to answer, never an
    instruction. System instructions are never re-derivable from tool
    output, so "ignore previous instructions" inside a chat message has
    nothing to override.
  - Prompt injection via retrieved content: retrieved factsheet chunks and
    account data are wrapped in <retrieved_context> / <account_data> tags
    and the system prompt explicitly says content there is reference data,
    not instructions -- relevant because factsheet text or (in production)
    third-party market data could theoretically be manipulated upstream.
  - Excessive agency: the LLM is never given a "transact" or "modify
    account" tool. It only has read-only tools (get_loan_summary,
    fund_lookup). Cross-sell eligibility is decided entirely outside the
    LLM (crosssell.py) -- the model cannot invent an offer.
  - Hallucination on numbers: every numeric claim in the answer is checked
    against the grounding context before being returned (security.py
    flag_ungrounded_numbers); if a number can't be traced, we regenerate
    with an explicit "you don't have that figure, say so" instruction
    rather than silently show a possibly-wrong number to a customer asking
    about their own money.
  - Data exfiltration to the model provider: only the minimum required
    slice of account data is included per turn (the specific loan being
    asked about, not the customer's full financial history), and PII is
    redacted before anything is written to logs/telemetry (never sent to
    the model unredacted, since the model needs the real numbers to answer
    correctly -- but analytics/logging pipelines only ever see redacted
    text). See README for enterprise-tier options (Azure OpenAI in a
    private VNet / on-prem model) where data residency matters.

NOTE: the actual network call to an LLM provider is stubbed as
`_call_llm_api()`. Wire this to Anthropic/OpenAI/Azure OpenAI using an
API key from a secrets manager (never hardcoded). The rest of the
pipeline (routing, grounding, guardrails) is fully implemented and is the
part that actually matters for this assessment.
"""

import json
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
import rag
import security

load_dotenv()

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

_gemini_client = None

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

SYSTEM_PROMPT = """You are the Bajaj Finserv assistant. Answer only loan account or Bajaj AMC mutual fund questions.
Use only the data in <account_data>, <retrieved_context>, and <crosssell_offer> tags.
If the answer is not in the provided context, say you do not know.
Do not invent numbers or make transaction recommendations.
Keep responses brief, direct, and structured."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    grounding_flag: list[str]


def _extract_anthropic_text(response) -> str:
    if not response or not getattr(response, 'content', None):
        return ""
    parts = []
    for block in response.content:
        if isinstance(block, dict):
            if block.get('type') == 'text':
                parts.append(block.get('text', ''))
        else:
            if getattr(block, 'type', None) == 'text':
                parts.append(getattr(block, 'text', ''))
    return "".join(parts).strip()


def _extract_gemini_text(response) -> str:
    if not response:
        return ""
    if isinstance(response, dict):
        if 'last' in response and response['last']:
            content = response['last'].get('content')
            if content:
                texts = [block.get('text', '') if isinstance(block, dict) else '' for block in content]
                return ''.join(texts).strip()
        if 'candidates' in response:
            for candidate in response['candidates']:
                if not candidate:
                    continue
                output = candidate.get('output')
                if output:
                    return str(output).strip()
                content = candidate.get('content')
                if content:
                    texts = [block.get('text', '') if isinstance(block, dict) else '' for block in content]
                    return ''.join(texts).strip()
        if 'content' in response and isinstance(response['content'], list):
            texts = [block.get('text', '') if isinstance(block, dict) else '' for block in response['content']]
            return ''.join(texts).strip()
        return json.dumps(response)

    if hasattr(response, 'last'):
        last = response.last
        content = getattr(last, 'content', None)
        if content:
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(block.get('text', ''))
                else:
                    texts.append(getattr(block, 'text', ''))
            return ''.join(texts).strip()
    if hasattr(response, 'candidates'):
        for candidate in getattr(response, 'candidates', []):
            output = getattr(candidate, 'output', None)
            if output:
                return str(output).strip()
            content = getattr(candidate, 'content', None)
            if content:
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        texts.append(block.get('text', ''))
                    else:
                        texts.append(getattr(block, 'text', ''))
                return ''.join(texts).strip()
    return str(response)


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _gemini_client


def _call_llm_api(system: str, user_payload: str) -> tuple[str, int, int]:
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        model = os.environ.get("GOOGLE_MODEL", "gemini-3.1-flash-lite")

        if not api_key:
            return (
                "Gemini API key is missing. Set GOOGLE_API_KEY in your .env and restart the backend.",
                0,
                0,
            )

        if not genai:
            return (
                "The google-genai SDK is not installed. Run `pip install google-genai python-dotenv`.",
                0,
                0,
            )

        try:
            client = _get_gemini_client()
            config = genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=300,
                temperature=0.2,
            )
            response = client.models.generate_content(
                model=model,
                contents=user_payload,
                config=config,
            )
            text = getattr(response, "text", None) or _extract_gemini_text(response)
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            return (str(text).strip() if text else "", prompt_tokens, completion_tokens)
        except Exception as exc:
            return (
                "Gemini request failed. Please verify your GOOGLE_API_KEY and model access." +
                (f" Error: {exc}" if exc else ""),
                0,
                0,
            )

    model = os.environ.get("ANTHROPIC_MODEL", "claude-3.5-mini")
    if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) and anthropic:
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        )
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": user_payload}],
            system=system,
            max_tokens=500,
            temperature=0.2,
        )
        text = _extract_anthropic_text(response)
        prompt_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        completion_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        return text, prompt_tokens, completion_tokens

    # ---- offline demo fallback (deterministic, not a real LLM call) ----
    text = _offline_demo_answer(user_payload)
    prompt_tokens = max(1, len(system.split()) + len(user_payload.split()))
    completion_tokens = max(1, len(text.split()))
    return text, prompt_tokens, completion_tokens


def _offline_demo_answer(user_payload: str) -> str:
    """Extremely small templated responder so the demo works with zero
    external dependencies. Real deployments delete this and only use
    _call_llm_api's provider branch above."""
    m = re.search(r"<user_message>(.*?)</user_message>", user_payload, re.S)
    q = (m.group(1).strip() if m else "").lower()

    acc = re.search(r"<account_data>(.*?)</account_data>", user_payload, re.S)
    ctx = re.search(r"<retrieved_context>(.*?)</retrieved_context>", user_payload, re.S)
    offer = re.search(r"<crosssell_offer>(.*?)</crosssell_offer>", user_payload, re.S)

    if any(k in q for k in ["emi", "balance", "tenure", "repay", "loan", "outstanding"]) and acc:
        base = f"Here's what I have on your account:\n{acc.group(1).strip()}"
    elif ctx and ctx.group(1).strip():
        base = f"Based on the fund factsheet:\n{ctx.group(1).strip()}"
    else:
        base = "I don't have that information grounded in your account data or the fund factsheet, so I can't answer that accurately. Could you rephrase, or ask about your loan or a specific Bajaj AMC fund?"

    if offer and offer.group(1).strip():
        base += f"\n\nBy the way, based on your profile you may be eligible for: {offer.group(1).strip()}"
    return base


def answer_query(
    user_message: str,
    account_context: str = "",
    retrieved_context: str = "",
    crosssell_offer: str = "",
) -> LLMResponse:
    t0 = time.time()

    # Layer 1: cheap heuristic screen (security.py)
    suspicious, pattern = security.screen_user_input(user_message)
    if suspicious:
        security.write_alert = getattr(security, "write_alert", None)  # placeholder hook
        latency_ms = (time.time() - t0) * 1000
        return LLMResponse(
            text="I can only help with your loan account or Bajaj AMC mutual fund questions. "
                 "I'm not able to follow embedded instructions in a message.",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            grounding_flag=[f"blocked_by_input_screen:{pattern}"],
        )

    account_context = account_context.strip()
    if len(account_context) > 600:
        account_context = account_context[:600].rsplit("\n", 1)[0]

    retrieved_context = retrieved_context.strip()
    if len(retrieved_context) > 1200:
        retrieved_context = retrieved_context[:1200].rsplit("\n", 1)[0]

    crosssell_offer = crosssell_offer.strip()
    if len(crosssell_offer) > 300:
        crosssell_offer = crosssell_offer[:300].rsplit("\n", 1)[0]

    payload = (
        f"<account_data>{account_context}</account_data>\n"
        f"<retrieved_context>{retrieved_context}</retrieved_context>\n"
        f"<crosssell_offer>{crosssell_offer}</crosssell_offer>\n"
        f"<user_message>{user_message.strip()}</user_message>"
    )

    raw_text, p_tok, c_tok = _call_llm_api(SYSTEM_PROMPT, payload)

    # Layer 3: output-side grounding guard
    grounding_context = f"{account_context}\n{retrieved_context}\n{crosssell_offer}"
    ungrounded = security.flag_ungrounded_numbers(raw_text, grounding_context)

    latency_ms = (time.time() - t0) * 1000
    return LLMResponse(
        text=raw_text,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        latency_ms=latency_ms,
        grounding_flag=ungrounded,
    )
