"""
observability.py
-----------------
Minimum viable observability for the hackathon demo: per-request latency,
LLM token usage, and error/blocked-request counters, exposed as a Prometheus
text-format /metrics endpoint plus a small in-memory ring buffer the frontend
polls to show "under the hood" stats in the UI.

In production this becomes: OpenTelemetry traces (one span per request,
child spans for auth / db / retrieval / llm_call) exported to an APM
(Datadog/New Relic/Grafana Tempo), metrics scraped by Prometheus and
dashboarded in Grafana, and structured JSON logs shipped to a log
aggregator / SIEM. See README "Observability at scale".
"""

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict


@dataclass
class RequestRecord:
    ts: float
    route: str
    customer_id: str | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    status: str  # "ok" | "blocked" | "error"


class Telemetry:
    def __init__(self, max_records: int = 500):
        self._lock = threading.Lock()
        self._records: deque[RequestRecord] = deque(maxlen=max_records)
        self._counters = defaultdict(int)

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._counters[f"requests_total{{route=\"{rec.route}\",status=\"{rec.status}\"}}"] += 1
            self._counters["tokens_prompt_total"] += rec.prompt_tokens
            self._counters["tokens_completion_total"] += rec.completion_tokens

    def recent(self, n: int = 20) -> list[dict]:
        with self._lock:
            return [asdict(r) for r in list(self._records)[-n:]]

    def summary(self) -> dict:
        with self._lock:
            if not self._records:
                return {"count": 0}
            lat = [r.latency_ms for r in self._records]
            return {
                "count": len(self._records),
                "avg_latency_ms": round(sum(lat) / len(lat), 2),
                "p95_latency_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1], 2) if len(lat) > 1 else round(lat[0], 2),
                "total_prompt_tokens": self._counters["tokens_prompt_total"],
                "total_completion_tokens": self._counters["tokens_completion_total"],
            }

    def prometheus_text(self) -> str:
        with self._lock:
            lines = ["# TYPE requests_total counter", "# TYPE tokens_prompt_total counter",
                     "# TYPE tokens_completion_total counter"]
            for k, v in self._counters.items():
                lines.append(f"{k} {v}")
            return "\n".join(lines) + "\n"


telemetry = Telemetry()


def timeit():
    """Context manager helper: `with timeit() as t: ...; t.ms`"""
    class _Timer:
        def __enter__(self):
            self._t0 = time.time()
            self.ms = 0.0
            return self

        def __exit__(self, *exc):
            self.ms = (time.time() - self._t0) * 1000
            return False

    return _Timer()
