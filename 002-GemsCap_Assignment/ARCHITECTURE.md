# Architecture and Design

This document explains the architecture, design choices, trade-offs, and instructions to run the Live Pairs Analytics demo.

**Overview**
- Purpose: end-to-end demo from real-time ingestion → storage → analytics → interactive visualization.
- Scope: ingest Binance futures trade streams, buffer ticks in-memory, compute pair analytics (OLS hedge ratio, spread, z-score, ADF, rolling correlation), display live charts and KPIs in a Dash frontend.

**High-level components**
- Backend ingestion: `core/ingestion.py` — a background WebSocket client using the `websockets` library. It connects to Binance combined streams (wss://fstream.binance.com/stream?streams=...) and normalizes trade messages into tick records.
- In-memory store: `core/state.py` — `TickStore` implements a thread-safe bounded buffer (collections.deque with a lock). Chosen for simplicity and low-latency reads/writes.
- Analytics: `core/analytics.py` — resampling utilities (`resample_ticks`), pair analytics (`compute_pair_analytics`) using `pandas`, `statsmodels` (OLS + ADF), and `scipy` where needed; a small mean-reversion backtest function.
- Alerts: `core/alerts.py` — simple rule engine that evaluates numeric metrics (e.g., |z|) and returns fired rules.
- Frontend: `app.py` — Dash + `dash-bootstrap-components` UI with charts (Plotly), KPI cards, upload/download, and interactive controls.

**Design choices & rationale**
- Dash (Plotly) frontend: quick to build interactive analytics dashboards without a separate SPA framework. Good for data-centric UIs and fast iteration.
- In-memory `TickStore`: low complexity and low latency for an interactive demo. Reasoning: the assignment focuses on analytics and visualization; keeping a bounded in-memory buffer avoids introducing DB setup friction.
  - Tradeoffs: not durable across restarts and not horizontally scalable. For production, swap `TickStore` with a lightweight persistence layer (SQLite/Parquet/Redis) or stream into Kafka.
- Websocket ingestion with `websockets`: simple async client that runs inside a dedicated thread so Dash callbacks (synchronous) can read snapshots safely. Backoff/retry logic in the ingest loop improves robustness for transient disconnects.
- Analytics using `pandas` + `statsmodels`: familiar tools for time-series regressions and stationarity testing (ADF). Rolling computations are implemented with pandas windowing to keep code concise.
- Export/Upload: `dcc.Upload` and `dcc.Download` let the UI ingest offline CSVs and export current buffered ticks.

**Security & Operational notes**
- Authentication in the demo is intentionally minimal (UI-level flag) to keep scope focused on analytics. Do not use demo auth for real deployments.
- The Flask development server (used by Dash) is not intended for production. For production, use a WSGI server (Gunicorn/uvicorn via ASGI wrapper) behind a reverse proxy.

**Scalability & Production adaptations**
- Persistence: persist ticks to Parquet/SQLite/TimescaleDB to enable historical analysis and replay.
- Horizontal scale: use a message broker (Kafka/RabbitMQ) or cloud streaming (Kinesis) to decouple ingestion and analytics; scale analytics workers separately.
- Stateful analytics: move rolling/ADF computations to worker processes and serve results via a small API to the Dash frontend.

**How to run (local)**
1. Install dependencies (Python 3.11+ recommended):

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
python app.py
```

3. Open http://localhost:8050 and press Start.

**Docker (reproducible run)**
- A `Dockerfile` is provided in the project root to build a containerized dev image. See `Dockerfile`.

**Files of interest**
- `app.py` (main Dash app)
- `core/ingestion.py` (Binance websocket ingestor)
- `core/state.py` (TickStore)
- `core/analytics.py` (pair analytics and resampling)
- `core/alerts.py` (alerts rule engine)
- `tests/smoke_analytics.py` (a smoke test verifying `compute_pair_analytics`)

**Next steps & extensions**
- Add persistence and historical replay.
- Add CI (unit tests and linting) and a GitHub Actions workflow.
- Add Docker Compose to include a lightweight DB (Redis/Postgres) if persistence is added.

---

This document aims to explain the current demo choices and provide a clear path to production hardening if desired.
