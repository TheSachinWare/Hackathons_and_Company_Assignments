import asyncio
import json
import threading
import time
from typing import List, Dict

import pandas as pd
import websockets

from .state import TickStore


class BinanceIngestor:
    """Background WebSocket client that streams trades into a TickStore."""

    def __init__(self, store: TickStore):
        self.store = store
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.symbols: List[str] = []

    def start(self, symbols: List[str]) -> None:
        self.stop()
        self.symbols = [s.lower() for s in symbols if s]
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen_loop())
        except Exception:
            # swallow exceptions so thread does not crash silently
            pass
        finally:
            loop.stop()
            loop.close()

    async def _listen_loop(self) -> None:
        if not self.symbols:
            return
        stream = "/".join(f"{s}@trade" for s in self.symbols)
        url = f"wss://fstream.binance.com/stream?streams={stream}"
        backoff = 1
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10
                ) as ws:
                    backoff = 1
                    while not self._stop_event.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        self._handle_message(msg)
            except asyncio.TimeoutError:
                continue
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _handle_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        data: Dict = payload.get("data") or payload
        if data.get("e") != "trade":
            return
        ts = data.get("E") or data.get("T") or data.get("eventTime")
        symbol = data.get("s")
        price = data.get("p")
        qty = data.get("q")
        if not (ts and symbol and price and qty):
            return
        try:
            pd_ts = pd.to_datetime(ts, unit="ms", utc=True)
        except Exception:
            return
        normalized = [
            {
                "ts": pd_ts,
                "symbol": symbol.lower(),
                "price": float(price),
                "size": float(qty),
            }
        ]
        self.store.append(normalized)


class FileIngestor:
    """Simple replayer that pushes rows (dicts) into a TickStore honoring original timestamps.

    Use `start(rows, speed=1.0)` to begin replay in a background thread. `speed` >1.0 speeds up playback.
    """
    def __init__(self, store: TickStore):
        self.store = store
        self._thread = None
        self._stop_event = threading.Event()
        self._rows = []

    def start(self, rows: List[Dict], speed: float = 1.0, repeat: bool = False) -> None:
        self.stop()
        self._rows = rows or []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(speed, repeat), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self, speed: float, repeat: bool) -> None:
        if not self._rows:
            return
        # normalize timestamps to pandas.Timestamp
        rows = []
        for r in self._rows:
            rr = dict(r)
            try:
                rr_ts = pd.to_datetime(rr.get("ts"))
            except Exception:
                rr_ts = None
            rr["ts"] = rr_ts
            rows.append(rr)

        while not self._stop_event.is_set():
            start_time = None
            for r in rows:
                if self._stop_event.is_set():
                    break
                ts = r.get("ts")
                if start_time is None and ts is not None:
                    start_time = ts
                    base = pd.Timestamp.utcnow()
                if ts is not None and start_time is not None:
                    # compute sleep relative to first timestamp scaled by speed
                    target = base + (ts - start_time) / speed
                    while pd.Timestamp.utcnow() < target and not self._stop_event.is_set():
                        time.sleep(0.01)
                # push single-row normalized by ingest format
                try:
                    norm = [{
                        "ts": r.get("ts") or pd.Timestamp.utcnow(),
                        "symbol": (r.get("symbol") or r.get("s") or "").lower(),
                        "price": float(r.get("price") or r.get("p") or r.get("price_usd") or 0.0),
                        "size": float(r.get("size") or r.get("q") or 0.0),
                    }]
                except Exception:
                    norm = [{"ts": pd.Timestamp.utcnow(), "symbol": "", "price": 0.0, "size": 0.0}]
                self.store.append(norm)
            if not repeat:
                break
        # finished

