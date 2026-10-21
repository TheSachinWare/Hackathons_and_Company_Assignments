import threading
from collections import deque
from typing import Deque, List, Dict, Any

import pandas as pd


class TickStore:
    """Thread-safe in-memory store for tick data."""

    def __init__(self, max_rows: int = 50000):
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._rows: Deque[Dict[str, Any]] = deque(maxlen=max_rows)

    def append(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            self._rows.extend(rows)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def snapshot(self) -> pd.DataFrame:
        """Return a copy of the buffered ticks as a DataFrame."""
        with self._lock:
            if not self._rows:
                return pd.DataFrame(columns=["ts", "symbol", "price", "size"])
            df = pd.DataFrame(list(self._rows))
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.sort_values("ts")


class AppState:
    """Holds global app state shared between callbacks and background workers."""

    def __init__(self):
        self.store = TickStore()
        self.active_symbols: List[str] = []
        self.running: bool = False

    def reset(self) -> None:
        self.store.clear()
        self.active_symbols = []
        self.running = False

