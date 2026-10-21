import sys
import os
import pandas as pd
import numpy as np
# Ensure repository root is on sys.path so `core` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.analytics import compute_pair_analytics


def make_df(seconds=300):
    now = pd.Timestamp.utcnow().floor('s') - pd.Timedelta(seconds=seconds)
    rows = []
    for i in range(seconds):
        ts = now + pd.Timedelta(seconds=i)
        x = 50000 + 10 * np.sin(i / 30.0) + np.random.normal(0, 0.5)
        y = 0.08 * x + 2000 + np.random.normal(0, 1.0)
        rows.append({"ts": ts, "symbol": "btcusdt", "price": float(x), "size": 0.01})
        rows.append({"ts": ts, "symbol": "ethusdt", "price": float(y), "size": 0.1})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = make_df(600)
    pa = compute_pair_analytics(df, 'btcusdt', 'ethusdt', lookback=300, corr_window=50)
    if pa is None:
        print('compute_pair_analytics returned None', file=sys.stderr)
        sys.exit(2)
    print('hedge_ratio:', pa.hedge_ratio)
    print('adf_pvalue:', pa.adf_pvalue)
    print('zscore tail:', pa.zscore.dropna().tail(5).tolist())
    sys.exit(0)
