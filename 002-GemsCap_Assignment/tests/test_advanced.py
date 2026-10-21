import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.analytics import (
    half_life_of_mean_reversion,
    hurst_exponent,
    kalman_filter_hedge_ratio,
    realized_volatility,
    portfolio_mean_reversion_backtest,
)


def make_df(seconds=200):
    now = pd.Timestamp.utcnow().floor('s') - pd.Timedelta(seconds=seconds)
    rows = []
    for i in range(seconds):
        ts = now + pd.Timedelta(seconds=i)
        x = 100 + 0.5 * np.sin(i / 10.0) + np.random.normal(0, 0.02)
        y = 0.3 * x + 10 + np.random.normal(0, 0.05)
        rows.append({'ts': ts, 'btcusdt': float(x), 'ethusdt': float(y)})
    df = pd.DataFrame(rows).set_index('ts')
    return df


def test_advanced_helpers():
    df = make_df(300)
    x = df['btcusdt']
    y = df['ethusdt']
    hl = half_life_of_mean_reversion(y - 0.3 * x)
    h = hurst_exponent(y - 0.3 * x)
    k = kalman_filter_hedge_ratio(x, y)
    rv = realized_volatility(y - 0.3 * x)
    back = portfolio_mean_reversion_backtest(x, y, 0.3, (y - 0.3 * x) / (y - 0.3 * x).rolling(50).std())
    assert hl is None or isinstance(hl, float)
    assert h is None or isinstance(h, float)
    assert isinstance(k, pd.Series)
    assert isinstance(rv, pd.Series)
    assert isinstance(back, dict)
