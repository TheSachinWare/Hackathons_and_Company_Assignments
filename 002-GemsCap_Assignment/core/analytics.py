from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller


def resample_ticks(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts", "symbol", "price", "size"])
    out = (
        df.set_index("ts")
        .groupby("symbol")
        .resample(rule)
        .agg({"price": "last", "size": "sum"})
        .reset_index()
        .sort_values("ts")
    )
    return out


@dataclass
class PairAnalytics:
    hedge_ratio: float
    spread: pd.Series
    zscore: pd.Series
    adf_pvalue: Optional[float]
    rolling_corr: pd.Series


def compute_pair_analytics(
    df: pd.DataFrame,
    sym_x: str,
    sym_y: str,
    lookback: int = 500,
    corr_window: int = 100,
    regression: str = "ols",
) -> Optional[PairAnalytics]:
    if df.empty or sym_x not in df["symbol"].unique() or sym_y not in df["symbol"].unique():
        return None
    pivot = df.pivot(index="ts", columns="symbol", values="price").dropna()
    pivot = pivot.tail(lookback)
    if pivot.empty:
        return None

    x = pivot[sym_x]
    y = pivot[sym_y]
    if x.empty or y.empty:
        return None
    # Estimate hedge ratio using requested regression type
    beta = None
    try:
        if regression is None:
            regression = "ols"
        reg = regression.lower()
        if reg == "ols":
            X = add_constant(x.values)
            model = OLS(y.values, X).fit()
            beta = float(model.params[1])
        elif reg == "theil-sen":
            from sklearn.linear_model import TheilSenRegressor

            ts = TheilSenRegressor(random_state=0)
            ts.fit(x.values.reshape(-1, 1), y.values)
            beta = float(ts.coef_[0])
        elif reg == "huber":
            from sklearn.linear_model import HuberRegressor

            hr = HuberRegressor()
            hr.fit(x.values.reshape(-1, 1), y.values)
            beta = float(hr.coef_[0])
        else:
            # fallback to OLS
            X = add_constant(x.values)
            model = OLS(y.values, X).fit()
            beta = float(model.params[1])
    except Exception:
        # last-resort: simple OLS via numpy
        try:
            beta = float(np.polyfit(x.values, y.values, 1)[0])
        except Exception:
            return None
    spread = y - beta * x
    spread_mean = spread.rolling(window=50, min_periods=10).mean()
    spread_std = spread.rolling(window=50, min_periods=10).std()
    zscore = (spread - spread_mean) / spread_std
    corr = x.rolling(corr_window, min_periods=10).corr(y)

    pvalue = None
    if len(spread.dropna()) > 20:
        try:
            pvalue = float(adfuller(spread.dropna(), maxlag=1, autolag="AIC")[1])
        except Exception:
            pvalue = None

    return PairAnalytics(
        hedge_ratio=beta,
        spread=spread,
        zscore=zscore,
        adf_pvalue=pvalue,
        rolling_corr=corr,
    )


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["symbol", "last", "pct_change_1m", "pct_change_5m", "vol_5m"])
    df = df.sort_values("ts")
    latest = df.groupby("symbol").tail(1).set_index("symbol")["price"]
    df["m1"] = df["ts"].dt.floor("1min")
    agg = df.groupby(["symbol", "m1"]).agg(last=("price", "last"), vol=("size", "sum")).reset_index()
    pct_1m = agg.groupby("symbol")["last"].apply(lambda s: s.pct_change().iloc[-1] if len(s) > 1 else 0)
    pct_5m = agg.groupby("symbol")["last"].apply(lambda s: s.pct_change(periods=5).iloc[-1] if len(s) > 5 else 0)
    vol_5m = agg.groupby("symbol").apply(lambda s: s["vol"].tail(5).sum())
    out = pd.DataFrame(
        {
            "symbol": latest.index,
            "last": latest.values,
            "pct_change_1m": pct_1m.reindex(latest.index).fillna(0).values,
            "pct_change_5m": pct_5m.reindex(latest.index).fillna(0).values,
            "vol_5m": vol_5m.reindex(latest.index).fillna(0).values,
        }
    )
    return out


def mini_mean_reversion_backtest(spread: pd.Series, entry: float = 2.0, exit: float = 0.0) -> Dict[str, float]:
    if spread.empty:
        return {"trades": 0, "pnl": 0.0, "avg_holding": 0}
    pos = 0
    pnl = 0.0
    trades = 0
    hold = []
    last_entry_idx = None
    for i, z in enumerate(spread.dropna()):
        if pos == 0 and abs(z) >= entry:
            pos = -1 if z > 0 else 1
            trades += 1
            last_entry_idx = i
        elif pos != 0 and abs(z) <= exit:
            pnl += pos * (spread.iloc[i] - spread.iloc[last_entry_idx])
            hold.append(i - last_entry_idx)
            pos = 0
    avg_hold = float(np.mean(hold)) if hold else 0.0
    return {"trades": trades, "pnl": float(pnl), "avg_holding": avg_hold}


def half_life_of_mean_reversion(spread: pd.Series) -> Optional[float]:
    """Estimate half-life of mean reversion from AR(1) on the spread.

    Returns half-life in number of periods (same units as spread index), or None.
    """
    s = spread.dropna()
    if len(s) < 10:
        return None
    # lagged series
    s_lag = s.shift(1).iloc[1:]
    s_ret = (s - s.shift(1)).iloc[1:]
    X = add_constant(s_lag.values)
    try:
        res = OLS(s_ret.values, X).fit()
        phi = res.params[1]
        if phi >= 0:
            # half-life = -log(2) / log(phi + 1)
            denom = np.log(1 + phi)
            if denom == 0:
                return None
            hl = -np.log(2) / denom
            return float(max(0.0, hl))
    except Exception:
        return None
    return None


def hurst_exponent(ts: pd.Series, min_lag: int = 2, max_lag: int = 100) -> Optional[float]:
    """Estimate Hurst exponent using R/S method (approx).

    Returns H in (0,1) where <0.5 indicates mean-reversion.
    """
    s = ts.dropna()
    if len(s) < max_lag:
        max_lag = max(min_lag + 1, int(len(s) / 2))
    lags = np.arange(min_lag, max_lag)
    tau = []
    for lag in lags:
        # standard deviation of lagged differences
        diff = np.subtract(s.values[lag:], s.values[:-lag])
        tau.append(np.std(diff))
    if len(tau) < 2:
        return None
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = reg[0] * 0.5
        return float(hurst)
    except Exception:
        return None


def kalman_filter_hedge_ratio(x: pd.Series, y: pd.Series) -> pd.Series:
    """Simple Kalman filter to estimate time-varying hedge ratio (beta) of y = beta * x + eps.

    Returns a pandas Series of beta estimates aligned to the intersection index of x and y.
    """
    xi = x.dropna()
    yi = y.dropna()
    df = pd.concat([xi, yi], axis=1, join="inner")
    if df.shape[0] < 2:
        return pd.Series(dtype=float)
    xvals = df.iloc[:, 0].values
    yvals = df.iloc[:, 1].values

    n = len(xvals)
    # allocate arrays
    beta = np.zeros(n)
    P = np.zeros(n)
    R = 0.001  # observation noise
    Q = 0.001  # state noise

    # initialize
    beta[0] = 0.0
    P[0] = 1.0

    for t in range(1, n):
        # prediction
        beta_pred = beta[t - 1]
        P_pred = P[t - 1] + Q

        # observation
        H = xvals[t]
        yhat = H * beta_pred
        S = H * P_pred * H + R
        if S == 0:
            K = 0
        else:
            K = (P_pred * H) / S
        # update
        beta[t] = beta_pred + K * (yvals[t] - yhat)
        P[t] = (1 - K * H) * P_pred

    return pd.Series(beta, index=df.index)


def realized_volatility(series: pd.Series, window: int = 60) -> pd.Series:
    """Return rolling realized volatility (std of log returns * sqrt(periods))."""
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    ret = np.log(s).diff().dropna()
    rv = ret.rolling(window).std() * np.sqrt(window)
    return rv


def volatility_target_position_size(price_series: pd.Series, target_vol: float = 0.01, window: int = 60) -> Optional[float]:
    """Simple volatility-targeted position size (fraction of notional).

    Returns a scalar fraction such that expected vol ~ target_vol, based on realized vol over `window`.
    """
    s = price_series.dropna()
    if s.empty or len(s) < 2:
        return None
    rv = realized_volatility(s, window=window).dropna()
    if rv.empty:
        return None
    latest = rv.iloc[-1]
    if latest <= 0:
        return None
    size = float(target_vol / latest)
    return size


def portfolio_mean_reversion_backtest(
    px: pd.Series,
    py: pd.Series,
    hedge_ratio: float | pd.Series,
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.0,
    notional: float = 1.0,
) -> Dict[str, Any]:
    """Simple portfolio backtest for a two-leg mean reversion pair.

    - `px`, `py`: price series for X and Y aligned by index (timestamps)
    - `hedge_ratio`: scalar or series aligned with index (beta applied to x)
    - `zscore`: z-score series of the spread
    Returns dict with trades, pnl, cumulative_pnl (Series), turnover, trades_count
    """
    # align inputs
    df = pd.concat([px, py, zscore], axis=1, join="inner")
    df.columns = ["px", "py", "z"]
    if isinstance(hedge_ratio, (int, float)):
        df["beta"] = float(hedge_ratio)
    else:
        df = df.join(hedge_ratio.rename("beta"), how="left")
    df = df.dropna()
    if df.empty:
        return {"trades": 0, "pnl": 0.0, "cumulative_pnl": pd.Series(dtype=float), "turnover": 0.0}

    pos = 0
    entry_idx = None
    pnl = 0.0
    pnl_series = []
    trades = 0
    notional_x = 0.0
    notional_y = 0.0
    turnover = 0.0

    spread = df["py"] - df["beta"] * df["px"]

    for i, (ts, row) in enumerate(df.iterrows()):
        z = row["z"]
        if pos == 0 and abs(z) >= entry:
            pos = -1 if z > 0 else 1
            entry_idx = i
            trades += 1
            # notional exposure on each leg (simple): allocate `notional` to spread
            notional_y = notional
            notional_x = abs(row["beta"]) * notional
            turnover += notional_x + notional_y
        elif pos != 0 and abs(z) <= exit:
            # close position: pnl computed on spread change scaled by notional
            delta_spread = spread.iloc[i] - spread.iloc[entry_idx]
            pnl += pos * delta_spread * notional
            pnl_series.append((ts, pnl))
            pos = 0
            entry_idx = None
    # build cumulative pnl series
    if pnl_series:
        idx, vals = zip(*pnl_series)
        cum = pd.Series(list(vals), index=list(idx))
    else:
        cum = pd.Series([0.0], index=[df.index[-1]])

    return {"trades": trades, "pnl": float(pnl), "cumulative_pnl": cum, "turnover": float(turnover)}

