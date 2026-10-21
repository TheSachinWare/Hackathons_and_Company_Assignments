import base64
import io
import os
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from core.alerts import AlertManager, AlertRule
from core.analytics import (
    compute_pair_analytics,
    mini_mean_reversion_backtest,
    resample_ticks,
    summary_stats,
    half_life_of_mean_reversion,
    hurst_exponent,
    kalman_filter_hedge_ratio,
    realized_volatility,
)

from core.ingestion import BinanceIngestor
from core.state import AppState


state = AppState()
ingestor = BinanceIngestor(state.store)
file_ingestor = None
uploaded_rows = []
from core.ingestion import FileIngestor
file_ingestor = FileIngestor(state.store)
alerts = AlertManager()


def seed_demo_data(store, seconds: int = 900):
    """Seed the TickStore with synthetic demo ticks if it's empty."""
    try:
        df = store.snapshot()
        if not df.empty:
            return
    except Exception:
        pass
    import numpy as np
    import pandas as pd
    now = pd.Timestamp.utcnow().floor("s") - pd.Timedelta(seconds=seconds)
    rows = []
    syms = ["btcusdt", "ethusdt"]
    for i in range(seconds):
        ts = now + pd.Timedelta(seconds=i)
        # simple sinusoidal price for demo
        btc_price = 50000 + 50 * np.sin(i / 30.0) + (i % 5)
        eth_price = 4000 + 10 * np.cos(i / 40.0) + (i % 3) * 0.2
        rows.append({"ts": ts.isoformat(), "symbol": "btcusdt", "price": float(btc_price), "size": float(0.01 + (i % 5) * 0.001)})
        rows.append({"ts": ts.isoformat(), "symbol": "ethusdt", "price": float(eth_price), "size": float(0.1 + (i % 3) * 0.01)})
    store.append(rows)


# Seed demo data so Dashboard shows charts on first load
seed_demo_data(state.store)

DEFAULT_SYMBOLS = ["btcusdt", "ethusdt"]
THEMES = {
    "dark": {
        "bg": "#0b0f14",
        "bg_gradient": "linear-gradient(135deg, #0b0f14 0%, #0f172a 50%, #0b1220 100%)",
        "card": "#0f172a",
        "text": "#e6edf3",
        "border": "#1e293b",
        "accent": "#0ea5e9",
        "muted": "#9ca3af",
        "plotly_template": "plotly_dark",
    },
    "light": {
        "bg": "#f5f7fb",
        "bg_gradient": "linear-gradient(135deg, #e8edf7 0%, #ffffff 50%, #e6efff 100%)",
        "card": "#ffffff",
        "text": "#0f172a",
        "border": "#d0d7e2",
        "accent": "#2563eb",
        "muted": "#6b7280",
        "plotly_template": "plotly_white",
    },
}


def make_price_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Waiting for data...")
        return fig
    for sym, sdf in df.groupby("symbol"):
        fig.add_trace(
            go.Scatter(
                x=sdf["ts"],
                y=sdf["price"],
                mode="lines",
                name=sym.upper(),
            )
        )
    fig.update_layout(
        title="Prices",
        legend_orientation="h",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def make_volume_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Volume")
        return fig
    for sym, sdf in df.groupby("symbol"):
        fig.add_trace(
            go.Bar(
                x=sdf["ts"],
                y=sdf["size"],
                name=sym.upper(),
                opacity=0.7,
            )
        )
    fig.update_layout(
        barmode="group",
        title="Volume",
        legend_orientation="h",
    )
    return fig


def make_spread_fig(analytics) -> go.Figure:
    fig = go.Figure()
    if not analytics:
        fig.update_layout(title="Spread / Z-Score")
        return fig
    fig.add_trace(
        go.Scatter(
            x=analytics.spread.index,
            y=analytics.spread.values,
            mode="lines",
            name="Spread",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=analytics.zscore.index,
            y=analytics.zscore.values,
            mode="lines",
            name="Z-Score",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Spread and Z-Score",
        yaxis=dict(title="Spread"),
        yaxis2=dict(
            title="Z",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    return fig


def make_corr_fig(analytics) -> go.Figure:
    fig = go.Figure()
    if not analytics:
        fig.update_layout(title="Rolling Correlation")
        return fig
    fig.add_trace(
        go.Scatter(
            x=analytics.rolling_corr.index,
            y=analytics.rolling_corr.values,
            mode="lines",
            name="Corr",
        )
    )
    fig.update_layout(title="Rolling Correlation")
    return fig


def kpi_card(title: str, value: str, sub: str = "", color: str = "info"):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(title, className="text-muted small"),
                    html.H5(value, className=f"text-{color} mb-0"),
                    html.Div(sub, className="small mb-2"),
                ]
            ),
            className="mb-2 kpi-card",
        ),
        md=2,
        sm=6,
        xs=12,
    )


def section_style(current: str, name: str) -> dict:
    return {"display": "block"} if current == name else {"display": "none"}


def render_kpis(df: pd.DataFrame, pair, alerts_fired) -> dbc.Row:
    total_ticks = len(df)
    syms = sorted(df["symbol"].unique()) if not df.empty else []
    last_ts = df["ts"].max() if not df.empty else None
    last_ts_str = last_ts.strftime("%H:%M:%S") if last_ts is not None else "—"
    latest_z = pair.zscore.dropna().iloc[-1] if pair and not pair.zscore.dropna().empty else None
    adf_txt = f"{pair.adf_pvalue:.4f}" if pair and pair.adf_pvalue is not None else "n/a"
    corr_last = pair.rolling_corr.dropna().iloc[-1] if pair and not pair.rolling_corr.dropna().empty else None
    alert_text = ", ".join(a["name"] for a in alerts_fired) if alerts_fired else "None"
    # Build main KPI cards
    main_cards = [
        kpi_card("Buffered ticks", f"{total_ticks:,}", f"Symbols: {', '.join(syms) if syms else '—'}"),
        kpi_card("Last update", last_ts_str, "UTC", color="secondary"),
        kpi_card("Latest |z|", f"{abs(latest_z):.2f}" if latest_z is not None else "—", "Pair z-score", color="warning"),
        kpi_card("ADF p-value", adf_txt, "Lower is better", color="primary"),
        kpi_card("Rolling corr", f"{corr_last:.3f}" if corr_last is not None else "—", "Pair corr", color="info"),
        kpi_card("Alerts", alert_text, "Live rules", color="danger" if alerts_fired else "success"),
    ]

    # Sparkline mini-cards showing last 30 prices per symbol
    spark_cols = []
    for sym in syms:
        s = df[df["symbol"] == sym].sort_values("ts").tail(30)
        if not s.empty:
            fig = go.Figure(go.Scatter(x=s["ts"], y=s["price"], mode="lines", line=dict(width=1, color="#2563eb")))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=44)
        else:
            fig = go.Figure()
            fig.update_layout(height=44)
        spark_cols.append(
            dbc.Col(
                dbc.Card(dbc.CardBody([html.Div(sym.upper(), className="text-muted small"), dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "44px"})]), className="mb-2"),
                md=2,
                sm=4,
                xs=6,
            )
        )

    return dbc.Row(main_cards + spark_cols, className="gy-2 mb-3")


def dashboard_layout():
    return html.Div(
        [
            dbc.Row([
                dbc.Col(dbc.Label("Data source:"), md=2),
                dbc.Col(
                    dcc.RadioItems(
                        id="data-source",
                        options=[{"label": "Live Binance", "value": "live"}, {"label": "Replay (uploaded NDJSON)", "value": "file"}],
                        value="live",
                        inline=True,
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Refresh ms"),
                        dbc.Input(id="refresh-ms", type="number", value=1000, step=100),
                    ]),
                    md=4,
                ),
            ], className="mb-2"),
            dbc.Card(
                dbc.CardBody(
                    [
                    
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText("Symbols"),
                                            dbc.Input(id="symbols-input", value="btcusdt,ethusdt"),
                                        ]
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Timeframe"),
                                        dcc.RadioItems(
                                            id="timeframe",
                                            options=[
                                                {"label": "Tick", "value": "tick"},
                                                {"label": "1s", "value": "1s"},
                                                {"label": "1m", "value": "1min"},
                                                {"label": "5m", "value": "5min"},
                                            ],
                                            value="1s",
                                            inline=True,
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Z-Score Alert"),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText(">|z|"),
                                                dbc.Input(id="z-threshold", type="number", value=2.0, step=0.1),
                                            ]
                                        ),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="gy-3",
                        ),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Pair (hedge / ADF)"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Input(id="pair-a", value="btcusdt")),
                                                        dbc.Col(dbc.Input(id="pair-b", value="ethusdt")),
                                                    ]
                                                ),
                                                html.Div(className="mt-2 d-flex align-items-center", children=[
                                                    dbc.Label("Regression:" , className="me-2 small"),
                                                    dcc.Dropdown(id="regression-type", options=[
                                                        {"label": "OLS", "value": "ols"},
                                                        {"label": "Theil-Sen", "value": "theil-sen"},
                                                        {"label": "Huber", "value": "huber"},
                                                    ], value="ols", clearable=False, style={"width": "220px"}),
                                                    dbc.Button("Run ADF", id="run-adf", color="secondary", className="ms-3", size="sm"),
                                                    ]),
                                                    html.Div(id="adf-result", className="mt-2"),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Rolling Window (ticks)"),
                                        dbc.Input(id="lookback", type="number", value=500),
                                        dbc.Label("Correlation Window", className="mt-2"),
                                        dbc.Input(id="corr-window", type="number", value=100),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Upload CSV"),
                                        dcc.Upload(
                                            id="upload-data",
                                            children=html.Div(["Upload OHLC/Trades CSV"]),
                                            style={
                                                "width": "100%",
                                                "height": "60px",
                                                "lineHeight": "60px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "5px",
                                                "textAlign": "center",
                                                "marginTop": "10px",
                                            },
                                            className="uploader-box",
                                        ),
                                        html.Small(
                                            "Columns: ts, symbol, price, size (UTC ts preferred)",
                                            className="text-muted",
                                        ),
                                        html.Div(id="upload-status", className="mt-1"),
                                        html.Div([
                                            dcc.Upload(id="upload-ndjson", children=html.Div(["Upload NDJSON (for replay)"]), style={"width":"100%","height":"60px","lineHeight":"60px","borderWidth":"1px","borderStyle":"dashed","borderRadius":"5px","textAlign":"center","marginTop":"10px"}),
                                            dbc.Button("Start Replay", id="start-replay", color="secondary", className="mt-2"),
                                        ]),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="gy-3",
                        ),
                    ]
                ),
                id="controls-card",
                className="border-0 shadow-sm mb-3",
            ),
            html.Div(id="kpi-cards"),
            dbc.Row(
                [
                    dbc.Col(dbc.Spinner(dcc.Graph(id="price-fig", style={"height": "480px"}), color="primary", size="lg"), md=8, className="mb-4"),
                    dbc.Col(dbc.Spinner(dcc.Graph(id="volume-fig", style={"height": "480px"}), color="secondary"), md=4, className="mb-4"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Spinner(dcc.Graph(id="spread-fig", style={"height": "380px"}), color="warning"), md=6, className="mb-4"),
                    dbc.Col(dbc.Spinner(dcc.Graph(id="corr-fig", style={"height": "380px"}), color="info"), md=6, className="mb-4"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Summary Stats"),
                                dbc.CardBody(
                                    dbc.Table(id="stats-table", bordered=True, dark=True, striped=True, hover=True),
                                    style={"maxHeight": "420px", "overflowY": "auto"},
                                ),
                            ],
                            id="stats-card",
                            className="border-0 shadow-sm",
                        ),
                        md=12,
                        className="mb-4",
                    ),
                ],
                className="my-3",
            ),
        ],
        id="page-dashboard",
    )


def login_layout():
    return html.Div(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Login", className="mb-3"),
                    dbc.InputGroup([dbc.InputGroupText("👤"), dbc.Input(id="username", placeholder="Username")], className="mb-2"),
                    dbc.InputGroup([dbc.InputGroupText("🔒"), dbc.Input(id="password", type="password", placeholder="Password")], className="mb-2"),
                    dbc.Button("Login", id="login-btn", color="primary", className="w-100"),
                    html.Div(id="auth-status", className="mt-2 small text-info text-center"),
                ]
            ),
            id="login-card",
            className="border-0 shadow-sm",
        ),
        id="page-login",
    )


def signup_layout():
    return html.Div(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Signup", className="mb-3"),
                    dbc.InputGroup([dbc.InputGroupText("👤"), dbc.Input(id="signup-username", placeholder="Username")], className="mb-2"),
                    dbc.InputGroup([dbc.InputGroupText("🔒"), dbc.Input(id="signup-password", type="password", placeholder="Password")], className="mb-2"),
                    dbc.Button("Sign up", id="signup-btn", color="secondary", className="w-100"),
                    html.Div(id="signup-status", className="mt-2 small text-info text-center"),
                ]
            ),
            id="signup-card",
            className="border-0 shadow-sm",
        ),
        id="page-signup",
    )


def exports_layout():
    return html.Div(
        dbc.Card(
            dbc.CardBody([
                html.H4("Exports", className="mb-3"),
                html.P("Use the navbar Download CSV button for current buffer."),
                html.P("Upload is available on Dashboard view."),
                dbc.Button("Download CSV", id="download-btn-2", color="info", className="mt-2"),
            ]),
            id="exports-card",
            className="border-0 shadow-sm",
        ),
        id="page-exports",
    )


def analysis_layout():
    return html.Div(html.H4("Analysis (placeholder)"), id="page-analysis")


def serve_layout():
    return dbc.Container(
        [
            dbc.Navbar(
                [
                    dbc.Button(html.Span(className="navbar-toggler-icon"), id="sidebar-toggle", className="me-3", color="light", outline=True),
                    dbc.NavbarBrand(
                        [
                            html.Img(src="/assets/logo.svg", height="36", style={"marginRight": "10px"}),
                            html.Div([
                                html.Div("Live Pairs Analytics", className="fw-bold fs-5", style={"lineHeight":"1"}),
                                html.Div("Binance Futures — real-time pairs dashboard", className="small text-muted", style={"lineHeight":"1"}),
                            ]),
                            dbc.Badge("LIVE", color="danger", className="ms-2"),
                        ],
                        className="d-flex align-items-center ms-1",
                    ),
                    # Centered search to mimic Instagram header
                    dbc.Input(id="nav-search", placeholder="Search symbols, e.g. btcusdt", className="mx-3 d-none d-md-block", style={"width":"320px","borderRadius":"20px","paddingLeft":"12px"}),
                    dbc.Nav(
                        [
                            dbc.Button("▶ Start", id="start-btn", color="success", className="me-2", size="sm"),
                            dbc.Button("■ Stop", id="stop-btn", color="danger", className="me-2", size="sm"),
                            dbc.Button("⬇ CSV", id="download-btn", color="info", className="me-2", size="sm"),
                            dbc.Button("⬇ Analytics", id="download-analytics", color="info", className="me-2", size="sm"),
                            dcc.Download(id="download-data"),
                            dcc.Download(id="download-analytics-data"),
                            html.Span(id="status-text", className="ms-2 fw-semibold"),
                            # small avatar placeholder on right
                            html.Img(src="/assets/logo.svg", height="32", style={"borderRadius":"50%","marginLeft":"12px","boxShadow":"0 6px 18px rgba(2,6,23,0.12)"}),
                        ],
                        className="ms-auto d-flex align-items-center",
                        navbar=True,
                    ),
                ],
                id="navbar",
                className="mb-3 rounded px-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H2("Trade faster with clearer signals", className="mb-1"),
                        html.P(
                            "Live spreads, hedge ratios, alerts, and exports — tuned for pairs traders.",
                            className="text-muted mb-0",
                        ),
                    ]
                ),
                id="hero-card",
                className="border-0 shadow-lg mb-4",
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Dashboard", href="/", className="me-2"),
                    dbc.NavLink("Login", href="/login", className="me-2"),
                    dbc.NavLink("Signup", href="/signup", className="me-2"),
                    dbc.NavLink("Analysis", href="/analysis", className="me-2"),
                    dbc.NavLink("Exports", href="/exports", className="me-2"),
                    dbc.Button("🌙 Dark", id="theme-toggle", color="info", size="sm", className="ms-auto"),
                ],
                pills=True,
                className="mb-3 align-items-center nav-top-menu",
            ),
            # Sidebar (fixed, collapsible)
            html.Div(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div("Menu", className="fw-bold mb-2"),
                                dbc.ListGroup(
                                    [
                                        dcc.Link(dbc.ListGroupItem("Dashboard", action=True), href="/"),
                                        dcc.Link(dbc.ListGroupItem("Analysis", action=True), href="/analysis"),
                                        dcc.Link(dbc.ListGroupItem("Exports", action=True), href="/exports"),
                                        dcc.Link(dbc.ListGroupItem("Login", action=True), href="/login"),
                                        dbc.ListGroupItem(dbc.Button("Toggle Theme", id="theme-toggle-sidebar", color="secondary", size="sm", className="w-100"), className="mt-2"),
                                    ],
                                    flush=True,
                                ),
                            ]
                        )
                    ],
                    id="sidebar-card",
                    className="shadow-sm",
                ),
                ),
                html.Div(id="sidebar-backdrop", n_clicks=0, className="sidebar-backdrop"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            dbc.InputGroup(
                                                                [
                                                                    dbc.InputGroupText("Symbols"),
                                                                    dbc.Input(id="symbols-input", value="btcusdt,ethusdt"),
                                                                ]
                                                            ),
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Timeframe"),
                                                                dcc.RadioItems(
                                                                    id="timeframe",
                                                                    options=[
                                                                        {"label": "Tick", "value": "tick"},
                                                                        {"label": "1s", "value": "1s"},
                                                                        {"label": "1m", "value": "1min"},
                                                                        {"label": "5m", "value": "5min"},
                                                                    ],
                                                                    value="1s",
                                                                    inline=True,
                                                                ),
                                                            ],
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Z-Score Alert"),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(">|z|"),
                                                                        dbc.Input(id="z-threshold", type="number", value=2.0, step=0.1),
                                                                    ]
                                                                ),
                                                            ],
                                                            md=4,
                                                        ),
                                                    ],
                                                    className="gy-3",
                                                ),
                                                html.Hr(),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Pair (hedge / ADF)"),
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(dbc.Input(id="pair-a", value="btcusdt")),
                                                                        dbc.Col(dbc.Input(id="pair-b", value="ethusdt")),
                                                                    ]
                                                                ),
                                                            ],
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Rolling Window (ticks)"),
                                                                dbc.Input(id="lookback", type="number", value=500),
                                                                dbc.Label("Correlation Window", className="mt-2"),
                                                                dbc.Input(id="corr-window", type="number", value=100),
                                                            ],
                                                            md=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Upload CSV"),
                                                                dcc.Upload(
                                                                    id="upload-data",
                                                                    children=html.Div(["Upload OHLC/Trades CSV"]),
                                                                    style={
                                                                        "width": "100%",
                                                                        "height": "60px",
                                                                        "lineHeight": "60px",
                                                                        "borderWidth": "1px",
                                                                        "borderStyle": "dashed",
                                                                        "borderRadius": "5px",
                                                                        "textAlign": "center",
                                                                        "marginTop": "10px",
                                                                    },
                                                                    className="uploader-box",
                                                                ),
                                                                html.Small(
                                                                    "Columns: ts, symbol, price, size (UTC ts preferred)",
                                                                    className="text-muted",
                                                                ),
                                                                html.Div(id="upload-status", className="mt-1"),
                                                            ],
                                                            md=4,
                                                        ),
                                                    ],
                                                    className="gy-3",
                                                ),
                                            ]
                                        ),
                                        id="controls-card",
                                        className="border-0 shadow-sm mb-3",
                                    ),
                                    html.Div(id="kpi-cards"),
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Spinner(dcc.Graph(id="price-fig", style={"height": "380px"}), color="primary", size="lg"), md=6, className="mb-4"),
                                            dbc.Col(dbc.Spinner(dcc.Graph(id="volume-fig", style={"height": "180px"}), color="secondary"), md=3, className="mb-4"),
                                            dbc.Col(dbc.Spinner(dcc.Graph(id="pnl-fig", style={"height": "180px"}), color="success"), md=3, className="mb-4"),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Spinner(dcc.Graph(id="spread-fig", style={"height": "320px"}), color="warning"), md=6, className="mb-4"),
                                            dbc.Col(dbc.Spinner(dcc.Graph(id="corr-fig", style={"height": "320px"}), color="info"), md=6, className="mb-4"),
                                        ]
                                    ),
                                    dbc.Row([
                                        dbc.Col(dbc.Spinner(dcc.Graph(id="beta-fig", style={"height": "220px"}), color="secondary"), md=6),
                                        dbc.Col(dbc.Spinner(dcc.Graph(id="rv-fig", style={"height": "220px"}), color="secondary"), md=6),
                                    ], className="mb-4"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Summary Stats"),
                                                        dbc.CardBody(
                                                            dbc.Table(
                                                                id="stats-table", bordered=True, dark=True, striped=True, hover=True
                                                            ),
                                                            style={"maxHeight": "420px", "overflowY": "auto"},
                                                        ),
                                                    ],
                                                    id="stats-card",
                                                    className="border-0 shadow-sm",
                                                ),
                                                md=12,
                                                className="mb-4",
                                            ),
                                        ],
                                        className="my-3",
                                    ),
                                ],
                                id="content-dashboard",
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Login", className="mb-3"),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("👤"),
                                                    dbc.Input(id="username", placeholder="Username"),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("🔒"),
                                                    dbc.Input(id="password", type="password", placeholder="Password"),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Button("Login", id="login-btn", color="primary", className="w-100"),
                                            html.Div(id="auth-status", className="mt-2 small text-info text-center"),
                                        ]
                                    ),
                                    id="login-card",
                                    className="border-0 shadow-sm",
                                ),
                                id="content-login",
                                style={"display": "none"},
                            ),
                            html.Div(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Signup", className="mb-3"),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("👤"),
                                                    dbc.Input(id="signup-username", placeholder="Username"),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("🔒"),
                                                    dbc.Input(id="signup-password", type="password", placeholder="Password"),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Button("Sign up", id="signup-btn", color="secondary", className="w-100"),
                                            html.Div(id="signup-status", className="mt-2 small text-info text-center"),
                                        ]
                                    ),
                                    id="signup-card",
                                    className="border-0 shadow-sm",
                                ),
                                id="content-signup",
                                style={"display": "none"},
                            ),
                            html.Div(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H4("Alerts & Backtest", className="mb-3"),
                                                dbc.Alert(id="alert-box", color="warning", className="mb-3", is_open=True),
                                                html.Div(id="backtest-stats", className="text-info"),
                                            ]
                                        ),
                                        id="alerts-card",
                                        className="border-0 shadow-sm",
                                    ),
                                id="content-alerts",
                                style={"display": "none"},
                            ),
                            html.Div(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H4("Exports", className="mb-3"),
                                                html.P("Use the navbar Download CSV button for current buffer."),
                                                html.P("Upload is available on Dashboard view."),
                                                dbc.Button("Download CSV", id="download-btn-2", color="info", className="mt-2"),
                                            ]
                                        ),
                                        id="exports-card",
                                        className="border-0 shadow-sm",
                                    ),
                                id="content-exports",
                                style={"display": "none"},
                            ),
                        ],
                        md=12,
                        className="pb-5",
                    ),
                ],
                className="g-3",
            ),
            dcc.Interval(id="refresh-interval", interval=1000, n_intervals=0),
            dcc.Store(id="run-state", data={"running": False, "symbols": DEFAULT_SYMBOLS}),
            dcc.Store(id="auth-state", data={"authed": False, "user": ""}),
            dcc.Store(id="sidebar-state", data={"open": False}),
            dcc.Store(id="nav", data={}),
            dcc.Store(id="nav-dummy", data=""),
            dcc.Store(id="theme-store", data={"mode": "dark"}),
            dcc.Store(id="view-state", data={"view": "dashboard"}),
            dcc.Store(id="ui-sync", data=""),
            # Global toast for transient alerts (shows when alert rules fire)
            dbc.Toast(id="alert-toast", header="Alert", is_open=False, duration=4500, icon="danger", style={"position":"fixed","top":"80px","right":"20px","zIndex":1100}),
            # Upload success toast
            dbc.Toast(id="upload-toast", header="Upload", is_open=False, duration=3000, icon="primary", style={"position":"fixed","top":"130px","right":"20px","zIndex":1100}),
        ],
        id="root-container",
        fluid=True,
        className="pb-5",
        style={"minHeight": "1400px"},
    )


# Use Bootstrap theme + Inter font for a modern look
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap",
    ],
)
server = app.server
app.title = "Live Pairs Analytics"
app.layout = serve_layout
# Include a validation layout that contains all ids used in callbacks
# so Dash doesn't validate callbacks against a missing id during multi-page setup.
app.validation_layout = serve_layout()
# Provide a static validation layout so callbacks referencing page components validate.
# This ensures callbacks registered before pages are rendered do not error.
app.validation_layout = serve_layout()


# Clientside UI sync: toggle body classes for dark mode and sidebar state
app.clientside_callback(
    """
    function(theme, sidebar_state) {
        var mode = (theme||{}).mode || 'dark';
        try {
            if(mode==='dark') document.body.classList.add('dark-mode');
            else document.body.classList.remove('dark-mode');
            var open = (sidebar_state||{}).open;
            if(open) document.body.classList.add('sidebar-open');
            else document.body.classList.remove('sidebar-open');
        } catch(e) { }
        return '';
    }
    """,
    Output('ui-sync','data'),
    Input('theme-store','data'),
    Input('sidebar-state','data')
)


@app.callback(
    Output('refresh-interval','interval'),
    Input('refresh-ms','value'),
)
def set_refresh_interval(ms):
    try:
        v = int(ms)
        if v < 100:
            v = 100
        return v
    except Exception:
        return 1000


@app.callback(
    Output("run-state", "data"),
    Output("status-text", "children"),
    Input("start-btn", "n_clicks"),
    Input("stop-btn", "n_clicks"),
    State("auth-state", "data"),
    State("symbols-input", "value"),
    State("data-source", "value"),
    State("run-state", "data"),
)
def control_stream(start_clicks, stop_clicks, auth_state, symbols_value, run_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    # allow starting without login (convenience) but require login for stop/actions
    if not auth_state.get("authed") and trig != "start-btn":
        return run_state, "Login first"
    symbols = [s.strip().lower() for s in (symbols_value or "").split(",") if s.strip()]
    if trig == "start-btn":
        if not symbols:
            return run_state, "Provide at least one symbol"
        # Decide source
        src = dash.callback_context.states.get("data-source.value") or "live"
        if src == "live":
            ingestor.start(symbols)
        else:
            # Start file replay if uploaded rows exist
            try:
                if uploaded_rows:
                    file_ingestor.start(uploaded_rows, speed=5.0, repeat=False)
            except Exception:
                pass
        state.active_symbols = symbols
        state.running = True
        status = f"Streaming {', '.join(symbols)}"
        if not auth_state.get("authed"):
            status += " (unauthenticated)"
        return {"running": True, "symbols": symbols}, status
    ingestor.stop()
    state.running = False
    return {"running": False, "symbols": symbols}, "Stopped"

    # Clientside navigation: when `nav` store contains {redirect: '/'} navigate browser
    app.clientside_callback(
        """
        function(nav) {
            if(nav && nav.redirect) {
                try { window.location.href = nav.redirect; } catch(e) {}
            }
            return '';
        }
        """,
        Output('nav-dummy','data'),
        Input('nav','data')
    )

@app.callback(
    Output("upload-status", "children"),
    Output("upload-toast", "children"),
    Output("upload-toast", "is_open"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def upload_data(contents, filename):
    if contents is None:
        raise PreventUpdate
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        expected = {"ts", "symbol", "price", "size"}
        if not expected.issubset(df.columns):
            return (
                "Missing columns. Expected ts,symbol,price,size",
                "Upload failed",
                True,
            )
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        rows = df.to_dict("records")
        state.store.append(rows)
        msg = f"Loaded {len(df)} rows from {filename}"
        return msg, msg, True
    except Exception as exc:
        err = f"Upload failed: {exc}"
        return err, err, True


@app.callback(
    Output("upload-status", "children"),
    Output("upload-toast", "children"),
    Output("upload-toast", "is_open"),
    Input("upload-ndjson", "contents"),
    State("upload-ndjson", "filename"),
)
def upload_ndjson(contents, filename):
    global uploaded_rows
    if contents is None:
        raise PreventUpdate
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        text = decoded.decode("utf-8")
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
        uploaded_rows = rows
        msg = f"Loaded {len(rows)} rows from {filename} (ready to replay)"
        return msg, msg, True
    except Exception as exc:
        err = f"NDJSON upload failed: {exc}"
        return err, err, True


@app.callback(
    Output('download-analytics-data', 'data'),
    Input('download-analytics', 'n_clicks'),
    State('pair-a', 'value'),
    State('pair-b', 'value'),
    State('regression-type', 'value'),
    prevent_initial_call=True,
)
def download_analytics(n_clicks, pair_a, pair_b, regression_type):
    # prepare analytics CSV (spread, zscore, rolling_corr) for selected pair
    df = state.store.snapshot()
    if df.empty or not pair_a or not pair_b:
        raise PreventUpdate
    pa = compute_pair_analytics(df, pair_a.lower(), pair_b.lower(), regression=regression_type)
    if pa is None:
        raise PreventUpdate
    out = pd.DataFrame({
        'ts': pa.spread.index,
        'spread': pa.spread.values,
        'zscore': pa.zscore.values,
        'rolling_corr': pa.rolling_corr.values,
    })
    return dcc.send_data_frame(out.to_csv, f"analytics_{pair_a}_{pair_b}.csv", index=False)


@app.callback(
    Output("kpi-cards", "children"),
    Output("price-fig", "figure"),
    Output("volume-fig", "figure"),
    Output("spread-fig", "figure"),
    Output("corr-fig", "figure"),
    Output("stats-table", "children"),
    Output("alert-box", "children"),
    Output("backtest-stats", "children"),
    Output("adf-result", "children"),
    Output("alert-toast", "children"),
    Output("alert-toast", "is_open"),
    Output("beta-fig", "figure"),
    Output("rv-fig", "figure"),
    Output("pnl-fig", "figure"),
    Input("refresh-interval", "n_intervals"),
    Input("run-adf", "n_clicks"),
    State("timeframe", "value"),
    State("pair-a", "value"),
    State("pair-b", "value"),
    State("lookback", "value"),
    State("corr-window", "value"),
    State("z-threshold", "value"),
    State("regression-type", "value"),
    State("theme-store", "data"),
)
def update_dashboard(n_intervals, run_adf_clicks, timeframe, pair_a, pair_b, lookback, corr_window, z_thresh, regression_type, theme_store):
    theme = THEMES.get((theme_store or {}).get("mode"), THEMES["dark"])
    df = state.store.snapshot()
    if timeframe != "tick":
        df = resample_ticks(df, timeframe)
    price_fig = make_price_fig(df)
    price_fig.update_layout(template=theme["plotly_template"])
    volume_fig = make_volume_fig(df)
    volume_fig.update_layout(template=theme["plotly_template"])

    pair = None
    if pair_a and pair_b:
        pair = compute_pair_analytics(
            df,
            sym_x=pair_a.lower(),
            sym_y=pair_b.lower(),
            lookback=int(lookback or 500),
            corr_window=int(corr_window or 100),
            regression=regression_type,
        )
    spread_fig = make_spread_fig(pair)
    spread_fig.update_layout(template=theme["plotly_template"])
    corr_fig = make_corr_fig(pair)
    corr_fig.update_layout(template=theme["plotly_template"])

    stats_df = summary_stats(df)
    if stats_df.empty:
        stats_table = html.Tbody([html.Tr([html.Td("Waiting for data")])])
    else:
        header = html.Thead(
            html.Tr([html.Th(col) for col in ["Symbol", "Last", "1m%", "5m%", "Vol5m"]])
        )
        body_rows = []
        for _, row in stats_df.iterrows():
            body_rows.append(
                html.Tr(
                    [
                        html.Td(row["symbol"].upper()),
                        html.Td(f"{row['last']:.4f}"),
                        html.Td(f"{row['pct_change_1m']*100:.2f}%"),
                        html.Td(f"{row['pct_change_5m']*100:.2f}%"),
                        html.Td(f"{row['vol_5m']:.2f}"),
                    ]
                )
            )
        stats_table = html.Table([header, html.Tbody(body_rows)], className="table table-dark")

    alerts.set_rules([AlertRule(name="Z-Score", metric="zscore", operator=">", threshold=float(z_thresh or 2))])
    metrics = {}
    backtest_text = ""
    if pair:
        latest_z = pair.zscore.dropna().iloc[-1] if not pair.zscore.dropna().empty else None
        if latest_z is not None:
            metrics["zscore"] = float(abs(latest_z))
        bt = mini_mean_reversion_backtest(pair.zscore.dropna())
        # advanced analytics helpers
        hl = half_life_of_mean_reversion(pair.spread)
        hurst = hurst_exponent(pair.spread)
        # kalman hedge ratio on the pivoted series
        try:
            kv = kalman_filter_hedge_ratio(
                df.pivot(index="ts", columns="symbol", values="price")[pair.spread.index.name].get(pair.hedge_ratio, pd.Series())
            )
        except Exception:
            kv = None
        adf_txt = f"{pair.adf_pvalue:.4f}" if pair.adf_pvalue is not None else "n/a"
        backtest_text = (
            f"Hedge ratio={pair.hedge_ratio:.4f} | ADF p={adf_txt} | "
            f"Mini MR: trades={bt['trades']} pnl={bt['pnl']:.4f} avg_hold={bt['avg_holding']:.1f} ticks"
        )
        if hl is not None:
            backtest_text += f" | Half-life={hl:.1f}"
        if hurst is not None:
            backtest_text += f" | Hurst={hurst:.3f}"
    fired = alerts.evaluate(metrics)
    alert_box = "No alerts" if not fired else f"Alert: {', '.join(a['name'] for a in fired)} @ {metrics}"

    kpi_row = render_kpis(df, pair, fired)

    # Prepare toast: show transient toast when alerts fired
    toast_children = alert_box
    toast_open = bool(fired)

    # ADF display: show p-value and stationarity badge
    try:
        if pair and pair.adf_pvalue is not None:
            pval = pair.adf_pvalue
            status = "Stationary" if pval < 0.05 else "Non-stationary"
            color = "success" if pval < 0.05 else "danger"
            adf_children = dbc.Row([
                dbc.Col(html.Div("ADF p-value", className="small text-muted"), width="auto"),
                dbc.Col(html.Div(f"{pval:.4g}", className="fw-bold ms-1"), width="auto"),
                dbc.Col(dbc.Badge(status, color=color, className="ms-2"), width="auto"),
            ], className="align-items-center")
        else:
            adf_children = html.Div("ADF p-value: n/a", className="small text-muted")
    except Exception:
        adf_children = html.Div("ADF p-value: n/a", className="small text-muted")

    # Additional analytics: kalman beta, realized vol, and portfolio backtest
    beta_fig = go.Figure()
    rv_fig = go.Figure()
    pnl_fig = go.Figure()
    try:
        pivot = df.pivot(index="ts", columns="symbol", values="price").dropna()
        if pair and pair.spread is not None:
            # Kalman beta on the two series
            x = pivot[pair_a.lower()]
            y = pivot[pair_b.lower()]
            kbeta = kalman_filter_hedge_ratio(x, y)
            if not kbeta.empty:
                beta_fig.add_trace(go.Scatter(x=kbeta.index, y=kbeta.values, mode="lines", name="Kalman Beta"))
                beta_fig.update_layout(title="Kalman Hedge Ratio")
            # realized vol on spread
            rv = realized_volatility(pair.spread.dropna())
            if not rv.empty:
                rv_fig.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines", name="Realized Vol"))
                rv_fig.update_layout(title="Realized Volatility (spread)")
            # portfolio backtest
            back = portfolio_mean_reversion_backtest(x, y, pair.hedge_ratio, pair.zscore.dropna(), entry=float(z_thresh or 2.0), exit=0.0, notional=1.0)
            cum = back.get("cumulative_pnl")
            if cum is not None and not cum.empty:
                pnl_fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Cumulative PnL"))
                pnl_fig.update_layout(title=f"Backtest PnL (trades={back.get('trades',0)})")
            # append analytics to backtest_text
            if "turnover" in back and back.get("turnover"):
                backtest_text += f" | Turnover={back.get('turnover'):.2f}"
        else:
            beta_fig.update_layout(title="Kalman Hedge Ratio")
            rv_fig.update_layout(title="Realized Volatility")
            pnl_fig.update_layout(title="Backtest PnL")
    except Exception:
        beta_fig.update_layout(title="Kalman Hedge Ratio")
        rv_fig.update_layout(title="Realized Volatility")
        pnl_fig.update_layout(title="Backtest PnL")

    return (
        kpi_row,
        price_fig,
        volume_fig,
        spread_fig,
        corr_fig,
        stats_table,
        alert_box,
        backtest_text,
        toast_children,
        toast_open,
        beta_fig,
        rv_fig,
        pnl_fig,
    )


@app.callback(
    Output("auth-state", "data"),
    Output("auth-status", "children"),
    Output("nav", "data"),
    Input("login-btn", "n_clicks"),
    State("username", "value"),
    State("password", "value"),
    State("auth-state", "data"),
)
def auth_login(login_clicks, username, password, auth_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if not username or not password or len(password) < 3:
        return auth_state, "Enter username and password (>=3 chars)", dash.no_update
    # on success, set auth and request clientside redirect to dashboard
    return {"authed": True, "user": username}, f"Logged in as {username}", {"redirect": "/"}


# Download current buffer as CSV (shared by navbar and exports button)
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    Input("download-btn-2", "n_clicks"),
    prevent_initial_call=True,
)
def handle_download(nav_clicks, exports_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    df = state.store.snapshot()
    if df.empty:
        raise PreventUpdate
    # Use Dash helper to send DataFrame as CSV
    return dcc.send_data_frame(df.to_csv, "ticks_export.csv", index=False)


@app.callback(
    Output("auth-state", "data", allow_duplicate=True),
    Output("signup-status", "children"),
    Output("nav", "data"),
    Input("signup-btn", "n_clicks"),
    State("signup-username", "value"),
    State("signup-password", "value"),
    State("auth-state", "data"),
    prevent_initial_call=True,
)
def auth_signup(signup_clicks, username, password, auth_state):
    if not signup_clicks:
        raise PreventUpdate
    if not username or not password or len(password) < 3:
        return auth_state, "Enter username and password (>=3 chars)", dash.no_update
    return {"authed": True, "user": username}, f"Signed up as {username}", {"redirect": "/"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting Dash server on http://0.0.0.0:{port} ...")
    app.run_server(host="0.0.0.0", port=port, debug=False)
