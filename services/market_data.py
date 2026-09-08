"""Best-effort public price retrieval. Call only from the page that needs it."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def fetch_price_history(symbols: list[str], period: str = "1y") -> tuple[pd.DataFrame, str, str | None]:
    """Fetch adjusted closes via yfinance, returning an empty frame rather than raising to the UI."""
    if not symbols:
        return pd.DataFrame(), "No symbols requested", None
    try:
        import yfinance as yf

        normalized = [symbol if symbol.upper().endswith((".NS", ".BO")) else f"{symbol}.NS" for symbol in symbols[:15]]
        raw = yf.download(normalized, period=period, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return pd.DataFrame(), "Yahoo Finance returned no data", None
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(normalized[0])
        return prices.dropna(how="all"), "Yahoo Finance (public, may be delayed)", datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        return pd.DataFrame(), f"Market data unavailable: {exc}", None
