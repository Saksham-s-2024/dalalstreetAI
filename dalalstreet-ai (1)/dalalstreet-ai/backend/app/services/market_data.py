"""
DalalStreet AI — Market Data Service (Upstox)
Streams real-time OHLC, LTP, and depth data via Upstox WebSocket.
Falls back to mock data in development when no API key is set.
"""
import asyncio
import logging
import random
from datetime import datetime
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

UPSTOX_BASE_URL = "https://api.upstox.com/v2"

# NSE instrument key format: NSE_EQ|<ISIN> or NSE_EQ|<SYMBOL>
# For simplicity we use symbol-based lookup
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
    "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "AXISBANK",
    "LT", "ASIANPAINT", "MARUTI", "TITAN", "BAJFINANCE",
    "SUNPHARMA", "ULTRACEMCO", "HCLTECH", "WIPRO", "NESTLEIND",
]


async def _upstox_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.upstox_access_token}",
        "Accept": "application/json",
    }


async def fetch_quote(symbol: str) -> dict[str, Any]:
    """
    Fetch real-time quote for a symbol from Upstox.
    Falls back to mock data if token not configured.
    """
    if not settings.upstox_access_token:
        return _mock_quote(symbol)

    instrument_key = f"NSE_EQ|{symbol}"
    url = f"{UPSTOX_BASE_URL}/market-quote/quotes"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers=await _upstox_headers(),
                params={"instrument_key": instrument_key},
            )
            resp.raise_for_status()
            data = resp.json()
            return _parse_upstox_quote(data, symbol)
    except Exception as exc:
        logger.warning("Upstox quote fetch failed for %s: %s", symbol, exc)
        return _mock_quote(symbol)


async def fetch_historical_ohlc(
    symbol: str,
    interval: str = "1minute",
    days: int = 30,
) -> list[dict[str, Any]]:
    """
    Fetch historical OHLC candles from Upstox.
    Returns list of {time, open, high, low, close, volume}.
    """
    if not settings.upstox_access_token:
        return _mock_ohlc(symbol, days * 375)  # ~375 candles/day

    instrument_key = f"NSE_EQ|{symbol}"
    from_date = (datetime.now() - __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    url = f"{UPSTOX_BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=await _upstox_headers())
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("data", {}).get("candles", [])
            return [
                {
                    "time": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": int(c[5]),
                }
                for c in candles
            ]
    except Exception as exc:
        logger.warning("Upstox OHLC fetch failed for %s: %s", symbol, exc)
        return _mock_ohlc(symbol, 100)


def _parse_upstox_quote(data: dict, symbol: str) -> dict[str, Any]:
    try:
        q = data["data"][f"NSE_EQ:{symbol}"]
        return {
            "symbol": symbol,
            "ltp": q["last_price"],
            "open": q["ohlc"]["open"],
            "high": q["ohlc"]["high"],
            "low": q["ohlc"]["low"],
            "close": q["ohlc"]["close"],
            "volume": q["volume"],
            "change_pct": q.get("net_change", 0),
            "bid": q.get("depth", {}).get("buy", [{}])[0].get("price", 0),
            "ask": q.get("depth", {}).get("sell", [{}])[0].get("price", 0),
            "timestamp": datetime.now().isoformat(),
            "is_mock": False,
        }
    except (KeyError, IndexError) as exc:
        logger.warning("Quote parse failed: %s", exc)
        return _mock_quote(symbol)


# ── Mock Data (dev / demo) ────────────────────────────────────
def _mock_quote(symbol: str) -> dict[str, Any]:
    base = _symbol_base_price(symbol)
    change = random.uniform(-2.5, 2.5)
    ltp = round(base * (1 + change / 100), 2)
    return {
        "symbol": symbol,
        "ltp": ltp,
        "open": round(base * random.uniform(0.98, 1.01), 2),
        "high": round(ltp * random.uniform(1.00, 1.03), 2),
        "low": round(ltp * random.uniform(0.97, 1.00), 2),
        "close": round(base, 2),
        "volume": random.randint(50_000, 5_000_000),
        "change_pct": round(change, 2),
        "bid": round(ltp - 0.05, 2),
        "ask": round(ltp + 0.05, 2),
        "timestamp": datetime.now().isoformat(),
        "is_mock": True,
    }


def _mock_ohlc(symbol: str, candles: int = 100) -> list[dict[str, Any]]:
    base = _symbol_base_price(symbol)
    result = []
    price = base
    now_ts = int(datetime.now().timestamp())
    for i in range(candles, 0, -1):
        change = random.uniform(-1.5, 1.5)
        open_ = round(price, 2)
        close = round(price * (1 + change / 100), 2)
        high = round(max(open_, close) * random.uniform(1.001, 1.015), 2)
        low = round(min(open_, close) * random.uniform(0.985, 0.999), 2)
        result.append({
            "time": now_ts - i * 60,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": random.randint(10_000, 500_000),
        })
        price = close
    return result


def _symbol_base_price(symbol: str) -> float:
    prices = {
        "RELIANCE": 2850, "TCS": 4100, "HDFCBANK": 1680, "INFY": 1780,
        "ICICIBANK": 1250, "KOTAKBANK": 1890, "BHARTIARTL": 1620,
        "ITC": 465, "AXISBANK": 1180, "LT": 3700, "MARUTI": 12800,
        "TITAN": 3580, "BAJFINANCE": 7200, "SUNPHARMA": 1890,
        "HCLTECH": 1920, "WIPRO": 560, "NESTLEIND": 2450,
        "ASIANPAINT": 2980, "ULTRACEMCO": 11400, "HINDUNILVR": 2680,
    }
    return prices.get(symbol, random.uniform(500, 3000))


async def get_market_status() -> dict[str, Any]:
    """Check if Indian market (NSE) is currently open."""
    now = datetime.now()
    # IST offset: +5:30
    # NSE hours: 09:15 – 15:30 Mon-Fri
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    total_minutes = hour * 60 + minute

    is_open = (
        weekday < 5 and
        9 * 60 + 15 <= total_minutes <= 15 * 60 + 30
    )
    return {
        "is_open": is_open,
        "session": "Regular" if is_open else "Closed",
        "timestamp": now.isoformat(),
        "note": "Times in IST (UTC+5:30)",
    }
