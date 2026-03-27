import json
import logging
from typing import Any
import anthropic

from app.core.config import settings
from app.services.market_data import fetch_quote, fetch_historical_ohlc
from app.services.report_cache import (
    ReportMode,
    acquire_report_lock,
    get_cached_report,
    release_report_lock,
    set_cached_report,
)

logger = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None


def get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


# ── Technical Indicators (lightweight, no extra deps) ─────────
def _calc_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[-(period + 1 - i + 1)] - closes[-(period + 1 - i)]
        if delta >= 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_vwap(candles: list[dict]) -> float:
    if not candles:
        return 0.0
    total_vol = sum(c["volume"] for c in candles)
    if total_vol == 0:
        return 0.0
    return round(
        sum(((c["high"] + c["low"] + c["close"]) / 3) * c["volume"] for c in candles) / total_vol,
        2,
    )


def _calc_bollinger(closes: list[float], period: int = 20) -> dict[str, float]:
    if len(closes) < period:
        mid = closes[-1] if closes else 0
        return {"upper": mid, "mid": mid, "lower": mid}
    window = closes[-period:]
    mean = sum(window) / period
    std = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
    return {
        "upper": round(mean + 2 * std, 2),
        "mid": round(mean, 2),
        "lower": round(mean - 2 * std, 2),
    }


def _build_technical_summary(quote: dict, candles: list[dict]) -> dict[str, Any]:
    closes = [c["close"] for c in candles]
    return {
        "ltp": quote["ltp"],
        "change_pct": quote["change_pct"],
        "volume": quote["volume"],
        "rsi_14": _calc_rsi(closes),
        "vwap": _calc_vwap(candles[-78:]),   # last ~78 min = intraday
        "bollinger": _calc_bollinger(closes),
        "high_52w": round(max(c["high"] for c in candles[-252:]) if candles else quote["high"], 2),
        "low_52w": round(min(c["low"] for c in candles[-252:]) if candles else quote["low"], 2),
    }


# ── Trader Mode Report ─────────────────────────────────────────
async def generate_trader_report(symbol: str, timestamp: str) -> dict[str, Any]:
    """
    Generate an intraday risk assessment report.
    Uses Redis cache + distributed lock to prevent duplicate generation.
    """
    cached = await get_cached_report(ReportMode.TRADER, symbol, timestamp[:16])
    if cached:
        return {**cached, "from_cache": True}

    lock_acquired = await acquire_report_lock(ReportMode.TRADER, symbol, timestamp[:16])
    if not lock_acquired:
        # Another request is generating — poll cache briefly
        import asyncio
        for _ in range(10):
            await asyncio.sleep(1)
            cached = await get_cached_report(ReportMode.TRADER, symbol, timestamp[:16])
            if cached:
                return {**cached, "from_cache": True}
        return {"error": "Report generation in progress, please retry in a moment.", "from_cache": False}

    try:
        quote = await fetch_quote(symbol)
        candles = await fetch_historical_ohlc(symbol, interval="1minute", days=5)
        tech = _build_technical_summary(quote, candles)

        prompt = f"""You are DalalStreet AI, a professional intraday trading risk analyst for Indian markets.

Analyze the following real-time data for {symbol} on NSE and generate a concise, actionable risk assessment report.

**CURRENT MARKET DATA (at {timestamp})**
- LTP: ₹{tech['ltp']}
- Change: {tech['change_pct']}%
- Volume: {tech['volume']:,}
- VWAP: ₹{tech['vwap']}
- RSI (14): {tech['rsi_14']}
- Bollinger Bands: Upper ₹{tech['bollinger']['upper']} | Mid ₹{tech['bollinger']['mid']} | Lower ₹{tech['bollinger']['lower']}
- 52W High: ₹{tech['high_52w']} | 52W Low: ₹{tech['low_52w']}
- Bid/Ask: ₹{quote['bid']} / ₹{quote['ask']}

**TASK**: Generate a structured JSON risk assessment with these exact fields:
{{
  "risk_level": "LOW|MEDIUM|HIGH|EXTREME",
  "risk_score": <0-100 integer>,
  "decision": "BUY|SELL|HOLD|AVOID",
  "confidence_pct": <0-100 integer>,
  "entry_price": <suggested entry or null>,
  "stop_loss": <price or null>,
  "target_price": <price or null>,
  "holding_period": "intraday|short-term|avoid",
  "summary": "<2-3 sentence concise analysis>",
  "key_signals": ["<signal 1>", "<signal 2>", "<signal 3>"],
  "warnings": ["<warning if any>"],
  "technical_bias": "BULLISH|BEARISH|NEUTRAL"
}}

Return ONLY valid JSON. No markdown, no preamble."""

        client = get_anthropic_client()
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ai_data = json.loads(raw)

        report = {
            "symbol": symbol,
            "timestamp": timestamp,
            "mode": "trader",
            "technical": tech,
            "assessment": ai_data,
            "data_source": "mock" if quote.get("is_mock") else "upstox",
            "from_cache": False,
        }

        await set_cached_report(ReportMode.TRADER, symbol, report, timestamp[:16])
        return report

    except json.JSONDecodeError as exc:
        logger.error("AI JSON parse error: %s", exc)
        return {"error": "Failed to parse AI response", "from_cache": False}
    except Exception as exc:
        logger.error("Trader report generation error: %s", exc)
        return {"error": str(exc), "from_cache": False}
    finally:
        await release_report_lock(ReportMode.TRADER, symbol, timestamp[:16])


# ── Investor Mode Report ───────────────────────────────────────
async def generate_investor_report(
    asset_type: str,
    risk_appetite: str,
    symbols: list[str],
) -> dict[str, Any]:
    """
    Generate a long-term investment recommendation report.
    """
    cache_key = f"{asset_type}:{risk_appetite}"
    cached = await get_cached_report(ReportMode.INVESTOR, cache_key)
    if cached:
        return {**cached, "from_cache": True}

    lock_acquired = await acquire_report_lock(ReportMode.INVESTOR, cache_key)
    if not lock_acquired:
        import asyncio
        for _ in range(10):
            await asyncio.sleep(1)
            cached = await get_cached_report(ReportMode.INVESTOR, cache_key)
            if cached:
                return {**cached, "from_cache": True}
        return {"error": "Report generation in progress, please retry in a moment.", "from_cache": False}

    try:
        # Fetch historical data for each symbol
        assets_data = {}
        for sym in symbols[:5]:  # limit to 5 symbols
            quote = await fetch_quote(sym)
            candles = await fetch_historical_ohlc(sym, interval="day", days=365)
            closes = [c["close"] for c in candles]

            # Calculate 1-year return
            ret_1y = round(((closes[-1] - closes[0]) / closes[0]) * 100, 2) if len(closes) > 1 else 0
            # Volatility (std dev of daily returns)
            if len(closes) > 20:
                daily_rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                vol = round((sum(r**2 for r in daily_rets) / len(daily_rets)) ** 0.5 * 100, 2)
            else:
                vol = 0

            assets_data[sym] = {
                "ltp": quote["ltp"],
                "change_pct": quote["change_pct"],
                "return_1y_pct": ret_1y,
                "volatility_pct": vol,
                "rsi": _calc_rsi(closes),
            }

        prompt = f"""You are DalalStreet AI, a financial advisor for Indian retail investors (laymen).

Generate a long-term investment recommendation report.

**INVESTOR PROFILE**
- Asset Type Interest: {asset_type} (Mutual Funds / ETFs / Large Cap / Small Cap)
- Risk Appetite: {risk_appetite} (LOW / MODERATE / HIGH)

**ASSETS ANALYSED** (historical data, past 1 year):
{json.dumps(assets_data, indent=2)}

**TASK**: Generate a structured JSON recommendation report:
{{
  "recommendation_summary": "<2-3 sentences for a layman>",
  "top_picks": [
    {{
      "symbol": "<symbol>",
      "type": "{asset_type}",
      "rationale": "<1 sentence why>",
      "risk_rating": "LOW|MODERATE|HIGH",
      "expected_return_range": "<e.g. 12-18% p.a.>",
      "suggested_allocation_pct": <integer 0-100>,
      "sip_suitable": true/false
    }}
  ],
  "diversification_tip": "<practical tip for Indian retail investor>",
  "risk_warning": "<SEBI-style disclaimer, 1 sentence>",
  "holding_horizon": "<e.g. 3-5 years>",
  "tax_note": "<brief note on LTCG/STCG relevant for India>"
}}

Use simple language. Return ONLY valid JSON."""

        client = get_anthropic_client()
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ai_data = json.loads(raw)

        report = {
            "mode": "investor",
            "asset_type": asset_type,
            "risk_appetite": risk_appetite,
            "assets_analysed": assets_data,
            "recommendation": ai_data,
            "from_cache": False,
        }

        await set_cached_report(ReportMode.INVESTOR, cache_key, report, ttl=3600)  # 1hr for investor
        return report

    except Exception as exc:
        logger.error("Investor report error: %s", exc)
        return {"error": str(exc), "from_cache": False}
    finally:
        await release_report_lock(ReportMode.INVESTOR, cache_key)
