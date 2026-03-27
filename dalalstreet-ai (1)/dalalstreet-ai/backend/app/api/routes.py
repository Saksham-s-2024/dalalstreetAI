"""
DalalStreet AI — API Routes (v1)
Rate-limited, input-sanitized endpoints.
"""
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.security import sanitize_symbol, sanitize_text, sanitize_ticker_list
from app.services.ai_reports import generate_investor_report, generate_trader_report
from app.services.market_data import (
    NIFTY50_SYMBOLS,
    fetch_historical_ohlc,
    fetch_quote,
    get_market_status,
)

router = APIRouter(prefix="/api/v1")
limiter = Limiter(key_func=get_remote_address)

# ── Request / Response Models ──────────────────────────────────

class TraderReportRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    timestamp: str | None = None

    @validator("symbol")
    def clean_symbol(cls, v):
        try:
            return sanitize_symbol(v)
        except ValueError as exc:
            raise ValueError(str(exc))

    @validator("timestamp", pre=True, always=True)
    def default_timestamp(cls, v):
        return v or datetime.now().isoformat()


class InvestorReportRequest(BaseModel):
    asset_type: str = Field(..., min_length=2, max_length=50)
    risk_appetite: str = Field(..., pattern="^(LOW|MODERATE|HIGH)$")
    symbols: list[str] = Field(default_factory=list, max_items=10)

    @validator("asset_type")
    def clean_asset_type(cls, v):
        return sanitize_text(v, max_len=50)

    @validator("symbols", each_item=True)
    def clean_symbols(cls, v):
        return sanitize_symbol(v)


class QuoteRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)

    @validator("symbol")
    def clean(cls, v):
        return sanitize_symbol(v)


# ── Market Status ─────────────────────────────────────────────

@router.get("/market/status")
@limiter.limit("30/minute")
async def market_status(request: Request):
    return await get_market_status()


@router.get("/market/symbols")
@limiter.limit("20/minute")
async def list_symbols(request: Request):
    return {"symbols": NIFTY50_SYMBOLS}


# ── Live Quote ────────────────────────────────────────────────

@router.get("/market/quote/{symbol}")
@limiter.limit("60/minute")
async def get_quote(request: Request, symbol: str):
    try:
        clean = sanitize_symbol(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid symbol format")
    return await fetch_quote(clean)


# ── Historical OHLC ───────────────────────────────────────────

@router.get("/market/ohlc/{symbol}")
@limiter.limit("20/minute")
async def get_ohlc(request: Request, symbol: str, days: int = 5, interval: str = "1minute"):
    try:
        clean = sanitize_symbol(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid symbol format")

    days = min(max(days, 1), 365)
    if interval not in ("1minute", "5minute", "15minute", "30minute", "60minute", "day"):
        interval = "1minute"

    candles = await fetch_historical_ohlc(clean, interval=interval, days=days)
    return {"symbol": clean, "interval": interval, "candles": candles}


# ── Trader Mode — Risk Report ─────────────────────────────────

@router.post("/trader/report")
@limiter.limit("10/minute")
async def trader_report(request: Request, body: TraderReportRequest):
    """
    Generate an intraday risk assessment report for a given symbol.
    Cached in Redis — duplicate requests within 5 min return cached result.
    """
    report = await generate_trader_report(body.symbol, body.timestamp)
    if "error" in report:
        raise HTTPException(status_code=503, detail=report["error"])
    return report


# ── Investor Mode — Recommendation Report ────────────────────

@router.post("/investor/report")
@limiter.limit("5/minute")
async def investor_report(request: Request, body: InvestorReportRequest):
    """
    Generate a long-term investment recommendation report.
    Cached for 1 hour — duplicate requests return cached result.
    """
    # If no symbols provided, pick defaults based on asset_type
    symbols = body.symbols or _default_symbols(body.asset_type)

    report = await generate_investor_report(
        asset_type=body.asset_type,
        risk_appetite=body.risk_appetite,
        symbols=symbols,
    )
    if "error" in report:
        raise HTTPException(status_code=503, detail=report["error"])
    return report


def _default_symbols(asset_type: str) -> list[str]:
    defaults = {
        "large_cap": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        "small_cap": ["TITAN", "BAJFINANCE", "AXISBANK", "ITC", "LT"],
        "etf": ["NIFTYBEES", "GOLDBEES", "BANKBEES", "JUNIORBEES", "ITBEES"],
        "mutual_fund": ["RELIANCE", "HDFCBANK", "TCS", "INFY", "BHARTIARTL"],
    }
    key = asset_type.lower().replace(" ", "_").replace("-", "_")
    return defaults.get(key, NIFTY50_SYMBOLS[:5])


# ── Health Check ─────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "service": "DalalStreet AI"}
