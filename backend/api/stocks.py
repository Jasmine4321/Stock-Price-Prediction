from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import SUPPORTED_STOCKS
from data.fetcher import get_latest_price
from ml.forecast_engine import run_forecast

router = APIRouter()


@router.get("/stocks")
async def list_stocks():
    """Powers the search bar -- just the fixed 15-stock list, no external call needed."""
    return [{"ticker": t, "name": n} for t, n in SUPPORTED_STOCKS.items()]


@router.get("/stocks/{ticker}/quote")
async def quote(ticker: str):
    ticker = ticker.upper()
    if ticker not in SUPPORTED_STOCKS:
        raise HTTPException(status_code=404, detail=f"{ticker} is not supported")
    return get_latest_price(ticker)


@router.get("/forecast/{ticker}")
async def forecast(ticker: str):
    """Used directly by the Forecast page (Predict button) -- same engine the chatbot calls."""
    ticker = ticker.upper()
    if ticker not in SUPPORTED_STOCKS:
        raise HTTPException(status_code=404, detail=f"{ticker} is not supported")
    return run_forecast(ticker)


class CompareRequest(BaseModel):
    tickers: list[str]


@router.post("/compare")
async def compare(req: CompareRequest):
    results = []
    for t in req.tickers:
        t = t.upper()
        if t not in SUPPORTED_STOCKS:
            continue
        results.append(run_forecast(t))
    return results

