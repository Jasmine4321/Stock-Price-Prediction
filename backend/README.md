# Stock Forecast Backend (college project)

Forecasts 1-week price movement for 15 fixed stocks using a trained LSTM.
The chatbot never predicts numbers itself — it only extracts the ticker the
user is asking about, calls the real forecasting pipeline, then rewords the
already-computed result. See `chat/intent_parser.py`, `ml/forecast_engine.py`,
and `chat/responder.py` for that boundary.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Anthropic API key (used only for the two small chat-wording calls,
never for forecasting):
```bash
export ANTHROPIC_API_KEY=your_key_here
```

## 1. Train the models (run once, or whenever you want to refresh)

```bash
python -m ml.train                 # trains all 15 tickers, ~a few minutes on CPU
python -m ml.train --ticker TSLA   # or just one, while iterating
```

This pulls 5 years of daily data per ticker via yfinance, trains an LSTM,
and saves `model.pt`, `scaler.json`, `metrics.json` into `ml/models/{ticker}/`.
`metrics.json` holds the backtested accuracy — this is what the API reports,
not a training-time number.

## 2. Run the API

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## Key endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/stocks` | list of the 15 supported stocks (search bar) |
| `GET /api/stocks/{ticker}/quote` | live-ish price (search page) |
| `GET /api/forecast/{ticker}` | run the real forecast directly (Predict button) |
| `POST /api/compare` `{"tickers": [...]}` | forecasts for multiple tickers (compare page) |
| `POST /api/chat` `{"message": "..."}` | chatbot: intent -> forecast -> worded reply |

## Notes on the "no internet in this sandbox" caveat

This project was scaffolded in an environment without internet access, so
`yfinance` calls and model training couldn't be executed here. What *was*
tested in-sandbox with synthetic data (see `data/fetcher.py`'s
`_synthetic_history`) and confirmed working end-to-end:
- feature engineering (`ml/features.py`)
- sequence windowing + chronological train/val/test split (`ml/dataset.py`)
- all files pass a Python syntax check

Run `python -m ml.train` on your own machine (with internet + `pip install -r
requirements.txt`) to actually train and validate the LSTM.
