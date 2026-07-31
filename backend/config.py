"""
Central config for the project.
Keeping this project scoped to 15 stocks keeps data collection, training,
and storage simple and fast enough to run/demo on a laptop.
"""

SUPPORTED_STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms",
    "NFLX": "Netflix Inc.",
    "JPM": "JPMorgan Chase",
    "V": "Visa Inc.",
    "DIS": "Walt Disney Co.",
    "INTC": "Intel Corp.",
    "AMD": "Advanced Micro Devices",
    "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.",
}

# How many past trading days of history to pull per stock
HISTORY_PERIOD = "5y"

# LSTM input window: how many past days the model looks at to make one prediction
LOOKBACK_DAYS = 60

# Forecast horizon in trading days (1 week ~ 5 trading days)
HORIZON_DAYS = 5

# Direction classification band: moves smaller than this % are called "flat"
FLAT_THRESHOLD_PCT = 1.0

# Where trained models are saved, one subfolder per ticker
MODEL_DIR = "ml/models"

# Database
DATABASE_URL = "sqlite:///./stock_predictor.db"  # swap for postgres in production

# LLM used for the chat layer (intent parsing + reply wording only, never forecasting)
LLM_MODEL = "llama-3.3-70b-versatile"  # free on Groq, good enough for extraction/wording
