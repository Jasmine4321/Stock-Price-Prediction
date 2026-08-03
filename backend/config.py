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
    "GESHIP.BO" : "The Great Eastern Shipping Company Limited",
    "TATAGOLD.NS" : "Tata Gold ETF",
    "TATASTEEL.BO" : "Tata Steel Limited",
    "TATACAP.NS" : "TATA CAPITAL LIMITED",
    "PTC" : "PTC Earnings: Resources From Divestiture Reinforce Ambition as System of Record",
    "JKLAKSHMI.BO" : "JK Lakshmi Cement Limited",
    "SOUTHBANK.NS" : "The South Indian Bank Limited",
    "BEL.NS" : "Bharat Electronics Limited",
    "JYOTHYLAB.BO" : "Jyothy Labs Limited",
    "SAIL" : "SailPoint, Inc.",
    "INFY" :  "Infosys Limited",
    "PARADEEP.BO" :  "Paradeep Phosphates Limited ",
    "HFCL.NS" : "HFCL Limited",
    "SUZLON.NS" : "Suzlon Energy Limited",
    "IOC.NS" : "Indian Oil Corporation Limited",
    "EXIDEIND.NS" : "Exide Industries Limited ",
    "FEDERALBNK.BO" : "The Federal Bank Limited",
    "NHPC.BO" : "NHPC Limited",
    "UNIMECH.BO" : "Unimech Aerospace and Manufacturing Limited",
    "ORIENTHOT.NS" : "Oriental Hotels Limited"
}

# History requested per ticker -- "max" pulls all available data rather than
# forcing exactly 5 years, since some tickers (e.g. recently listed ones)
# don't have 5 years of history and would otherwise fail outright.
HISTORY_PERIOD = "max"

# Train/val/test split fractions used in ml/dataset.py's chronological_split:
# 80% train, 10% validation, 10% held-out test (chronological, never shuffled).

# LSTM input window: how many past days the model looks at to make one prediction
LOOKBACK_DAYS = 60

# Supported forecast horizons, in trading days (counted directly -- since the
# underlying price data only exists on trading days, we treat "N days" as
# "N trading days ahead" rather than converting calendar days, for simplicity).
# Each horizon gets its own "flat" band -- a 0.5% move over 1 day is meaningful,
# but over 1 year it's noise, so a fixed threshold across horizons would mislead.
HORIZONS = {
    "1d":   {"days": 1,   "flat_threshold_pct": 0.5},
    "5d":   {"days": 5,   "flat_threshold_pct": 1.0},
    "7d":   {"days": 7,   "flat_threshold_pct": 1.3},
    "15d":  {"days": 15,  "flat_threshold_pct": 2.0},
    "30d":  {"days": 30,  "flat_threshold_pct": 3.0},
    "180d": {"days": 180, "flat_threshold_pct": 7.0},
    "1y":   {"days": 252, "flat_threshold_pct": 10.0},
}
DEFAULT_HORIZON = "7d"

# Where trained models are saved: ml/models/{ticker}/{horizon}/
MODEL_DIR = "ml/models"

# Database
DATABASE_URL = "sqlite:///./stock_predictor.db"  # swap for postgres in production

# LLM used for the chat layer (intent parsing + reply wording only, never forecasting)
LLM_MODEL = "llama-3.3-70b-versatile"