"""
LLM call #1 of 2 in the chat pipeline.

Its only job is: read the user's free-text message, figure out which
supported ticker (if any) they're asking about. It is explicitly forbidden
from answering the actual question -- the system prompt below constrains it
to JSON extraction only, and we never even show it any price data.
"""

import json
import re

from groq import Groq

from config import SUPPORTED_STOCKS, LLM_MODEL

client = Groq()

SYSTEM_PROMPT = f"""You extract a stock ticker from a user's message. You do not
answer questions, you do not predict prices, you only extract.

Valid tickers (company name -> ticker):
{json.dumps(SUPPORTED_STOCKS, indent=2)}

Rules:
- Match on ticker OR company name (e.g. "apple" -> AAPL, "tesla" -> TSLA).
- If the message mentions a company/ticker NOT in the list, set "ticker" to null
  and "unsupported_mention" to the name they used.
- If no company is mentioned at all, set "ticker" to null and "unsupported_mention" to null.
- Respond with ONLY raw JSON, nothing else, in this exact shape:
  {{"ticker": "AAPL" | null, "unsupported_mention": "string" | null}}
"""


class Intent:
    def __init__(self, ticker: str | None, unsupported_mention: str | None):
        self.ticker = ticker
        self.unsupported_mention = unsupported_mention


def extract_intent(message: str) -> Intent:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=100,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Intent(ticker=None, unsupported_mention=None)

    ticker = data.get("ticker")
    if ticker not in SUPPORTED_STOCKS:
        ticker = None

    return Intent(ticker=ticker, unsupported_mention=data.get("unsupported_mention"))
