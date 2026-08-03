"""
LLM call #2 of 2 in the chat pipeline.

Takes the structured dict already produced by ml/forecast_engine.py and turns
it into a short, natural-sounding, point-based reply. It is given the numbers
as fixed context and instructed never to introduce numbers that aren't in
that payload -- its job is wording, not math.
"""

import json

from groq import Groq

from config import LLM_MODEL

client = Groq()

SYSTEM_PROMPT = """You explain a stock forecast to a beginner investor.

FORMAT: Reply in short bullet points, not paragraphs. Use a "-" at the start
of each line. Keep it natural and conversational in tone, not robotic --
each bullet can be a short phrase or a full sentence, whichever reads more
naturally. Aim for 4-6 bullets total. No headers, no bold text, just plain
bullet lines.

Suggested structure (adapt naturally, don't force every bullet to exist if it
doesn't fit):
- current price and predicted price/direction
- the up/down/flat probability split
- the predicted high/low range
- the model's backtested accuracy vs. its baseline, in plain terms
- a closing one-line reminder that this is a backtested estimate, not
  financial advice

CRITICAL: Every number in your reply must come directly from the JSON you are
given. Do not calculate, round differently, or invent any number that isn't
already in the payload.
"""


def format_reply(forecast: dict) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(forecast)},
        ],
    )
    return response.choices[0].message.content.strip()


def format_unsupported_reply(mention: str | None) -> str:
    from config import SUPPORTED_STOCKS

    names = ", ".join(f"{t} ({n})" for t, n in SUPPORTED_STOCKS.items())
    if mention:
        return (
            f"I can't forecast \"{mention}\" -- this project only covers 15 stocks: {names}. "
            "Try asking about one of those."
        )
    return (
        f"Ask me about one of the 15 stocks I can forecast: {names}. "
        "For example: \"What will TSLA be next week?\""
    )