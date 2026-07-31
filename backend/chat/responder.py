"""
LLM call #2 of 2 in the chat pipeline.

Takes the structured dict already produced by ml/forecast_engine.py and turns
it into a friendly sentence or two. It is given the numbers as fixed context
and instructed never to introduce numbers that aren't in that payload -- its
job is wording, not math.
"""

import json

from groq import Groq

from config import LLM_MODEL

client = Groq()

SYSTEM_PROMPT = """You explain a stock forecast to a beginner investor in
plain, friendly language, 3-4 sentences.

CRITICAL: Every number in your reply must come directly from the JSON you are
given. Do not calculate, round differently, or invent any number that isn't
already in the payload. If you want to mention a percentage, use the
probabilities and accuracy figures exactly as given.

Always include a brief reminder that this is a backtested statistical
estimate, not financial advice, and that beginners should not treat it as a
guarantee.
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
