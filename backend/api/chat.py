from fastapi import APIRouter
from pydantic import BaseModel

from chat.intent_parser import extract_intent
from chat.responder import format_reply, format_unsupported_reply
from ml.forecast_engine import run_forecast

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    data: dict | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Step 1: LLM extracts intent only -- no numbers shown to it, no prediction made here
    intent = extract_intent(req.message)

    if intent.ticker is None:
        reply = format_unsupported_reply(intent.unsupported_mention)
        return ChatResponse(reply=reply, data=None)

    # Step 2: real forecasting pipeline, zero LLM involvement
    forecast = run_forecast(intent.ticker)

    # Step 3: LLM reword-only pass over the already-computed numbers
    reply = format_reply(forecast)

    return ChatResponse(reply=reply, data=forecast)
