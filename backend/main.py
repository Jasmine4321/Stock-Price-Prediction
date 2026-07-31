from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import stocks, chat

app = FastAPI(title="Stock Forecast API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Stock Forecast API is running"}
