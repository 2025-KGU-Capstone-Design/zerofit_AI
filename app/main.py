# app/main.py
from fastapi import FastAPI
from app.setting.startup import load_resources
from app.endpoints import recommend
import logging
import os

# CORS 설정
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="ESG 개선활동 추천 서비스", version="1.0")

app.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])


@app.on_event("startup")
async def startup_event():
    load_resources()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
