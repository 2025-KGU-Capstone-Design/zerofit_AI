# app/main.py
from fastapi import FastAPI
from app.setting.startup import load_resources
from app.endpoints import recommend, comment
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

origins = ["*"]

app = FastAPI(title="ESG 개선활동 추천 서비스", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="ESG 개선활동 추천 서비스", version="1.0")

app.include_router(recommend.router, tags=["Recommendation"])
app.include_router(comment.router, tags=["Comment Generation"])


@app.on_event("startup")
async def startup_event():
    load_resources()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
