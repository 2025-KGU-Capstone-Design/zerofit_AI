from fastapi import FastAPI

import logging

# 원하는 타이틀 및 버전 정보를 추가
app = FastAPI(
    title="AI Model API",
    description="ZeroFit: 기업 맞춤형 온실가스 감축 솔루션 서비스",
    version="0.1.0",
)


# 서버 상태 점검용
@app.get("/health", tags=["Health"])
def health():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
