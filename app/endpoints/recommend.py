from fastapi import APIRouter
from pydantic import BaseModel

from ..services.inference import recommend_improvements

router = APIRouter()


class RecommendRequest(BaseModel):
    업종: str
    대상설비: str
    투자비: float
    절감액: float
    투자비회수기간: float
    온실가스감축량: float


@router.post("")
async def recommend(request: RecommendRequest):
    """추천 API 엔드포인트"""
    results = recommend_improvements(request.dict())
    return {"recommendations": results}
