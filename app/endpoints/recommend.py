from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict
from ..services.inference import recommend_all

from ..services.inference import recommend_improvements

router = APIRouter()


class RecommendRequest(BaseModel):
    industry: str = Field(
        ...,
        title="산업군",
        description="개선활동을 원하는 산업군",
        example="발전 / 에너지",
    )
    ownedFacilities: List[str] = Field(
        ...,
        title="보유 설비 목록",
        description="사용자가 보유한 설비 리스트",
        example=["동력설비"],
    )
    investmentBudget: float = Field(
        ...,
        title="투자가능금액",
        description="투자가능한 예산(백 만원 단위)",
        example=30.0,
    )
    currentEmission: float = Field(
        ..., title="현재 배출량", description="현재 연간 배출량(tCO2eq)", example=100.0
    )
    targetEmission: float = Field(
        ..., title="목표 배출량", description="목표 연간 배출량(tCO2eq)", example=80.0
    )
    targetRoiPeriod: float = Field(
        ..., title="목표 ROI 기간", description="목표 회수 기간(년)", example=2.0
    )

    class Config:
        allow_population_by_field_name = True


@router.post(
    "/recommend",
    response_model=Dict[str, List[Dict]],
    summary="ESG 개선활동 추천",
    description="균형, ROI, 절감액, 온실가스 관점별로 Top-N 결과를 반환합니다.",
)
async def recommend(request: RecommendRequest):
    return recommend_all(request.dict(), per_k=4)
