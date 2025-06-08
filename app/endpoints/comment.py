# app/endpoints/comment.py

from typing import List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import asyncio

from app.services.gpt_client import generate_comparison_comment_async


# ─────────────────────────────────────────────────────────────────────────────
# 1) Pydantic 모델 정의
# ─────────────────────────────────────────────────────────────────────────────


class LLMParam(BaseModel):
    id: int
    type: str
    rank: int
    industry: str
    improvementType: str
    facility: str
    activity: str
    emissionReduction: float
    costSaving: float
    roiPeriod: float
    investmentCost: float
    bookmark: bool


class CommentRequest(BaseModel):
    llmParams: List[LLMParam]


class CommentResponse(BaseModel):
    type: str
    top1: str
    comparison: str


# ─────────────────────────────────────────────────────────────────────────────
# 2) APIRouter 생성 및 엔드포인트 구현
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="", tags=["Comment Generation"])

# “기본값”으로 사용할 예시 JSON을 미리 정의해 둡니다.
default_body = {
    "llmParams": [
        {
            "id": 1,
            "type": "total_optimization",
            "rank": 1,
            "industry": "제조업",
            "improvementType": "설비 개선",
            "facility": "보일러",
            "activity": "고효율 보일러 교체",
            "emissionReduction": 120.5,
            "costSaving": 3000.0,
            "roiPeriod": 2.5,
            "investmentCost": 750.0,
            "bookmark": False,
        },
        {
            "id": 2,
            "type": "total_optimization",
            "rank": 2,
            "industry": "제조업",
            "improvementType": "설비 개선",
            "facility": "보일러",
            "activity": "고효율 보일러 교체",
            "emissionReduction": 120.5,
            "costSaving": 3000.0,
            "roiPeriod": 2.5,
            "investmentCost": 750.0,
            "bookmark": False,
        },
        {
            "id": 3,
            "type": "total_optimization",
            "rank": 3,
            "industry": "제조업",
            "improvementType": "설비 개선",
            "facility": "보일러",
            "activity": "고효율 보일러 교체",
            "emissionReduction": 120.5,
            "costSaving": 3000.0,
            "roiPeriod": 2.5,
            "investmentCost": 750.0,
            "bookmark": False,
        },
        {
            "id": 4,
            "type": "total_optimization",
            "rank": 4,
            "industry": "제조업",
            "improvementType": "설비 개선",
            "facility": "보일러",
            "activity": "고효율 보일러 교체",
            "emissionReduction": 120.5,
            "costSaving": 3000.0,
            "roiPeriod": 2.5,
            "investmentCost": 750.0,
            "bookmark": False,
        },
    ]
}


@router.post(
    "/comment",
    response_model=CommentResponse,
    summary="LLM 설명 생성",
    description="최대 4개의 개선 활동 데이터를 받아 비동기로 LLM 호출을 수행한 뒤, "
    "`top1`/`comparison`을 반환합니다.",
)
async def comment_endpoint(
    request: CommentRequest = Body(
        default=default_body,
        example=default_body,  # Swagger UI에 ‘예시 값(example)’으로도 노출됩니다.
    )
):
    """
    클라이언트로부터 llmParams(최대 4개 객체 리스트)를 입력받아,
    비동기 LLM 호출을 통해 'top1' 및 'comparison' 설명을 생성하여 반환합니다.

    - Path: POST /comment
    - 기본 request body 예시(default)와 Swagger UI 예시(example)를 위에 지정해 두었습니다.
    """

    params = request.llmParams

    # 1) llmParams가 비어 있으면 400 에러 반환
    if not params:
        raise HTTPException(status_code=400, detail="llmParams 배열이 비어 있습니다.")

    # 2) rank 순서대로 정렬 (rank=1부터 차례대로)
    sorted_params = sorted(params, key=lambda x: x.rank)

    # 3) 최대 4개까지만 사용
    top_items = sorted_params[:4]

    # 4) focus는 첫 번째 요소의 type
    focus_type = top_items[0].type

    # 5) LLM 호출 유틸에 넘겨줄 때는 dict 형태로 변환
    top_items_dicts = [item.dict() for item in top_items]

    # 6) 비동기 LLM 호출
    result_dict = await generate_comparison_comment_async(focus_type, top_items_dicts)

    # 7) result_dict에 "top1"과 "comparison" 키가 없는 경우 오류 처리
    if "top1" not in result_dict or "comparison" not in result_dict:
        raise HTTPException(
            status_code=500, detail="LLM 응답 형식이 올바르지 않습니다."
        )

    # 8) 최종 응답 생성
    return CommentResponse(
        type=focus_type,
        top1=result_dict["top1"],
        comparison=result_dict["comparison"],
    )
