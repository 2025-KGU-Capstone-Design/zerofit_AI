from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()


# 사용자 입력 스키마 (Pydantic 모델)
class UserInput(BaseModel):
    업종: str
    대상설비: List[str]
    투자비: float
    투자비회수기간: float
    온실가스감축량: float


@router.post("/")
def get_recommendations(input_data: UserInput, request: Request):
    # 필요한 보조 데이터 및 모델을 app.state에서 가져옵니다.
    model = request.app.state.model
    candidate_df = request.app.state.candidate_df
    user_cat_vocab = request.app.state.user_cat_vocab
    user_num_cols = request.app.state.user_num_cols
    candidate_text_cols = request.app.state.candidate_text_cols
    target_scaler = request.app.state.target_scaler
    device = request.app.state.device

    # 추천 함수 호출 (inference.py에 정의된 recommend_improvements 함수)
    from app.services.inference import recommend_improvements

    recommendations = recommend_improvements(
        input_data.dict(),
        candidate_df,
        model,
        user_cat_vocab,
        user_num_cols,
        candidate_text_cols,
        target_scaler,
        text_max_len=10,
        device=device,
    )
    return recommendations
