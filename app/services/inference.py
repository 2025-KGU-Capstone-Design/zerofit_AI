# app/services/inference.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator

from ..setting.startup import (
    encoder,
    latent_vectors,
    df,
    ohe,
    scaler,
    categorical_cols,
    numeric_cols,
)


# app/services/inference.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator

from ..setting.startup import encoder, latent_vectors, df, ohe, scaler


def recommend_improvements(input_data: dict):
    """
    1) AutoEncoder+코사인 유사도로 후보군 생성
       Top-K는 엘보 포인트(knee)로 결정됩니다.
    """
    # — 전처리 —
    industry = input_data.get("industry", "")
    facilities = input_data.get("ownedFacilities", [])
    # 대표 설비 하나만 사용
    cat_row = [[industry, facilities[0]] if facilities else [industry, ""]]
    user_cat = ohe.transform(cat_row)
    print(f"사용자 카테고리 벡터: {user_cat}")

    invest = input_data.get("investmentBudget", 0.0)
    roi_months = input_data.get("targetRoiPeriod", 0.0)

    if roi_months > 0:
        reduction = invest / roi_months
    else:
        reduction = 0.0

    # 온실가스 감축량은 배출량 차이로 계산
    ghg_reduction = input_data.get("currentEmission", 0.0) - input_data.get(
        "targetEmission", 0.0
    )

    print(f"사용자 입력: {industry}, {facilities}")
    print(
        f"투자비: {invest}, 절감액: {reduction}, ROI(개월): {roi_months}, 온실가스감축량: {ghg_reduction}"
    )
    user_num = scaler.transform([[invest, reduction, roi_months, ghg_reduction]])
    print(f"사용자 수치 벡터: {user_num}")

    # — latent 예측 & 유사도 계산 —
    user_vec = np.hstack([user_cat, user_num]).astype("float32")
    print(f"사용자 전체 벡터: {user_vec}")
    user_latent = encoder.predict(user_vec)
    cos_sim = cosine_similarity(user_latent, latent_vectors)[0]

    # — 엘보 포인트로 K 결정 —
    sims_sorted = np.sort(cos_sim)[::-1]
    ranks = np.arange(1, len(cos_sim) + 1)
    kneedle = KneeLocator(ranks, sims_sorted, curve="convex", direction="decreasing")
    elbow_k = kneedle.knee
    if elbow_k is None:
        # 엘보가 감지되지 않으면 기본값(예: 10) 사용
        elbow_k = 10

    # — 후보군 DataFrame 생성 —
    idxs = np.argsort(cos_sim)[-elbow_k:][::-1]
    cand_df = df.iloc[idxs].copy()
    cand_df["similarity"] = cos_sim[idxs]
    return cand_df


def recommend_by_focus(cand_df, focus: str, k: int = 4):
    """
    cand_df: recommend_improvements 반환 DataFrame
    focus: "balanced" | "roi" | "saving" | "ghg"
    """
    if focus == "balanced":
        sorted_df = cand_df.sort_values("similarity", ascending=False)
    elif focus == "roi":
        sorted_df = cand_df.assign(
            roi=cand_df["절감액"] / cand_df["투자비"]
        ).sort_values("roi", ascending=False)
    elif focus == "saving":
        sorted_df = cand_df.sort_values("절감액", ascending=False)
    elif focus == "ghg":
        sorted_df = cand_df.sort_values("온실가스감축량", ascending=False)
    else:
        raise ValueError(f"Unknown focus: {focus}")
    return sorted_df.head(k).to_dict(orient="records")


def recommend_all(input_data: dict, per_k: int = 4):
    """
    4가지 관점별로 각각 per_k개씩 추천 결과 반환
    """
    cand_df = recommend_improvements(input_data)
    return {
        "balanced": recommend_by_focus(cand_df, "balanced", per_k),
        "roi": recommend_by_focus(cand_df, "roi", per_k),
        "saving": recommend_by_focus(cand_df, "saving", per_k),
        "ghg": recommend_by_focus(cand_df, "ghg", per_k),
    }
