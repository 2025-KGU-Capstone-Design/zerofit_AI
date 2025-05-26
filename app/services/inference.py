# app/services/inference.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator

from ..setting.startup import (
    encoder,
    latent_vectors,
    df,
    ohe,
    scaler,
)


def recommend_improvements(input_data: dict, per_k: int = 10):
    """
    1) AutoEncoder + cosine similarity 로 후보군 생성
       Top-K 는 엘보 포인트(knee) 로 결정하고, 감지되지 않으면 per_k 사용
    """
    # — 입력값 파싱 및 수치 계산 —
    industry = input_data.get("industry", "")
    facilities = input_data.get("ownedFacilities", [])
    invest = float(input_data.get("investmentBudget", 0.0))
    roi_months = float(input_data.get("targetRoiPeriod", 0.0))
    current_em = float(input_data.get("currentEmission", 0.0))
    target_em = float(input_data.get("targetEmission", 0.0))

    # 절감액 계산 (투자비 ÷ ROI 기간), ROI 기간이 0 이면 0.0
    reduction = invest / roi_months if roi_months > 0 else 0.0
    # 온실가스 감축량 계산 (현재 배출량 – 목표 배출량)
    ghg_reduction = current_em - target_em

    all_cands = []

    for facility in facilities:
        # — 1) 카테고리 벡터 —
        user_cat = ohe.transform([[industry, facility]])  # shape (1, cat_dim)

        # — 2) 수치 벡터 —
        user_num = scaler.transform(
            [[invest, reduction, roi_months, ghg_reduction]]
        )  # shape (1, num_dim)

        # — 3) 결합 후 latent 벡터 & 유사도 계산 —
        user_vec = np.hstack([user_cat, user_num]).astype(
            "float32"
        )  # shape (1, total_dim)
        user_latent = encoder.predict(user_vec)  # shape (1, latent_dim)
        cos_sim = cosine_similarity(user_latent, latent_vectors)[
            0
        ]  # shape (n_candidates,)

        # — 4) KneeLocator 로 Top-K 결정 —
        sims_sorted = np.sort(cos_sim)[::-1]
        ranks = np.arange(1, len(cos_sim) + 1)
        kneedle = KneeLocator(
            ranks, sims_sorted, curve="convex", direction="decreasing"
        )
        k = kneedle.knee or per_k

        # — 5) Top-K 후보 추출 —
        idxs = np.argsort(cos_sim)[-k:][::-1]
        cand_d = df.iloc[idxs].copy()
        cand_d["similarity"] = cos_sim[idxs]
        cand_d["facility"] = facility
        all_cands.append(cand_d)

    # — 6) 모든 설비 결과 합치기 및 중복 제거 —
    final_df = (
        pd.concat(all_cands, ignore_index=True)
        .sort_values("similarity", ascending=False)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if industry:
        final_df = final_df[final_df["업종"] == industry]

    return final_df


def recommend_by_focus(cand_df, focus: str, k: int):
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


def recommend_all(input_data: dict, per_k: int):
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
