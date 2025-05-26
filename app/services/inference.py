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
       2) 클러스터 단위로 집계(aggregation)하여 중복 제거 및 평균화
    """
    # — 입력값 파싱 및 수치 계산 —
    industry = input_data.get("industry", "")
    facilities = input_data.get("ownedFacilities", [])
    invest = float(input_data.get("investmentBudget", 0.0))
    roi_months = float(input_data.get("targetRoiPeriod", 0.0))
    current_em = float(input_data.get("currentEmission", 0.0))
    target_em = float(input_data.get("targetEmission", 0.0))

    # 절감액 계산 및 GHG 감축량 계산
    reduction = invest / roi_months if roi_months > 0 else 0.0
    ghg_reduction = current_em - target_em

    all_cands = []

    for facility in facilities:
        user_cat = ohe.transform([[industry, facility]])
        user_num = scaler.transform([[invest, reduction, roi_months, ghg_reduction]])
        user_vec = np.hstack([user_cat, user_num]).astype("float32")
        user_latent = encoder.predict(user_vec)
        cos_sim = cosine_similarity(user_latent, latent_vectors)[0]

        sims_sorted = np.sort(cos_sim)[::-1]
        ranks = np.arange(1, len(cos_sim) + 1)
        kneedle = KneeLocator(
            ranks, sims_sorted, curve="convex", direction="decreasing"
        )
        k = kneedle.knee or per_k
        print(f"[recommend_improvements] facility={facility}, elbow_k={k}")

        idxs = np.argsort(cos_sim)[-k:][::-1]
        cand = df.iloc[idxs].copy()
        cand["similarity"] = cos_sim[idxs]
        cand["facility"] = facility
        all_cands.append(cand)

    combined = pd.concat(all_cands, ignore_index=True)

    # — 클러스터 단위 집계 —
    # 노이즈(-1) 분리
    noise = combined[combined["cluster"] == -1]
    valid = combined[combined["cluster"] != -1]

    if not valid.empty:
        agg_funcs = {
            "similarity": "mean",
            "투자비": "mean",
            "절감액": "mean",
            "투자비회수기간": "mean",
            "온실가스감축량": "mean",
        }
        first_funcs = {
            "개선활동명_요약": "first",
            "업종": "first",
            "대상설비": "first",
            "개선구분": "first",
            "facility": "first",
            "cluster": "first",
        }
        aggregated = valid.groupby("cluster", as_index=False).agg(
            {**agg_funcs, **first_funcs}
        )
        combined = pd.concat([aggregated, noise], ignore_index=True)
    # 정렬 및 업종 필터
    combined = combined.sort_values("similarity", ascending=False)
    if industry:
        combined = combined[combined["업종"] == industry]

    return combined.reset_index(drop=True)


def recommend_by_focus(cand_df, focus: str, k: int):
    """
    cand_df: recommend_improvements 반환 DataFrame
    focus: "balanced" | "roi" | "saving" | "ghg"
    """
    if focus == "balanced":
        df_sorted = cand_df.sort_values("similarity", ascending=False)
    elif focus == "roi":
        df_sorted = cand_df.assign(roi=lambda d: d["절감액"] / d["투자비"]).sort_values(
            "roi", ascending=False
        )
    elif focus == "saving":
        df_sorted = cand_df.sort_values("절감액", ascending=False)
    elif focus == "ghg":
        df_sorted = cand_df.sort_values("온실가스감축량", ascending=False)
    else:
        raise ValueError(f"Unknown focus: {focus}")

    cols = [
        "cluster",
        "개선활동명_요약",
        "업종",
        "대상설비",
        "개선구분",
        "similarity",
        "투자비",
        "절감액",
        "투자비회수기간",
        "온실가스감축량",
    ]
    result = df_sorted[cols].head(k).copy()
    result[["similarity", "투자비", "절감액", "투자비회수기간", "온실가스감축량"]] = (
        result[
            ["similarity", "투자비", "절감액", "투자비회수기간", "온실가스감축량"]
        ].round(1)
    )
    return result.to_dict(orient="records")


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
