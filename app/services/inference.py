import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator
from .gpt_client import generate_comparison_comment

from ..setting.startup import (
    encoder,
    latent_vectors,
    df,
    ohe,
    scaler,
    categorical_cols,
    numeric_cols,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recommend_improvements(input_data: dict, per_k: int = 10):
    """
    1) AutoEncoder + cosine similarity 로 후보군 생성
       Top-K 는 엘보 포인트(knee) 로 결정하고, 감지되지 않으면 per_k 사용
       2) 클러스터 단위로 집계(aggregation)하여 중복 제거 및 평균화
    """
    logger.info(
        f"recommend_improvements 호출 - input_data: {input_data}, per_k: {per_k}"
    )

    # — 입력값 파싱 및 수치 계산 —
    industry = input_data.get("industry", "")
    facilities = input_data.get("targetFacilities", [])
    invest = float(input_data.get("availableInvestment") or 0.0)
    roi_months = float(input_data.get("targetRoiPeriod") or 0.0)
    current_em = float(input_data.get("currentEmission", 0.0))
    target_em = float(input_data.get("targetEmission", 0.0))

    # 절감액 계산 및 GHG 감축량 계산
    reduction = invest / roi_months if roi_months > 0 else 0.0
    ghg_reduction = current_em - target_em
    logger.debug(f"계산된 reduction: {reduction}, ghg_reduction: {ghg_reduction}")

    all_cands = []

    for facility in facilities:
        logger.info(f"추천 후보 생성 - 처리 중인 설비: {facility}")
        input_cat = pd.DataFrame([[industry, facility]], columns=categorical_cols)
        user_cat = ohe.transform(input_cat)
        logger.debug(f"카테고리 원-핫 인코딩 결과 shape: {user_cat.shape}")

        input_num = pd.DataFrame(
            [[invest, reduction, roi_months, ghg_reduction]], columns=numeric_cols
        )
        user_num = scaler.transform(input_num)
        logger.debug(f"스케일링된 수치형 데이터 shape: {user_num.shape}")

        user_vec = np.hstack([user_cat, user_num]).astype("float32")
        logger.debug(f"결합된 사용자 벡터 shape: {user_vec.shape}")

        user_latent = encoder.predict(user_vec)
        logger.debug(f"잠재 공간 벡터 user_latent shape: {user_latent.shape}")

        cos_sim = cosine_similarity(user_latent, latent_vectors)[0]
        logger.debug(f"유사도 샘플 (상위 5): {np.sort(cos_sim)[-5:][::-1]}")

        sims_sorted = np.sort(cos_sim)[::-1]
        ranks = np.arange(1, len(cos_sim) + 1)
        kneedle = KneeLocator(
            ranks, sims_sorted, curve="convex", direction="decreasing"
        )
        k = kneedle.knee or per_k
        logger.info(f"[recommend_improvements] facility={facility}, elbow_k={k}")

        idxs = np.argsort(cos_sim)[-k:][::-1]
        logger.info(f"facility별 후보 개수: {len(idxs)}")
        cand = df.iloc[idxs].copy()
        cand["similarity"] = cos_sim[idxs]
        cand["facility"] = facility
        all_cands.append(cand)

    combined = pd.concat(all_cands, ignore_index=True)
    logger.info(f"모든 후보 결합 후 shape: {combined.shape}")

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
        logger.info(f"클러스터별 집계 결과 shape: {aggregated.shape}")
        combined = pd.concat([aggregated, noise], ignore_index=True)

    # 정렬 및 업종 필터
    combined = combined.sort_values("similarity", ascending=False)
    if industry:
        combined = combined[combined["업종"] == industry]
    logger.info(f"업종 필터 적용 후 후보 shape: {combined.shape}")

    return combined.reset_index(drop=True)


def recommend_by_focus(cand_df: pd.DataFrame, focus: str, k: int):
    """
    cand_df: recommend_improvements 반환 DataFrame
    focus: "similarity" (정렬: 유사도) | "balanced" (정렬: 복합지표) | "roi" | "saving" | "ghg"
    """

    if focus == "similarity":
        sorted_df = cand_df.sort_values("similarity", ascending=False)
    elif focus == "balanced":
        dfc = cand_df.copy()
        feats = ["투자비", "절감액", "투자비회수기간", "온실가스감축량"]

        # 1) 각 지표별 평균을 추출
        means = {f: dfc[f].mean() for f in feats}
        logger.debug(f"지표별 평균값: {means}")

        # 2) 평균값을 min–max 정규화 (투자비, 투자비회수기간은 역정규화)
        norm_means = {}
        for f, m in means.items():
            col = dfc[f]
            mn, mx = col.min(), col.max()
            if mx > mn:
                norm = (m - mn) / (mx - mn)
            else:
                norm = 0
            if f in ["투자비", "투자비회수기간"]:
                norm = 1 - norm
            norm_means[f] = norm
        total = sum(norm_means.values())
        weights = {
            f: (norm_means[f] / total if total > 0 else 1 / len(feats)) for f in feats
        }
        logger.debug(f"가중치: {weights}")

        # 4) 다시 전체 행을 정규화하고 balanced_score 계산
        for f in feats:
            mn, mx = dfc[f].min(), dfc[f].max()
            if mx > mn:
                norm_col = (dfc[f] - mn) / (mx - mn)
            else:
                norm_col = 0
            if f in ["투자비", "투자비회수기간"]:
                dfc[f + "_norm"] = 1 - norm_col
            else:
                dfc[f + "_norm"] = norm_col

        dfc["balanced_score"] = sum(dfc[f + "_norm"] * w for f, w in weights.items())
        sorted_df = dfc.sort_values("balanced_score", ascending=False)
    elif focus == "roi":
        sorted_df = cand_df.assign(roi=lambda d: d["절감액"] / d["투자비"]).sort_values(
            "roi", ascending=False
        )
    elif focus == "saving":
        sorted_df = cand_df.sort_values("절감액", ascending=False)
    elif focus == "ghg":
        sorted_df = cand_df.sort_values("온실가스감축량", ascending=False)
    else:
        logger.error(f"알 수 없는 focus: {focus}")
        raise ValueError(f"Unknown focus: {focus}")

    cols = ["cluster", "개선활동명_요약", "업종", "대상설비", "개선구분"]
    if focus == "balanced":
        cols += ["balanced_score"]
    cols += ["similarity", "투자비", "절감액", "투자비회수기간", "온실가스감축량"]
    result = sorted_df[cols].head(k).copy()

    round_cols = [
        c
        for c in [
            "투자비",
            "절감액",
            "투자비회수기간",
            "온실가스감축량",
        ]
        if c in result.columns
    ]
    result[round_cols] = result[round_cols].round(1)

    if "balanced_score" in result.columns:
        result["balanced_score"] = result["balanced_score"].round(3)
    return result.to_dict(orient="records")


def recommend_all(input_data: dict, per_k: int):
    logger.info(f"recommend_all 호출 - input_data: {input_data}, per_k: {per_k}")
    df_cand = recommend_improvements(input_data, per_k)
    solution = []

    type_mapping = {
        "balanced": "total_optimization",
        "ghg": "emission_reduction",
        "saving": "cost_saving",
        "roi": "roi",
    }

    for focus in ["balanced", "ghg", "saving", "roi"]:
        recs = recommend_by_focus(df_cand, focus, per_k)

        for idx, item in enumerate(recs, start=1):
            solution_item = {
                "id": None,
                "type": type_mapping[focus],
                "rank": idx,
                "industry": item.get("업종"),
                "improvementType": item.get("개선구분"),
                "facility": item.get("대상설비"),
                "activity": item.get("개선활동명_요약"),
                "emissionReduction": item.get("온실가스감축량"),
                "costSaving": item.get("절감액"),
                "roiPeriod": item.get("투자비회수기간"),
                "investmentCost": item.get("투자비"),
                "bookmark": None,
            }
            solution.append(solution_item)

    logger.info(f"recommend_all 반환 solution 아이템 총 개수: {len(solution)}")
    return {"solution": solution}
