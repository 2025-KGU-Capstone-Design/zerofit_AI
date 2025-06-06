# app/services/inference.py
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


def recommend_improvements(input_data: dict, per_k: int = 10):
    """
    1) AutoEncoder + cosine similarity 로 후보군 생성
       Top-K 는 엘보 포인트(knee) 로 결정하고, 감지되지 않으면 per_k 사용
       2) 클러스터 단위로 집계(aggregation)하여 중복 제거 및 평균화
    """
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

    all_cands = []

    for facility in facilities:
        input_cat = pd.DataFrame([[industry, facility]], columns=categorical_cols)
        user_cat = ohe.transform(input_cat)

        input_num = pd.DataFrame(
            [[invest, reduction, roi_months, ghg_reduction]], columns=numeric_cols
        )
        user_num = scaler.transform(input_num)
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

        # 2) 평균값을 min–max 정규화 (투자비, 투자비회수기간은 역정규화)
        norm_means = {}
        for f, m in means.items():
            col = dfc[f]
            mn, mx = col.min(), col.max()
            if mx > mn:
                norm = (m - mn) / (mx - mn)
            else:
                norm = 0
            # cost & payback: lower is better => invert
            if f in ["투자비", "투자비회수기간"]:
                norm = 1 - norm
            norm_means[f] = norm

        # 3) 정규화된 평균값을 합산해서 가중치로 변환
        total = sum(norm_means.values())
        weights = {
            f: (norm_means[f] / total if total > 0 else 1 / len(feats)) for f in feats
        }

        # 4) 다시 전체 행을 정규화하고 balanced_score 계산
        for f in feats:
            mn, mx = dfc[f].min(), dfc[f].max()
            if mx > mn:
                norm_col = (dfc[f] - mn) / (mx - mn)
            else:
                norm_col = 0
            # invert cost & payback
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
        raise ValueError(f"Unknown focus: {focus}")

    # 공통 반환 컬럼
    cols = ["cluster", "개선활동명_요약", "업종", "대상설비", "개선구분"]
    if focus == "balanced":
        cols += ["balanced_score"]
    cols += ["similarity", "투자비", "절감액", "투자비회수기간", "온실가스감축량"]
    result = sorted_df[cols].head(k).copy()

    # 수치형(balanced_score, 투자비, 절감액, 투자비회수기간, 온실가스감축량)만 반올림
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

    # 점수 (balanced_score)도 반올림
    if "balanced_score" in result.columns:
        result["balanced_score"] = result["balanced_score"].round(3)

    return result.to_dict(orient="records")


def recommend_all(input_data: dict, per_k: int):
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

        top4 = recs[:4] if len(recs) >= 4 else recs
        comment_for_focus = None
        if len(top4) >= 2:
            comment_for_focus = generate_comparison_comment(focus, top4)

        for idx, item in enumerate(recs, start=1):
            # 1위 활동의 경우에만 comment_for_focus 사용
            score_value = (
                item.get("balanced_score") if focus == "balanced" else item.get(focus)
            )
            # 랭크가 1일 때만 comment_for_focus, 아니면 None
            comment_value = comment_for_focus if idx == 1 else None

            solution_item = {
                "id": None,
                "type": type_mapping[focus],
                "rank": idx,
                "score": score_value,
                "industry": item.get("업종"),
                "improvementType": item.get("개선구분"),
                "facility": item.get("대상설비"),
                "activity": item.get("개선활동명_요약"),
                "emissionReduction": item.get("온실가스감축량"),
                "costSaving": item.get("절감액"),
                "roiPeriod": item.get("투자비회수기간"),
                "investmentCost": item.get("투자비"),
                "bookmark": None,
                "comment": comment_value,
            }
            solution.append(solution_item)

    return {"solution": solution}
