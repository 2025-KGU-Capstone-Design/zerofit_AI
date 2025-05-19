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


def recommend_improvements(input_data: dict, default_k: int = 10) -> list:
    """
    사용자 입력(input_data)에 대한 추천 결과를 반환합니다.
    """
    # 1) 입력 전처리
    user_cat = ohe.transform([[input_data["업종"], input_data["대상설비"]]])
    user_num = scaler.transform(
        [
            [
                input_data["투자비"],
                input_data["절감액"],
                input_data["투자비회수기간"],
                input_data["온실가스감축량"],
            ]
        ]
    )
    user_vec = np.hstack([user_cat, user_num]).astype("float32")

    # 2) 잠재벡터 예측
    user_latent = encoder.predict(user_vec)

    # 3) 코사인 유사도 계산
    cos_sim = cosine_similarity(user_latent, latent_vectors)[0]

    # 4) 엘보 포인트로 k 결정
    sims_sorted = np.sort(cos_sim)[::-1]
    ranks = np.arange(1, len(cos_sim) + 1)
    kneedle = KneeLocator(ranks, sims_sorted, curve="convex", direction="decreasing")
    top_k = kneedle.knee if kneedle.knee is not None else default_k

    # 5) Top-K 후보 추출
    idxs = np.argsort(cos_sim)[-top_k:][::-1]
    candidates = df.iloc[idxs].copy()
    candidates["similarity"] = cos_sim[idxs]

    return candidates.to_dict(orient="records")
