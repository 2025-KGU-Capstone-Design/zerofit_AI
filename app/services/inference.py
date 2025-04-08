# app/services/inference.py

import torch
import numpy as np
from transformers import AutoTokenizer

# HuggingFace 토크나이저 로드 (beomi/KcELECTRA-base)
hf_tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")


def electra_tokenizer(text, max_length=10):
    tokens = hf_tokenizer.encode(
        text, add_special_tokens=True, max_length=max_length, truncation=True
    )
    return tokens


# 추천 함수: 사용자 입력과 후보 솔루션 정보를 받아 모델 예측 후, 역정규화 및 정렬하여 추천
def recommend_improvements(
    user_input: dict,
    candidate_df,
    model,
    user_cat_vocab,
    user_num_cols,
    candidate_text_cols,
    target_scaler,
    text_max_len=10,
    device="cuda",
):
    """
    user_input: dict, 예:
          {
              '업종': '제조업 - 식품',
              '대상설비': ['공기 압축기 설비'],
              '투자비': 50,
              '투자비회수기간': 3,
              '온실가스감축량': 18
          }
    candidate_df: DataFrame, 후보 솔루션 정보
    추천 기준:
      1. 투자비회수기간 낮은 순
      2. 예측 투자비 낮은 순
      3. 예측 온실가스감축량 높은 순
    """
    model.eval()
    recommendations = []

    # 사용자 입력 처리 (복수 대상설비일 경우, 첫 번째 설비 사용)
    user_input_single = user_input.copy()
    if isinstance(user_input_single["대상설비"], list):
        user_input_single["대상설비"] = user_input_single["대상설비"][0]

    # 사용자 범주형 데이터
    user_cat = {}
    for col in user_cat_vocab:
        idx = user_cat_vocab[col].get(user_input_single[col], 0)
        user_cat[col] = torch.tensor([idx], dtype=torch.long).to(device)
    user_num = torch.tensor(
        [[user_input_single[col] for col in user_num_cols]], dtype=torch.float32
    ).to(device)

    # 후보 솔루션 각각에 대해 예측 수행
    with torch.no_grad():
        for i, row in candidate_df.iterrows():
            candidate_text = {}
            for col in candidate_text_cols:
                tokens = electra_tokenizer(str(row[col]), max_length=text_max_len)
                if len(tokens) < text_max_len:
                    tokens = tokens + [0] * (text_max_len - len(tokens))
                else:
                    tokens = tokens[:text_max_len]
                candidate_text[col] = torch.tensor([tokens], dtype=torch.long).to(
                    device
                )
            # 모델 예측 (출력: (1,4): [예측투자비, 예측투자비회수기간, 예측절감액, 예측온실가스감축량])
            pred = model(user_cat, user_num, candidate_text)
            pred_array = np.array(
                [
                    [
                        pred[0, 0].item(),
                        pred[0, 1].item(),
                        pred[0, 2].item(),
                        pred[0, 3].item(),
                    ]
                ]
            )
            pred_orig = target_scaler.inverse_transform(pred_array)[0]

            recommendations.append(
                {
                    "대상설비": user_input_single["대상설비"],
                    "투자비": user_input_single["투자비"],
                    "온실가스감축량": user_input_single["온실가스감축량"],
                    "예측투자비": float(pred_orig[0]),
                    "예측투자비회수기간": float(pred_orig[1]),
                    "예측절감액": float(pred_orig[2]),
                    "예측온실가스감축량": float(pred_orig[3]),
                    "개선구분": row["개선구분"],
                    "개선활동명": row["개선활동명"],
                }
            )

    # 추천 기준별 상위 10개 항목 추출
    sorted_by_recovery = sorted(recommendations, key=lambda x: x["예측투자비회수기간"])[
        :10
    ]
    sorted_by_investment = sorted(recommendations, key=lambda x: x["예측투자비"])[:10]
    sorted_by_ghg = sorted(recommendations, key=lambda x: -x["예측온실가스감축량"])[:10]

    return {
        "투자비회수기간_상위10": sorted_by_recovery,
        "투자비낮은_상위10": sorted_by_investment,
        "온실가스감축량높은_상위10": sorted_by_ghg,
    }
