import os
import torch
import pandas as pd
from sklearn.preprocessing import RobustScaler
from app.models.model import load_model
from app.setting.config import MODEL_PATH, TRAIN_DATA_PATH


def load_resources(app):
    # 1. 디바이스 결정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device

    # 2. 모델 로드
    app.state.model = load_model(MODEL_PATH, device)

    # 3. 후보 데이터셋 및 보조 데이터 로드
    df = pd.read_csv(TRAIN_DATA_PATH, encoding="cp949")

    # 숫자형 컬럼 정제
    for col in ["투자비", "온실가스감축량"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # 보조 데이터 설정
    user_cat_cols = ["업종", "대상설비"]
    user_num_cols = ["투자비", "온실가스감축량"]
    candidate_text_cols = ["개선구분", "개선활동명"]
    target_cols = ["투자비", "투자비회수기간", "절감액", "온실가스감축량"]

    # 사용자 범주형 변수 사전 생성
    user_cat_vocab = {}
    for col in user_cat_cols:
        unique_vals = df[col].unique().tolist()
        user_cat_vocab[col] = {val: idx for idx, val in enumerate(unique_vals)}
    app.state.user_cat_vocab = user_cat_vocab

    # 타깃 스케일러 준비 (역정규화를 위해)
    target_scaler = RobustScaler()
    target_scaler.fit(df[target_cols].values)
    app.state.target_scaler = target_scaler

    # 기타 보조 데이터 저장
    app.state.candidate_df = df
    app.state.user_num_cols = user_num_cols
    app.state.candidate_text_cols = candidate_text_cols

    print("Startup complete: 모델 및 보조 데이터가 로드되었습니다.")
