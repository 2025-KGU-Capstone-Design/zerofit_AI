import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tensorflow.keras.models import load_model
from .config import VEC_DIR, CLUSTERING_DIR
import joblib


def load_resources():
    global autoencoder, encoder, latent_vectors, df, ohe, scaler, categorical_cols, numeric_cols
    autoencoder = load_model(f"{VEC_DIR}/autoencoder_model.keras")
    encoder = load_model(f"{VEC_DIR}/encoder_model.keras")
    latent_vectors = np.load(f"{VEC_DIR}/latent_vectors.npy")
    df = pd.read_parquet(f"{CLUSTERING_DIR}/final_upscaled_with_clusters.parquet")

    ohe = joblib.load(f"{VEC_DIR}/ohe.pkl")  # ← 저장해둔 OHE 그대로 사용
    scaler = joblib.load(f"{VEC_DIR}/scaler.pkl")  # ← 저장해둔 Scaler 그대로 사용

    if hasattr(ohe, "feature_names_in_"):
        categorical_cols = list(ohe.feature_names_in_)
    else:
        categorical_cols = ["업종", "대상설비"]

    if hasattr(scaler, "feature_names_in_"):
        numeric_cols = list(scaler.feature_names_in_)
    else:
        numeric_cols = ["투자비", "절감액", "투자비회수기간", "온실가스감축량"]


# 서버 시작 시 load_resources()를 호출하도록 변경
load_resources()
