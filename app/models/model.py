from tensorflow.keras.models import load_model
import numpy as np

from ..setting.config import VEC_DIR


def load_autoencoder_models():
    """AutoEncoder 모델과 latent 벡터를 로드합니다."""
    ae = load_model(f"{VEC_DIR}/autoencoder_model.keras")
    enc = load_model(f"{VEC_DIR}/encoder_model.keras")
    vectors = np.load(f"{VEC_DIR}/latent_vectors.npy")
    return ae, enc, vectors
