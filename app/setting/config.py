import os

# 현재 파일의 상위 디렉토리 경로 (프로젝트 루트 또는 app 폴더 기준)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 모델 파일 경로
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "two_tower_improvement_model.pth"),
)

# 데이터셋 파일 경로
TRAIN_DATA_PATH = os.getenv(
    "TRAIN_DATA_PATH", os.path.join(BASE_DIR, "data", "train_data.csv")
)
