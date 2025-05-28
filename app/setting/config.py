import os

# 현재 파일의 상위 디렉토리 경로 (프로젝트 루트 또는 app 폴더 기준)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 모델 파일 경로
VEC_DIR = os.getenv(
    "VEC_DIR",
    os.path.join(BASE_DIR, "data2"),
)
CLUSTERING_DIR = os.getenv(
    "CLUSTERING_DIR",
    os.path.join(BASE_DIR, "data2"),
)
