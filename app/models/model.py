import torch
from app.models.model_arch import (
    UserTower,
    HierarchicalCandidateTower,
    TwoTowerImprovementModel,
)


def load_model(save_path: str, device):
    # 학습 시 사용했던 범주형 변수의 vocab 크기 (체크포인트에 맞게 수정)
    user_cat_vocab_sizes = {"업종": 13, "대상설비": 24}
    user_cat_embed_dims = {"업종": 8, "대상설비": 8}
    num_numerical_features = 2  # 예: '투자비', '온실가스감축량'
    user_out_dim = 64
    text_vocab_size = 30000  # 학습 시 사용한 값에 맞춰야 함
    text_embed_dim = 32
    candidate_out_dim = 64

    # 모델 인스턴스 생성
    user_tower = UserTower(
        user_cat_vocab_sizes,
        user_cat_embed_dims,
        num_numerical_features,
        out_dim=user_out_dim,
    )
    candidate_tower = HierarchicalCandidateTower(
        text_vocab_size, text_embed_dim, max_len=10, out_dim=candidate_out_dim
    )
    model = TwoTowerImprovementModel(user_tower, candidate_tower, joint_hidden=64)

    # 저장된 weight 불러오기
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
