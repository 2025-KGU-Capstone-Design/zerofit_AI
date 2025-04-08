# app/models/model_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(
        self,
        user_cat_vocab_sizes,
        user_cat_embed_dims,
        num_numerical_features,
        out_dim=64,
    ):
        super(UserTower, self).__init__()
        self.cat_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(
                    user_cat_vocab_sizes[col], user_cat_embed_dims.get(col, 8)
                )
                for col in user_cat_vocab_sizes
            }
        )
        total_cat_dim = sum(
            user_cat_embed_dims.get(col, 8) for col in user_cat_vocab_sizes
        )
        total_input_dim = total_cat_dim + num_numerical_features
        self.fc1 = nn.Linear(total_input_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, user_cat, user_num):
        embeds = []
        for col in user_cat:
            emb = self.cat_embeddings[col](user_cat[col])
            embeds.append(emb)
        x_cat = torch.cat(embeds, dim=1)
        x = torch.cat([x_cat, user_num], dim=1)
        x = F.relu(self.fc1(x))
        user_embedding = F.relu(self.fc2(x))
        return user_embedding


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class HierarchicalCandidateTower(nn.Module):
    def __init__(self, text_vocab_size, text_embed_dim, max_len, out_dim=64):
        super(HierarchicalCandidateTower, self).__init__()
        self.text_embedding = nn.Embedding(
            text_vocab_size, text_embed_dim, padding_idx=0
        )
        self.attention = AttentionLayer(text_embed_dim)
        self.max_len = max_len
        self.fc_group = nn.Linear(text_embed_dim, 128)
        self.fc_leaf = nn.Linear(text_embed_dim, 128)
        self.fc_weight = nn.Linear(128, 1)
        self.fc_combine = nn.Linear(128, out_dim)

    def forward(self, candidate_text):
        emb_group_tokens = self.text_embedding(candidate_text["개선구분"])
        emb_group = self.attention(emb_group_tokens)
        emb_group = F.relu(self.fc_group(emb_group))
        emb_leaf_tokens = self.text_embedding(candidate_text["개선활동명"])
        emb_leaf = self.attention(emb_leaf_tokens)
        emb_leaf = F.relu(self.fc_leaf(emb_leaf))
        weight = torch.sigmoid(self.fc_weight(emb_group))
        combined = emb_group + weight * emb_leaf
        candidate_embedding = F.relu(self.fc_combine(combined))
        return candidate_embedding


class TwoTowerImprovementModel(nn.Module):
    def __init__(self, user_tower, candidate_tower, joint_hidden=64):
        super(TwoTowerImprovementModel, self).__init__()
        self.user_tower = user_tower
        self.candidate_tower = candidate_tower
        self.fc_joint1 = nn.Linear(
            user_tower.fc2.out_features + candidate_tower.fc_combine.out_features,
            joint_hidden,
        )
        self.fc_joint2 = nn.Linear(joint_hidden, 4)  # 4개 회귀 출력

    def forward(self, user_cat, user_num, candidate_text):
        user_emb = self.user_tower(user_cat, user_num)
        candidate_emb = self.candidate_tower(candidate_text)
        joint_input = torch.cat([user_emb, candidate_emb], dim=1)
        x = F.relu(self.fc_joint1(joint_input))
        output = self.fc_joint2(x)
        return output
