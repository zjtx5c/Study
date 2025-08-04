# import torch

# def metrics(user_id, item_id, all_users_embed, all_items_embed, top_k):
#     '''
#     输入的是一个批次的数据
#     user_id: [B,]
#     item_id: [B,]
#     all_users_embed: [n_users, D]
#     all_items_embed: [n_items, D]
    
#     返回:
#         hit_total: 命中数量 (int)
#         ndcg_total: NDCG 累计值 (float)
#     '''


#     device = all_users_embed.device  # 获取模型 embedding 所在设备

#     user_embed = all_users_embed[user_id]  # [B, D]
#     candidate_scores = torch.matmul(user_embed, all_items_embed.t())  # [B, n_items]
#     topk_indices = torch.topk(candidate_scores, k = top_k, dim = 1).indices  # [B, K]

#     hits_mask = (item_id.unsqueeze(1) == topk_indices)  # [B, K]
#     hit_total = hits_mask.any(dim = 1).sum().item()

#     # 保证 rank 在同一设备
#     rank = torch.full((user_id.size(0),), -1, dtype=torch.long, device=device)
#     hit_rows = hits_mask.any(dim=1)
#     rank[hit_rows] = hits_mask[hit_rows].int().argmax(dim=1)

#     valid_ranks = rank[rank != -1].float()
#     ndcg_total = torch.sum(1.0 / torch.log2(valid_ranks + 2.0)).item()

#     return hit_total, ndcg_total

import torch
import numpy as np

def metrics(user_id, item_id, all_users_embed, all_items_embed, top_k, train_matrix = None):
    '''
    user_id: [B,]
    item_id: [B,]
    all_users_embed: [n_users, D]
    all_items_embed: [n_items, D]
    train_matrix: scipy.sparse.csr_matrix 用户历史交互矩阵 (num_users, num_items)
    '''
    device = all_users_embed.device

    user_embed = all_users_embed[user_id]  # [B, D]
    candidate_scores = torch.matmul(user_embed, all_items_embed.t())  # [B, n_items]

    # ---- 屏蔽用户历史交互 ----
    if train_matrix is not None:
        user_np = user_id.cpu().numpy()
        for row_idx, u in enumerate(user_np):
            interacted_items = train_matrix[u].indices  # 历史交互物品ID
            candidate_scores[row_idx, interacted_items] = -1e9  # 设为极小值

    # ---- 计算 Top-K ----
    topk_indices = torch.topk(candidate_scores, k = top_k, dim = 1).indices  # [B, K]

    hits_mask = (item_id.unsqueeze(1) == topk_indices)  # [B, K]
    hit_total = hits_mask.any(dim=1).sum().item()

    rank = torch.full((user_id.size(0),), -1, dtype=torch.long, device = device)
    hit_rows = hits_mask.any(dim = 1)
    rank[hit_rows] = hits_mask[hit_rows].int().argmax(dim = 1)

    valid_ranks = rank[rank != -1].float()
    ndcg_total = torch.sum(1.0 / torch.log2(valid_ranks + 2.0)).item()

    return hit_total, ndcg_total
