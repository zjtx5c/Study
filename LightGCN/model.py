import torch
import torch.nn as nn
import torch.functional as F

import dgl
import dgl.function as fn


class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()
        '''
        不需要任何操作
        '''
    
    def forward(self, g, users_emb, items_emb):
        '''
        g: 表示整个图数据
        users_emb: 用户节点的嵌入   [n_users, D]
        items_emb: 物品节点的嵌入   [n_items, D]
        一次图卷积操作
        '''
        with g.local_scope():
            all_emb = torch.cat([users_emb, items_emb], dim = 0)    # [N, D]
            deg = g.out_degrees().to(users_emb.device).float().clamp(min = 1)   # [N,]
            normed_deg = torch.pow(deg, -0.5).view(-1, 1)   # [N, 1]

            g.ndata["n_feat"] = all_emb * normed_deg        # [N, D]


            g.update_all(
                message_func = fn.copy_u("n_feat", "m"),
                reduce_func = fn.sum("m", "n_feat")
            )

            n_feat = g.ndata["n_feat"] * normed_deg         # [N, D]
            return n_feat
        


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim = 64, K = 3):
        super(LightGCN, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.embedding_dim = embedding_dim
        self.K = K

        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)

        self.layers = nn.ModuleList([LightGCNConv() for _ in range(K)])

    def forward(self, g):
        users_emb = self.users_emb.weight   # [N_user, D]
        items_emb = self.items_emb.weight   # [N_item, D]

        all_embs = [torch.cat([users_emb, items_emb], dim = 0)] # [ [N, D] ]

        for i in range(self.K):
            emb = self.layers[i](g, all_embs[-1][: self.num_users], all_embs[-1][self.num_users: ])     # [N, D]
            all_embs.append(emb)                                       

        # 平均， [N, D]
        all_embs = torch.stack(all_embs, dim = 0).mean(dim = 0)
        users_emb_final = all_embs[: self.num_users]
        items_emb_final = all_embs[self.num_users: ]

        return users_emb_final, items_emb_final
    

if __name__ == '__main__':
    pass