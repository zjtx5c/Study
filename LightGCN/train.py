import os
import argparse

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl

from utils import load_data, EarlyStopping_simple, calc_innerProduct
from model import LightGCN
from BPRDataset import BPRDataset

from tqdm import tqdm

# 我们这里的任务只是一个简单的 DEMO，还未做出推荐，只是找到模型能训练出更高质量的初始 `embedding`
# 并未做具体的推荐任务


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_data(args.raw_data_root)
    u2i_mat = sp.load_npz(args.u2i_data_file)
    
    num_users, num_items = u2i_mat.shape
    i2u_mat = u2i_mat.T

    # 这里默认是 0 矩阵，可以根据要求进行适当扩充
    u2u_mat = sp.csr_matrix((num_users, num_users))
    i2i_mat = sp.csr_matrix((num_items, num_items))
    ADJ = sp.vstack([sp.hstack([u2u_mat, u2i_mat]), sp.hstack([i2u_mat, i2i_mat])])

    graph = dgl.from_scipy(ADJ, idtype = torch.int32, device = device)

    train_dataset = dataset["train"]
    val_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    train_dataset = BPRDataset(train_dataset["inter"], num_items, train_dataset["matrix"], is_training = True)
    # 对训练集进行负采样
    train_dataset.neg_sample()
    val_dataset = BPRDataset(val_dataset["inter"], num_items, val_dataset["matrix"])
    test_dataset = BPRDataset(test_dataset["inter"], num_items, test_dataset["matrix"])

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    train_step = len(train_dataset) // args.batch_size + 1

    stopper = EarlyStopping_simple(patience = args.patience, save_path = os.path.join(args.model_save_path, "LightGCN_model.pt"))
    model = LightGCN(num_users, num_items, embedding_dim = args.embedding_dim, K = args.K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)


    best_val_score = 0.0
    for e in range(args.epoch):
        epoch_loss = 0.0
        model.train()
        for user, item_pos, item_neg in tqdm(train_dataloader, total = len(train_dataloader)):
            user_id = user.long().to(device)
            item_pos_id = item_pos.long().to(device)
            item_neg_id = item_neg.long().to(device)
            all_users_embed, all_items_embed = model(graph)


            user_embed = all_users_embed[user_id]       # [B, D]
            item_pos_embed = all_items_embed[item_pos_id]
            item_neg_embed = all_items_embed[item_neg_id]


            # 分别计算 user 与 正样本 和 负样本的 相似度
            pred_i, pred_j = calc_innerProduct(user_embed, item_pos_embed, item_neg_embed)  # [B,]

            bpr_loss = -torch.sum(torch.log(torch.sigmoid(pred_i - pred_j)))    # [1]
            reg_loss = (torch.norm(user_embed) ** 2 + torch.norm(item_pos_embed) ** 2 + torch.norm(item_neg_embed) ** 2)    # [1]

            loss = 0.5 * (bpr_loss + args.reg * reg_loss) / args.batch_size

            epoch_loss = epoch_loss + bpr_loss.item() / train_step

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            scores = []
            all_users_embed, all_items_embed = model(graph)

            for user, item in val_dataloader:
                user_id = user.long().to(device)    # [B,]
                item_id = item.long().to(device)    # [B,]

                user_embed = all_users_embed[user_id]   # [B, D]
                item_embed = all_items_embed[item_id]   # [B, D]

                sim = F.cosine_similarity(user_embed, item_embed, dim = 1)
                scores.append(sim.cpu().numpy())
            
            val_score = np.mean(np.concatenate(scores))
            best_val_score = max(best_val_score, val_score)
            stop = stopper.step(val_score, e, model)
            if stop:
                print('best epoch :', stopper.best_epoch)
                break
        
        print(f"In epoch {e}, Train_loss: {epoch_loss}, Val_score: {val_score:.4f}, Best_Val_score: {best_val_score:.4f}")
        

    # 测试模式
    model = LightGCN(num_users, num_items, embedding_dim = args.embedding_dim, K = args.K).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, "LightGCN_model.pt")))
    model.eval()
    with torch.no_grad():
        scores = []
        all_users_embed, all_items_embed = model(graph)
        
        for user, item in test_dataloader:
            user_id = user.long().to(device)    # [B,]
            item_id = item.long().to(device)    # [B,]

            user_embed = all_users_embed[user_id]   # [B, D]
            item_embed = all_items_embed[item_id]   # [B, D]

            sim = F.cosine_similarity(user_embed, item_embed, dim = 1)
            scores.append(sim.cpu().numpy())
        
        test_score = np.mean(np.concatenate(scores))

    print(f"test_score: {test_score:.4f}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "LightGCN Training")
    
    parser.add_argument("--raw-data-root", type = str, default = r"D:\PY\JUPYTER\study\datasets\LightGCN\phone\raw")
    parser.add_argument("--u2i-data-file", type = str, default = r"D:\PY\JUPYTER\study\datasets\LightGCN\phone\inter.npz")
    parser.add_argument("--model-save-path", type = str, default = r"D:\PY\JUPYTER\study\LightGCN\ckpt")

    parser.add_argument("--epoch", type = int, default = 10000)
    parser.add_argument("--batch-size", type = int, default = 4096)
    parser.add_argument("--embedding-dim", type = int, default = 64)
    parser.add_argument("--K", type = int, default = 3)
    parser.add_argument("--patience", type = int, default = 100)

    parser.add_argument("--lr", type = float, default = 0.05)
    parser.add_argument("--weight-decay", type = float, default = 0)
    parser.add_argument("--reg", type = float, default = 0.001)

    args = parser.parse_args()
    main(args)