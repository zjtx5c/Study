import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch

def load_data(data_root = None):
    '''
    data_root: path
                传入 train, valid, test 的路径

    '''

    dataset = {}
    files = os.listdir(data_root)
    files = [f for f in files if os.path.isfile(os.path.join(data_root, f))]
    for file in files:
        df = pd.read_csv(os.path.join(data_root, file))
        df = df[["user_id", "item_id"]]
        
        row_indices = []
        col_indices = []
        value = [1] * len(df)
        for _, row in df.iterrows():
            uid = row["user_id"] - 1
            iid = row["item_id"] - 1
            row_indices.append(uid)
            col_indices.append(iid)

        max_users_num = max(row_indices) + 1
        max_items_num = max(col_indices) + 1
        matrix = sp.csr_matrix((value, (row_indices, col_indices)), shape = (max_users_num, max_items_num))

        for typ in ["train", "valid", "test"]:
            if typ not in file:
                continue
            dataset[typ] = {}
            dataset[typ]["inter"] = df
            dataset[typ]["matrix"] = matrix
    
    return dataset



def calc_innerProduct(user, item_i, item_j):
    '''
    user, item_i, item_j 均为 [1, D] 的 tensor
    '''

    pred_i = torch.mul(user, item_i).sum(dim = -1)
    pred_j = torch.mul(user, item_j).sum(dim = -1)
    return pred_i, pred_j



class EarlyStopping_simple:
    def __init__(self, patience = 50, save_path = None, min_epoch = -1):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.save_path = save_path
        self.min_epoch = min_epoch

    def step(self, acc, epoch, model):
        score = acc
        if epoch < self.min_epoch:
            return self.early_stop
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop
    
    def step_regression(self, acc, epoch, model):
        score = acc
        if epoch < self.min_epoch:
            return self.early_stop
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.save_path)


