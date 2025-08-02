import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import save_npz


def GetAllData_Df(raw_data_path, save_path):
    files = os.listdir(raw_data_path)

    files = [f for f in files if os.path.isfile(os.path.join(raw_data_path, f))]

    all_df = []
    for f in files:
        df = pd.read_csv(os.path.join(raw_data_path, f))
        df = df[["user_id", "item_id"]]
        all_df.append(df)
    all_df = pd.concat(all_df, axis = 0)
    all_df.to_csv(os.path.join(save_path, "inter.csv"), index = False)


def GetAllData_Metrix(file_path, save_path):
    '''
    file_path: pd.DataFrame
    将 .csv 转化成 .npz
    '''

    df = pd.read_csv(file_path)
    # df = df[["user_id", "item_id"]]
    row_indices = []
    col_indices = []
    data = []
    print("开始将数据转化为稀疏矩阵...")
    for _, row in tqdm(df.iterrows(), total = len(df)):
        u_id = row["user_id"] - 1
        i_id = row["item_id"] - 1
        d = 1 if "rating" not in row.keys() else row["rating"]
        row_indices.append(u_id)
        col_indices.append(i_id)
        data.append(d)

    users_num = max(row_indices) + 1
    items_num = max(col_indices) + 1
    matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape = (users_num, items_num))
    save_npz(os.path.join(save_path, "inter.npz"), matrix)
    print(f"数据转化与保存成功！")



def main(args):
    GetAllData_Df(args.raw_data_path, args.save_path)

    df_path = os.path.join(args.save_path, "inter.csv")
    matrix_save_path = args.save_path
    GetAllData_Metrix(df_path, matrix_save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "dataProcess.py")
    
    parser.add_argument('--raw-data-path', type = str, default = r"D:\PY\JUPYTER\study\datasets\LightGCN\phone\raw")
    parser.add_argument('--save-path', type = str, default = r"D:\PY\JUPYTER\study\datasets\LightGCN\phone")

    args = parser.parse_args()
    main(args)
