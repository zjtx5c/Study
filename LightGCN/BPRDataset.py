from torch.utils.data import Dataset
import numpy as np

class BPRDataset(Dataset):
    def __init__(self, inter_list, item_num, matrix = None, num_neg = 1, is_training = None):
        super(BPRDataset, self).__init__()
        '''
        传入的基准数据还是交互记录
        inter_list: List
            [[user_1, item_1], [user_2, item_2], ..., [user_n, item_n]]

        matrix: sp  u2i 矩阵(R 矩阵, 非大矩阵 ADJ)

        这里默认对应一条记录只采样 1 个负样本
        '''
        ### 将值都减 1 以到达从 0 开始映射!!!
        inter_list = inter_list - 1
        ### ----------------------


        self.inter_data = np.array(inter_list)
        self.item_num = item_num
        self.u2i_matrix = matrix
        self.num_neg = num_neg
        self.is_training = is_training
    

    def neg_sample(self):
        assert self.is_training, "No sampling is required when testing"
        # 为每一组交互数据的正样本找一个负样本
        inter_matrix = self.u2i_matrix.todok()
        inter_data_length = self.inter_data.shape[0]
        self.neg_data = np.random.randint(low = 0, high = self.item_num, size = inter_data_length)
        for i in range(inter_data_length):
            uid = self.inter_data[i][0]
            iid = self.neg_data[i]
            while (uid, iid) in inter_matrix:
                iid = np.random.randint(low = 0, high = self.item_num)
            self.neg_data[i] = iid


    def __len__(self):
        return len(self.inter_data)
    
    
    def __getitem__(self, index):
        # index 是针对 交互记录表的 索引
        # 若是训练集，则需要返回对应的 user, item_pos, item_neg
        # 若是测试集，则只需要返回对应的 user, item_pos 即可
        user = self.inter_data[index][0]
        item_pos = self.inter_data[index][1]
        if self.is_training:
            item_neg = self.neg_data[index]
            return user, item_pos, item_neg
        else:
            return user, item_pos