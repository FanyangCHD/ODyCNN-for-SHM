# -*-coding:utf-8-*-
"""
Created on 2022.3.14
programing language:python
@author:夜剑听雨
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, feature_path, label_path):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))   # glob.glob(pathname),返回所有匹配的文件路径列表。
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))       # os.path.join，拼接文件路径

    # 返回数据集大小
    def __len__(self):
        return len(self.feature_paths)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        label_data = np.load(self.label_paths[index])
        feature_data = torch.from_numpy(feature_data)  # numpy转成张量
        label_data = torch.from_numpy(label_data)                           
        feature_data.unsqueeze_(0)  # 增加一个维度  128*128 =>1*128*128
        label_data.unsqueeze_(0)
        return feature_data, label_data

if __name__ == "__main__":       #  在Python中，if __name__ == "__main__":是一个常见的代码块，
                                 #  它用于判断当前模块是否是主模块（即直接执行的脚本），如果是ZHEJIAN则执行该代码块，否则不执行。

    feature_path = "E:\SHM_Data\GNTT_DATA_missing4"
    label_path = "E:\SHM_Data\GNTT_DATA_clean4"
    seismic_dataset = MyDataset(feature_path, label_path)
    train_loader = torch.utils.data.DataLoader(dataset=seismic_dataset,     # 导入的训练集
                                               batch_size=32,               # 每批训练的样本数WOKBUGUO 
                                               shuffle=True)                # 是否打乱训练集
    # Img = train_loader.numpy().astype(np.float32)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Dataset size:', len(seismic_dataset))
    print('train_loader:', len(train_loader))
