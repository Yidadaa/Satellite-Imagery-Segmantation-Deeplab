'''
用于存放数据读取程序
'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import config

class SegDataset(Dataset):
    def __init__(self, data_paths:list, label_paths:list, mean:tuple, std:tuple):
        '''自定义数据集
        '''
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index:int):
        '''根据索引获取数据

        Args:
            index(int): 索引

        Return:
            x(tensor): 训练数据
            y(tensor): 标签
            size(tuple): 图片的尺寸
        '''
        ipath = self.data_paths[index]
        lpath = self.label_paths[index]
        img = Image.open(ipath).convert('RGB')
        label = Image.open(lpath)
        return self.transformX(img), self.transformY(label), img.size

    def transformX(self, img):
        return transforms.Compose([
            transforms.Resize((config.im_w, config.im_h)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])(img)

    def transformY(self, label):
        return transforms.Compose([transforms.ToTensor()])(label)