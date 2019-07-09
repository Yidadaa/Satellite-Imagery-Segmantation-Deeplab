'''
用于存放数据读取程序
'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np

import config

class SegDataset(Dataset):
    def __init__(self, data_paths:list, label_paths:list, mean:list, std:list):
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

    def transformX(self, img:Image.Image)->torch.Tensor:
        img = img.resize((config.im_w, config.im_h), resample=Image.BILINEAR)
        params = self.get_random_color_jitter_params()
        return transforms.Compose([
            transforms.ColorJitter(**params),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])(img)

    def transformY(self, label:Image.Image)->Image.Image:
        label = label.resize((config.im_w, config.im_h), resample=Image.NEAREST) # type: Image.Image
        return torch.Tensor(np.array(label, dtype=np.uint8)).long()

    def get_random_color_jitter_params(self)->dict:
        '''随机生成ColorJitter函数的参数
        '''
        params = {}
        for key in ['brightness', 'contrast', 'saturation', 'hue']:
            params[key] = np.random.rand() * 0.1
        return params