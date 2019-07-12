'''
用于存放数据读取程序
'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFilter
import numpy as np

import config

class SegDataset(Dataset):
    def __init__(self, data_paths:list, label_paths:list, name:str, mean:list, std:list,
            use_jitter:bool = False):
        '''自定义数据集

        Args:
            data_paths(str): 图像路径列表
            label_paths(str): 标签路径列表
            name(str): 数据集名字
            mean, str(list): 均值和方差
            use_jitter(bool): 是否使用jitter数据增强，默认False
        '''
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.name = name
        self.mean = mean
        self.std = std
        self.use_jitter = use_jitter

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
        if self.name == 'train':
            X, y = self.transform(img, label)
        elif self.name == 'val':
            X, y = self.transform_on_eval(img, label)
        else:
            raise NotImplementedError
        return X, y, img.size

    def transform(self, img:Image.Image, label:Image.Image)->(torch.Tensor, torch.Tensor):
        label = label.resize((config.im_w, config.im_h), resample=Image.NEAREST) # type: Image.Image
        img = img.resize((config.im_w, config.im_h), resample=Image.BILINEAR) # type: Image.Image

        # # 随机高斯模糊
        # if np.random.rand() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=np.random.rand()))

        # 随机旋转一定角度或者翻转
        for flag in [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]:
            if np.random.rand() < 0.5: continue
            img = img.transpose(flag)
            label = label.transpose(flag)
        transform_funcs = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        if self.use_jitter:
            params = self.get_random_color_jitter_params() # 随机调整亮度、对比度、色调等数值
            transform_funcs.insert(0, transforms.ColorJitter(**params))
        return transforms.Compose(transform_funcs)(img), torch.Tensor(np.array(label, dtype=np.uint8)).long()

    def transform_on_eval(self, img:Image.Image, label:Image.Image = None)->(torch.Tensor, torch.Tensor):
        img = img.resize((config.im_w, config.im_h), resample=Image.BILINEAR) # type: Image.Image
        if label is not None:
            label = label.resize((config.im_w, config.im_h), resample=Image.NEAREST) # type: Image.Image
            label = torch.Tensor(np.array(label, dtype=np.uint8)).long()

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])(img), label

    def get_random_color_jitter_params(self)->dict:
        '''随机生成ColorJitter函数的参数
        '''
        params = {}
        for key in ['brightness', 'contrast', 'saturation', 'hue']:
            params[key] = np.random.rand() * 0.1
        return params