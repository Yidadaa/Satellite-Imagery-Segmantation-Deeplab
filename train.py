'''
用于存放训练代码
'''
import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import argparse
import os
import json
import time

from dataloader import SegDataset
from model import DeepLabV3Res101, DeepLabV3Res50
from utils import setup, get_metrics, restore_from, save_to
from losses import FocalLoss2d, CrossEntropyLoss2d, mIoULoss2d, LovaszLoss2d
import config

class Trainer(object):
    def __init__(self, data_path:str, ckpt_path:str = None, model_name:str = 'deeplab',
            loss_name:str = 'focal', log_dir:str = 'logs', lr:float = 1e-3,
            weight:list = None, max_epoch:int = 30):
        '''
        Args:
            data_path(str): 训练数据存放位置
            ckpt_path(str): 存档点文件存放位置
        '''
        self.date_id = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        self.model_name = model_name
        self.loss_name = loss_name
        self.lr = lr
        self.max_epoch = max_epoch
        self.weight = torch.Tensor(weight) if weight is not None else weight
        self.train_loader, self.val_loader = self.setup_dataloader(data_path)
        self.model, self.device, self.start_ep = self.setup_model(model_name, ckpt_path)
        self.ckpt = ckpt_path
        self.optimizer = Adam(self.model.parameters(), self.lr)
        self.criterion = self.setup_loss(loss_name)
        self.writer = SummaryWriter(self.get_log_dir(log_dir), flush_secs=30)

    def get_log_dir(self, log_dir:str):
        return os.path.join(log_dir, self.get_comment())

    def get_comment(self):
        weight_flag = 'noweight' if self.weight is None else 'withweight'
        return '{}_{}_startfrom_{}_{}_{}_{}'.format(self.model_name, self.loss_name,
            self.start_ep, self.lr, weight_flag, self.date_id)

    def setup_loss(self, loss_name:str):
        '''配置loss函数
        '''
        loss_map = {
            'focal': FocalLoss2d,
            'ce': CrossEntropyLoss2d,
            'miou': mIoULoss2d,
            'lovasz': LovaszLoss2d
        }
        if loss_name not in loss_map:
            raise NotImplementedError(loss_name)
        params = {}
        if loss_name in ['focal', 'ce', 'miou'] and self.weight is not None:
            params['weight'] = self.weight
        loss = loss_map[loss_name](**params).to(self.device)
        return loss

    def setup_model(self, model_name:str, ckpt_path:str)->(nn.Module, int):
        '''配置模型

        Args:
            model_name(str): 模型名字
            ckpt_path(str): 模型检查点路径
        '''
        model_map = {
            'deeplab': DeepLabV3Res101,
            'deeplabres50': DeepLabV3Res50
        }
        if model_name not in model_map:
            raise NotImplementedError(model_name)
        model = model_map[model_name]()
        model, device = setup(model)

        start_ep = 0
        if ckpt_path is not None:
            model, start_ep = restore_from(model, ckpt_path)

        return model, device, start_ep

    def setup_dataloader(self, train_path:str):
        '''构建训练用数据集，包含训练集和测试集

        Args:
            train_path(str): 训练数据集路径
        
        Return:
            train_loader(Dataloader): 训练集加载器
            val_loader(Dataloader): 验证集加载器
        '''
        list_full_path = lambda path: [os.path.join(path, f) for f in os.listdir(path)]
        split_size = 0.2 # 选取0.2作为验证集
        X_list = list_full_path(os.path.join(train_path, 'img'))
        y_list = list_full_path(os.path.join(train_path, 'label'))
        X_train, X_val, y_train, y_val = train_test_split(X_list, y_list, test_size=split_size)
        dataset = {}
        for name, data_list, label_list in [('train', X_train, y_train), ('val', X_val, y_val)]:
            dataset[name] = DataLoader(SegDataset(data_list, label_list, name, **config.dataset_config), **config.dataloader_config)
        return dataset['train'], dataset['val']

    def train_on_epochs(self, start_ep:int = 0):
        '''在整个数据集上进行训练
        '''
        print('Size of training set: {}'.format(len(self.train_loader.dataset)))
        print('Size of val set: {}'.format(len(self.val_loader.dataset)))
        # 开始执行训练
        for ep in range(self.start_ep, self.max_epoch):
            self.train(ep)
            self.validate(ep)
            self.save_model(ep)
        self.writer.close()

    def train(self, ep:int):
        '''训练模型

        Args:
            ep(int): 当前epoch
        '''
        self.model.train()
        size = len(self.train_loader)
        # 执行训练
        for step, (X, y, _) in tqdm(enumerate(self.train_loader), desc='Epoch {:3d}'.format(ep), total=size):
            X = X.to(self.device) # type: torch.Tensor
            y = y.to(self.device) # type: torch.Tensor
            self.optimizer.zero_grad()
            y_ = self.model(X) # type: torch.Tensor
            loss = self.criterion(y_, y) # type: torch.Tensor
            loss.backward()
            self.optimizer.step()

            y_ = y_.argmax(dim=1).cpu().numpy()
            y = y.cpu().numpy()
            
            # 计算运行时指标
            miou, _, pacc = get_metrics(y, y_)
            # 输出到tensorboard
            n_iter = ep * size + step
            self.writer.add_scalar('train/pacc', pacc, n_iter)
            self.writer.add_scalar('train/mIoU', miou, n_iter)
            self.writer.add_scalar('train/loss', loss.item(), n_iter)

    def validate(self, ep:int):
        '''验证模型

        Args:
            ep(int): 当前epoch
        '''
        mious, paccs = [], []
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for X, y, _ in tqdm(self.val_loader, desc='Validating'):
                X, y = X.to(self.device), y.to(self.device)
                y_ = self.model(X)
                loss = self.criterion(y_, y)
                total_loss += loss.item()
                y_ = y_.argmax(dim=1)
                y_gd = y.cpu().numpy()
                y_pred = y_.cpu().numpy()
                miou, _, pacc = get_metrics(y_gd, y_pred)
                mious.append(miou)
                paccs.append(pacc)

        avg_loss = total_loss / len(self.val_loader)
        miou = np.average(mious)
        pacc = np.average(paccs)

        print(ep, miou, pacc)

        # 输出信息
        self.writer.add_scalar('test/pacc', pacc, ep)
        self.writer.add_scalar('test/mIoU', miou, ep)
        self.writer.add_scalar('test/avg_loss', avg_loss, ep)

    def save_model(self, ep:int, save_path:str = 'checkpoints'):
        '''保存模型
        '''
        save_to(self.model, os.path.join(save_path, self.get_comment()), ep)

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -t path/to/train/data -v path/to/val/data -r path/to/checkpoint')
    parser.add_argument('-t', '--train_path', help='path to your datasets', default='./data/train')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    names = ['train', 'val']
    args = parse_args()
    assert os.path.exists(args.train_path), '请指定训练数据集路径'
    params = {
        'loss_name': 'focal',
        'weight': [1 - x for x in [0.6826871849591719, 0.09160297945818623, 0.07519184557703008, 0.15051799000561178]],
        'lr': 1e-4,
        'max_epoch': 30
    }
    trainer = Trainer(args.train_path, args.restore_from, **params)
    trainer.train_on_epochs()