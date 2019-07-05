'''
用于存放训练代码
'''
import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse

import os
import json

from model import DeepLabV3Res101
from utils import setup, restore_from, save_to, mIoU
import config

def train_on_epochs(train_loader:DataLoader, val_loader:DataLoader, ckpt:str=None):
    '''在整个数据集上进行训练
    
    Args:
        train_loader(Dataloader): 训练集加载器
        val_loader(DataLoader): 验证集加载器
        restore_from(str): 从断点恢复
    '''
    model = DeepLabV3Res101()
    model, device = setup(model)

    start_ep = 0 # 从指定的epoch开始训练
    if restore_from is not None:
        model, start_ep = restore_from(model, ckpt)

    # 训练时的各种指标
    info = {'train': [], 'val': []}

    save_path = './checkpints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 设定优化器
    optimizer = Adam(model.parameters(), lr=config.train_config['lr'])

    # 开始执行训练
    for ep in range(start_ep, config.train_config['max_epoch']):
        train_info = train(model, train_loader, optimizer, ep, device)
        val_info = validate(model, val_loader, optimizer, ep, device)
        # 保存信息
        info['train'] += train_info
        info['val'] += val_info
        # 保存模型
        save_to(model, save_path, ep)

    with open('./info.json', 'w') as f:
        json.dump(info, f)
    
    print('Done.')

def train(model:nn.Module, dataloader:DataLoader, optimizer:Optimizer, ep:int, device:torch.device):
    '''训练模型

    Args:
        model(nn.Module): 待训练模型
        dataloader(DataLoader): 数据加载器
        optimizer(Optimizer): 优化器
        device(torch.Device): 运行环境

    Return:
        train_info(list): 训练时的log
    '''
    model.train()

    train_info = []

    print('Size of training set: {}'.format(len(dataloader.dataset)))

    # 执行训练
    for step, (X, y, size) in enumerate(dataloader):
        X = X.to(device) # type: torch.Tensor
        y = y.to(device) # type: torch.Tensor
        optimizer.zero_grad()
        y_ = model(X)
        y_ = F.interpolate(y_, size=size) # 将输出上采样回原大小
        loss = F.cross_entropy(y_, y) # type: torch.Tensor
        loss.backward()
        optimizer.step()

        y_ = y_.argmax(dim=1) # TODO: 需要确认维度
        acc = accuracy_score(y_.cpu.numpy(), y.cpu().numpy())
        miou = mIoU(y_.cpu().numpy(), y.cpu().numpy())

        # 保存训练时数据
        train_info.append([ep, loss.item(), acc, miou])

        # 输出信息
        if (step + 1) % config.train_config['log_interval'] == 0:
            print('[Epoch %2d - %2d of %2d]acc: %.2f, miou: %.2f, loss: %.2f'\
                % (ep, step + 1, len(dataloader), acc, miou, loss.item()))
    return train_info

def validate(model:nn.Module, test_dataloader:DataLoader, optimizer:Optimizer, ep:int, device:torch.device):
    '''验证模型

    Args:
        model(nn.Module): 待测试模型
        dataloader(DataLoader): 数据加载器
        optimizer(Optimizer): 优化器
        device(torch.Device): 运行环境

    Return:
        test_info(list): 测试时的log
    '''
    print('Size of test set: ', len(test_dataloader))

    y_gd, y_pred = [], []
    test_info = []
    total_loss = 0

    for X, y, size in tqdm(test_dataloader, desc='Validating'):
        X, y = X.to(device), y.to(device)
        y_ = model(X)
        y_ = F.interpolate(y_, size=size)
        loss = F.cross_entropy(y_, y, reduction='sum')
        total_loss += loss.item()
        y_ = y_.argmax(dim=1)
        y_gd += y.cpu().numpy().tolist()
        y_pred += y_.cpu().numpy().tolist()

    avg_loss = total_loss / len(test_dataloader)
    acc = accuracy_score(y_gd, y_pred)
    miou = mIoU(y_gd, y_pred)
    test_info.append([ep, avg_loss, acc, miou])

    print('[Epoch %2d]Test avg loss: %.4f, acc: %.2f, mIoU: %.2f' % (ep, avg_loss, acc, miou))

    return test_info

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 train.py -i path/to/data -r path/to/checkpoint')
    parser.add_argument('-i', '--data_path', help='path to your datasets', default='./data')
    parser.add_argument('-r', '--restore_from', help='path to the checkpoint', default=None)
    args = parser.parse_args()
    return args