import os

from torch import nn, Tensor, torch
from torch.utils.data import DataLoader
import numpy as np

from dataloader import SegDataset
import config

def setup(model:nn.Module)->nn.Module:
    '''配置模型的GPU/CPU环境
    
    Args:
        model(nn.Module): 需要设置的模型

    Return:
        model(nn.Module): 配置好的模型
        device(torch.device): 模型可用的环境
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)

    # 检测多GPU环境
    if use_cuda:
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            model = nn.DataParallel(model)

    return model, device

def restore_from(model:nn.Module, restore_from:str)->nn.Module:
    '''从存档点恢复模型

    Args:
        model(nn.Module): 模型
        restore_from(str): 存档路径

    Return:
        model(nn.Module): 恢复的模型
    '''
    assert os.path.exists(restore_from), '不存在的路径！{}'.format(restore_from)

    ckpt = torch.load(restore_from)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt['epoch']


def save_to(model:nn.Module, path:str, ep:int):
    '''保存模型到指定路径

    Args:
        model(nn.Module): 模型
        path(str): 存档路径
        ep(int): 当前所处的epoch
    '''
    ckpt_path = os.path.join(path, 'ep-%d.pth' % ep)
    torch.save({
        'epoch': ep,
        'model_state_dict': model.state_dict()
    }, ckpt_path)
    print('Model trained after %d epochs has been saved to: %s.' % (ep, ckpt_path))

def build_dataloader(data_path:str)->DataLoader:
    '''构建数据集加载器

    Args:
        data_path(str): 数据集所在位置

    Return:
        dataloader(Dataloader): pytorch的数据加载器
    '''
    assert os.path.exists(data_path), '不存在的路径！{}'.format(data_path)

    folders = os.listdir(data_path)
    assert set(folders) == set(['img', 'label']), '请检查数据集是否完整'

    img_list = os.listdir(os.path.join(data_path, 'img'))
    label_list = os.listdir(os.path.join(data_path, 'label'))

    the_dataset = SegDataset(img_list, label_list, **config.dataset_config)
    return DataLoader(the_dataset, **config.dataloader_config)

def mIoU(a:np.ndarray, b:np.ndarray)->float:
    '''计算两者的mIoU
    '''
    # TODO: 需要实现
    return 0.0
